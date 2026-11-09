import json
import sqlite3
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.catalog import fixture_catalog
from trustworthy_recsys.serving.components import FixtureRetrieval, PassThroughReranker
from trustworthy_recsys.serving.config import Settings
from trustworthy_recsys.serving.contracts import FeedbackEvent, PreferenceState, RecommendationRequest
from trustworthy_recsys.serving.feedback import FeedbackStore, FeedbackError, probabilities, WINDOW, LATE_ALLOWANCE
from trustworthy_recsys.serving.service import RecommendationService
from trustworthy_recsys.ope.movie import prepare, evaluate, outcome, assess, fit


def request(key=None):
    return RecommendationRequest(session_id='session-test', preference_revision=0, preferences=PreferenceState(),
                                 result_count=5, assignment_key=key)


def service(tmp_path, clock, mode='randomized'):
    catalog = fixture_catalog()
    store = FeedbackStore(tmp_path/'movie.sqlite3', mode=mode, clock=lambda: clock[0])
    return RecommendationService(catalog, FixtureRetrieval.build(catalog), PassThroughReranker(), 'fixture', 100, store)


def event(assignment, kind='details_open', elapsed=1000, **kwargs):
    return FeedbackEvent(event_id=str(uuid.uuid4()), decision_id=assignment.decision_id, token=assignment.token,
                         movie_id=assignment.movie_id, kind=kind, elapsed_ms=elapsed, **kwargs)


def test_known_probabilities_and_fixed_grid_sampling(tmp_path):
    assert probabilities(10, .1) == pytest.approx([.91]+[.01]*9)
    assert probabilities(1, .1) == [1.]
    clock = [1_800_000_000.]
    rec = service(tmp_path, clock)
    fixed, counts = None, {}
    for _ in range(250):
        response = rec.recommend(request())
        ids = [r.movie.movie_id for r in response.recommendations]
        fixed = fixed or ids[1:]
        assert ids[1:] == fixed and ids[0] not in fixed
        assert len(ids) == len(set(ids)) == 5
        counts[ids[0]] = counts.get(ids[0], 0)+1
        assert response.feedback.synthetic
    # Broad, non-flaky bound; distinguishes missing exploitation/marginal errors.
    assert 180 < max(counts.values()) < 250


def test_assignment_idempotency_event_binding_duplicate_and_fail_closed(tmp_path):
    clock = [1_800_000_000.]
    rec = service(tmp_path, clock)
    req = request(str(uuid.uuid4()))
    first, repeated = rec.recommend(req), rec.recommend(req)
    assert first.feedback.decision_id == repeated.feedback.decision_id
    assert first.recommendations == repeated.recommendations
    changed = req.model_copy(update={'preference_revision': 1})
    with pytest.raises(FeedbackError, match='reused'):
        rec.recommend(changed)
    clock[0] += 2
    e = event(first.feedback)
    assert rec.feedback_store.event(e) is False
    assert rec.feedback_store.event(e) is True
    with pytest.raises(FeedbackError, match='token'):
        rec.feedback_store.event(e.model_copy(update={'token': 'x'*64}))
    with pytest.raises(FeedbackError, match='featured'):
        rec.feedback_store.event(e.model_copy(update={'movie_id': 'not-assigned'}))
    with pytest.raises(FeedbackError, match='different content'):
        rec.feedback_store.event(e.model_copy(update={'kind': 'hide'}))
    with pytest.raises(FeedbackError, match='future'):
        rec.feedback_store.event(event(first.feedback, elapsed=100_000))
    # Simulate a storage failure before a decision can be returned.
    with rec.feedback_store.connect() as db:
        db.execute('DROP TABLE events')
        db.execute('DROP TABLE decisions')
    with pytest.raises(sqlite3.Error):
        rec.recommend(request())


def test_unknown_is_not_negative_and_visibility_not_an_inclusion_rule():
    assert outcome({}, [])['reward'] is None
    assert outcome({}, [{'kind': 'details_open', 'elapsed_ms': 1, 'stream_ok': True}])['reward'] == 1
    events = [{'kind': 'heartbeat', 'elapsed_ms': t*1000, 'stream_ok': True} for t in range(0, 1800, 30)]
    events.append({'kind': 'window_end', 'elapsed_ms': 1800000, 'stream_ok': True})
    assert outcome({}, events)['reward'] == 0
    assert outcome({}, events)['display_acknowledged'] is False
    assert outcome({}, events[:5]+events[10:])['reward'] is None
    events[-1]['stream_ok'] = False
    assert outcome({}, events)['reward'] is None


def test_empty_singleton_and_late_events(tmp_path):
    clock = [1_800_000_000.]
    rec = service(tmp_path, clock)
    ids = list(rec.catalog.movies)
    one = request().model_copy(update={'preferences': PreferenceState(excluded_movie_ids=ids[1:])})
    response = rec.recommend(one)
    assert response.feedback.eligible_count == 1 and response.feedback.propensity == 1
    assert [r.movie.movie_id for r in response.recommendations] == [ids[0]]
    empty = request().model_copy(update={'preferences': PreferenceState(excluded_movie_ids=ids)})
    assert rec.recommend(empty).feedback is None
    clock[0] += WINDOW+LATE_ALLOWANCE+1
    with pytest.raises(FeedbackError, match='late-arrival'):
        rec.feedback_store.event(event(response.feedback))


def test_real_api_token_and_server_failure_contract(tmp_path, monkeypatch):
    settings = Settings(feedback_mode='randomized', feedback_db=tmp_path/'api.sqlite3')
    with TestClient(create_app(settings)) as client:
        response = client.post('/api/v1/recommendations', json=request().model_dump()).json()
        assert response['feedback']['synthetic']
        e = FeedbackEvent(**{**event(type('Assignment', (), response['feedback'])()).model_dump(), 'elapsed_ms': 0})
        assert client.post('/api/v1/feedback/events', json=e.model_dump()).json()['duplicate'] is False
        assert client.post('/api/v1/feedback/events', json=e.model_dump()).json()['duplicate'] is True
        assert client.post('/api/v1/feedback/events', json={**e.model_dump(), 'propensity': .5}).status_code == 422
        def fail(*args):
            raise sqlite3.OperationalError('disk full')
        monkeypatch.setattr(client.app.state.runtime.service.feedback_store, 'assign', fail)
        assert client.post('/api/v1/recommendations', json=request().model_dump()).status_code == 503


def test_export_evaluate_synthetic_roundtrip_and_missing_bounds(tmp_path):
    clock = [1_800_000_000.]
    rec = service(tmp_path, clock)
    for i in range(60):
        clock[0] = 1_800_000_000.+i*172800
        req = request().model_copy(update={'session_id': f'session-{i}'})
        response = rec.recommend(req)
        if i % 3 != 0:
            rec.feedback_store.event(event(response.feedback, elapsed=0))
    as_of = clock[0]+WINDOW+LATE_ALLOWANCE+1
    excluded = prepare(rec.feedback_store.path, tmp_path/'no_synthetic', backend='fixture', as_of=as_of)
    assert excluded['eligible_mature_assignments'] == 0
    assert evaluate(tmp_path/'no_synthetic', tmp_path/'empty_report', bootstrap=100)['status'] == 'insufficient_data'
    stats = prepare(rec.feedback_store.path, tmp_path/'inputs', backend='fixture', include_synthetic=True, as_of=as_of)
    assert [p['rows'] for p in stats['partitions'].values()] == [48, 6, 6]
    assert stats['partitions']['test']['rewards'] == {'unknown': 2, 'positive': 4}
    report = evaluate(tmp_path/'inputs', tmp_path/'report', bootstrap=100)
    assert report['synthetic'] and report['selected_policy'] is None
    logging = report['test']['logging']
    assert logging['ips_missing_outcome_interval'] == pytest.approx([4/6, 1.])
    assert logging['complete_outcome_estimates'] is None
    assert logging['effective_sample_size'] == pytest.approx(6)
    assert not report['policy_selection_uses_test']
    with pytest.raises(FileExistsError):
        evaluate(tmp_path/'inputs', tmp_path/'report', bootstrap=100)


def test_future_outcome_availability_is_purged(tmp_path):
    clock = [1_800_000_000.]
    rec = service(tmp_path, clock)
    for i in range(30):
        clock[0] = 1_800_000_000.+i*60
        response = rec.recommend(request())
        rec.feedback_store.event(event(response.feedback, elapsed=0))
    stats = prepare(rec.feedback_store.path, tmp_path/'inputs', backend='fixture',
                    include_synthetic=True, as_of=clock[0]+WINDOW+LATE_ALLOWANCE+1)
    assert stats['partitions']['train']['rows'] == 0
    assert stats['partitions']['validation']['rows'] == 0
    assert stats['outcome_availability_purge'] == {'train': 24, 'validation': 3}
    assert evaluate(tmp_path/'inputs', tmp_path/'report', bootstrap=100)['status'] == 'insufficient_data'


def test_complete_reward_known_value_and_padded_support():
    rows = [{'action_index': i % 2, 'action': str(i % 2), 'propensity': .5, 'probabilities': [.5, .5],
             'candidates': [{'movie_id': '0'}, {'movie_id': '1'}], 'reward': 1-i % 2,
             'session': str(i//2)} for i in range(100)]
    model = {'prior': .1, 'q': {}}
    result = assess(rows, model, 'rank_first', 100)
    assert result['complete_outcome_estimates']['ips'] == pytest.approx(1.)
    assert result['complete_outcome_estimates']['dr'] == pytest.approx(1.)
    assert result['complete_outcome_estimates']['snips'] == pytest.approx(1.)
    trained = fit(rows[:50])
    assert trained['q']['0'] > trained['q']['1']
