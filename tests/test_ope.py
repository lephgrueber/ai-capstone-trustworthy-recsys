"""Known-value estimators, temporal boundaries, and actual pipeline integration."""

import json

import numpy as np
import pandas as pd
import pytest

from trustworthy_recsys.ope.common import verify
from trustworthy_recsys.ope.analyze import analyze
from trustworthy_recsys.ope.estimators import contributions, point_estimates, summarize
from trustworthy_recsys.ope.evaluate import evaluate
from trustworthy_recsys.ope.policy import fit_reward_model, predict_rewards
from trustworthy_recsys.ope.prepare import prepare, split_temporally
from trustworthy_recsys.ope.validate import validate


def test_known_policy_value_and_dr_correction():
    # Exact balanced logging: action 0 always clicks, action 1 never clicks.
    action = np.tile([0, 1], 100)
    pi = np.tile([.8, .2], (200, 1))
    # Deliberately wrong reward model; correct propensities still correct DR.
    q = np.full((200, 2), .1)
    c = contributions(1-action, action, np.full(200, .5), pi, q)
    values = point_estimates(c)
    assert values['dm'] == pytest.approx(.1)
    for key in ['ips', 'snips', 'dr']:
        assert values[key] == pytest.approx(.8)
    result = summarize(c, np.repeat(np.arange(10), 20), repetitions=100)
    assert result['effective_sample_size'] == pytest.approx(200 / 1.36)


@pytest.mark.parametrize('defect', ['zero_propensity', 'bad_probabilities', 'nan', 'bad_action'])
def test_invalid_estimator_inputs_fail(defect):
    p, pi, q, action = np.full(2, .5), np.full((2, 2), .5), np.full((2, 2), .1), np.array([0, 1])
    if defect == 'zero_propensity':
        p[0] = 0
    elif defect == 'bad_probabilities':
        pi[0] = [.8, .8]
    elif defect == 'nan':
        q[0, 0] = np.nan
    else:
        action[0] = -1
    with pytest.raises(ValueError):
        contributions([1, 0], action, p, pi, q)


def test_equal_timestamps_never_cross_partitions():
    times = pd.date_range('2020-01-01', periods=10, freq='h', tz='UTC').repeat(10)
    frame = pd.DataFrame({'timestamp_utc': times, 'source_row': np.arange(100)})
    parts, _ = split_temporally(frame)
    assert sum(map(len, parts.values())) == 100
    assert parts['train'].timestamp_utc.max() < parts['validation'].timestamp_utc.min()
    assert parts['validation'].timestamp_utc.max() < parts['test'].timestamp_utc.min()


def make_source(root):
    directory = root / 'random' / 'all'
    directory.mkdir(parents=True)
    n = 1200
    frame = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='10min', tz='UTC').astype(str),
        'item_id': np.arange(n) % 2, 'position': 1, 'click': (np.arange(n) % 7 == 0).astype(int),
        'propensity_score': .5, 'user_feature_0': np.where(np.arange(n) % 3 == 0, 'a', 'b'),
    })
    frame.to_csv(directory / 'all.csv', index=True)
    pd.DataFrame({'item_id': [0, 1], 'item_feature_0': [1., 2.]}).to_csv(directory / 'item_context.csv', index=True)
    (root / 'README').write_text('Synthetic Open Bandit-shaped test fixture, not observed data. CC BY 4.0')
    (root / 'VERSION').write_text('test fixture')
    return frame


def test_pipeline_evaluation_roundtrip_and_tamper(tmp_path):
    raw, inputs, results = tmp_path / 'raw', tmp_path / 'inputs', tmp_path / 'results'
    make_source(raw)
    stats = prepare(raw, inputs, chunk_size=83)
    assert stats['selected_rows'] == 1200
    assert [stats['partitions'][s]['rows'] for s in ['train', 'validation', 'test']] == [960, 120, 120]
    verify(inputs)
    report = evaluate(inputs, results, bootstrap=100)
    assert report['integrity']['uniform_ips_snips_equal_observed_ctr']
    assert report['integrity']['test_used_for_selection'] is False
    verify(results)
    analysis = analyze(inputs, results, tmp_path / 'analysis')
    assert analysis['partitions']['test']['rows'] == report['test_rows']
    assert analysis['partitions']['test']['clicks'] == report['test_clicks']
    assert analysis['selection']['selected_epsilon'] == report['selected_epsilon']
    verify(tmp_path / 'analysis')
    with pytest.raises(FileExistsError):
        evaluate(inputs, results, bootstrap=100)
    (inputs / 'action_ids.json').write_text('[42, 99]')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate(inputs, tmp_path / 'bad_results', bootstrap=100)


def test_preparation_rejects_mismatched_random_support(tmp_path):
    raw = tmp_path / 'raw'
    frame = make_source(raw)
    frame.loc[0, 'propensity_score'] = .1
    frame.to_csv(raw / 'random' / 'all' / 'all.csv', index=True)
    with pytest.raises(ValueError, match='catalog size'):
        prepare(raw, tmp_path / 'out')
    assert (tmp_path / 'out' / 'INCOMPLETE').exists()


def test_semantic_validation_reproduction_and_rehashed_corruption(tmp_path):
    from trustworthy_recsys.data.reporting import hashes
    raw = tmp_path/'raw'
    make_source(raw)
    first, second = tmp_path/'first', tmp_path/'second'
    prepare(raw, first, chunk_size=83)
    prepare(raw, second, chunk_size=101)
    result = validate(second, tmp_path/'audit', compare_dir=first)
    assert result['logical_reproduction_verified']
    # A hash-consistent but semantically invalid artifact must still fail.
    frame = pd.read_parquet(second/'test.parquet')
    frame.loc[0, 'action_index'] = 1-int(frame.loc[0, 'action_index'])
    frame.to_parquet(second/'test.parquet', index=False)
    manifest = json.loads((second/'manifest.json').read_text())
    manifest['outputs']['test.parquet'] = hashes(second/'test.parquet')['sha256']
    (second/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Action IDs misaligned'):
        validate(second, tmp_path/'invalid')


def test_model_reload_unknown_context_and_training_isolation(tmp_path):
    train = pd.DataFrame({'action_index': [0, 1, 0, 1], 'reward': [1, 0, 1, 0],
                          'user_feature_0': ['a', 'a', 'b', 'b']})
    model = fit_reward_model(train, ['user_feature_0'], 2)
    unknown = pd.DataFrame({'user_feature_0': ['unknown']})
    prediction = predict_rewards(model, unknown)
    np.testing.assert_allclose(prediction[0], model['prior'])
    restored = json.loads(json.dumps(model))
    np.testing.assert_array_equal(predict_rewards(restored, unknown), prediction)
    # Evaluation rewards cannot affect prediction: only named context fields are read.
    unknown['reward'] = 0
    np.testing.assert_array_equal(predict_rewards(restored, unknown), prediction)
    unknown['reward'] = 1
    np.testing.assert_array_equal(predict_rewards(restored, unknown), prediction)
