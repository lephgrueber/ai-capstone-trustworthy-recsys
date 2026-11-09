"""Labeled synthetic end-to-end movie feedback pilot; never user-quality evidence."""

import argparse
import json
from pathlib import Path
import uuid

from trustworthy_recsys.serving.catalog import fixture_catalog
from trustworthy_recsys.serving.components import FixtureRetrieval, PassThroughReranker
from trustworthy_recsys.serving.contracts import FeedbackEvent, PreferenceState, RecommendationRequest
from trustworthy_recsys.serving.feedback import FeedbackStore, WINDOW, LATE_ALLOWANCE
from trustworthy_recsys.serving.service import RecommendationService
from .movie import prepare, evaluate


def pilot(output_dir, decisions=100):
    if decisions < 30:
        raise ValueError('Use at least 30 decisions for the synthetic split smoke check')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    clock = [1_800_000_000.]
    store = FeedbackStore(root/'synthetic.sqlite3', synthetic=True, clock=lambda: clock[0])
    catalog = fixture_catalog()
    service = RecommendationService(catalog, FixtureRetrieval.build(catalog), PassThroughReranker(), 'fixture', 100, store)
    for i in range(decisions):
        gaps = int(i >= int(.8*decisions)) + int(i >= int(.9*decisions))
        clock[0] = 1_800_000_000.+i*3600+gaps*(WINDOW+LATE_ALLOWANCE)
        request = RecommendationRequest(session_id=f'synthetic-session-{i}', preference_revision=0,
            assignment_key=str(uuid.uuid4()), preferences=PreferenceState(), result_count=5)
        result = service.recommend(request)
        assignment = result.feedback
        assigned = clock[0]
        def send(kind, seconds):
            clock[0] = assigned+seconds
            store.event(FeedbackEvent(event_id=str(uuid.uuid4()), decision_id=assignment.decision_id,
                token=assignment.token, movie_id=assignment.movie_id, kind=kind, elapsed_ms=seconds*1000))
        send('heartbeat', 0)
        send('display', 1)
        if i % 4 == 0:
            send('details_open', 2)
        elif i % 4 != 1:
            for second in range(30, WINDOW, 30):
                send('heartbeat', second)
            send('window_end', WINDOW)
        # Remaining assignments deliberately model incomplete/censored windows.
    as_of = clock[0]+WINDOW+LATE_ALLOWANCE+1
    stats = prepare(store.path, root/'inputs', backend='fixture', include_synthetic=True, as_of=as_of)
    report = evaluate(root/'inputs', root/'evaluation', bootstrap=100)
    summary = {'synthetic': True, 'decisions': decisions, 'partitions': stats['partitions'],
               'evaluation_status': report['status'], 'test_logging': report['test']['logging'],
               'claim': 'Technical plumbing only; generated outcomes are not observed movie-user preferences.'}
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=100)
    pilot(**vars(parser.parse_args()))
