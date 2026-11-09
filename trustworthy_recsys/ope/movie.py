"""Export mature movie assignments and evaluate supported featured-slot policies.

MovieLens ratings are never converted into fake logged propensities. This module
consumes the application's transactional decision/event store, not Open Bandit.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from trustworthy_recsys.data.reporting import write_json
from trustworthy_recsys.serving.feedback import WINDOW, LATE_ALLOWANCE, POLICY, canonical, digest, probabilities
from .common import begin, finish, verify
from .estimators import point_estimates, summarize


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def check_decision(row):
    candidates = row['candidates']
    ids = [item['movie_id'] for item in candidates]
    index = row['action_index']
    if not ids or len(ids) != len(set(ids)) or set(ids)&set(row['fixed_grid']):
        raise ValueError('Invalid or overlapping featured pool/grid')
    if len(row['fixed_grid']) != len(set(row['fixed_grid'])):
        raise ValueError('Duplicate grid movies')
    if digest(candidates) != row['candidate_hash']:
        raise ValueError('Candidate hash mismatch')
    if not isinstance(index, int) or not 0 <= index < len(ids) or ids[index] != row['action']:
        raise ValueError('Action/mapping mismatch')
    expected = probabilities(len(ids), row['epsilon'])
    if not np.allclose(expected, row['probabilities'], rtol=1e-12, atol=1e-12) or not np.isclose(row['propensity'], expected[index], rtol=1e-12, atol=1e-12):
        raise ValueError('Logging probabilities differ from policy')
    if set(ids+row['fixed_grid']) & set(row['context']['preferences']['history_items']+row['context']['preferences']['excluded_movie_ids']):
        raise ValueError('Previously liked or hidden movie entered the experiment')


def outcome(row, events):
    # Assignment-level outcome, NEVER conditioned on whether the hero was visible.
    inside = [e for e in events if e['elapsed_ms'] <= WINDOW*1000]
    positive = any(e['kind'] == 'details_open' for e in inside)
    pulses = sorted(e['elapsed_ms']/1000 for e in events if e['kind'] in {'heartbeat', 'window_end'})
    ended = any(e['kind'] == 'window_end' and e['elapsed_ms'] >= WINDOW*1000 for e in events)
    # Conservative negative: a complete acknowledged stream, no gaps >45 seconds.
    complete = bool(ended and pulses and pulses[0] <= 45 and
                    max(np.diff([0., *pulses])) <= 45 and all(e['stream_ok'] for e in events))
    reward = 1 if positive else 0 if complete else None
    return {'reward': reward, 'reward_status': 'positive' if positive else 'negative' if complete else 'unknown',
            'stream_complete': complete, 'display_acknowledged': any(e['kind'] == 'display' for e in inside),
            'event_count': len(events),
            'favorite_add': any(e['kind'] == 'favorite_add' for e in inside),
            'hide': any(e['kind'] == 'hide' for e in inside)}


def prepare(db_path, output_dir, backend='learned', include_synthetic=False, as_of=None):
    as_of = time.time() if as_of is None else float(as_of)
    db_path = Path(db_path).resolve()
    if not db_path.is_file():
        raise ValueError('Feedback database does not exist; start collection first')
    db = sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        decisions = [json.loads(r[0]) for r in db.execute('SELECT payload FROM decisions ORDER BY assigned,id')]
        events = [{'received': r[0], **json.loads(r[1])} for r in db.execute('SELECT received,payload FROM events ORDER BY received,id')]
        duplicates = db.execute("SELECT value FROM metadata WHERE key='duplicate_events'").fetchone()
    finally:
        db.close()
    # Snapshot digest excludes the private store secret and capability tokens.
    snapshot_hash = digest({'decisions': decisions, 'events': events,
                            'duplicate_events': int(duplicates[0]) if duplicates else 0})
    by_decision = defaultdict(list)
    ids = {row['decision_id'] for row in decisions}
    if len(ids) != len(decisions):
        raise ValueError('Duplicate decision IDs')
    for event in events:
        if event['decision_id'] not in ids:
            raise ValueError('Orphan event')
        if event['received'] <= as_of:
            by_decision[event['decision_id']].append(event)
    excluded, rows = Counter(), []
    for row in decisions:
        check_decision(row)
        if row['backend'] != backend:
            excluded['other_backend'] += 1
        elif row['synthetic'] and not include_synthetic:
            excluded['synthetic'] += 1
        elif row['policy'] != POLICY or not row['ope_eligible']:
            excluded['no_alternative_support'] += 1
        elif row['assigned'] + WINDOW + LATE_ALLOWANCE > as_of:
            excluded['immature'] += 1
        else:
            joined = by_decision[row['decision_id']]
            if any(e['movie_id'] != row['action'] or e['received'] > row['assigned']+WINDOW+LATE_ALLOWANCE for e in joined):
                raise ValueError('Invalid event attribution or late receipt')
            clean = {k: v for k, v in row.items() if k != 'response'}
            clean.update(outcome(row, joined))
            rows.append(clean)
    # A file's ordering is not trusted; equal assignment timestamps stay together.
    rows.sort(key=lambda r: (r['assigned'], r['decision_id']))
    n = len(rows)
    cut1 = rows[min(n-1, int(.8*n))]['assigned'] if n else 0
    cut2 = rows[min(n-1, int(.9*n))]['assigned'] if n else 0
    parts = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        name = 'train' if row['assigned'] < cut1 else 'validation' if row['assigned'] < cut2 else 'test'
        parts[name].append(row)
    # Prevent attribution/late-arrival windows crossing a later decision boundary.
    # Purging uses timestamps only, never reward, visibility or selected action.
    purge = {}
    for name, boundary in [('train', cut1), ('validation', cut2)]:
        retained = [r for r in parts[name] if r['assigned']+WINDOW+LATE_ALLOWANCE < boundary]
        purge[name] = len(parts[name])-len(retained)
        parts[name] = retained
    out = begin(output_dir)
    stats = {'source_decisions': len(decisions), 'source_events': len(events), 'eligible_mature_assignments': n,
        'exported_assignments': sum(map(len, parts.values())), 'outcome_availability_purge': purge,
        'duplicate_event_retries': int(duplicates[0]) if duplicates else 0,
        'source_decisions_with_events': sum(bool(by_decision[r['decision_id']]) for r in decisions),
        'candidate_counts': dict(Counter(len(r['candidates']) for r in rows)),
        'minimum_logging_probability': min((min(r['probabilities']) for r in rows), default=None),
        'model_provenance_groups': dict(Counter(digest(r['provenance']) for r in rows)),
        'excluded': dict(excluded), 'backend': backend, 'include_synthetic': include_synthetic,
        'contains_synthetic': any(r['synthetic'] for r in rows), 'as_of_utc': datetime.fromtimestamp(as_of, timezone.utc).isoformat(),
        'snapshot_sha256': snapshot_hash, 'window_seconds': WINDOW, 'late_allowance_seconds': LATE_ALLOWANCE,
        'outcome': 'assignment-level featured details_open within 30 minutes', 'partitions': {},
        'negative_observation_rule': 'window_end + healthy heartbeats with maximum 45s gap from assignment through window end',
        'session_overlap': {}}
    for name, part in parts.items():
        (out/f'{name}.jsonl').write_text(''.join(canonical(row)+'\n' for row in part), encoding='utf-8')
        stats['partitions'][name] = {'rows': len(part), 'sessions': len({r['session'] for r in part}),
            'rewards': dict(Counter(r['reward_status'] for r in part)),
            'display_acknowledged': sum(r['display_acknowledged'] for r in part),
            'stream_complete': sum(r['stream_complete'] for r in part),
            'start': part[0]['assigned'] if part else None, 'end': part[-1]['assigned'] if part else None}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        stats['session_overlap'][a+'_'+b] = len({r['session'] for r in parts[a]} & {r['session'] for r in parts[b]})
    write_json(out/'statistics.json', stats)
    write_json(out/'schema.json', {'version': 1, 'format': 'JSONL',
        'unit': 'server-generated featured assignment', 'reward': '0, 1, or null (unknown)',
        'action_index': 'zero-based index into this row candidates, not a global movie index',
        'session': 'store-specific HMAC pseudonym; not a verified unique person',
        'context': 'frozen pre-assignment preference snapshot and revision',
        'probabilities': 'complete conditional logging distribution over the saved candidate pool',
        'splits': 'chronological approximate 80/10/10 before purging; equal timestamps kept together; sessions may overlap',
        'availability_gap_seconds': WINDOW+LATE_ALLOWANCE})
    (out/'DATA_CARD.md').write_text(movie_card(stats), encoding='utf-8')
    finish(out, {'stage': 'movie_feedback_prepare', 'snapshot_sha256': snapshot_hash})
    return stats


def movie_card(stats):
    return f'''# Movie feedback Data Card

## Overview
Application-collected featured-slot assignments, not reconstructed MovieLens ratings.
Mature eligible assignments: {stats['eligible_mature_assignments']}. Backend: {stats['backend']}.
Contains synthetic traffic: {stats['contains_synthetic']}. A synthetic run is plumbing evidence only.

## Provenance and collection
Server-generated decisions are persisted before serving. The browser reports display,
details-open, favorites, hides and telemetry heartbeats. SHA-256 of the consistent
logical database snapshot: {stats['snapshot_sha256']}. No capability tokens or store
secrets are exported. Context and HMAC session identifiers remain sensitive research data.

## Licensing and usage
Movie metadata and derived model artifacts retain MovieLens terms. Newly collected
behavior is local pilot research data, not automatically covered by the software MIT
license or released for redistribution. Do not publish raw sessions/preferences.

## Data structure and splits
See schema.json and statistics.json. Each row has its own frozen candidate set and
exact action probabilities. Chronological approximate 80/10/10 partitions keep ties
together. Purge train/validation assignments whose outcome/late-arrival windows extend
to the next partition's first assignment. Actual post-purge ratios differ. Fit reward
models on train; select policies on validation; report test once.
Sessions can cross boundaries and are not model features. Bootstrap test sessions as clusters.

## Limitations and risks
The outcome is opening featured details within 30 minutes, not watching or satisfaction.
Wait another 24 hours for delayed events. Unknown outcomes remain null; no display
acknowledgment is required for inclusion. Negative outcomes require a complete healthy
event stream. Browser closure, throttling, offline loss and self-report errors can censor
outcomes. Report missing-outcome bounds instead of treating unknowns as zero.
These logs support alternate featured choices within the same pool and fixed grid,
not changes to retrieval, the full slate, or long-term recommendation behavior.
Session pseudonyms are not authenticated people; this is a local single-service pilot.
'''


def fit(rows):
    observed = [r for r in rows if r['reward'] is not None]
    prior = (sum(r['reward'] for r in observed)+1)/(len(observed)+2)
    counts, clicks = Counter(), Counter()
    for row in observed:
        counts[row['action']] += 1
        clicks[row['action']] += row['reward']
    return {'prior': prior, 'strength': 20, 'q': {item: (clicks[item]+20*prior)/(count+20) for item, count in counts.items()},
            'training_observed': len(observed), 'features': 'movie ID only; no session ID or held-out rewards'}


def policy_arrays(rows, model, policy):
    # Ragged pools are padded with zero-probability actions, never widened support.
    width = max(len(r['candidates']) for r in rows)
    pi, q = np.zeros((len(rows), width)), np.zeros((len(rows), width))
    for i, row in enumerate(rows):
        size = len(row['candidates'])
        q[i, :size] = [model['q'].get(c['movie_id'], model['prior']) for c in row['candidates']]
        if policy == 'logging':
            pi[i, :size] = row['probabilities']
        elif policy == 'uniform':
            pi[i, :size] = 1/size
        elif policy == 'rank_first':
            pi[i, 0] = 1
        elif policy == 'trained_movie_ctr':
            # Fixed 10% exploration, not fitted using test outcomes.
            pi[i, :size] = .1/size
            pi[i, int(np.argmax(q[i, :size]))] += .9
        else:
            raise ValueError('Unknown target policy')
    return pi, q


def assess(rows, model, policy, bootstrap):
    from .estimators import contributions
    pi, q = policy_arrays(rows, model, policy)
    action = np.array([r['action_index'] for r in rows], dtype=int)
    ps = np.array([r['propensity'] for r in rows])
    unknown = np.array([r['reward'] is None for r in rows])
    lower = np.array([r['reward'] if r['reward'] is not None else 0 for r in rows])
    upper = np.where(unknown, 1, lower)
    lo = contributions(lower, action, ps, pi, q)
    hi = contributions(upper, action, ps, pi, q)
    w = lo['weight']
    result = {'rows': len(rows), 'unknown_outcomes': int(unknown.sum()),
              'known_positives': int(lower.sum()), 'mean_weight': float(w.mean()), 'max_weight': float(w.max()),
              'effective_sample_size': float(w.sum()**2/np.dot(w, w)) if np.dot(w, w) else 0,
              'ips_missing_outcome_interval': [float(lo['ips'].mean()), float(hi['ips'].mean())],
              'interpretation': 'Sensitivity endpoints, not a confidence interval; unknown rewards assigned 0 and 1 respectively.',
              'complete_outcome_estimates': None}
    clusters = [r['session'] for r in rows]
    if w.sum() == 0:
        result['warning'] = 'No sampled target-policy support'
        return result
    if not unknown.any():
        result['complete_outcome_estimates'] = point_estimates(lo)
    try:
        bounds = [summarize(c, clusters, repetitions=bootstrap) for c in [lo, hi]]
        for item in bounds:
            item['session_clusters'] = item.pop('time_blocks')
        result['lower_reward_scenario'] = bounds[0]
        result['upper_reward_scenario'] = bounds[1]
        result['ips_envelope_with_sampling_uncertainty'] = [bounds[0]['estimates']['ips']['ci95'][0], bounds[1]['estimates']['ips']['ci95'][1]]
    except ValueError as error:
        result['uncertainty_unavailable'] = str(error)
    return result


def evaluate(inputs_dir, output_dir, bootstrap=1000):
    if bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    root = Path(inputs_dir)
    manifest = verify(root)
    required = {'train.jsonl', 'validation.jsonl', 'test.jsonl', 'statistics.json', 'schema.json', 'DATA_CARD.md'}
    if not required <= set(manifest['outputs']):
        raise ValueError('Missing movie dataset artifacts')
    stats = json.loads((root/'statistics.json').read_text(encoding='utf-8'))
    parts = {name: read_rows(root/f'{name}.jsonl') for name in ['train', 'validation', 'test']}
    seen, previous = set(), None
    for name, rows in parts.items():
        if len(rows) != stats['partitions'][name]['rows']:
            raise ValueError('Split statistics mismatch')
        for row in rows:
            check_decision(row)
            if row['decision_id'] in seen or row['reward'] not in [None, 0, 1]:
                raise ValueError('Duplicate decision or invalid reward')
            seen.add(row['decision_id'])
        if rows:
            if rows != sorted(rows, key=lambda r: (r['assigned'], r['decision_id'])) or previous is not None and rows[0]['assigned'] <= previous:
                raise ValueError('Temporal overlap or unsorted records')
            previous = rows[-1]['assigned']+WINDOW+LATE_ALLOWANCE
    out = begin(output_dir)
    model = fit(parts['train'])
    policies = ['logging', 'rank_first', 'uniform', 'trained_movie_ctr']
    report = {'scope': 'featured-slot selection conditional on logged pool/fixed grid; not whole-retriever value',
              'synthetic': stats['contains_synthetic'], 'backend': stats['backend'],
              'source_snapshot_sha256': stats['snapshot_sha256'], 'policy_selection_uses_test': False,
              'statistics': stats, 'validation': {}, 'test': {}, 'selected_policy': None,
              'bootstrap_unit': 'HMAC browser session (not verified person)',
              'status': 'insufficient_data'}
    if all(parts.values()) and model['training_observed']:
        report['validation'] = {p: assess(parts['validation'], model, p, bootstrap) for p in policies}
        # Selection is permitted only when validation outcomes are fully observed.
        if all(r['reward'] is not None for r in parts['validation']):
            report['selected_policy'] = max(policies, key=lambda p: report['validation'][p]['ips_missing_outcome_interval'][0])
        report['test'] = {p: assess(parts['test'], model, p, bootstrap) for p in policies}
        report['status'] = 'synthetic_validation_only' if report['synthetic'] else 'reported_with_missing_outcome_bounds' if any(r['reward'] is None for r in parts['test']) else 'offline_estimates_not_deployment_approval'
    write_json(out/'reward_model.json', model)
    write_json(out/'report.json', report)
    (out/'REPORT.md').write_text('# Movie application OPE\n\nStatus: '+report['status']+'\n\n'+
        'No movie-policy improvement is established by this run alone. Read report.json for each policy, support, uncertainty and unknown-outcome bounds. '+
        'DM/IPS/SNIPS/DR scenario estimates with missing outcomes are sensitivity analyses, not identified policy values. '+
        'Selection uses validation only and is withheld if validation outcomes are missing.\n', encoding='utf-8')
    finish(out, {'stage': 'movie_feedback_ope'})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--db-path', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--backend', choices=['learned', 'baseline', 'fixture'], default='learned')
    prep.add_argument('--include-synthetic', action='store_true')
    prep.add_argument('--as-of', type=float, help='UTC Unix seconds; defaults to current time')
    ev = sub.add_parser('evaluate')
    ev.add_argument('--inputs-dir', type=Path, required=True)
    ev.add_argument('--output-dir', type=Path, required=True)
    ev.add_argument('--bootstrap', type=int, default=1000)
    args = vars(parser.parse_args())
    command = args.pop('command')
    result = prepare(**args) if command == 'prepare' else evaluate(**args)
    print(json.dumps(result if command == 'prepare' else {'status': result['status'], 'selected_policy': result['selected_policy']}))


if __name__ == '__main__':
    main()
