"""Explain a frozen OPE run without refitting or selecting a new policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import hashes, write_json
from .common import begin, finish, verify
from .policy import predict_rewards


def analyze(inputs_dir, evaluation_dir, output_dir):
    inputs, evaluation = Path(inputs_dir), Path(evaluation_dir)
    verify(inputs)
    manifest = verify(evaluation)
    if manifest['input_manifest_sha256'] != hashes(inputs / 'manifest.json')['sha256']:
        raise ValueError('Evaluation and input runs do not match')
    model = json.loads((evaluation / 'reward_model.json').read_text())
    policy = json.loads((evaluation / 'policy.json').read_text())
    saved = pd.read_parquet(evaluation / 'contributions.parquet')
    actions = np.array(json.loads((inputs / 'action_ids.json').read_text()))
    out = begin(output_dir)
    summary = {'scope': 'Post-hoc descriptive analysis only; no fitting or policy reselection',
               'selection': policy, 'partitions': {}, 'test_weight_concentration': {}}
    for name in ['train', 'validation', 'test']:
        frame = pd.read_parquet(inputs / f'{name}.parquet')
        q = predict_rewards(model, frame)
        preferred = q.argmax(axis=1)
        matched = frame.action_index.to_numpy() == preferred
        detail = pd.DataFrame({'action_id': actions[preferred], 'matched': matched,
                               'matched_click': matched * frame.reward.to_numpy()})
        groups = detail.groupby('action_id').agg(
            preferred_contexts=('matched', 'size'), logged_matches=('matched', 'sum'),
            matched_clicks=('matched_click', 'sum')).sort_values('preferred_contexts', ascending=False)
        groups.to_csv(out / f'{name}_preferred_actions.csv')
        summary['partitions'][name] = {
            'rows': len(frame), 'clicks': int(frame.reward.sum()),
            'preferred_action_matches': int(matched.sum()),
            'clicks_on_preferred_action': int(frame.loc[matched, 'reward'].sum()),
            'distinct_preferred_actions': len(groups),
            'top_preferred_actions': groups.head(10).reset_index().to_dict('records')}
    for name, part in saved.groupby('policy'):
        assert len(part) == summary['partitions']['test']['rows']
        assert part.reward.sum() == summary['partitions']['test']['clicks']
        favored = part.weight > 1 + 1e-10
        clicks = part.loc[part.reward == 1, 'ips'].sort_values(ascending=False)
        total = float(part.ips.sum())
        summary['test_weight_concentration'][name] = {
            'high_weight_exposures': int(favored.sum()),
            'high_weight_clicks': int(part.loc[favored, 'reward'].sum()),
            'high_weight_share_of_ips_click_total': float(part.loc[favored, 'ips'].sum()/total) if total else None,
            'largest_single_click_share': float(clicks.iloc[0]/total) if total else None,
            'ips_without_largest_single_click_reward': float((total-clicks.iloc[0])/len(part)) if total else None,
            'sensitivity_definition': 'Change one largest-contribution click to zero; same exposures, weights and denominator. Not a corrected estimate.'}
    write_json(out / 'analysis.json', summary)
    lines = ['# What the first OPE experiment tells us', '',
             'This analysis uses the saved model and completed evaluation. It changes no model, policy or test split.', '',
             '## Why the result is uncertain', '',
             '| Partition | Recorded recommendations | All clicks | Records matching the model\'s favorite item | Clicks on that favorite |',
             '|---|---:|---:|---:|---:|']
    for name, values in summary['partitions'].items():
        lines.append(f"| {name} | {values['rows']:,} | {values['clicks']} | {values['preferred_action_matches']} | {values['clicks_on_preferred_action']} |")
    concentration = summary['test_weight_concentration']['contextual_epsilon_0.1']
    lines += ['', f"For the strongest personalized candidate, {concentration['high_weight_clicks']} clicks account for "
              f"{100*concentration['high_weight_share_of_ips_click_total']:.2f}% of the weighted click total. "
              'The old system picked randomly among 80 items, so it rarely displayed the particular item this candidate favored. '
              'Each such observation receives substantial influence when estimating the alternative approach.', '',
              f"One of those clicks represents {100*concentration['largest_single_click_share']:.2f}% of the weighted total. "
              f"Changing that one click to a non-click would move its IPS estimate to {concentration['ips_without_largest_single_click_reward']:.6f}. "
              'This is a sensitivity illustration, not a reason to remove a legitimate click.', '',
              '## Which items were favored', '',
              'IDs below belong to the Open Bandit fashion catalog, not MovieLens. Counts refer to the saved model\'s top-scoring action '
              'for each context before exploration, not actual recommendations delivered by our application.', '',
              '| Action ID | Test contexts preferring it | Times it was actually shown in those contexts | Clicks in those matches |',
              '|---|---:|---:|---:|']
    for row in summary['partitions']['test']['top_preferred_actions']:
        lines.append(f"| {row['action_id']} | {row['preferred_contexts']} | {row['logged_matches']} | {row['matched_clicks']} |")
    lines += ['', '## What to conclude', '',
              'Validation chose uniform selection. Keep that recorded decision. Higher personalized point estimates on the final test '
              'do not justify switching winners after inspecting it. The simple model may be inaccurate, preferences may vary over time, '
              'and there are few informative clicks; these data do not isolate which explanation caused the result.', '',
              'The narrow intervals from the prediction-only method do not mean the model is accurate: they hold its predictions fixed '
              'and omit model bias and fitting uncertainty. The importance-weighted estimates and wide intervals are a useful warning.', '',
              '## Next experimental steps', '',
              'Preserve this experiment as the initial benchmark. Any revised model should be developed on training/validation data and '
              'assessed with a predeclared fresh evaluation window or replication campaign. Label further analysis of this test as exploratory. '
              'For the movie app, implement the separate logging design in movie_feedback_plan.md before claiming movie-policy OPE. '
              'Do not apply Open Bandit item IDs, click rates, or the selected uniform policy to the movie application.', '']
    (out / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    finish(out, {'stage': 'ope_descriptive_analysis',
                 'input_manifest_sha256': hashes(inputs / 'manifest.json')['sha256'],
                 'evaluation_manifest_sha256': hashes(evaluation / 'manifest.json')['sha256']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    analyze(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
