"""Fit training-only policies, select on validation, and report held-out OPE."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import hashes, write_json
from .common import begin, finish, verify
from .estimators import contributions, point_estimates, summarize
from .policy import fit_reward_model, predict_rewards, policy_probabilities


def render_report(report):
    lines = ['# Open Bandit OPE report', '',
             'This is a position-specific fashion click benchmark, not evaluation of the MovieLens retriever.', '',
             f"Campaign: {report['campaign']}; source position: {report['position']}; logging policy: random.",
             f"Test exposures: {report['test_rows']:,}; clicks: {report['test_clicks']}; observed CTR: {report['observed_ctr']:.6f}.",
             f"Validation-selected policy: {report['selected_policy']} (epsilon {report['selected_epsilon']}). Selection metric: validation IPS; no test tuning.",
             'All predeclared policy candidates are reported below as diagnostics; test results do not change selection.', '',
             '| Policy | Estimator | Value | 95% interval (hour-block bootstrap) |',
             '|---|---|---:|---|']
    for name, result in report['policies'].items():
        for estimator, estimate in result['estimates'].items():
            lo, hi = estimate['ci95']
            lines.append(f"| {name} | {estimator.upper()} | {estimate['value']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '## Diagnostics and interpretation', '']
    for name, result in report['policies'].items():
        lines.append(f"- {name}: ESS {result['effective_sample_size']:.1f}/{result['rows']}; "
                     f"mean weight {result['mean_weight']:.4f}; maximum weight {result['max_weight']:.2f}; "
                     f"{result['time_blocks']} hourly blocks.")
    lines += ['', 'The uniform-policy IPS and SNIPS estimates reproduce the observed click rate; this is an algebraic sanity check, '
              'not independent proof of estimator accuracy for the contextual candidates. The contextual candidates have no matched on-policy ground truth. '
              'If epsilon 1.0 wins validation, the selected policy is exactly uniform and no personalized improvement is demonstrated.', '',
              'Intervals resample observed UTC hour blocks and condition on the fitted model and selected policy. '
              'They do not include training/selection uncertainty or ensure independence across repeated users. '
              'The held-out window is short; block-size sensitivity (15 minutes and 2 hours) is included in report.json. '
              'Intervals are not clipped to [0,1]. Clipped-weight estimates are sensitivity analyses, not primary estimates.', '',
              'Inspect error_analysis.json for high-weight events and daily/context slices. contributions.parquet preserves '
              'per-event reward, target probability, weight, reward prediction and estimator contributions for every candidate. '
              'Test outcomes are used only for reporting, never selection. The policy uses four (or the discovered set of) '
              'anonymized categorical user fields; no demographic meaning or identity is inferred. '
              'Averages of smoothed feature/action rates are a simple baseline, not a two-tower or calibrated causal model.', '',
              'No improvement or statistical-significance claim follows from point estimates alone. '
              'Position-wise marginal propensities do not identify arbitrary joint-slate interventions; other displayed items may affect clicks.', '']
    return '\n'.join(lines)


def evaluate(inputs_dir, output_dir, bootstrap=1000, seed=42):
    if bootstrap < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    inputs = Path(inputs_dir)
    verify(inputs)
    fingerprint = hashes(inputs / 'manifest.json')['sha256']
    schema = json.loads((inputs / 'schema.json').read_text())
    stats = json.loads((inputs / 'statistics.json').read_text())
    actions = json.loads((inputs / 'action_ids.json').read_text())
    out = begin(output_dir)
    train = pd.read_parquet(inputs / 'train.parquet')
    validation = pd.read_parquet(inputs / 'validation.parquet')
    if train.timestamp_utc.max() >= validation.timestamp_utc.min():
        raise ValueError('Training/validation temporal leakage')
    model = fit_reward_model(train, schema['context_columns'], len(actions))
    write_json(out / 'reward_model.json', model)
    q_validation = predict_rewards(model, validation)
    candidates = []
    for epsilon in [.1, .3, .5, 1.]:
        c = contributions(validation.reward, validation.action_index, validation.propensity_score,
                          policy_probabilities(q_validation, epsilon), q_validation)
        candidates.append({'epsilon': epsilon, **point_estimates(c)})
    # Prefer greater exploration if validation scores tie.
    selected = max(candidates, key=lambda row: (row['ips'], row['epsilon']))['epsilon']
    selection = {'candidates': candidates, 'selected_epsilon': selected,
                 'selection_metric': 'validation IPS', 'reward_model_train_rows': len(train),
                 'reward_model_shrinkage': 100., 'validation_rows': len(validation),
                 'test_used_for_selection': False, 'tie_break': 'greater epsilon',
                 'policy': 'epsilon-greedy over training-only predicted click rates; action ties use ascending action ID'}
    write_json(out / 'policy.json', selection)
    # First read held-out outcomes only after fitting and selection are complete.
    test = pd.read_parquet(inputs / 'test.parquet')
    if validation.timestamp_utc.max() >= test.timestamp_utc.min():
        raise ValueError('Validation/test temporal leakage')
    if not all(part.event_id.is_unique for part in [train, validation, test]):
        raise ValueError('Repeated source event IDs')
    if len(set(train.event_id) | set(validation.event_id) | set(test.event_id)) != len(train)+len(validation)+len(test):
        raise ValueError('Cross-partition event leakage')
    if not np.allclose(test.propensity_score, 1/len(actions), rtol=1e-7, atol=1e-12):
        raise ValueError('Uniform support assumption violated')
    q = predict_rewards(model, test)
    report = {'campaign': stats['campaign'], 'position': stats['position'], 'selected_epsilon': selected,
              'selected_policy': 'uniform' if selected == 1. else f'contextual_epsilon_{selected}',
              'test_rows': len(test), 'test_clicks': int(test.reward.sum()), 'observed_ctr': float(test.reward.mean()),
              'input_manifest_sha256': fingerprint, 'policies': {},
              'integrity': {'input_hashes_verified': True, 'strict_temporal_splits': True,
                            'event_ids_disjoint': True, 'test_used_for_selection': False,
                            'uniform_logging_probability_matches_full_catalog': True},
              'scope': 'single-position marginal click reward; no MovieLens or joint-slate policy claim'}
    saved, errors = [], {}
    for name, epsilon in [('uniform', 1.), *[(f'contextual_epsilon_{e}', e) for e in [.1, .3, .5]]]:
        pi = policy_probabilities(q, epsilon)
        c = contributions(test.reward, test.action_index, test.propensity_score, pi, q)
        result = summarize(c, test.timestamp_utc.dt.floor('h').astype(str), bootstrap, seed)
        result['block_size_sensitivity'] = {
            period: summarize(c, test.timestamp_utc.dt.floor(period).astype(str), bootstrap, seed)['estimates']
            for period in ['15min', '2h']}
        result['epsilon'] = epsilon
        report['policies'][name] = result
        frame = test[['event_id', 'timestamp_utc', 'action_id', 'propensity_score']].copy()
        for key, value in c.items():
            frame[key] = value
        frame['policy'] = name
        saved.append(frame)
        slices = {}
        for day, group in frame.groupby(frame.timestamp_utc.dt.strftime('%Y-%m-%d')):
            slices[day] = {'rows': len(group), 'clicks': int(group.reward.sum()),
                           **point_estimates({k: group[k].to_numpy() for k in c})}
        high = frame.nlargest(20, 'weight').drop(columns='timestamp_utc').to_dict('records')
        context_slices = {}
        for column in schema['context_columns']:
            context_slices[column] = {}
            for value, indices in test.groupby(column).indices.items():
                subset = {key: array[indices] for key, array in c.items()}
                context_slices[column][str(value)] = {
                    'rows': len(indices), 'clicks': int(subset['reward'].sum()), **point_estimates(subset)}
        errors[name] = {'daily_slices': slices, 'context_slices': context_slices,
                        'largest_weights': high,
                        'warning': 'Descriptive test diagnostics only; context values have no inferred demographic meaning.'}
    uniform = report['policies']['uniform']
    if not all(np.isclose(uniform['estimates'][key]['value'], report['observed_ctr']) for key in ['ips', 'snips']):
        raise ValueError('On-policy identity check failed')
    report['integrity']['uniform_ips_snips_equal_observed_ctr'] = True
    pd.concat(saved, ignore_index=True).to_parquet(out / 'contributions.parquet', index=False)
    write_json(out / 'report.json', report)
    write_json(out / 'error_analysis.json', errors)
    (out / 'REPORT.md').write_text(render_report(report), encoding='utf-8')
    (out / 'MODEL_CARD.md').write_text(
        '# Model Card: Open Bandit contextual baseline\n\n'
        '## Purpose and scope\n\nPosition-specific fashion click OPE research. Separate from MovieLens and Weaviate.\n\n'
        '## Model and training\n\nTraining-only smoothed click rates per categorical feature/action, averaged across features. '
        'Per-action prior shrinks toward global CTR; feature/action rates shrink toward that prior (strength 100). '
        'Unknown categories fall back to the action prior. No affinity, item metadata, user identity or test outcomes enter fitting. '
        'This is a transparent baseline, not an interaction-rich or neural model.\n\n'
        '## Policy selection\n\nEpsilon-greedy with candidates 0.1, 0.3, 0.5, 1.0 selected by validation IPS; '
        f'selected epsilon {selected}. No train+validation refit. All target actions have uniform logging support.\n\n'
        '## Evaluation\n\nSee REPORT.md for held-out DM/IPS/SNIPS/DR, time-block bootstrap intervals and diagnostics. '
        'policy.json records all validation candidates; reward_model.json is sufficient to reload the fitted model.\n\n'
        '## Limitations\n\nSparse clicks, high-variance importance weights, short time coverage, repeated-user dependence, '
        'unknown feature semantics, position interactions and possible temporal drift constrain inference. '
        'No matched on-policy ground truth for the contextual target; no online, demographic fairness, '
        'joint-slate or movie-model conclusions. Confidence intervals condition on the fitted policy, not the full learning procedure.\n\n'
        '## Provenance and usage\n\nDataset terms are CC BY 4.0 as stated in its README. Preserve attribution and source citation. '
        'Model and report manifests record exact input and code fingerprints.\n', encoding='utf-8')
    verify(inputs)
    if hashes(inputs / 'manifest.json')['sha256'] != fingerprint:
        raise ValueError('Input manifest changed')
    finish(out, {'stage': 'open_bandit_evaluate', 'input_manifest_sha256': fingerprint,
                 'configuration': {'bootstrap': bootstrap, 'seed': seed}})
    print(render_report(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    evaluate(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
