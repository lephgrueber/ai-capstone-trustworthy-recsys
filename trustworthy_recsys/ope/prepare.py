"""Prepare random-policy Open Bandit logs; never infer missing propensities."""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import hashes, write_json
from .common import begin, finish


def split_temporally(frame):
    frame = frame.sort_values(['timestamp_utc', 'source_row'], kind='stable').reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError('Too few observations for temporal splits')
    times = frame.timestamp_utc
    validation_start, test_start = times.iloc[int(.8 * len(frame))], times.iloc[int(.9 * len(frame))]
    splits = {'train': frame[times < validation_start],
              'validation': frame[(times >= validation_start) & (times < test_start)],
              'test': frame[times >= test_start]}
    if any(part.empty for part in splits.values()):
        raise ValueError('Timestamp ties leave an empty partition')
    return splits, {'validation_start': str(validation_start), 'test_start': str(test_start)}


def prepare(raw_dir, output_dir, campaign='all', position=1, chunk_size=100000):
    if campaign not in {'all', 'men', 'women'} or position not in {1, 2, 3}:
        raise ValueError('Unsupported campaign or position')
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    raw = Path(raw_dir)
    source = raw / 'random' / campaign / f'{campaign}.csv'
    item_source = source.with_name('item_context.csv')
    sources = [source, item_source, raw / 'README', raw / 'VERSION']
    fingerprints = {str(p.relative_to(raw)): hashes(p)['sha256'] for p in sources}
    out = begin(output_dir)
    items = pd.read_csv(item_source)
    if items.item_id.isna().any() or items.item_id.duplicated().any():
        raise ValueError('Missing or duplicate catalog IDs')
    if not np.equal(items.item_id, np.floor(items.item_id)).all():
        raise ValueError('Nonintegral catalog IDs')
    actions = sorted(items.item_id.astype(int).tolist())
    mapping = {item: i for i, item in enumerate(actions)}
    if len(actions) < 2:
        raise ValueError('Need at least two candidate actions')
    header = pd.read_csv(source, nrows=0).columns.tolist()
    context = sorted(c for c in header if c.startswith('user_feature_'))
    if not context:
        raise ValueError('No user context columns')
    columns = ['timestamp', 'item_id', 'position', 'click', 'propensity_score', *context]
    parts, total, position_counts = [], 0, {}
    for chunk in pd.read_csv(source, usecols=columns, dtype={c: 'string' for c in context}, chunksize=chunk_size):
        chunk['source_row'] = np.arange(total, total + len(chunk), dtype=np.int64)
        total += len(chunk)
        if chunk[columns].isna().any().any():
            raise ValueError('Missing required source values; no silent deletion')
        if not chunk.position.isin([1, 2, 3]).all() or not chunk.click.isin([0, 1]).all():
            raise ValueError('Invalid position or binary reward')
        p = chunk.propensity_score.to_numpy(float)
        if not (np.isfinite(p).all() and ((p > 0) & (p <= 1)).all()):
            raise ValueError('Invalid propensity')
        if not chunk.item_id.isin(actions).all():
            raise ValueError('Logged action absent from item catalog')
        for key, count in chunk.position.value_counts().items():
            position_counts[str(int(key))] = position_counts.get(str(int(key)), 0) + int(count)
        chunk['timestamp_utc'] = pd.to_datetime(chunk.timestamp, utc=True, format='ISO8601', errors='raise')
        if chunk.timestamp_utc.isna().any():
            raise ValueError('Missing timestamp')
        selected = chunk.loc[chunk.position == position].copy()
        # This implementation requires the complete uniform action support at this position.
        if not np.allclose(selected.propensity_score, 1 / len(actions), rtol=1e-7, atol=1e-12):
            raise ValueError('Random propensity does not match catalog size; investigate availability before OPE')
        parts.append(selected.drop(columns='timestamp'))
    frame = pd.concat(parts, ignore_index=True).rename(columns={'item_id': 'action_id', 'click': 'reward'})
    frame['action_index'] = frame.action_id.map(mapping).astype('int32')
    frame['event_id'] = 'random/' + campaign + '/' + frame.source_row.astype(str)
    frame['campaign'], frame['behavior_policy'] = campaign, 'random'
    splits, boundaries = split_temporally(frame)
    if set(splits['train'].action_id) != set(actions):
        raise ValueError('Training partition does not cover the complete action catalog')
    statistics = {'source_rows': total, 'source_position_counts': position_counts,
                  'selected_rows': len(frame), 'excluded_other_positions': total-len(frame),
                  'n_actions': len(actions), 'context_columns': context,
                  'campaign': campaign, 'position': position, 'boundaries': boundaries, 'partitions': {}}
    for name, part in splits.items():
        part.to_parquet(out / f'{name}.parquet', index=False)
        statistics['partitions'][name] = {
            'rows': len(part), 'clicks': int(part.reward.sum()), 'ctr': float(part.reward.mean()),
            'first_timestamp': str(part.timestamp_utc.min()), 'last_timestamp': str(part.timestamp_utc.max()),
            'actions': int(part.action_id.nunique()),
            'propensity_min': float(part.propensity_score.min()), 'propensity_max': float(part.propensity_score.max()),
            'context_cardinality': {c: int(part[c].nunique()) for c in context},
            'unseen_context_values': {c: int((~part[c].isin(splits['train'][c])).sum()) for c in context},
            'action_counts': {str(k): int(v) for k, v in part.action_id.value_counts().sort_index().items()},
        }
    if sum(len(p) for p in splits.values()) != len(frame) or not frame.event_id.is_unique:
        raise ValueError('Event accounting failed')
    items.to_parquet(out / 'item_context.parquet', index=False)
    write_json(out / 'action_ids.json', actions)
    write_json(out / 'statistics.json', statistics)
    write_json(out / 'schema.json', {
        'version': 1, 'columns': {c: str(t) for c, t in frame.dtypes.items()},
        'context_columns': context, 'position_convention': 'original 1-based display position',
        'reward': 'binary click; all nonclick exposures retained',
        'action_mapping': 'action_ids.json[index] gives original campaign-local item ID',
        'event_id': 'source policy/campaign/zero-based data-row ordinal; not a user or slate ID',
        'excluded_features': 'affinity features excluded pending temporal provenance review; item features archived, not modeled',
    })
    for name in ['README', 'VERSION']:
        shutil.copyfile(raw / name, out / f'source_{name}.txt')
    (out / 'DATA_CARD.md').write_text(
        '# Open Bandit Data Card\n\n'
        '## Overview and provenance\n\n'
        f'Open Bandit Dataset, ZOZO/ZOZOTOWN. Random policy, {campaign} campaign, position {position}. '
        f'{total:,} source rows; {len(frame):,} selected exposures; {len(actions)} catalog actions. '
        'Local source files are immutable; source hashes, version and README are preserved. '
        'Download date and downloading party are not known. Hashes identify the local files, not an independently authenticated download.\n\n'
        '## Licensing and usage\n\n'
        'The supplied README states CC BY 4.0. Retain attribution, license and modification notices. '
        'Cite Saito, Aihara, Matsutani and Narita, Open Bandit Dataset and Pipeline (arXiv:2008.07146). '
        'These Parquet tables, action mappings and temporal partitions are project transformations.\n\n'
        '## Structure and splits\n\n'
        'Click reward, original logging propensity, categorical user context, action and source-row identity are retained. '
        'Approximate chronological 80/10/10 event splits keep equal timestamps together. '
        f'Validation begins {boundaries["validation_start"]}; test begins {boundaries["test_start"]}. '
        'See schema.json and statistics.json for exact types, counts and missingness-related checks. '
        'No clicks or nonclicks are selectively removed. Affinities and item features are not model inputs.\n\n'
        '## Limitations and risks\n\n'
        'This is a position-specific fashion click benchmark, not evaluation of the MovieLens model. '
        'Other displayed items may influence clicks: this is not joint-slate policy evaluation. '
        'Propensity checks assume a fixed uniform action catalog at the selected position. '
        'No verified user/slate identifier is available in the selected schema; anonymized features are not user IDs. '
        'Context features are treated as pre-decision covariates according to the source description; their upstream construction is not independently audited. '
        'Short time coverage, repeated-user dependence, temporal drift, and limited clicks constrain inference. '
        'Campaign names do not justify demographic fairness claims.\n', encoding='utf-8')
    if fingerprints != {str(p.relative_to(raw)): hashes(p)['sha256'] for p in sources}:
        raise ValueError('Source changed during preparation')
    finish(out, {'stage': 'open_bandit_prepare', 'sources': fingerprints,
                 'configuration': {'campaign': campaign, 'position': position, 'ratios': [.8, .1, .1]}})
    print(f'Prepared {len(frame):,} exposures in {out}', flush=True)
    return statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--campaign', choices=['all', 'men', 'women'], default='all')
    parser.add_argument('--position', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--chunk-size', type=int, default=100000)
    prepare(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
