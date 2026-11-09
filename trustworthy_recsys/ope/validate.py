"""Independent semantic audit of prepared Open Bandit artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import hashes, write_json
from .common import begin, finish, verify


def validate(inputs_dir, output_dir, compare_dir=None):
    root = Path(inputs_dir)
    manifest = verify(root)
    required = {'train.parquet', 'validation.parquet', 'test.parquet', 'schema.json',
                'statistics.json', 'action_ids.json', 'item_context.parquet',
                'source_README.txt', 'source_VERSION.txt', 'DATA_CARD.md'}
    if not required <= set(manifest['outputs']):
        raise ValueError('Manifest omits required OPE artifacts')
    if compare_dir:
        compare_dir = Path(compare_dir)
        verify(compare_dir)
    stats = json.loads((root/'statistics.json').read_text(encoding='utf-8'))
    schema = json.loads((root/'schema.json').read_text(encoding='utf-8'))
    actions = json.loads((root/'action_ids.json').read_text(encoding='utf-8'))
    if len(actions) != len(set(actions)) or len(actions) != stats['n_actions']:
        raise ValueError('Invalid action mapping')
    if set(pd.read_parquet(root/'item_context.parquet').item_id) != set(actions):
        raise ValueError('Item context/catalog mismatch')
    previous_end, seen, rows, clicks = None, set(), 0, 0
    checks = {}
    for name in ['train', 'validation', 'test']:
        frame = pd.read_parquet(root/f'{name}.parquet')
        if frame.empty or frame.isna().any().any() or not frame.event_id.is_unique:
            raise ValueError(f'{name}: empty, missing or duplicate events')
        if set(frame.columns) != set(schema['columns']):
            raise ValueError('Schema columns differ')
        if any(str(frame[c].dtype) != dtype for c, dtype in schema['columns'].items()):
            raise ValueError('Schema types differ')
        if seen.intersection(frame.event_id):
            raise ValueError('Cross-partition event leakage')
        seen.update(frame.event_id)
        if not frame.timestamp_utc.is_monotonic_increasing:
            raise ValueError('Unsorted timestamps')
        if previous_end is not None and previous_end >= frame.timestamp_utc.min():
            raise ValueError('Temporal overlap')
        previous_end = frame.timestamp_utc.max()
        indices = frame.action_index.to_numpy()
        if (indices < 0).any() or (indices >= len(actions)).any():
            raise ValueError('Action index out of bounds')
        if not np.array_equal(np.array(actions)[indices], frame.action_id):
            raise ValueError('Action IDs misaligned')
        if not frame.reward.isin([0, 1]).all() or not np.allclose(frame.propensity_score, 1/len(actions)):
            raise ValueError('Invalid reward or uniform propensity')
        if not frame.position.eq(stats['position']).all() or not frame.campaign.eq(stats['campaign']).all() or not frame.behavior_policy.eq('random').all():
            raise ValueError('Scope differs from declared experiment')
        recorded = stats['partitions'][name]
        if recorded['rows'] != len(frame) or recorded['clicks'] != int(frame.reward.sum()):
            raise ValueError('Statistics differ from artifact records')
        if compare_dir:
            pd.testing.assert_frame_equal(frame, pd.read_parquet(compare_dir/f'{name}.parquet'))
        rows += len(frame)
        clicks += int(frame.reward.sum())
        checks[name] = {'rows': len(frame), 'clicks': int(frame.reward.sum()), 'passed': True}
    if rows != stats['selected_rows'] or rows+stats['excluded_other_positions'] != stats['source_rows']:
        raise ValueError('Source row accounting failed')
    if compare_dir:
        for file in ['action_ids.json', 'statistics.json', 'schema.json']:
            if json.loads((root/file).read_text()) != json.loads((compare_dir/file).read_text()):
                raise ValueError(f'Reproducibility mismatch: {file}')
    out = begin(output_dir)
    result = {'rows': rows, 'clicks': clicks, 'checks': checks,
              'hashes_verified': True, 'schema_verified': True, 'temporal_separation': True,
              'event_ids_disjoint': True, 'action_mapping_verified': True,
              'logical_reproduction_verified': bool(compare_dir),
              'input_manifest_sha256': hashes(root/'manifest.json')['sha256'],
              'comparison_manifest_sha256': hashes(compare_dir/'manifest.json')['sha256'] if compare_dir else None}
    write_json(out/'validation.json', result)
    finish(out, {'stage': 'open_bandit_semantic_validation'})
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--compare-dir', type=Path)
    validate(**vars(p.parse_args()))
