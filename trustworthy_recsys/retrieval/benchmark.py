"""Repeated ANN/exact validation benchmark and portable model reload audit."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from eval.harness import RecommendationRequest
from trustworthy_recsys.data.reporting import hashes, write_json
from trustworthy_recsys.ope.common import begin, finish
from .evaluate import percentiles
from .index import ExactIndex, filtered_search
from .recommender import RetrievalRecommender
from .train import sampled_examples


def benchmark(inputs_dir, model_dir, output_dir, reload_dir, users=200, repetitions=3):
    if users < 1 or repetitions < 1:
        raise ValueError('Users and repetitions must be positive')
    torch.set_num_threads(2)
    out = begin(output_dir)
    started = time.perf_counter()
    with RetrievalRecommender.load(inputs_dir, model_dir) as rec:
        original_load = time.perf_counter()-started
        source = Path(rec.inputs.meta['source_run'])
        path = source/rec.inputs.meta['scenario']/'validation_warm.jsonl'
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if hashes(source/'manifest.json')['sha256'] != rec.inputs.meta['source_manifest_sha256'] or hashes(path)['sha256'] != manifest['outputs'][path.relative_to(source).as_posix()]['sha256']:
            raise ValueError('Validation source hash mismatch')
        rows = sampled_examples(path, users, 42)
        exact = ExactIndex(rec.item_vectors)
        prepared = []
        for row in rows:
            request = RecommendationRequest(row['user_id'], row['history_items'])
            query = rec.query_vector(request.history_items)
            if query is not None:
                prepared.append((request, query, rec.excluded(request)))
        if not prepared:
            raise ValueError('No validation users have usable histories')
        for _, query, excluded in prepared[:10]:
            for index in (rec.index, exact):
                filtered_search(index, query, 500, excluded)
        measurements = []
        for repeat in range(repetitions):
            for number, (request, query, excluded) in enumerate(prepared):
                for k in (100, 500):
                    order = [('ann', rec.index), ('exact', exact)]
                    if (number+repeat) % 2:
                        order.reverse()
                    results, timings = {}, {}
                    for name, index in order:
                        start = time.perf_counter()
                        result = filtered_search(index, query, k, excluded)
                        timings[name+'_ms'] = (time.perf_counter()-start)*1000
                        if len(result) != len(set(result)) or set(result)&excluded or any(i < 0 or i >= len(rec.inputs.item_ids) for i in result):
                            raise ValueError('Invalid search output')
                        results[name] = result
                    measurements.append({'user_id': request.user_id, 'repeat': repeat, 'k': k,
                        'overlap': len(set(results['ann'])&set(results['exact']))/len(results['exact']), **timings})
        # A separately extracted bundle must load against the same remote index.
        started = time.perf_counter()
        with RetrievalRecommender.load(Path(reload_dir)/'inputs', Path(reload_dir)/'model') as restored:
            restored_load = time.perf_counter()-started
            if rec.inputs.item_ids != restored.inputs.item_ids or rec.inputs.user_ids != restored.inputs.user_ids:
                raise ValueError('Reloaded mappings differ')
            np.testing.assert_array_equal(rec.item_vectors, restored.item_vectors)
            np.testing.assert_allclose(restored.model.all_item_vectors(), restored.item_vectors, atol=1e-6)
            probes = [r for r, _, _ in prepared[:10]] + [RecommendationRequest('__new__', []), RecommendationRequest('__new__', ['__unknown__'])]
            for request in probes:
                if rec(request, 100) != restored(request, 100):
                    raise ValueError('Reloaded predictions differ')
        summary = {}
        for k in (100, 500):
            values = [m for m in measurements if m['k'] == k]
            overlap = float(np.mean([m['overlap'] for m in values]))
            ann = percentiles([m['ann_ms'] for m in values])
            reference = percentiles([m['exact_ms'] for m in values])
            summary[str(k)] = {'queries': len(values), 'mean_exact_overlap': overlap,
                'ann_ms': ann, 'exact_ms': reference, 'agreement_target_passed': overlap >= .99,
                'latency_target_passed': ann['p95'] <= 100,
                'ann_faster_at_median': ann['p50'] < reference['p50']}
        report = {'protocol': {'users_requested': users, 'users_with_history': len(prepared),
            'repetitions': repetitions, 'seed': 42, 'warmups_per_index': min(10, len(prepared)),
            'timing': 'search and seen filtering including RPC; precomputed query; alternating order; sequential local requests',
            'targets': 'New engineering acceptance targets: mean exact overlap >= .99 and local ANN p95 <= 100 ms at each k. These are not historical SLOs or concurrent-load guarantees.',
            'platform': platform.platform(), 'processor': platform.processor(), 'torch': torch.__version__,
            'numpy': np.__version__, 'torch_threads': 2, 'index': rec.index.descriptor,
            'validation_sha256': hashes(path)['sha256'], 'model_manifest_sha256': hashes(Path(model_dir)/'manifest.json')['sha256']},
            'results': summary,
            'reload': {'mappings_equal': True, 'embeddings_equal': True, 'model_vectors_match': True,
                'remote_mappings_and_vectors_verified': True, 'identical_prediction_probes': len(probes),
                'original_load_seconds': original_load, 'bundle_load_seconds': restored_load},
            'measurements': measurements}
        write_json(out/'benchmark.json', report)
    finish(out, {'stage': 'retrieval_ann_and_reload_validation'})
    print(json.dumps({'results': summary, 'reload': report['reload']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-dir', 'model-dir', 'output-dir', 'reload-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--users', type=int, default=200)
    parser.add_argument('--repetitions', type=int, default=3)
    benchmark(**vars(parser.parse_args()))
