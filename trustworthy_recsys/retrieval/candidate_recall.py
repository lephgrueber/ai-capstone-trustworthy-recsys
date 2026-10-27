"""Measure two-tower candidate recall at reranker-sized depths (e.g. 100 vs 500).

Recall@K of the retrieval stage is the ceiling for any reranker that reorders its top K.
Exact cosine search over the saved embeddings is used (ANN agreement with exact top-100 was
99.9%), so no Weaviate instance is needed. Defaults to validation examples so reranker
design decisions never consult test labels.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from eval.harness import EvalExample, RecommendationRequest, load_evaluation_examples
from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json
from trustworthy_recsys.evaluation.metrics import recall_at_k
from .index import ExactIndex
from .recommender import RetrievalRecommender
from .train import sampled_examples

DEFAULT_KS = (10, 50, 100, 200, 300, 500)


def history_bucket(length):
    return 'history_0' if length == 0 else 'history_1_20' if length <= 20 else 'history_21_plus'


def summarize(rows, ks):
    """Macro user means of each recall depth for one population."""
    if not rows:
        return {'users': 0}
    return {'users': len(rows), **{f'recall_at_{k}': float(np.mean([row[k] for row in rows])) for k in ks}}


def measure(rec, examples, ks):
    ks = sorted(set(ks))
    if not ks or ks[0] <= 0:
        raise ValueError('ks must be positive')
    depth = ks[-1]
    catalog = set(rec.inputs.item_ids)
    populations = defaultdict(list)
    catalog_ceiling, latency = [], []
    for example in examples:
        request = RecommendationRequest(example.user_id, list(example.history_items))
        started = time.perf_counter()
        indices, fallback = rec.recommend_indices(request, depth)
        latency.append((time.perf_counter() - started) * 1000)
        ranked = [rec.inputs.item_ids[i] for i in indices]
        popular = [rec.inputs.item_ids[i] for i in rec.popularity_indices(rec.excluded(request), depth)]
        relevant = set(example.relevant_items)
        catalog_ceiling.append(len(relevant & catalog) / len(relevant))
        model_row = {k: recall_at_k(relevant, ranked, k) for k in ks}
        popularity_row = {k: recall_at_k(relevant, popular, k) for k in ks}
        path = 'fallback' if fallback else 'learned'
        for name in ('all', path, history_bucket(len(example.history_items))):
            populations[f'two_tower/{name}'].append(model_row)
            populations[f'popularity/{name}'].append(popularity_row)
    report = {name: summarize(rows, ks) for name, rows in sorted(populations.items())}
    return {
        'ks': ks,
        'metrics': report,
        'mean_in_catalog_label_fraction': float(np.mean(catalog_ceiling)) if catalog_ceiling else None,
        'exact_search_latency_ms': dict(zip(['p50', 'p95', 'p99'], map(float, np.percentile(latency, [50, 95, 99])))) if latency else {},
    }


def run(inputs_dir, model_dir, examples_path, output, ks=DEFAULT_KS, max_users=0, seed=42, threads=2):
    output = Path(output)
    if output.exists():
        raise ValueError('Output report already exists; choose a new path')
    if max_users < 0:
        raise ValueError('max_users must be non-negative')
    torch.set_num_threads(threads)
    inputs, model, vectors = RetrievalRecommender.load_verified(inputs_dir, model_dir)
    rec = RetrievalRecommender(inputs, model, ExactIndex(vectors))
    examples_path = Path(examples_path)
    if max_users:
        # Same label-independent hash sample used for checkpoint selection.
        records = sampled_examples(examples_path, max_users, seed)
        examples = [load_record(record) for record in records]
    else:
        examples = load_evaluation_examples(examples_path)
    result = measure(rec, examples, ks)
    result.update({
        'examples': str(examples_path), 'examples_sha256': hashes(examples_path)['sha256'],
        'max_users': max_users, 'seed': seed, 'search': 'exact cosine over saved item_embeddings.npy',
        'model_manifest_sha256': hashes(Path(model_dir) / 'manifest.json')['sha256'],
        'code': code_provenance(),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def load_record(record):
    return EvalExample(user_id=record['user_id'], history_items=record['history_items'],
                       relevant_items=record['relevant_items'])


def format_table(result):
    ks = result['ks']
    lines = ['| Method / population | Users | ' + ' | '.join(f'R@{k}' for k in ks) + ' |',
             '|---|---:|' + '---:|' * len(ks)]
    for name, row in result['metrics'].items():
        cells = ' | '.join(f"{row[f'recall_at_{k}']:.4f}" for k in ks) if row['users'] else ' | '.join('-' for _ in ks)
        lines.append(f"| {name} | {row['users']} | {cells} |")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--examples', type=Path, required=True, help='e.g. <run>/ratio_80_10_10/validation_warm.jsonl')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ks', type=int, nargs='+', default=list(DEFAULT_KS))
    p.add_argument('--max-users', type=int, default=0, help='0 evaluates every example')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    result = run(a.inputs_dir, a.model_dir, a.examples, a.output, a.ks, a.max_users, a.seed, a.threads)
    print(format_table(result))
    print(json.dumps({k: result[k] for k in ('mean_in_catalog_label_fraction', 'exact_search_latency_ms')}))


if __name__ == '__main__':
    main()
