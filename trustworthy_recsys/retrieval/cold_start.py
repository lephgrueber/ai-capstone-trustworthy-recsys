"""Optional experiment: recommend movies the towers never saw (feedback §8).

Trial group: movies first rated in Train B, i.e. absent from the Train A tower catalog. Their
vectors come from metadata alone (a zero movie-ID vector plus genre weights), are added to
an exact index next to the original catalog, and dev viewers with earlier favorites are
served from the combined index. Reported for the same dev requests:

- new-movie recall@K: share of each viewer's dev likes *of new movies* that reach the top K
  (viewers with at least one such like);
- original-catalog recall@K with the original index vs the combined index, to see whether
  adding new movies changes quality for the original catalog.

Train one tower set with ``--identity-dropout`` (e.g. 0.2) and one without, then compare.
Genre-only towers are supported; towers with credit/runtime/tag inputs would also need
those rows for the new movies. Reranker scoring of new movies is not covered: the
reranker's movie statistics are indexed by the tower catalog.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from trustworthy_recsys.data.reporting import code_provenance, write_json
from trustworthy_recsys.data.windows import load_events, load_records, read_window_meta
from .index import ExactIndex, filtered_search
from .recommender import RetrievalRecommender, request_from_record


def genre_rows(genres_lists, vocabulary):
    index = {g: i for i, g in enumerate(vocabulary)}
    rows = np.zeros((len(genres_lists), len(vocabulary)), dtype="float32")
    for row, genres in enumerate(genres_lists):
        known = [g for g in genres if g in index]
        for g in known:
            rows[row, index[g]] = 1 / len(genres)  # same fractional weights as the inputs
    return rows


def run(run_dir, inputs_dir, model_dir, examples, output, k=100, max_viewers=0):
    inputs, model, vectors = RetrievalRecommender.load_verified(inputs_dir, model_dir)
    if model.side is not None:
        raise ValueError("Cold-start vectors currently support genre-input towers only")
    window = read_window_meta(examples)
    if window is None:
        raise ValueError("Use a window export (e.g. dev.jsonl)")
    movies = pq.read_table(Path(run_dir) / "movies.parquet", columns=["movie_id", "genres"]).to_pandas()
    catalog = set(inputs.item_ids)
    # Movies first rated in Train B: rated before the dev start but absent from the Train A catalog.
    events = load_events(run_dir, inputs.meta["scenario"], tuple(window["partitions"]), end=window["history_end"])
    rated_before_dev = {str(int(m)) for m in np.unique(events.item)}
    new_ids = sorted((rated_before_dev - catalog) & set(movies["movie_id"]), key=int)
    records = list(load_records(examples))
    genres = dict(zip(movies["movie_id"], map(list, movies["genres"])))
    new_vectors = model.metadata_only_vectors(genre_rows([genres[m] for m in new_ids], inputs.meta["genres"]))
    combined_ids = list(inputs.item_ids) + new_ids
    original_index, combined_index = ExactIndex(vectors), ExactIndex(np.vstack([vectors, new_vectors]))
    combined_map = {m: i for i, m in enumerate(combined_ids)}
    rec = RetrievalRecommender(inputs, model, original_index)
    new_set = set(new_ids)
    new_recall, original_before, original_after = [], [], []
    viewers = 0
    for record in records:
        request = request_from_record(record, window["history_end"])
        query = rec.query_vector(request.history_items, request)
        if query is None:
            continue
        viewers += 1
        if max_viewers and viewers > max_viewers:
            break
        rated = set(request.history_items) | set(request.exclude_items)
        excluded_original = {inputs.item_map[m] for m in rated if m in inputs.item_map}
        excluded_combined = {combined_map[m] for m in rated if m in combined_map}
        before = {inputs.item_ids[i] for i in filtered_search(original_index, query, k, excluded_original)}
        after = {combined_ids[i] for i in filtered_search(combined_index, query, k, excluded_combined)}
        relevant = set(record["relevant_items"])
        old_likes, new_likes = relevant & catalog, relevant & new_set
        if old_likes:
            original_before.append(len(old_likes & before) / len(old_likes))
            original_after.append(len(old_likes & after) / len(old_likes))
        if new_likes:
            new_recall.append(len(new_likes & after) / len(new_likes))
    report = {
        "k": k, "new_movies": len(new_ids), "personalized_viewers": min(viewers, max_viewers or viewers),
        "new_movie_recall": {"viewers": len(new_recall), f"recall_at_{k}": float(np.mean(new_recall)) if new_recall else None},
        "original_catalog_recall": {"viewers": len(original_before),
                                    f"recall_at_{k}_original_index": float(np.mean(original_before)),
                                    f"recall_at_{k}_with_new_movies": float(np.mean(original_after))},
        "identity_dropout": model.identity_dropout, "model": str(model_dir), "code": code_provenance(),
    }
    write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--examples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max-viewers", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    print(json.dumps(run(a.run_dir, a.inputs_dir, a.model_dir, a.examples, a.output, a.k, a.max_viewers), indent=2))


if __name__ == "__main__":
    main()
