"""Build a reranker table: one row per (viewer, two-tower candidate) with features and a label.

Proposed workflow (M3 feedback §1): build one table from Train B examples (reranker
fitting; history and features from Train A) and one from dev examples (selection; history
and features from Train A + B). Example files exported by trustworthy_recsys.data.windows
carry a sidecar with their history cutoff, which fixes the feature window here, so a table
can never use ratings at or after its own label window.

Legacy example files (M3 validation_warm/test_warm) have no sidecar and keep the original
behavior: features from the full training partition.

Outputs:
- rows.parquet: candidate rows (label = candidate is a later like of the viewer).
- users.parquet: one row per viewer, including viewers with no hit in the candidates,
  so end-to-end metrics keep full denominators and per-group counts.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from eval.harness import EvalExample, RecommendationRequest, load_evaluation_examples
from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json
from trustworthy_recsys.data.windows import history_bucket, load_records, read_window_meta
from trustworthy_recsys.retrieval.index import ExactIndex
from trustworthy_recsys.retrieval.recommender import RetrievalRecommender
from trustworthy_recsys.retrieval.train import sampled_examples
from .features import FEATURE_NAMES, RankingFeatures

POPULARITY_DEPTH = 50


def verify_run_matches_inputs(run_dir, inputs):
    """The run must contain the same data the retrieval model was fitted on."""
    scenario = inputs.meta["scenario"]
    # JSON files may differ only in line endings across platforms; Parquet bytes must match.
    for name, digest in inputs.meta["source_hashes"].items():
        if name.endswith(".parquet") and hashes(Path(run_dir) / name)["sha256"] != digest:
            raise ValueError(f"Pipeline run does not match retrieval inputs: {name}")
    if "window" in inputs.meta:
        return
    with (Path(run_dir) / f"{scenario}/train_item_ids.json").open(encoding="utf-8") as handle:
        if [str(item) for item in json.load(handle)] != list(inputs.item_ids):
            raise ValueError("Pipeline training catalog does not match retrieval inputs")


def _field(example, name, default=()):
    return example.get(name, default) if isinstance(example, dict) else getattr(example, name, default)


def popularity_order(features):
    """Catalog indices by descending positive count in the feature window (ties by index)."""
    counts = features.positive_counts
    return np.lexsort((np.arange(len(counts)), -counts))


def build_rows(rec, features, examples, depth, progress_every=1000, popularity=None):
    """Yield per-viewer column blocks; labels are only read after features are built."""
    started = time.perf_counter()
    for count, example in enumerate(examples, start=1):
        user_id = _field(example, "user_id")
        history = list(_field(example, "history_items"))
        exclude = tuple(_field(example, "other_rated_items"))
        request = RecommendationRequest(user_id, history, exclude)
        indices, scores, fallback = rec.recommend_scored(request, depth)
        matrix = features.build(history, indices, scores, user_id)
        movie_ids = [rec.inputs.item_ids[index] for index in indices]
        relevant = set(_field(example, "relevant_items"))
        labels = np.array([movie in relevant for movie in movie_ids], dtype="int8")
        viewer = {"user_id": user_id, "relevant_count": len(relevant),
                  "history_length": len(features.history_indices(history)),
                  "fallback": bool(fallback), "candidate_hits": int(labels.sum())}
        if popularity is not None:
            seen = set(features.history_indices(history)) | {
                features.item_map[x] for x in exclude if x in features.item_map}
            ranked = [i for i in popularity[:POPULARITY_DEPTH + len(seen)] if int(i) not in seen][:POPULARITY_DEPTH]
            viewer["popularity_labels"] = [int(features.item_ids[i] in relevant) for i in ranked]
        yield {
            "user_id": [user_id] * len(indices),
            "movie_id": movie_ids,
            "label": labels,
            "fallback": np.full(len(indices), fallback),
            "user_relevant_count": np.full(len(indices), len(relevant), dtype="int32"),
            "features": matrix,
            "viewer": viewer,
        }
        if progress_every and count % progress_every == 0:
            print(json.dumps({"users": count, "seconds": round(time.perf_counter() - started, 1)}), flush=True)


def to_table(blocks):
    if not blocks:
        raise ValueError("No examples produced rows")
    features = np.concatenate([block["features"] for block in blocks])
    columns = {
        "user_id": pa.array([u for block in blocks for u in block["user_id"]], pa.string()),
        "movie_id": pa.array([m for block in blocks for m in block["movie_id"]], pa.string()),
        "label": np.concatenate([block["label"] for block in blocks]),
        "fallback": np.concatenate([block["fallback"] for block in blocks]),
        "user_relevant_count": np.concatenate([block["user_relevant_count"] for block in blocks]),
    }
    columns.update({name: features[:, i] for i, name in enumerate(FEATURE_NAMES)})
    return pa.table(columns)


def to_users_table(blocks):
    viewers = [block["viewer"] for block in blocks]
    frame = pd.DataFrame(viewers)
    frame["history_bucket"] = [history_bucket(n) for n in frame["history_length"]]
    return pa.Table.from_pandas(frame, preserve_index=False)


def load_examples(examples_path, max_users=0, seed=42):
    """Window exports load as dicts (with exclusion fields); legacy files as EvalExample."""
    meta = read_window_meta(examples_path)
    if meta is not None:
        if max_users:
            return meta, sampled_examples(examples_path, max_users, seed)
        return meta, list(load_records(examples_path))
    if max_users:
        return None, [EvalExample(r["user_id"], r["history_items"], r["relevant_items"])
                      for r in sampled_examples(examples_path, max_users, seed)]
    return None, load_evaluation_examples(examples_path)


def build(run_dir, inputs_dir, model_dir, examples_path, output_dir, depth=500, max_users=0, seed=42, threads=2):
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    if depth <= 0 or max_users < 0:
        raise ValueError("depth must be positive and max_users non-negative")
    torch.set_num_threads(threads)
    inputs, model, vectors = RetrievalRecommender.load_verified(inputs_dir, model_dir)
    verify_run_matches_inputs(run_dir, inputs)
    rec = RetrievalRecommender(inputs, model, ExactIndex(vectors))
    examples_path = Path(examples_path)
    window, examples = load_examples(examples_path, max_users, seed)
    tower_cutoff = int(pd.Timestamp(inputs.meta["training_cutoff_utc"]).timestamp())
    if window is None:
        cutoff, partitions = tower_cutoff, ("train",)
    else:
        cutoff, partitions = window["history_end"], tuple(window["partitions"])
        if cutoff < tower_cutoff:
            raise ValueError("Examples' history ends before the towers' fitting window")
    load_started = time.perf_counter()
    features = RankingFeatures.from_run(
        run_dir, inputs.meta["scenario"], inputs.item_ids, inputs.item_features,
        cutoff, partitions=partitions, threshold=inputs.meta["positive_threshold"],
    )
    load_seconds = time.perf_counter() - load_started

    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("Reranker table build in progress.\n")
    started = time.perf_counter()
    blocks = list(build_rows(rec, features, examples, depth, popularity=popularity_order(features)))
    table = to_table(blocks)
    users = to_users_table(blocks)
    del blocks
    build_seconds = time.perf_counter() - started
    pq.write_table(table, out / "rows.parquet", compression="zstd")
    pq.write_table(users, out / "users.parquet", compression="zstd")

    hits = users["candidate_hits"].to_numpy()
    summary = {
        "users": int(users.num_rows),
        "rows": int(table.num_rows),
        "positive_rows": int(hits.sum()),
        "users_with_positive_candidate": int((hits > 0).sum()),
        "fallback_users": int(users["fallback"].to_numpy().sum()),
        "history_buckets": {str(k): int(v) for k, v in
                            pd.Series(users["history_bucket"].to_pylist()).value_counts().sort_index().items()},
    }
    feature_cutoff = pd.Timestamp(cutoff, unit="s", tz="UTC").isoformat()
    write_json(out / "manifest.json", {
        "description": "Reranker rows: two-tower candidates, pre-window features, later-window labels",
        "depth": depth, "max_users": max_users, "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "label": "1 if candidate is in the example's relevant_items (later rating >= threshold)",
        "feature_cutoff_utc": feature_cutoff,
        "feature_partitions": list(partitions),
        "tower_cutoff_utc": inputs.meta["training_cutoff_utc"],
        "window": None if window is None else {k: window[k] for k in
                                               ("history_end_utc", "label_start_utc", "label_end_utc")},
        "popularity_baseline": f"top {POPULARITY_DEPTH} by positive count before the feature cutoff, "
                               "excluding the viewer's rated movies",
        "examples": str(examples_path), "examples_sha256": hashes(examples_path)["sha256"],
        "run_dir": str(run_dir),
        "input_manifest_sha256": hashes(Path(inputs_dir) / "manifest.json")["sha256"],
        "model_manifest_sha256": hashes(Path(model_dir) / "manifest.json")["sha256"],
        "rows_sha256": hashes(out / "rows.parquet")["sha256"],
        "users_sha256": hashes(out / "users.parquet")["sha256"],
        "summary": summary,
        "timing_seconds": {"feature_fit": load_seconds, "build": build_seconds,
                           "per_user_ms": build_seconds * 1000 / max(1, summary["users"])},
        "code": code_provenance(),
    })
    (out / "INCOMPLETE").unlink()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="Completed pipeline run, e.g. data/processed/movielens_phase1")
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--examples", type=Path, required=True,
                   help="Window export (train_b.jsonl to fit, dev.jsonl to select, test.jsonl to evaluate)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--depth", type=int, default=500)
    p.add_argument("--max-users", type=int, default=0, help="0 uses every example")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    summary = build(a.run_dir, a.inputs_dir, a.model_dir, a.examples, a.output_dir, a.depth, a.max_users, a.seed, a.threads)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
