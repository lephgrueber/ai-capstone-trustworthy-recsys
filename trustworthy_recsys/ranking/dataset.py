"""Build a reranker table: one row per (user, two-tower candidate) with features and a label.

Run once on validation examples (reranker training data) and once on test examples
(final evaluation). Features use training-period data only; the label is whether the
candidate is in the example's held-out relevant_items.
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
from trustworthy_recsys.retrieval.index import ExactIndex
from trustworthy_recsys.retrieval.recommender import RetrievalRecommender
from trustworthy_recsys.retrieval.train import sampled_examples
from .features import FEATURE_NAMES, RankingFeatures


def verify_run_matches_inputs(run_dir, inputs):
    """The run must contain the same training data the retrieval model was fitted on."""
    scenario = inputs.meta["scenario"]
    # JSON files may differ only in line endings across platforms; Parquet bytes must match.
    for name in ("movies.parquet", f"{scenario}/train.parquet"):
        if hashes(Path(run_dir) / name)["sha256"] != inputs.meta["source_hashes"][name]:
            raise ValueError(f"Pipeline run does not match retrieval inputs: {name}")
    with (Path(run_dir) / f"{scenario}/train_item_ids.json").open(encoding="utf-8") as handle:
        if [str(item) for item in json.load(handle)] != list(inputs.item_ids):
            raise ValueError("Pipeline training catalog does not match retrieval inputs")


def build_rows(rec, features, examples, depth, progress_every=1000):
    """Yield per-user column blocks; labels are only read after features are built."""
    started = time.perf_counter()
    for count, example in enumerate(examples, start=1):
        request = RecommendationRequest(example.user_id, list(example.history_items))
        indices, scores, fallback = rec.recommend_scored(request, depth)
        matrix = features.build(example.history_items, indices, scores, example.user_id)
        movie_ids = [rec.inputs.item_ids[index] for index in indices]
        relevant = set(example.relevant_items)
        yield {
            "user_id": [example.user_id] * len(indices),
            "movie_id": movie_ids,
            "label": np.array([movie in relevant for movie in movie_ids], dtype="int8"),
            "fallback": np.full(len(indices), fallback),
            "user_relevant_count": np.full(len(indices), len(relevant), dtype="int32"),
            "features": matrix,
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
    cutoff = pd.Timestamp(inputs.meta["training_cutoff_utc"])
    load_started = time.perf_counter()
    features = RankingFeatures.from_run(
        run_dir, inputs.meta["scenario"], inputs.item_ids, inputs.item_features,
        int(cutoff.timestamp()), threshold=inputs.meta["positive_threshold"],
    )
    load_seconds = time.perf_counter() - load_started
    examples_path = Path(examples_path)
    if max_users:
        examples = [EvalExample(r["user_id"], r["history_items"], r["relevant_items"])
                    for r in sampled_examples(examples_path, max_users, seed)]
    else:
        examples = load_evaluation_examples(examples_path)

    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("Reranker table build in progress.\n")
    started = time.perf_counter()
    table = to_table(list(build_rows(rec, features, examples, depth)))
    build_seconds = time.perf_counter() - started
    pq.write_table(table, out / "rows.parquet", compression="zstd")

    labels = table["label"].to_numpy()
    per_user = pd.DataFrame({"user": table["user_id"].to_numpy(zero_copy_only=False), "label": labels}).groupby("user")["label"].sum()
    summary = {
        "users": int(len(per_user)),
        "rows": int(table.num_rows),
        "positive_rows": int(labels.sum()),
        "users_with_positive_candidate": int((per_user > 0).sum()),
        "fallback_users": int(len(set(table.filter(table["fallback"])["user_id"].to_pylist()))),
    }
    write_json(out / "manifest.json", {
        "description": "Reranker rows: two-tower candidates, training-period features, held-out labels",
        "depth": depth, "max_users": max_users, "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "label": "1 if candidate is in the example's relevant_items (held-out rating >= threshold)",
        "feature_cutoff_utc": inputs.meta["training_cutoff_utc"],
        "examples": str(examples_path), "examples_sha256": hashes(examples_path)["sha256"],
        "run_dir": str(run_dir),
        "input_manifest_sha256": hashes(Path(inputs_dir) / "manifest.json")["sha256"],
        "model_manifest_sha256": hashes(Path(model_dir) / "manifest.json")["sha256"],
        "rows_sha256": hashes(out / "rows.parquet")["sha256"],
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
    p.add_argument("--examples", type=Path, required=True, help="validation_warm.jsonl to train, test_warm.jsonl to evaluate")
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
