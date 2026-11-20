"""Train a LightGBM LambdaRank reranker on a candidate table.

Proposed workflow (M3 feedback §1, §7): fit on the Train B table and pass the dev table as
``valid_table_dir``. All Train B lists with at least one hit are used for fitting (lists
without a positive supply no LambdaRank pairs). Dev labels never fit weights; they only
choose the number of boosting rounds by *end-to-end* NDCG@K on every dev viewer, whose
denominator counts all later likes, including those retrieval missed.

Final refit: pass ``holdout=0`` and no dev table to train exactly ``rounds`` rounds.

Legacy M3 mode (no dev table, holdout > 0): a hash-selected share of the table's own users
is held out for early stopping on candidate-only NDCG@10.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json
from .features import FEATURE_NAMES

DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "lambdarank_truncation_level": 30,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def lightgbm_threads(requested):
    """Thread count that is safe in this process; call before creating any Booster.

    On macOS, PyTorch bundles its own OpenMP runtime while LightGBM may resolve a
    different copy (e.g. Anaconda's). When both runtimes use several threads the
    process segfaults, including inside model-file parsing. So whenever torch is
    already imported, LightGBM is capped process-wide at one thread.
    """
    if "torch" not in sys.modules:
        return requested
    lgb.basic._LIB.LGBM_SetMaxThreads(ctypes.c_int(1))
    return 1


def holdout_mask(user_ids, fraction, seed):
    """Label-independent user split: the same user always lands on the same side."""
    unique = dict.fromkeys(user_ids)
    held = {u for u in unique
            if int.from_bytes(hashlib.sha256(f"{seed}:{u}".encode()).digest()[:8], "big") / 2**64 < fraction}
    return np.fromiter((u in held for u in user_ids), dtype=bool, count=len(user_ids))


def group_sizes(user_ids):
    """Sizes of contiguous user blocks; rows must already be grouped by user."""
    change = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(user_ids)]])
    sizes = np.diff(bounds)
    if len(set(user_ids[bounds[:-1]])) != len(sizes):
        raise ValueError("Rows for a user are not contiguous")
    return sizes


def ranking_dataset(table, features, mask, reference=None):
    users = table["user_id"][mask]
    labels = table["label"][mask]
    sizes = group_sizes(users)
    # Drop users whose candidates contain no positive; they cannot produce lambdas.
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    keep_group = np.add.reduceat(labels, starts) > 0
    keep_rows = np.repeat(keep_group, sizes)
    matrix = np.column_stack([table[name][mask] for name in features])[keep_rows]
    data = lgb.Dataset(matrix, labels[keep_rows], group=sizes[keep_group],
                       feature_name=list(features), reference=reference, free_raw_data=False)
    return data, int(keep_group.sum()), int((~keep_group).sum())


def load_columns(path, features):
    table = pq.read_table(path, columns=["user_id", "label", *features])
    return {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}


RETRIEVAL_FILES = ("model.pt", "item_embeddings.npy")


def retrieval_fingerprint(model_dir):
    """Hashes that identify a two-tower model across bundle and published directories."""
    return {name: hashes(Path(model_dir) / name)["sha256"] for name in RETRIEVAL_FILES}


def group_ids(user_ids):
    sizes = group_sizes(user_ids)
    return np.repeat(np.arange(len(sizes)), sizes), sizes


def ndcg_by_group(groups, labels, scores, relevant_counts, k):
    """Per-group binary NDCG@k; ties keep table order; denominators use relevant_counts."""
    order = np.lexsort((-scores, groups))
    g = groups[order]
    starts = np.searchsorted(g, np.arange(len(relevant_counts)))
    rank = np.arange(len(g)) - starts[g]
    top = rank < k
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.bincount(g[top], weights=labels[order][top] * discounts[rank[top]], minlength=len(relevant_counts))
    ideal = np.cumsum(discounts)[np.minimum(relevant_counts, k) - 1]
    return dcg / ideal


def round_grid(rounds, points=40):
    grid = np.unique(np.linspace(1, rounds, min(points, rounds)).round().astype(int))
    return [int(x) for x in grid]


def select_rounds(booster, valid, features, k, grid):
    """End-to-end NDCG@k on every dev viewer for each round count in `grid` (one pass)."""
    groups, sizes = group_ids(valid["user_id"])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    relevant = valid["user_relevant_count"][starts].astype(np.int64)
    labels = valid["label"].astype(float)
    matrix = np.column_stack([valid[name] for name in features])
    raw = np.zeros(len(labels))
    done, curve = 0, {}
    for target in grid:
        raw += booster.predict(matrix, start_iteration=done, num_iteration=target - done,
                               raw_score=True, num_threads=lightgbm_threads(4))
        done = target
        curve[target] = float(ndcg_by_group(groups, labels, raw, relevant, k).mean())
    best = max(curve, key=lambda r: (curve[r], -r))
    baseline = float(ndcg_by_group(groups, labels, -valid["retrieval_rank"].astype(float), relevant, k).mean())
    return best, curve, baseline, len(sizes)


def train(table_dir, output_dir, drop_features=(), holdout=0.1, seed=42, rounds=2000,
          early_stopping=100, threads=4, params=None, retrieval_model_dir=None,
          valid_table_dir=None, display_k=12, grid_points=40):
    """Fit and save a reranker.

    `retrieval_model_dir` (the model that generated the table's candidates) is recorded
    so serving can refuse to pair this reranker with a different retrieval model.
    """
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Model directory already exists; choose a new directory")
    unknown = set(drop_features) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    features = [name for name in FEATURE_NAMES if name not in set(drop_features)]
    rows = Path(table_dir) / "rows.parquet"
    manifest = json.loads((Path(table_dir) / "manifest.json").read_text())
    if hashes(rows)["sha256"] != manifest["rows_sha256"]:
        raise ValueError("Reranker table hash mismatch")
    retrieval = None
    if retrieval_model_dir is not None:
        if hashes(Path(retrieval_model_dir) / "manifest.json")["sha256"] != manifest["model_manifest_sha256"]:
            raise ValueError("Table candidates were not generated by this retrieval model")
        retrieval = {"input_manifest_sha256": manifest["input_manifest_sha256"],
                     **retrieval_fingerprint(retrieval_model_dir)}
    table = load_columns(rows, features)
    if valid_table_dir is not None or holdout == 0:
        return _train_full(table, manifest, features, out, table_dir, valid_table_dir, rounds, seed, threads,
                           params, retrieval, drop_features, display_k, grid_points)
    held = holdout_mask(table["user_id"], holdout, seed)
    fit, fit_users, fit_skipped = ranking_dataset(table, features, ~held)
    valid, valid_users, valid_skipped = ranking_dataset(table, features, held, reference=fit)
    config = {**DEFAULT_PARAMS, **(params or {}), "seed": seed, "num_threads": lightgbm_threads(threads)}

    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("Reranker training in progress.\n")
    history = {}
    started = time.perf_counter()
    booster = lgb.train(
        config, fit, num_boost_round=rounds, valid_sets=[valid], valid_names=["holdout"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                   lgb.record_evaluation(history), lgb.log_evaluation(100)],
    )
    seconds = time.perf_counter() - started
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    result = {
        "table": str(table_dir), "table_rows_sha256": manifest["rows_sha256"],
        "candidate_depth": manifest["depth"], "feature_cutoff_utc": manifest["feature_cutoff_utc"],
        "retrieval": retrieval,
        "features": features, "dropped_features": sorted(drop_features),
        "params": config, "rounds": rounds, "early_stopping_rounds": early_stopping,
        "best_iteration": booster.best_iteration,
        "holdout": {"fraction": holdout, "seed": seed, "metric": "LightGBM ndcg@10 on users with a positive candidate",
                    "best_ndcg_at_10": float(booster.best_score["holdout"]["ndcg@10"])},
        "users": {"fit": fit_users, "fit_skipped_no_positive": fit_skipped,
                  "holdout": valid_users, "holdout_skipped_no_positive": valid_skipped},
        "feature_importance_gain": dict(sorted(
            ((name, float(value)) for name, value in zip(features, gain)), key=lambda item: -item[1])),
        "training_seconds": seconds,
        "holdout_curve": history["holdout"]["ndcg@10"],
        "lightgbm_version": lgb.__version__,
        "code": code_provenance(),
    }
    write_json(out / "training.json", result)
    write_json(out / "manifest.json", {"outputs": {p.name: hashes(p)["sha256"] for p in out.iterdir()
                                                    if p.is_file() and p.name != "INCOMPLETE"}})
    (out / "INCOMPLETE").unlink()
    return result


def _verified_table(table_dir):
    manifest = json.loads((Path(table_dir) / "manifest.json").read_text())
    if hashes(Path(table_dir) / "rows.parquet")["sha256"] != manifest["rows_sha256"]:
        raise ValueError(f"Reranker table hash mismatch: {table_dir}")
    return manifest


def _train_full(table, manifest, features, out, table_dir, valid_table_dir, rounds, seed, threads,
                params, retrieval, drop_features, display_k, grid_points):
    """Fit on every table user with a hit; choose rounds on a separate dev table if given."""
    valid_manifest = None
    if valid_table_dir is not None:
        valid_manifest = _verified_table(valid_table_dir)
        if valid_manifest["model_manifest_sha256"] != manifest["model_manifest_sha256"]:
            raise ValueError("Fitting and dev tables use different retrieval models")
        if valid_manifest["rows_sha256"] == manifest["rows_sha256"]:
            raise ValueError("The dev table must differ from the fitting table")
        if valid_manifest.get("feature_sources", {}) != manifest.get("feature_sources", {}):
            raise ValueError("Fitting and dev tables use different metadata sources")
    fit, fit_users, fit_skipped = ranking_dataset(table, features, np.ones(len(table["label"]), dtype=bool))
    config = {**DEFAULT_PARAMS, **(params or {}), "seed": seed, "num_threads": lightgbm_threads(threads)}
    config.pop("eval_at", None)
    config["metric"] = "None"
    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("Reranker training in progress.\n")
    started = time.perf_counter()
    booster = lgb.train(config, fit, num_boost_round=rounds)
    seconds = time.perf_counter() - started
    selection = None
    best = rounds
    if valid_table_dir is not None:
        valid = load_columns(Path(valid_table_dir) / "rows.parquet",
                             list(dict.fromkeys([*features, "retrieval_rank", "user_relevant_count"])))
        grid = round_grid(rounds, grid_points)
        best, curve, two_tower, viewers = select_rounds(booster, valid, features, display_k, grid)
        selection = {"table": str(valid_table_dir), "table_rows_sha256": valid_manifest["rows_sha256"],
                     "metric": f"end-to-end NDCG@{display_k}, macro over every dev viewer "
                               "(denominator: all later likes, including retrieval misses)",
                     "viewers": viewers, "best_rounds": best, "best_ndcg": curve[best],
                     "two_tower_order_ndcg": two_tower, "curve": {str(k): v for k, v in curve.items()}}
        print(json.dumps({k: selection[k] for k in ("best_rounds", "best_ndcg", "two_tower_order_ndcg")}), flush=True)
    booster.save_model(str(out / "model.txt"), num_iteration=best)
    gain = booster.feature_importance("gain", iteration=best)
    result = {
        "table": str(table_dir), "table_rows_sha256": manifest["rows_sha256"],
        "candidate_depth": manifest["depth"], "feature_cutoff_utc": manifest["feature_cutoff_utc"],
        "feature_partitions": manifest.get("feature_partitions", ["train"]),
        "feature_sources": manifest.get("feature_sources", {}),
        # Serving uses every rating the pipeline was fitted on: the fitting table's history
        # plus its label window (e.g. Final A + Final B), never anything later.
        "serving_features": None if not manifest.get("window") else {
            "partitions": manifest.get("feature_partitions", ["train"]),
            "end_utc": manifest["window"]["label_end_utc"]},
        "retrieval": retrieval, "features": features, "dropped_features": sorted(drop_features),
        "params": config, "rounds": rounds, "best_iteration": best, "display_k": display_k,
        "selection": selection, "holdout": None,
        "users": {"fit": fit_users, "fit_skipped_no_positive": fit_skipped},
        "feature_importance_gain": dict(sorted(
            ((name, float(value)) for name, value in zip(features, gain)), key=lambda item: -item[1])),
        "training_seconds": seconds, "lightgbm_version": lgb.__version__, "code": code_provenance(),
    }
    write_json(out / "training.json", result)
    write_json(out / "manifest.json", {"outputs": {p.name: hashes(p)["sha256"] for p in out.iterdir()
                                                    if p.is_file() and p.name != "INCOMPLETE"}})
    (out / "INCOMPLETE").unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-dir", type=Path, required=True, help="Validation table from ranking.dataset")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--retrieval-model-dir", type=Path, required=True,
                   help="Two-tower model that generated the table's candidates (recorded for serving checks)")
    p.add_argument("--drop-features", nargs="*", default=[], help="Ablation: features to leave out")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--valid-table-dir", type=Path,
                   help="Dev table: choose boosting rounds by end-to-end NDCG@K (proposed workflow)")
    p.add_argument("--display-k", type=int, default=12, help="Browser display length used for selection")
    p.add_argument("--truncation-level", type=int, help="LambdaRank truncation level (default 30)")
    a = p.parse_args()
    params = {"lambdarank_truncation_level": a.truncation_level} if a.truncation_level else None
    result = train(a.table_dir, a.output_dir, a.drop_features, a.holdout, a.seed, a.rounds, a.early_stopping, a.threads,
                   params=params, retrieval_model_dir=a.retrieval_model_dir, valid_table_dir=a.valid_table_dir,
                   display_k=a.display_k)
    print(json.dumps({k: result[k] for k in ("best_iteration", "holdout", "selection", "users", "training_seconds")
                      if k != "selection" or result[k] is None} | (
        {"selection": {k: result["selection"][k] for k in ("best_rounds", "best_ndcg", "two_tower_order_ndcg")}}
        if result.get("selection") else {}), indent=2))
    print(json.dumps(result["feature_importance_gain"], indent=2))


if __name__ == "__main__":
    main()
