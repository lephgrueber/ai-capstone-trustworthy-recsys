"""Train a LightGBM LambdaRank reranker on a validation-period candidate table.

Validation users are split by a stable hash into a fitting set and an early-stopping
set; test tables are never read here. Users without any relevant candidate carry no
ranking signal and are excluded from fitting (they remain in evaluation).
"""

import argparse
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
    """Thread count that is safe in this process.

    On macOS, PyTorch bundles its own OpenMP runtime while LightGBM may resolve a
    different copy (e.g. Anaconda's). With both loaded, multi-threaded LightGBM can
    segfault, so LightGBM runs single-threaded whenever torch is already imported.
    """
    return 1 if "torch" in sys.modules else requested


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


def train(table_dir, output_dir, drop_features=(), holdout=0.1, seed=42, rounds=2000,
          early_stopping=100, threads=4, params=None):
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
    table = load_columns(rows, features)
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-dir", type=Path, required=True, help="Validation table from ranking.dataset")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--drop-features", nargs="*", default=[], help="Ablation: features to leave out")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    result = train(a.table_dir, a.output_dir, a.drop_features, a.holdout, a.seed, a.rounds, a.early_stopping, a.threads)
    print(json.dumps({k: result[k] for k in ("best_iteration", "holdout", "users", "training_seconds")}, indent=2))
    print(json.dumps(result["feature_importance_gain"], indent=2))


if __name__ == "__main__":
    main()
