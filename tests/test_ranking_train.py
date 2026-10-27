"""Reranker training split, grouping, metrics parity and an end-to-end fit."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("lightgbm")

from trustworthy_recsys.data.reporting import hashes, write_json
from trustworthy_recsys.evaluation.metrics import ndcg_at_k, recall_at_k
from trustworthy_recsys.ranking.evaluate import evaluate, paired_bootstrap, user_metrics
from trustworthy_recsys.ranking.features import FEATURE_NAMES
from trustworthy_recsys.ranking.train import group_sizes, holdout_mask, train


def test_holdout_is_per_user_and_deterministic():
    users = np.array(["a", "a", "b", "c", "c", "c"] * 1)
    first = holdout_mask(users, .5, 1)
    assert np.array_equal(first, holdout_mask(users, .5, 1))
    for user in "abc":
        assert len(set(first[users == user])) == 1


def test_group_sizes_require_contiguous_users():
    assert list(group_sizes(np.array(["a", "a", "b"]))) == [2, 1]
    with pytest.raises(ValueError, match="contiguous"):
        group_sizes(np.array(["a", "b", "a"]))


def test_user_metrics_match_repository_metrics():
    rng = np.random.default_rng(0)
    sizes, relevant_counts, labels, scores, cases = [], [], [], [], []
    for user in range(30):
        n = int(rng.integers(5, 40))
        label = (rng.random(n) < .2).astype(float)
        extra = int(rng.integers(0, 4))  # relevant items retrieval never returned
        if label.sum() + extra == 0:
            extra = 1
        score = rng.random(n)
        sizes.append(n), relevant_counts.append(int(label.sum()) + extra)
        labels.append(label), scores.append(score)
        ranked = [f"{user}-{i}" for i in np.argsort(-score, kind="stable")]
        relevant = [f"{user}-{i}" for i in np.flatnonzero(label)] + [f"{user}-missed{j}" for j in range(extra)]
        cases.append((relevant, ranked))
    out = user_metrics(np.concatenate(labels), np.concatenate(scores), np.array(sizes), np.array(relevant_counts), ks=(5, 10))
    for user, (relevant, ranked) in enumerate(cases):
        for k in (5, 10):
            assert out[f"recall_at_{k}"][user] == pytest.approx(recall_at_k(relevant, ranked, k))
            assert out[f"ndcg_at_{k}"][user] == pytest.approx(ndcg_at_k(relevant, ranked, k))


def test_paired_bootstrap_interval_contains_mean():
    result = paired_bootstrap(np.array([.3, .5, .4, .6]), np.array([.1, .5, .2, .4]), samples=200)
    assert result["mean_difference"] == pytest.approx(.15)
    assert result["ci95"][0] <= .15 <= result["ci95"][1]
    assert (result["users_improved"], result["users_worse"]) == (3, 0)


def write_table(path, users=60, depth=20, seed=0):
    """Synthetic table where one feature carries the label signal."""
    rng = np.random.default_rng(seed)
    n = users * depth
    label = (rng.random(n) < .15).astype("int8")
    columns = {
        "user_id": pa.array(np.repeat([f"u{i}" for i in range(users)], depth)),
        "movie_id": pa.array([str(i) for i in range(n)]),
        "label": label,
        "fallback": np.zeros(n, dtype=bool),
        "user_relevant_count": np.repeat(np.maximum(1, np.add.reduceat(label, np.arange(0, n, depth))), depth).astype("int32"),
    }
    for name in FEATURE_NAMES:
        columns[name] = rng.random(n).astype("float32")
    columns["retrieval_rank"] = np.tile(np.arange(depth), users).astype("float32")
    columns["recent_cosine_max"] = (label + rng.normal(0, .3, n)).astype("float32")
    columns["user_log_history_length"] = np.full(n, np.log1p(30), dtype="float32")
    path.mkdir()
    pq.write_table(pa.table(columns), path / "rows.parquet")
    write_json(path / "manifest.json", {"rows_sha256": hashes(path / "rows.parquet")["sha256"], "depth": depth,
                                        "feature_cutoff_utc": "2018-10-04T00:00:00+00:00"})


def test_train_and_evaluate_learn_informative_feature(tmp_path):
    write_table(tmp_path / "valid")
    write_table(tmp_path / "test", seed=1)
    result = train(tmp_path / "valid", tmp_path / "model", holdout=.3, rounds=50, early_stopping=10, threads=1,
                   params={"min_data_in_leaf": 5, "num_leaves": 7})
    assert next(iter(result["feature_importance_gain"])) == "recent_cosine_max"
    report = evaluate(tmp_path / "test", tmp_path / "model", tmp_path / "report.json", bootstrap=50)
    assert report["population"] == "all table users"
    assert report["lambdarank_minus_pass_through"]["ndcg_at_10"]["mean_difference"] > 0
    holdout = evaluate(tmp_path / "valid", tmp_path / "model", tmp_path / "holdout.json", bootstrap=50)
    assert holdout["population"].startswith("validation") and holdout["users"] < 60
    with pytest.raises(ValueError, match="already exists"):
        train(tmp_path / "valid", tmp_path / "model")
    with (tmp_path / "model" / "model.txt").open("a") as handle:
        handle.write("tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate(tmp_path / "test", tmp_path / "model", tmp_path / "again.json")
