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


def write_users(path, users=60, seed=0):
    """users.parquet for a write_table() table, including a popularity list."""
    table = pq.read_table(path / "rows.parquet").to_pandas()
    viewers = table.groupby("user_id", sort=False).agg(relevant_count=("user_relevant_count", "first"),
                                                       candidate_hits=("label", "sum")).reset_index()
    rng = np.random.default_rng(seed)
    viewers["history_length"] = np.where(np.arange(len(viewers)) % 2, 0, 30)
    viewers["history_bucket"] = np.where(viewers["history_length"] == 0, "history_0", "history_21_100")
    viewers["fallback"] = viewers["history_length"] == 0
    viewers["popularity_labels"] = [list((rng.random(50) < .05).astype(int)) for _ in range(len(viewers))]
    pq.write_table(pa.Table.from_pandas(viewers, preserve_index=False), path / "users.parquet")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["users_sha256"] = hashes(path / "users.parquet")["sha256"]
    manifest["model_manifest_sha256"] = "towers"
    write_json(path / "manifest.json", manifest)


def test_rounds_selected_on_dev_and_fixed_rounds(tmp_path):
    from trustworthy_recsys.ranking.train import ndcg_by_group
    write_table(tmp_path / "train_b")
    write_table(tmp_path / "dev", seed=1)
    for name in ("train_b", "dev"):
        write_users(tmp_path / name)
    result = train(tmp_path / "train_b", tmp_path / "model", rounds=30, threads=1, valid_table_dir=tmp_path / "dev",
                   params={"min_data_in_leaf": 5, "num_leaves": 7}, grid_points=6)
    selection = result["selection"]
    assert selection["viewers"] == 60 and result["holdout"] is None
    assert str(selection["best_rounds"]) in selection["curve"] and result["best_iteration"] == selection["best_rounds"]
    assert selection["best_ndcg"] == max(selection["curve"].values()) > selection["two_tower_order_ndcg"]
    report = evaluate(tmp_path / "dev", tmp_path / "model", tmp_path / "report.json", bootstrap=20)
    assert report["population"] == "all table users"
    assert report["viewer_groups"]["history_0"] == 30 and "popularity/all" in report["metrics"]
    assert "ndcg_at_12" in report["metrics"]["lambdarank/all"]
    fixed = train(tmp_path / "train_b", tmp_path / "final", rounds=7, holdout=0, threads=1,
                  params={"min_data_in_leaf": 5, "num_leaves": 7})
    assert fixed["best_iteration"] == 7 and fixed["selection"] is None
    with pytest.raises(ValueError, match="must differ"):
        train(tmp_path / "train_b", tmp_path / "same", rounds=3, threads=1, valid_table_dir=tmp_path / "train_b")
    # Vectorized NDCG agrees with the repository metric, including a zero-hit viewer.
    groups = np.array([0, 0, 0, 1, 1])
    out = ndcg_by_group(groups, np.array([0, 1, 0, 0, 0.]), np.array([.9, .5, .1, 1, 0]), np.array([2, 1]), 2)
    assert out[0] == pytest.approx(ndcg_at_k({"b", "z"}, ["a", "b", "c"], 2)) and out[1] == 0


def test_paired_comparison_of_feature_variants(tmp_path):
    from trustworthy_recsys.ranking.compare import compare
    write_table(tmp_path / "train_b")
    write_table(tmp_path / "dev", seed=1)
    for name in ("train_b", "dev"):
        write_users(tmp_path / name)
    common = dict(rounds=20, threads=1, valid_table_dir=tmp_path / "dev", params={"min_data_in_leaf": 5, "num_leaves": 7})
    train(tmp_path / "train_b", tmp_path / "with", **common)
    train(tmp_path / "train_b", tmp_path / "without", drop_features=["recent_cosine_max"], **common)
    train(tmp_path / "train_b", tmp_path / "with_seed2", seed=2, **common)
    report = compare(tmp_path / "dev", [tmp_path / "without"], [tmp_path / "with", tmp_path / "with_seed2"],
                     k=12, samples=200, output=tmp_path / "cmp.json")
    gain = report["difference"]["ndcg_at_12"]["all"]
    # The informative feature helps, and the interval is around the observed mean.
    assert gain["mean_difference"] > 0 and gain["ci95"][0] <= gain["mean_difference"] <= gain["ci95"][1]
    assert gain["viewers"] == 60 and set(report["difference"]["ndcg_at_12"]) >= {"all", "history_0"}
    assert len(report["seed_spread"]["ndcg_at_12"]["b"]["runs"]) == 2
    baseline = compare(tmp_path / "dev", ["pass_through"], [tmp_path / "with"], samples=50)
    assert baseline["scores"]["recall_at_12"]["a"] >= 0


def test_graded_ndcg_uses_star_gains_and_full_ideal():
    from trustworthy_recsys.ranking.labels import five_star_labels, graded_gain, graded_ndcg
    gains = graded_gain([5.0, 0, 4.0]) * (np.array([5.0, 0, 4.0]) >= 4)
    # Ideal includes a missed 4.5-star like (gain 1.5): [2, 1.5, 1].
    ideal = [np.array([2.0, 1.5, 1.0])]
    out = graded_ndcg(np.zeros(3, dtype=int), gains, np.array([3.0, 2.0, 1.0]), ideal, 2)
    discounts = 1 / np.log2([2, 3])
    assert out[0] == pytest.approx(2 * discounts[0] / (2 * discounts[0] + 1.5 * discounts[1]))
    assert five_star_labels(np.array([0, 4.0, 4.5, 5.0])).tolist() == [0, 1, 1, 2]
