"""Chronological Train A / Train B / dev windows and their example exports."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trustworthy_recsys.data.windows import (
    Events, ab_share, export_examples, history_bucket, read_window_meta, share_cutoff,
)


def events(rows):
    user, item, rating, ts = (np.array(col) for col in zip(*sorted(rows, key=lambda r: (r[0], r[3], r[1]))))
    return Events(user, item, rating.astype("float32"), ts)


def test_share_cutoff_keeps_equal_timestamps_together():
    ts = np.array([1, 2, 3, 3, 3, 3, 4, 5])
    cutoff = share_cutoff(ts, .5)
    assert cutoff == 3
    # All four events at t=3 land after the cutoff, none are split.
    assert (ts < cutoff).sum() == 2
    with pytest.raises(ValueError):
        share_cutoff(np.array([7, 7, 7]), .5)


def test_ab_ratio_share():
    assert ab_share(7) == pytest.approx(.875)
    with pytest.raises(ValueError):
        ab_share(0)


def test_history_buckets():
    assert [history_bucket(n) for n in (0, 1, 20, 21, 100, 101)] == [
        "history_0", "history_1_20", "history_1_20", "history_21_100", "history_21_100", "history_101_plus"]


def test_export_uses_only_pre_window_inputs(tmp_path):
    table = events([
        (1, 10, 5.0, 1), (1, 11, 2.0, 2), (1, 12, 4.0, 5), (1, 13, 3.0, 6), (1, 14, 4.5, 9),
        (2, 10, 1.0, 6),                      # label-window event, but no like: excluded
        (3, 12, 4.0, 7),                      # no history: scored via fallback
    ])
    path = tmp_path / "dev.jsonl"
    meta = export_examples(table, history_end=5, label_start=5, label_end=9, path=path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["user_id"] for r in records] == ["1", "3"]
    first = records[0]
    assert first["history_items"] == ["10"] and first["history_times"] == [1]
    assert first["other_rated_items"] == ["11"] and first["other_rated_ratings"] == [2.0]
    # Label-window likes are answers only; t=9 is outside the window.
    assert first["relevant_items"] == ["12"] and first["label_other_items"] == ["13"]
    assert not set(first["relevant_items"]) & set(first["history_items"] + first["other_rated_items"])
    stats = meta["statistics"]
    assert stats["examples"] == 2 and stats["excluded_no_label_likes"] == 1
    assert stats["history_buckets"]["history_0"] == 1
    assert read_window_meta(path)["history_end"] == 5
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_window_meta(path)


def test_export_rejects_labels_overlapping_history(tmp_path):
    with pytest.raises(ValueError):
        export_examples(events([(1, 1, 5.0, 1)]), history_end=5, label_start=4, label_end=9, path=tmp_path / "x.jsonl")


def test_table_viewers_keep_zero_hit_users_and_popularity(tmp_path):
    pytest.importorskip("scipy")
    from test_ranking_features import fitted
    from trustworthy_recsys.ranking.dataset import build_rows, popularity_order, to_users_table

    class Stub:
        inputs = SimpleNamespace(item_ids=["a", "b", "c", "d"])

        def recommend_scored(self, request, depth):
            assert request.exclude_items == ("d",)
            return [2, 1], [.2, .1], False

    features = fitted()
    record = {"user_id": "u9", "history_items": ["a"], "other_rated_items": ["d"], "relevant_items": ["x"]}
    blocks = list(build_rows(Stub(), features, [record], 2, progress_every=0, popularity=popularity_order(features)))
    viewer = to_users_table(blocks).to_pylist()[0]
    assert viewer["candidate_hits"] == 0 and viewer["relevant_count"] == 1
    assert viewer["history_bucket"] == "history_1_20"
    # Popularity skips the rated movies a and d.
    assert len(viewer["popularity_labels"]) == 2


def test_development_workflow_end_to_end(tmp_path):
    """Windows -> Train A towers selected on dev -> Train B and dev candidate tables."""
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import pyarrow.parquet as pq
    import test_data_pipeline as fixtures
    from trustworthy_recsys.data.retrieval_inputs import ModelInputs, prepare_inputs
    from trustworthy_recsys.data.windows import prepare
    from trustworthy_recsys.ranking.dataset import build
    from trustworthy_recsys.retrieval.train import train

    fixture = fixtures.PipelineTests()
    fixture.setUp()
    try:
        rng = np.random.default_rng(0)
        records = [(u, m, float(rng.choice([2, 4, 5])), int(day))
                   for u in range(1, 9) for m, day in zip(rng.permutation(6)[:5] + 1, sorted(rng.choice(np.arange(1, 13), 5, replace=False)))]
        fixture.write_ratings(sorted(records))
        fixture.refresh_manifest()
        run = tmp_path / "run"
        fixtures.run_pipeline(fixture.raw, run, fixture.manifest, fixture.readme,
                              explicit=("2020-01-08", "2020-01-10"), chunk_size=4, dataset_name="synthetic-test")
        windows = prepare(run, "calendar", tmp_path / "windows", ab_ratio=1, names=("train_b", "dev"))
        a_end = windows["windows"]["train_a"]["end"]
        dev_start = windows["windows"]["dev"]["start"]
        assert a_end < dev_start == windows["windows"]["train_b"]["end"]

        prepare_inputs(run, "calendar", tmp_path / "inputs_a", partitions=("train",), window_end=a_end)
        inputs = ModelInputs(tmp_path / "inputs_a", verify=True)
        assert int(inputs.target_times.max()) < a_end
        assert inputs.meta["window"]["end"] == a_end

        dev = tmp_path / "windows" / "dev.jsonl"
        selected = train(tmp_path / "inputs_a", tmp_path / "towers", epochs=2, batch_size=3, threads=1,
                         selection_examples=dev)
        assert selected["validation_users"] == windows["exports"]["dev"]["statistics"]["examples"]
        assert "validation_recall_at_500" in selected["epochs_log"][0]
        fixed = train(tmp_path / "inputs_a", tmp_path / "towers_fixed", epochs=2, batch_size=3, threads=1, select=False)
        assert fixed["selected_epoch"] == 2 and fixed["validation_users"] == 0
        # Labels that start inside the fitting window cannot select checkpoints.
        with pytest.raises(ValueError, match="overlap"):
            train(tmp_path / "inputs_a", tmp_path / "bad", epochs=1, batch_size=3, threads=1,
                  selection_examples=_shifted(tmp_path, dev, a_end))

        for name in ("train_b", "dev"):
            summary = build(run, tmp_path / "inputs_a", tmp_path / "towers", tmp_path / "windows" / f"{name}.jsonl",
                            tmp_path / f"table_{name}", depth=10, threads=1)
            manifest = json.loads((tmp_path / f"table_{name}" / "manifest.json").read_text())
            assert manifest["feature_cutoff_utc"] == windows["exports"][name]["history_end_utc"]
            users = pq.read_table(tmp_path / f"table_{name}" / "users.parquet")
            assert users.num_rows == summary["users"] == windows["exports"][name]["statistics"]["examples"]
    finally:
        fixture.doCleanups()


def _shifted(tmp_path, dev, label_start):
    """A copy of dev whose sidecar claims labels starting before the fitting window ends."""
    path = tmp_path / "early.jsonl"
    path.write_bytes(dev.read_bytes())
    meta = json.loads(dev.with_suffix(".json").read_text())
    meta["label_start"] = label_start - 1
    path.with_suffix(".json").write_text(json.dumps(meta))
    return path


def test_tower_sweep_summarizes_variants(tmp_path):
    pytest.importorskip("torch")
    import test_data_pipeline as fixtures
    from trustworthy_recsys.data.retrieval_inputs import prepare_inputs
    from trustworthy_recsys.data.windows import prepare
    from trustworthy_recsys.retrieval.sweep import run

    fixture = fixtures.PipelineTests()
    fixture.setUp()
    try:
        rng = np.random.default_rng(0)
        records = [(u, m, float(rng.choice([2, 4, 5])), int(day))
                   for u in range(1, 9) for m, day in zip(rng.permutation(6)[:5] + 1, sorted(rng.choice(np.arange(1, 13), 5, replace=False)))]
        fixture.write_ratings(sorted(records))
        fixture.refresh_manifest()
        run_dir = tmp_path / "run"
        fixtures.run_pipeline(fixture.raw, run_dir, fixture.manifest, fixture.readme,
                              explicit=("2020-01-08", "2020-01-10"), chunk_size=4, dataset_name="synthetic-test")
        windows = prepare(run_dir, "calendar", tmp_path / "windows", ab_ratio=1, names=("dev",))
        prepare_inputs(run_dir, "calendar", tmp_path / "inputs", 0, ("train",), windows["windows"]["train_a"]["end"])
        rows = run(tmp_path / "inputs", tmp_path / "windows" / "dev.jsonl", tmp_path / "sweep",
                   {"baseline": {"target_sampling": {"mode": "spaced", "cap": 4}},
                    "rotate": {"target_sampling": {"mode": "rotate", "cap": 1}, "learning_rate": .004}},
                   base={"epochs": 2, "batch_size": 3, "threads": 1})
        assert [r["variant"] for r in rows] == ["baseline", "rotate"]
        assert "| rotate |" in (tmp_path / "sweep" / "summary.md").read_text()
        assert len(rows[1]["curve"]) == 2
    finally:
        fixture.doCleanups()


def test_cold_start_vectors_for_movies_first_rated_in_train_b(tmp_path):
    pytest.importorskip("torch")
    import test_data_pipeline as fixtures
    from trustworthy_recsys.data.retrieval_inputs import prepare_inputs
    from trustworthy_recsys.data.windows import prepare
    from trustworthy_recsys.retrieval.cold_start import run
    from trustworthy_recsys.retrieval.train import train

    fixture = fixtures.PipelineTests()
    fixture.setUp()
    try:
        rng = np.random.default_rng(0)
        records = [(u, m, float(rng.choice([2, 4, 5])), int(day))
                   for u in range(1, 9) for m, day in zip(rng.permutation(6)[:5] + 1, sorted(rng.choice(np.arange(1, 13), 5, replace=False)))]
        records += [(u, 6, 5.0, 6) for u in range(1, 9) if (u, 6) not in {(r[0], r[1]) for r in records}]
        fixture.write_ratings(sorted({(r[0], r[1]): r for r in records}.values()))
        fixture.refresh_manifest()
        run_dir = tmp_path / "run"
        fixtures.run_pipeline(fixture.raw, run_dir, fixture.manifest, fixture.readme,
                              explicit=("2020-01-08", "2020-01-10"), chunk_size=4, dataset_name="synthetic-test")
        windows = prepare(run_dir, "calendar", tmp_path / "windows", ab_ratio=1, names=("dev",))
        prepare_inputs(run_dir, "calendar", tmp_path / "inputs", 4, ("train",), windows["windows"]["train_a"]["end"])
        train(tmp_path / "inputs", tmp_path / "towers", epochs=1, batch_size=3, threads=1,
              selection_examples=tmp_path / "windows" / "dev.jsonl", identity_dropout=.5)
        report = run(run_dir, tmp_path / "inputs", tmp_path / "towers", tmp_path / "windows" / "dev.jsonl",
                     tmp_path / "cold.json", k=3)
        assert report["identity_dropout"] == .5 and report["new_movies"] >= 0
        assert report["original_catalog_recall"]["viewers"] >= 0
    finally:
        fixture.doCleanups()
