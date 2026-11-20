"""Development -> final -> deployment fits on a synthetic run (M3 feedback §1)."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightgbm")

import test_data_pipeline as fixtures
from trustworthy_recsys.workflow import run_stage

TINY = {"depth": 10, "ab_ratio": 3,
        "tower": {"epochs": 2, "batch_size": 4, "threads": 1},
        "reranker": {"rounds": 6, "params": {"min_data_in_leaf": 1, "num_leaves": 3}}}


@pytest.fixture
def run(tmp_path):
    yield from synthetic_run(tmp_path)


def synthetic_run(tmp_path):
    """Generator: yields a completed synthetic pipeline run, then cleans up."""
    fixture = fixtures.PipelineTests()
    fixture.setUp()
    rng = np.random.default_rng(3)
    records = []
    for user in range(1, 16):
        movies = rng.permutation(8)[:7] + 1
        days = np.sort(rng.choice(np.arange(1, 21), 7, replace=False))
        records += [(user, int(m), float(rng.choice([2, 4, 4.5, 5])), int(d)) for m, d in zip(movies, days)]
    fixture.write_ratings(sorted(records))
    (fixture.raw / "movies.csv").write_text("movieId,title,genres\n" + "".join(
        f"{i},Movie {i} (200{i}),{'Drama' if i % 2 else 'Comedy'}\n" for i in range(1, 9)), encoding="utf-8")
    (fixture.raw / "links.csv").write_text("movieId,imdbId,tmdbId\n" + "".join(f"{i},00{i},{i}\n" for i in range(1, 9)))
    fixture.refresh_manifest()
    out = tmp_path / "run"
    fixtures.run_pipeline(fixture.raw, out, fixture.manifest, fixture.readme,
                          explicit=("2020-01-13", "2020-01-17"), chunk_size=8, dataset_name="synthetic-test")
    yield out
    fixture.doCleanups()


def test_stages_lock_settings_and_use_their_windows(run, tmp_path):
    root = tmp_path / "workflow"
    with pytest.raises(ValueError, match="development stage first"):
        run_stage("final", run, root, scenario="calendar")
    dev = run_stage("development", run, root, TINY, scenario="calendar")
    assert set(dev["report"]) == {"popularity/all", "pass_through/all", "lambdarank/all"}
    locked = json.loads((root / "locked.json").read_text())
    assert locked["tower"]["selected_epoch"] in (1, 2) and locked["reranker"]["selected_rounds"] >= 1
    with pytest.raises(ValueError, match="locked settings"):
        run_stage("final", run, root, {"depth": 5}, scenario="calendar")

    final = run_stage("final", run, root, scenario="calendar")
    windows = json.loads((root / "windows_ab3" / "windows.json").read_text())["windows"]
    inputs = json.loads((root / "final" / "inputs_a" / "metadata.json").read_text())
    assert inputs["window"]["end"] == windows["final_a"]["end"] and inputs["window"]["partitions"] == ["train", "validation"]
    towers = json.loads((root / "final" / "towers" / "training.json").read_text())
    assert towers["epochs"] == locked["tower"]["selected_epoch"] and towers["validation_users"] == 0
    reranker = json.loads((root / "final" / "reranker" / "training.json").read_text())
    assert reranker["best_iteration"] == locked["reranker"]["selected_rounds"] and reranker["selection"] is None
    assert reranker["serving_features"]["end_utc"] == windows["final_b"]["end_utc"]
    test_table = json.loads((root / "final" / "table_test" / "manifest.json").read_text())
    assert test_table["feature_cutoff_utc"] == windows["test"]["start_utc"]
    assert final["viewer_groups"]["all"] == json.loads((root / "windows_ab3" / "test.json").read_text())["statistics"]["examples"]

    deployment = run_stage("deployment", run, root, scenario="calendar")
    assert deployment["report"] is None
    package = json.loads((root / "deployment" / "pipeline.json").read_text())
    assert package["evaluation"].startswith("none") and "towers/model.pt" in package["files_sha256"]


def test_history_aware_towers_run_through_development(run, tmp_path):
    settings = json.loads(json.dumps(TINY))
    settings["tower"].update({"history": {"favorites": True, "pooling": "learned", "dislikes": True},
                              "dislike_penalty": 0.5})
    result = run_stage("development", run, tmp_path / "workflow", settings, scenario="calendar")
    towers = json.loads((tmp_path / "workflow" / "development" / "towers" / "training.json").read_text())
    assert towers["architecture"]["history"]["pooling"] == "learned" and "softplus" in towers["loss"]
    assert result["report"]["lambdarank/all"] >= 0


def test_five_star_weighting_and_graded_ndcg(run, tmp_path):
    settings = json.loads(json.dumps(TINY))
    settings["tower"]["five_star_weight"] = 2.0
    settings["reranker"]["five_star_weight"] = 3.0
    run_stage("development", run, tmp_path / "workflow", settings, scenario="calendar")
    report = json.loads((tmp_path / "workflow" / "development" / "dev_report.json").read_text())
    assert "graded_ndcg_at_12" in report["metrics"]["lambdarank/all"]
    reranker = json.loads((tmp_path / "workflow" / "development" / "reranker" / "training.json").read_text())
    assert reranker["params"]["label_gain"] == [0.0, 1.0, 3.0] and reranker["fit_labels"].startswith("graded")
    towers = json.loads((tmp_path / "workflow" / "development" / "towers" / "training.json").read_text())
    assert towers["five_star_weight"] == 2.0


def test_dislike_threshold_flows_to_inputs_and_tables(run, tmp_path):
    settings = json.loads(json.dumps(TINY))
    settings["dislike_threshold"] = 2.5
    run_stage("development", run, tmp_path / "workflow", settings, scenario="calendar")
    inputs = json.loads((tmp_path / "workflow" / "development" / "inputs_a" / "metadata.json").read_text())
    table = json.loads((tmp_path / "workflow" / "development" / "table_dev" / "manifest.json").read_text())
    assert inputs["dislike_threshold"] == table["dislike_threshold"] == 2.5
