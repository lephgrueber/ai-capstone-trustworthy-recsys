"""Movie metadata in the towers: credits, runtime and tags (M3 feedback §2, §4, §5)."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from trustworthy_recsys.data.reporting import hashes, write_json
from trustworthy_recsys.data.tags import build as build_tags
from trustworthy_recsys.retrieval.model import TwoTower, load_model
from trustworthy_recsys.retrieval.movie_side import UNKNOWN, MovieSide, MovieSideConfig
from test_metadata_tags import FakeEncoder

ITEMS = ["1", "2", "3", "4"]
DAY = 86_400


def write_metadata(path, item_ids=ITEMS):
    path.mkdir(parents=True)
    rows = [{"movie_id": m, "imdb_id": f"tt{int(m):07d}",
             "runtime_minutes": [100.0, 120.0, None, 140.0][i % 4],
             "directors": [["nm1"], ["nm1"], [], ["nm9"]][i % 4],
             "writers": [["nm2"], [], [], ["nm2"]][i % 4],
             "cast": [["nm3", "nm4"], ["nm3"], [], ["nm5"]][i % 4]} for i, m in enumerate(item_ids)]
    pq.write_table(pa.Table.from_pylist(rows), path / "movie_metadata.parquet")
    write_json(path / "manifest.json", {"output_sha256": hashes(path / "movie_metadata.parquet")["sha256"]})


def write_tags(root, item_ids=ITEMS, start=None):
    start = start or pd.Timestamp("2020-01-01", tz="UTC")
    run = root / "tag_run"
    run.mkdir(parents=True)
    pq.write_table(pa.table({"user_id": ["1"] * 3, "movie_id": [item_ids[0], item_ids[0], item_ids[1]],
                             "tag": ["funny", "family", "space travel"],
                             "timestamp_utc": [start, start + pd.Timedelta(days=400), start]}), run / "tags.parquet")
    build_tags(run, root / "tags", encoder_name="fake", encoder=FakeEncoder())
    return root / "tags"


def side(tmp_path, **flags):
    config = MovieSideConfig(metadata_dir=str(tmp_path / "meta"), tags_dir=str(tmp_path / "tags"), **flags)
    return MovieSide(config, ITEMS)


def test_credits_runtime_and_missing_values(tmp_path):
    write_metadata(tmp_path / "meta")
    s = side(tmp_path, credits=True, runtime=True)
    # nm1 and nm3 appear on two catalog movies; nm9/nm4/nm5 once -> shared UNKNOWN.
    assert s.credit_ids["directors"][:, 0].tolist() == [s.vocab["directors"]["nm1"]] * 2 + [0, UNKNOWN]
    assert s.credit_ids["cast"][0, :2].tolist() == [s.vocab["cast"]["nm3"], UNKNOWN]
    assert s.runtime_scale["median"] == 120
    assert s.runtime[2].tolist() == [0, 1]  # missing -> median (scaled 0) + flag
    model = TwoTower(np.eye(4, dtype="float32"), side=s).eval()
    features = model.item_side(torch.tensor([2]))[0]
    # Movie 3 lists nobody: zero role vectors with the missing flag set for each role.
    width = s.config.person_dim + 1
    assert all(features[i * width + s.config.person_dim] == 1 for i in range(3))
    assert not features[: s.config.person_dim].any()


def test_tags_are_as_of_and_model_reloads_from_verified_sources(tmp_path):
    write_metadata(tmp_path / "meta")
    write_tags(tmp_path)
    s = side(tmp_path, credits=True, runtime=True, tags=True)
    early = int(pd.Timestamp("2020-02-01", tz="UTC").timestamp())
    late = int(pd.Timestamp("2021-06-01", tz="UTC").timestamp())
    assert s.tag_matrix(early)[:, -1].tolist() == [1, 1, 0, 0]
    assert not np.allclose(s.tag_matrix(early)[0], s.tag_matrix(late)[0])  # 'family' appears later
    torch.manual_seed(0)
    model = TwoTower(np.eye(4, dtype="float32"), embedding_dim=4, side=s).set_as_of(early)
    before = model.all_item_vectors()
    torch.save({"config": model.config, "state_dict": model.state_dict(), "as_of": early}, tmp_path / "model.pt")
    loaded = load_model(tmp_path / "model.pt", np.eye(4, dtype="float32"), ITEMS)
    assert np.allclose(loaded.all_item_vectors(), before)
    with pytest.raises(ValueError, match="item_ids"):
        load_model(tmp_path / "model.pt", np.eye(4, dtype="float32"))
    meta = tmp_path / "meta" / "movie_metadata.parquet"
    table = pq.read_table(meta).to_pandas()
    table.loc[0, "runtime_minutes"] = 99.0
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), meta)
    write_json(tmp_path / "meta" / "manifest.json", {"output_sha256": hashes(meta)["sha256"]})
    with pytest.raises(ValueError, match="differ"):
        load_model(tmp_path / "model.pt", np.eye(4, dtype="float32"), ITEMS)


def test_metadata_towers_run_through_development(tmp_path):
    pytest.importorskip("lightgbm")
    from test_workflow import TINY, synthetic_run
    from trustworthy_recsys.workflow import run_stage
    gen = synthetic_run(tmp_path)
    run = next(gen)
    try:
        catalog = [str(i) for i in range(1, 9)]
        write_metadata(tmp_path / "meta", catalog)
        tags = write_tags(tmp_path, catalog, pd.Timestamp("2019-06-01", tz="UTC"))
        settings = json.loads(json.dumps(TINY))
        settings["tower"].update({
            "history": {"favorites": True, "metadata": True},
            "movie_side": {"metadata_dir": str(tmp_path / "meta"), "tags_dir": str(tags),
                           "credits": True, "runtime": True, "tags": True}})
        result = run_stage("development", run, tmp_path / "workflow", settings, scenario="calendar")
        towers = json.loads((tmp_path / "workflow" / "development" / "towers" / "training.json").read_text())
        assert towers["movie_side"]["tags"] and result["report"]["lambdarank/all"] >= 0
        final = run_stage("final", run, tmp_path / "workflow", scenario="calendar")
        exported = json.loads((tmp_path / "workflow" / "final" / "towers" / "training.json").read_text())
        windows = json.loads((tmp_path / "workflow" / "windows_ab3" / "windows.json").read_text())["windows"]
        assert exported["export_as_of_utc"] == windows["final_b"]["end_utc"] and final["report"]
    finally:
        next(gen, None)
