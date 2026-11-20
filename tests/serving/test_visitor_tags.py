"""Visitor review tags (M3 feedback §4): contract, opt-in store and tag summaries."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

pytest.importorskip("scipy")

from trustworthy_recsys.data.tags import build as build_tags
from trustworthy_recsys.serving.config import Settings
from trustworthy_recsys.serving.contracts import PreferenceState, RatingRecord
from trustworthy_recsys.serving.visitor_tags import VisitorTagStore

NOW = datetime(2026, 11, 20, tzinfo=timezone.utc)


def record(movie, tags, rating=4.5):
    return {"movie_id": movie, "rating": rating, "rated_at": NOW.isoformat(),
            **({"tags": tags, "tagged_at": NOW.isoformat()} if tags else {})}


def test_tags_need_a_utc_time_and_are_bounded():
    assert RatingRecord(**record("1", ["space travel"])).tags == ["space travel"]
    with pytest.raises(ValidationError):
        RatingRecord(movie_id="1", rating=4, rated_at=NOW, tags=["x"])
    with pytest.raises(ValidationError):
        RatingRecord(**record("1", ["a", "b", "c", "d", "e", "f"]))
    with pytest.raises(ValidationError):
        RatingRecord(**record("1", ["x" * 41]))


def test_visitor_tags_are_off_by_default():
    assert Settings().visitor_tags == "off"
    with pytest.raises(ValueError):
        Settings(visitor_tags="everything")


def test_store_records_once_persists_and_updates_summaries(tmp_path):
    from test_metadata_tags import FakeEncoder
    from test_ranking_features import fitted
    run = tmp_path / "run"
    run.mkdir()
    pq.write_table(pa.table({"user_id": ["1"], "movie_id": ["a"], "tag": ["funny"],
                             "timestamp_utc": [pd.Timestamp(0, unit="s", tz="UTC")]}), run / "tags.parquet")
    build_tags(run, tmp_path / "tags", encoder_name="fake", encoder=FakeEncoder())
    features = fitted().attach_metadata(tags_dir=tmp_path / "tags")
    assert not features.has_tags[2]
    store = VisitorTagStore(tmp_path / "db" / "tags.sqlite3", features, FakeEncoder())
    prefs = PreferenceState(ratings=[record("c", ["Space  Travel", "family"]), record("b", None)])
    assert sorted(store.record(prefs)) == [("c", "family"), ("c", "space travel")]
    assert store.record(prefs) == []  # already known
    assert features.has_tags[2] and features.tag_vectors[2] @ FakeEncoder().encode(["family"])[0] > .5
    out = features.build(["a"], [2], user_id="u1")
    assert np.isfinite(out[0, features.names.index("tag_phrase_match_max")])
    store.close()
    reopened = VisitorTagStore(tmp_path / "db" / "tags.sqlite3")
    assert set(reopened.seen) == {("c", "family"), ("c", "space travel")}
    reopened.close()
