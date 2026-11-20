"""IMDb credits/runtime and tag-phrase summaries (M3 feedback §4-§5)."""

import gzip

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trustworthy_recsys.data import metadata
from trustworthy_recsys.data.tags import TagIndex, build as build_tags, normalize


def write_gz(path, text):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(text)


def test_imdb_credits_and_runtime_keyed_by_movielens_id(tmp_path):
    run, imdb = tmp_path / "run", tmp_path / "imdb"
    run.mkdir(), imdb.mkdir()
    pq.write_table(pa.table({"movie_id": ["1", "2", "3"], "imdb_id": ["0000010", "0000020", None],
                             "tmdb_id": ["1", "2", None]}), run / "links.parquet")
    write_gz(imdb / "title.basics.tsv.gz", "tconst\ttitleType\truntimeMinutes\n"
             "tt0000010\tmovie\t121\ntt0000020\tmovie\t\\N\ntt0000099\tmovie\t90\n")
    write_gz(imdb / "title.crew.tsv.gz", "tconst\tdirectors\twriters\n"
             "tt0000010\tnm1,nm2\tnm3\ntt0000020\tnm1\t\\N\n")
    write_gz(imdb / "title.principals.tsv.gz", "tconst\tordering\tnconst\tcategory\tjob\tcharacters\n"
             "tt0000010\t1\tnm7\tactress\t\\N\t\\N\ntt0000010\t2\tnm1\tdirector\t\\N\t\\N\n"
             "tt0000010\t3\tnm8\tactor\t\\N\t\\N\ntt0000020\t1\tnm9\tself\t\\N\t\\N\n")
    manifest = metadata.build(run, imdb, tmp_path / "meta", "2026-11-20", catalog_ids=["1", "2"])
    frame = metadata.load(tmp_path / "meta")
    assert frame.loc["1", "runtime_minutes"] == 121 and frame.loc["1", "directors"].tolist() == ["nm1", "nm2"]
    assert frame.loc["1", "cast"].tolist() == ["nm7", "nm8"]  # directors and 'self' rows are not cast
    assert np.isnan(frame.loc["2", "runtime_minutes"]) and frame.loc["2", "writers"].tolist() == []
    assert frame.loc["3", "imdb_id"] is None and frame.loc["3", "directors"].tolist() == []
    assert manifest["coverage"]["catalog"]["positive_runtime_pct"] == 50
    assert manifest["snapshot_date"] == "2026-11-20"


class FakeEncoder:
    """Deterministic unit vectors: 'funny'/'hilarious' close, 'space travel' orthogonal."""
    BASIS = {"funny": [1, 0, 0], "hilarious": [.9, .1, 0], "space travel": [0, 0, 1], "family": [0, 1, 0]}

    def encode(self, phrases, **_):
        out = np.array([self.BASIS.get(p, [.5, .5, .5]) for p in phrases], dtype="float32")
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_tag_summaries_respect_as_of_time(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    stamp = lambda day: pd.Timestamp(f"2020-01-{day:02d}", tz="UTC")
    pq.write_table(pa.table({
        "user_id": ["1"] * 5, "movie_id": ["a", "a", "a", "b", "b"],
        "tag": ["  Funny ", "hilarious", "funny", "space   travel", "family"],
        "timestamp_utc": [stamp(1), stamp(5), stamp(9), stamp(2), stamp(8)]}), run / "tags.parquet")
    manifest = build_tags(run, tmp_path / "tags", encoder_name="fake", encoder=FakeEncoder())
    assert manifest["phrases"] == 4 and manifest["movie_phrase_pairs"] == 4  # 'funny' first used day 1
    index = TagIndex(tmp_path / "tags", ["a", "b", "c"])
    vectors, has = index.movie_vectors(stamp(3).timestamp())
    assert has.tolist() == [True, True, False] and not vectors[2].any()
    assert vectors[0] @ np.array([1, 0, 0]) == pytest.approx(1, abs=1e-2)  # float16 storage; only 'funny' visible before day 3
    later, _ = index.movie_vectors(stamp(10).timestamp())
    assert later[0] @ np.array([1, 0, 0]) < 1
    lists = index.phrase_lists(stamp(10).timestamp())
    assert index.phrase_match(lists[0], lists[1]) < .2 and np.isnan(index.phrase_match(lists[0], lists[2]))
    projected, _ = index.movie_vectors(stamp(10).timestamp(), projected=True)
    assert projected.shape[1] <= 32 and not projected[2].any()
    assert normalize("  Space   TRAVEL ") == "space travel"
    assert np.allclose(index.encode_new(["FUNNY", "new phrase"], FakeEncoder())[0], index.vectors[index.phrase_id["funny"]])
