"""Reranker features on a hand-checkable catalog, plus table labels."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from eval.harness import EvalExample
from trustworthy_recsys.ranking.dataset import build_rows, to_table
from trustworthy_recsys.ranking.features import FEATURE_NAMES, RankingFeatures, release_year
from trustworthy_recsys.serving.contracts import RatingStatistics

DAY = 86_400
CUTOFF = 400 * DAY
COLUMN = {name: index for index, name in enumerate(FEATURE_NAMES)}


def fitted(**options):
    # Likes: u1 -> a, b; u2 -> a, c; u3 -> b, c. u3 rated d 2.0 (not a like).
    interactions = [
        ("u1", 0, 5.0, 0), ("u1", 1, 5.0, 390 * DAY),
        ("u2", 0, 5.0, 380 * DAY), ("u2", 2, 4.0, 390 * DAY),
        ("u3", 1, 4.0, 395 * DAY), ("u3", 2, 4.5, 396 * DAY), ("u3", 3, 2.0, 397 * DAY),
    ]
    users = ["u1", "u2", "u3"]
    codes, items, ratings, stamps = zip(*[(users.index(u), i, r, t) for u, i, r, t in interactions])
    return RankingFeatures(
        ["a", "b", "c", "d"],
        np.array([[1, 0], [1, 0], [.5, .5], [0, 1]], dtype="float32"),
        np.array(codes), users, np.array(items), np.array(ratings), np.array(stamps),
        {"a": "A (1990)", "b": "B (2000)", "c": "C", "d": "D (2010) "},
        CUTOFF, prior_weight=1.0, **options,
    )


def test_cross_features_match_hand_computation():
    features = fitted()
    out = features.build(["a"], [1, 2, 3], retrieval_scores=[.9, .8, .1], user_id="u1")
    col = lambda name: out[:, COLUMN[name]]
    assert col("retrieval_score") == pytest.approx([.9, .8, .1])
    assert list(col("retrieval_rank")) == [0, 1, 2]
    # Likers of a are u1, u2. b's likers (u1, u3) overlap once; c's (u2, u3) once; d has none.
    assert col("co_like_log_sum") == pytest.approx(np.log1p([1, 1, 0]))
    assert col("co_like_per_liker") == pytest.approx([.5, .5, 0])
    # cosine(a, b) = 1 shared liker / sqrt(2 * 2).
    assert col("recent_cosine_max") == pytest.approx([.5, .5, 0])
    assert col("genre_match") == pytest.approx([1, .5, 0])
    assert col("year_gap")[[0, 2]] == pytest.approx([10, 20])
    assert np.isnan(col("year_gap")[1])  # "C" has no parsable year
    assert col("user_mean_rating") == pytest.approx([5, 5, 5])
    assert col("user_days_inactive") == pytest.approx([10, 10, 10])


def test_item_statistics_use_only_training_window():
    features = fitted()
    out = features.build([], [0, 3])
    col = lambda name: out[:, COLUMN[name]]
    assert col("item_log_positive_count") == pytest.approx(np.log1p([2, 0]))
    # a's first like is 400 days before the cutoff, outside the 365-day window.
    assert col("item_log_recent_positive_count") == pytest.approx(np.log1p([1, 0]))
    # Bayesian mean with one prior rating at the global mean (29.5 / 7).
    prior = 29.5 / 7
    assert col("item_bayesian_rating") == pytest.approx([(10 + prior) / 3, (2 + prior) / 2])
    assert col("item_release_year") == pytest.approx([1990, 2010])
    # No history and no known user: personalization features stay missing.
    for name in ("genre_match", "co_like_log_sum", "user_mean_rating", "retrieval_score"):
        assert np.isnan(col(name)).all()


def test_history_is_deduplicated_and_unknown_ids_ignored():
    features = fitted()
    assert list(features.history_indices(["b", "zz", "a", "b"])) == [1, 0]
    same = features.build(["a", "zz", "a"], [1, 2])
    assert np.array_equal(same, features.build(["a"], [1, 2]), equal_nan=True)


def test_browser_rating_statistics_override_only_request_user_columns():
    features = fitted()
    statistics = RatingStatistics(mean_rating=3.5, rating_count=0, days_inactive=2.25, complete=True)
    out = features.build(["a"], [1, 2], user_id="u1", rating_statistics=statistics)
    col = lambda name: out[:, COLUMN[name]]
    assert col("user_mean_rating") == pytest.approx([3.5, 3.5])
    assert col("user_log_rating_count") == pytest.approx([0, 0])
    assert col("user_days_inactive") == pytest.approx([2.25, 2.25])
    assert col("genre_match") == pytest.approx([1, .5])


def test_mixed_browser_profile_keeps_user_statistics_missing():
    features = fitted()
    statistics = RatingStatistics(complete=False)
    out = features.build(["a"], [1], user_id="u1", rating_statistics=statistics)
    assert np.isnan(out[:, [COLUMN["user_mean_rating"], COLUMN["user_log_rating_count"], COLUMN["user_days_inactive"]]]).all()


def test_rejects_interactions_at_or_after_cutoff():
    with pytest.raises(ValueError, match="cutoff"):
        RankingFeatures(["a"], np.ones((1, 1)), np.array([0]), ["u"], np.array([0]),
                        np.array([5.0]), np.array([CUTOFF]), {}, CUTOFF)


class StubRetrieval:
    inputs = SimpleNamespace(item_ids=["a", "b", "c", "d"])

    def recommend_scored(self, request, depth):
        if not request.history_items:
            return [0, 1][:depth], None, True
        return [3, 2, 1][:depth], [.3, .2, .1][:depth], False


def test_table_labels_mark_held_out_relevant_candidates():
    examples = [EvalExample("u1", ["a"], ["c", "x"]), EvalExample("new", [], ["a"])]
    table = to_table(list(build_rows(StubRetrieval(), fitted(), examples, 3, progress_every=0)))
    rows = table.to_pydict()
    assert rows["movie_id"] == ["d", "c", "b", "a", "b"]
    assert rows["label"] == [0, 1, 0, 1, 0]
    assert rows["fallback"] == [False, False, False, True, True]
    assert rows["user_relevant_count"] == [2, 2, 2, 1, 1]
    assert table.num_columns == 5 + len(FEATURE_NAMES)


def test_release_year_parsing():
    assert release_year("Toy Story (1995)") == 1995
    assert np.isnan(release_year("Untitled"))


def test_tower_similarity_matches_feedback_example():
    # Candidate d has cosine 0.8 with favorite a and 0.2 with favorite b.
    vectors = np.array([[.8, .6], [.2, np.sqrt(1 - .04)], [0, 1], [1, 0]], dtype="float32")
    features = fitted().attach_item_vectors(vectors)
    out = features.build(["a", "b"], [3], user_id="u1")
    assert out[0, COLUMN["tower_cosine_max"]] == pytest.approx(.8)
    assert out[0, COLUMN["tower_cosine_mean"]] == pytest.approx(.5)  # mean of scores, not of vectors
    assert np.isnan(fitted().build(["a"], [3])[0, COLUMN["tower_cosine_max"]])  # no vectors attached


def test_dislike_features_are_separate_and_missing_is_not_zero():
    features = fitted().attach_item_vectors(np.eye(4, dtype="float32"))
    unknown = features.build(["a"], [2, 3])
    assert np.isnan(unknown[:, COLUMN["dislike_count_log"]]).all()
    none = features.build(["a"], [2, 3], disliked_items=[])
    assert (none[:, COLUMN["dislike_count_log"]] == 0).all()
    assert np.isnan(none[:, COLUMN["dislike_tower_cosine_max"]]).all()
    some = features.build(["a"], [2, 3], disliked_items=["c"])
    assert some[0, COLUMN["dislike_count_log"]] == pytest.approx(np.log1p(1))
    assert some[:, COLUMN["dislike_tower_cosine_max"]].tolist() == [1, 0]
    # Audience cosine of c with itself: 2 likers shared of 2 -> 1.
    assert some[0, COLUMN["dislike_audience_cosine_max"]] == pytest.approx(1)
    # Favorite features are unchanged by dislikes.
    assert np.array_equal(none[:, :20], some[:, :20], equal_nan=True)


def test_metadata_comparisons_and_missing_rules(tmp_path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from trustworthy_recsys.data.reporting import hashes, write_json
    from trustworthy_recsys.data.tags import build as build_tags
    from test_metadata_tags import FakeEncoder

    meta = tmp_path / "meta"
    meta.mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"movie_id": "a", "imdb_id": None, "runtime_minutes": 100.0, "directors": ["d1"], "writers": ["w1"], "cast": []},
        {"movie_id": "b", "imdb_id": None, "runtime_minutes": 140.0, "directors": ["d2"], "writers": [], "cast": ["c1"]},
        {"movie_id": "c", "imdb_id": None, "runtime_minutes": None, "directors": ["d1", "d3"], "writers": ["w9"], "cast": ["c1"]},
        {"movie_id": "d", "imdb_id": None, "runtime_minutes": 90.0, "directors": [], "writers": ["w1"], "cast": []},
    ]), meta / "movie_metadata.parquet")
    write_json(meta / "manifest.json", {"output_sha256": hashes(meta / "movie_metadata.parquet")["sha256"]})
    run = tmp_path / "run"
    run.mkdir()
    early = pd.Timestamp(0, unit="s", tz="UTC")
    pq.write_table(pa.table({"user_id": ["1"] * 4, "movie_id": ["a", "c", "c", "d"],
                             "tag": ["funny", "hilarious", "space travel", "family"],
                             "timestamp_utc": [early, early, early, pd.Timestamp(CUTOFF + DAY, unit="s", tz="UTC")]}),
                   run / "tags.parquet")
    build_tags(run, tmp_path / "tags", encoder_name="fake", encoder=FakeEncoder())
    features = fitted().attach_metadata(meta, tmp_path / "tags")

    out = features.build(["a", "b"], [2, 3], user_id="u1")
    c, d = out
    # c shares director d1 with a (1 of 2 favorites) and cast c1 with b (1 of 1 favorite with cast).
    assert c[COLUMN["director_shared_count"]] == 1 and c[COLUMN["director_share"]] == pytest.approx(.5)
    assert c[COLUMN["cast_share"]] == 1 and c[COLUMN["writer_share"]] == 0
    # d lists no director (unknown -> NaN) but shares writer w1 with a.
    assert np.isnan(d[COLUMN["director_share"]]) and d[COLUMN["writer_share"]] == 1
    # Runtime gap to mean favorite runtime 120; c's runtime is unknown.
    assert np.isnan(c[COLUMN["runtime_gap"]]) and d[COLUMN["runtime_gap"]] == 30
    # Only favorite a is tagged before the cutoff; d's tag comes later, so d is untagged.
    assert c[COLUMN["tag_comparable_favorites"]] == 1
    assert c[COLUMN["tag_cosine_max"]] > 0.5 and np.isnan(d[COLUMN["tag_cosine_max"]])
    assert c[COLUMN["tag_phrase_match_max"]] == pytest.approx(
        np.mean([FakeEncoder().encode(["hilarious"])[0] @ FakeEncoder().encode(["funny"])[0],
                 FakeEncoder().encode(["space travel"])[0] @ FakeEncoder().encode(["funny"])[0]]), abs=1e-2)
    disliked = features.build(["b"], [2], user_id="u1", disliked_items=["a"])[0]
    assert disliked[COLUMN["dislike_director_share"]] == 1 and disliked[COLUMN["dislike_tag_cosine_max"]] > .5


def test_record_rating_statistics_match_movielens_lookup():
    from trustworthy_recsys.ranking.features import RATING_HISTORY, rating_statistics_from_record
    # u3 before the cutoff: liked b (day 395, 4.0) and c (day 396, 4.5), rated d 2.0 on day 397.
    record = {"history_items": ["b", "c"], "history_ratings": [4.0, 4.5], "history_times": [395 * DAY, 396 * DAY],
              "other_rated_items": ["d"], "other_rated_ratings": [2.0], "other_rated_times": [397 * DAY]}
    stats = rating_statistics_from_record(record, CUTOFF)
    assert stats.days_since_like == pytest.approx(4) and stats.days_inactive == pytest.approx(3)
    features = fitted()
    looked_up = features.build(["b", "c"], [0], user_id="u3")
    passed = features.build(["b", "c"], [0], rating_statistics=stats)
    columns = [COLUMN[name] for name in RATING_HISTORY]
    assert np.allclose(looked_up[:, columns], passed[:, columns])
    never_liked = rating_statistics_from_record({"other_rated_ratings": [1.0], "other_rated_times": [5]}, CUTOFF)
    assert never_liked.days_since_like is None and never_liked.rating_count == 1


def test_masking_is_per_viewer_and_deterministic():
    from trustworthy_recsys.ranking.train import mask_columns
    users = np.repeat([f"u{i}" for i in range(200)], 3)
    table = {"user_id": users, "user_mean_rating": np.ones(600, dtype="float32")}
    half = mask_columns(dict(table), {"user_mean_rating": .5}, 0)["user_mean_rating"]
    assert 60 < np.isnan(half).sum() / 3 < 140
    assert all(len(set(np.isnan(half[i:i + 3]))) == 1 for i in range(0, 600, 3))  # whole viewers
    assert np.array_equal(np.isnan(half), np.isnan(mask_columns(dict(table), {"user_mean_rating": .5}, 0)["user_mean_rating"]))
    assert np.isnan(mask_columns(dict(table), {"user_mean_rating": 1.0}, 0)["user_mean_rating"]).all()
