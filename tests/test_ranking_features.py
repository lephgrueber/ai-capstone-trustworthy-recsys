"""Reranker features on a hand-checkable catalog, plus table labels."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("scipy")

from eval.harness import EvalExample
from trustworthy_recsys.ranking.dataset import build_rows, to_table
from trustworthy_recsys.ranking.features import FEATURE_NAMES, RankingFeatures, release_year

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
