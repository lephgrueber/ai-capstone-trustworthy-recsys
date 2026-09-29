from datetime import timedelta

import pandas as pd
import pytest

from eval.harness import RecommendationRequest, evaluate, load_synthetic_examples
from trustworthy_recsys.baselines.popularity import (
    PopularityRecommender,
    fit_popularity,
    fit_top_rated,
    load_interactions,
)


def _interactions(rows):
    """Build an interactions frame from (user, movie, rating, 'YYYY-MM-DD') rows."""

    frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp_utc"])
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame


# "1" is an old favorite; "2" is popular only recently; "3" is rated highly
# but rarely; "10" ties "2" on all-time count to exercise numeric tie-breaks.
INTERACTIONS = _interactions([
    ("u1", "1", 4.0, "2020-01-01"),
    ("u2", "1", 4.0, "2020-01-02"),
    ("u3", "1", 3.0, "2020-01-03"),
    ("u4", "1", 2.0, "2020-01-04"),
    ("u1", "2", 3.0, "2023-06-01"),
    ("u2", "2", 3.0, "2023-06-02"),
    ("u3", "2", 3.0, "2023-06-03"),
    ("u1", "10", 5.0, "2020-02-01"),
    ("u2", "10", 5.0, "2020-02-02"),
    ("u3", "10", 1.0, "2020-02-03"),
    ("u4", "3", 5.0, "2023-06-04"),
])


def _request(*history):
    return RecommendationRequest(user_id="user", history_items=list(history))


def test_rejects_empty_ranking():
    with pytest.raises(ValueError):
        PopularityRecommender(ranking=())


def test_rejects_duplicate_ranking_items():
    with pytest.raises(ValueError):
        PopularityRecommender(ranking=("1", "1"))


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        PopularityRecommender(ranking=("1",))(_request(), 0)


def test_skips_history_items_and_respects_k():
    recommender = PopularityRecommender(ranking=("1", "2", "3", "4", "5"))

    assert recommender(_request("1", "3"), 2) == ["2", "4"]
    assert recommender(_request(), 10) == ["1", "2", "3", "4", "5"]


def test_all_time_popularity_ranks_by_count_then_numeric_id():
    recommender = fit_popularity(INTERACTIONS)

    # "2" and "10" both have 3 interactions; numeric order puts "2" first
    # even though "10" < "2" as strings.
    assert recommender.ranking == ("1", "2", "10", "3")


def test_windowed_popularity_favors_recent_items_and_keeps_the_tail():
    recommender = fit_popularity(INTERACTIONS, window=timedelta(days=30))

    # Only "2" and "3" fall in the window; older items follow by all-time
    # count so rankings can still be filled for heavy users.
    assert recommender.ranking == ("2", "3", "1", "10")


def test_window_anchored_at_explicit_as_of():
    as_of = pd.Timestamp("2020-02-05", tz="UTC")
    early = INTERACTIONS[INTERACTIONS["timestamp_utc"] < as_of]

    recommender = fit_popularity(early, window=timedelta(days=10), as_of=as_of)

    assert recommender.ranking == ("10", "1")


def test_rejects_interactions_after_as_of():
    with pytest.raises(ValueError, match="past as_of"):
        fit_popularity(INTERACTIONS, as_of=pd.Timestamp("2021-01-01", tz="UTC"))


def test_rejects_naive_as_of():
    with pytest.raises(ValueError, match="timezone-aware"):
        fit_popularity(INTERACTIONS, as_of=pd.Timestamp("2030-01-01"))


def test_rejects_non_positive_window():
    with pytest.raises(ValueError):
        fit_popularity(INTERACTIONS, window=timedelta(0))


def test_min_rating_counts_only_positive_interactions():
    recommender = fit_popularity(INTERACTIONS, min_rating=4.0)

    # Positives: "1" x2, "10" x2, "3" x1; "2" has none and is dropped.
    assert recommender.ranking == ("1", "10", "3")


def test_top_rated_shrinks_sparse_items_toward_global_mean():
    # "1" has a single perfect rating; "2" is rated 4.5 by twenty users;
    # "3" is widely disliked, pulling the global mean down to about 2.8.
    rows = [("u0", "1", 5.0, "2024-01-01")]
    rows += [(f"u{n}", "2", 4.5, "2024-01-01") for n in range(20)]
    rows += [(f"u{n}", "3", 1.0, "2024-01-01") for n in range(20)]
    interactions = _interactions(rows)

    assert fit_top_rated(interactions, prior_weight=0).ranking == ("1", "2", "3")
    assert fit_top_rated(interactions, prior_weight=5).ranking == ("2", "1", "3")


def test_top_rated_rejects_negative_prior_weight():
    with pytest.raises(ValueError):
        fit_top_rated(INTERACTIONS, prior_weight=-1)


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        fit_popularity(INTERACTIONS.drop(columns=["rating"]))


def test_load_interactions_round_trips_parquet(tmp_path):
    path = tmp_path / "train.parquet"
    INTERACTIONS.to_parquet(path)

    loaded = load_interactions(path)

    assert fit_popularity(loaded).ranking == fit_popularity(INTERACTIONS).ranking


def test_runs_end_to_end_through_the_harness():
    examples = load_synthetic_examples()
    rows = [
        (example.user_id, item, 5.0, "2024-01-01")
        for example in examples
        for item in (*example.history_items, *example.relevant_items)
    ]
    recommender = fit_popularity(_interactions(rows))

    aggregate, per_example = evaluate(examples, recommender, recall_k=2, ndcg_k=2)

    assert "recall@2" in aggregate
    assert "ndcg@2" in aggregate
    assert len(per_example) == len(examples)
