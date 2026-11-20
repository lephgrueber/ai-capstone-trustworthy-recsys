from datetime import datetime, timedelta, timezone

import pytest

from trustworthy_recsys.serving.contracts import PreferenceState
from trustworthy_recsys.serving.profile import derive_model_query


NOW = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)


def rating(movie_id, value, days=0):
    return {"movie_id": movie_id, "rating": value, "rated_at": (NOW - timedelta(days=days)).isoformat()}


def test_complete_profile_derives_all_ratings_positive_history_and_recency():
    preferences = PreferenceState(ratings=[rating("3", 2.5, 4), rating("2", 5, 1), rating("1", 4, 1)])
    query = derive_model_query(preferences, None, NOW)
    assert query.history_items == ["1", "2"]  # chronological, then movie ID tie-breaker
    assert query.rated_movie_ids == ["3", "1", "2"]
    assert query.rating_statistics.mean_rating == pytest.approx(11.5 / 3)
    assert query.rating_statistics.rating_count == 3
    assert query.rating_statistics.days_inactive == pytest.approx(1)
    assert query.rating_statistics.complete


def test_explicit_empty_omitted_and_mixed_profiles_remain_distinct():
    omitted = derive_model_query(PreferenceState(history_items=["9"]), None, NOW)
    assert omitted.rating_statistics is None and omitted.history_items == ["9"]

    empty = derive_model_query(PreferenceState(ratings=[]), None, NOW)
    assert empty.rating_statistics.rating_count == 0
    assert empty.rating_statistics.mean_rating is None
    assert empty.rating_statistics.days_inactive is None
    assert empty.rating_statistics.complete

    mixed = derive_model_query(
        PreferenceState(history_items=["9", "2"], ratings=[rating("2", 5)]), None, NOW
    )
    assert mixed.history_items == ["9", "2"]
    assert mixed.rating_statistics.rating_count is None
    assert mixed.rating_statistics.mean_rating is None
    assert mixed.rating_statistics.days_inactive is None
    assert not mixed.rating_statistics.complete


def test_explicit_rating_supersedes_legacy_and_threshold_changes_history():
    low = derive_model_query(
        PreferenceState(history_items=["1"], ratings=[rating("1", 3.5)]), None, NOW
    )
    assert low.history_items == [] and low.rated_movie_ids == ["1"]
    positive = derive_model_query(
        PreferenceState(history_items=["1"], ratings=[rating("1", 4)]), None, NOW
    )
    assert positive.history_items == ["1"]


def test_materially_future_rating_is_rejected_and_small_clock_skew_clamps_to_zero():
    slight = PreferenceState(ratings=[{
        "movie_id": "1", "rating": 4, "rated_at": (NOW + timedelta(minutes=1)).isoformat()
    }])
    assert derive_model_query(slight, None, NOW).rating_statistics.days_inactive == 0
    future = PreferenceState(ratings=[{
        "movie_id": "1", "rating": 4, "rated_at": (NOW + timedelta(minutes=6)).isoformat()
    }])
    with pytest.raises(ValueError, match="future"):
        derive_model_query(future, None, NOW)


def test_model_context_aligns_ratings_times_and_dislikes():
    preferences = PreferenceState(history_items=["9"],
                                  ratings=[rating("3", 2.0, 4), rating("4", 3.0, 3), rating("2", 5, 1)])
    query = derive_model_query(preferences, None, NOW)
    assert query.history_items == ["9", "2"]
    # The legacy favorite keeps no fabricated star rating or date.
    assert query.history_ratings == [None, 5]
    assert query.history_times == [None, (NOW - timedelta(days=1)).timestamp()]
    assert query.disliked_items == ["3"]  # 3.0 stars is neither a like nor a dislike
    assert query.as_of == NOW.timestamp()
