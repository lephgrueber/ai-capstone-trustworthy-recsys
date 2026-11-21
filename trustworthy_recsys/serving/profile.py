"""Authoritative per-request browser-rating derivation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trustworthy_recsys.serving.contracts import ModelQuery, PreferenceState, RatingStatistics

FUTURE_TOLERANCE = timedelta(minutes=5)
DISLIKE_THRESHOLD = 2.0
SECONDS_PER_DAY = 86_400


def derive_model_query(preferences: PreferenceState, known_user_id: str | None, as_of: datetime) -> ModelQuery:
    """Build one coherent model query from validated preferences at one UTC reference time.

    Legacy favorites have no fabricated dates. They precede dated positives so genuine
    timestamp ordering and the meaning of recent positive history remain intact.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must include a UTC offset")
    as_of = as_of.astimezone(timezone.utc)
    records = preferences.ratings
    if records is None:
        return ModelQuery(history_items=preferences.history_items, known_user_id=known_user_id)

    rated_ids = {record.movie_id for record in records}
    legacy = [movie_id for movie_id in preferences.history_items if movie_id not in rated_ids]
    normalized = []
    for record in records:
        rated_at = record.rated_at.astimezone(timezone.utc)
        if rated_at > as_of + FUTURE_TOLERANCE:
            raise ValueError(f"rating timestamp is materially in the future: {record.movie_id}")
        normalized.append((rated_at, record.movie_id, record.rating))
    normalized.sort(key=lambda row: (row[0], row[1]))
    positives = [movie_id for _, movie_id, rating in normalized if rating >= 4.0]
    if len(legacy) + len(positives) > 500:
        raise ValueError("effective positive history exceeds the supported limit of 500 movies")

    if legacy:
        statistics = RatingStatistics(complete=False)
    elif normalized:
        latest = max(row[0] for row in normalized)
        liked = [row[0] for row in normalized if row[2] >= 4.0]
        statistics = RatingStatistics(
            mean_rating=sum(row[2] for row in normalized) / len(normalized),
            rating_count=len(normalized),
            days_inactive=max(0.0, (as_of - latest).total_seconds() / SECONDS_PER_DAY),
            days_since_like=max(0.0, (as_of - max(liked)).total_seconds() / SECONDS_PER_DAY) if liked else None,
            complete=True,
        )
    else:
        statistics = RatingStatistics(rating_count=0, complete=True)

    dated = [(movie_id, rating, rated_at.timestamp()) for rated_at, movie_id, rating in normalized if rating >= 4.0]
    return ModelQuery(
        history_items=[*legacy, *positives],
        known_user_id=known_user_id,
        rated_movie_ids=[movie_id for _, movie_id, _ in normalized],
        rating_statistics=statistics,
        # Legacy favorites carry no fabricated stars or dates.
        history_ratings=[None] * len(legacy) + [rating for _, rating, _ in dated],
        history_times=[None] * len(legacy) + [stamp for _, _, stamp in dated],
        disliked_items=[movie_id for _, movie_id, rating in normalized if rating <= DISLIKE_THRESHOLD],
        as_of=as_of.timestamp(),
    )
