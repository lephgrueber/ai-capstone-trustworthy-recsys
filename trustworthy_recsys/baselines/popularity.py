"""Non-personalized popularity baseline recommenders.

Every user gets the same ranked list of movies, minus the ones already in
their history. These baselines answer "does personalization help at all?":
a personalized model that can't beat them isn't earning its complexity.

Three ways to build the shared ranking, all from training data only:

1. All-time popularity (`fit_popularity`). Rank movies by how many
   training interactions they have. `min_rating` can restrict the count
   to positive ratings (e.g. 4.0, the pipeline's relevance threshold).
2. Recent popularity (`fit_popularity` with `window`). Rank by
   interactions in the last `window` before `as_of`, normally the
   training cutoff. On a temporal split, all-time counts favor old
   classics, while recent counts better reflect what people watch next.
   Movies outside the window follow, ordered by all-time count, so long
   histories can still be filled to k. Interactions after `as_of` are
   rejected rather than silently leaking future popularity.
3. Top rated (`fit_top_rated`). Rank by Bayesian-average rating: each
   movie's mean rating is pulled toward the overall mean as if it had
   `prior_weight` extra average ratings, so a movie with one 5-star
   rating doesn't outrank one rated 4.5 by thousands. This usually scores
   below popularity on Recall/NDCG, which is a useful contrast.

Remaining ties are broken by numeric movie ID so rankings are
deterministic. `popularity_scores` exposes the underlying counts for
models that build on popularity, such as the genre baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_COLUMNS = ["user_id", "movie_id", "rating", "timestamp_utc"]


def load_interactions(path: str | Path) -> pd.DataFrame:
    """Load a split's interactions, such as a scenario's train.parquet."""

    return pd.read_parquet(path, columns=_COLUMNS)


@dataclass(frozen=True)
class PopularityRecommender:
    """Recommends the same global item ranking to every user.

    Items the user has already interacted with are skipped, so each user
    receives the top-k unseen items of `ranking`. Build instances with
    `fit_popularity` or `fit_top_rated` rather than by hand.
    """

    ranking: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ranking:
            raise ValueError("ranking must contain at least one item")
        # The harness's metrics reject ranked lists with repeated item IDs.
        if len(self.ranking) != len(set(self.ranking)):
            raise ValueError("ranking must not contain duplicate item IDs")

    def __call__(self, request, k: int) -> Sequence[str]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        seen = set(request.history_items)
        ranked = []
        for item in self.ranking:
            if item not in seen:
                ranked.append(item)
                if len(ranked) == k:
                    break
        return ranked


def fit_popularity(
    interactions: pd.DataFrame,
    *,
    window: timedelta | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    min_rating: float | None = None,
) -> PopularityRecommender:
    """Rank items by interaction count, optionally within a recent window.

    Args:
        interactions: Training interactions with the pipeline's columns.
        window: If given, rank by counts in the `window` ending at `as_of`.
            Items outside the window still follow, ordered by all-time
            count, so users with long histories can always be filled to k.
        as_of: End of the popularity window. Defaults to the latest
            interaction; pass the split's validation start to anchor the
            window at the exact training cutoff.
        min_rating: If given, count only ratings at or above this value
            (e.g. the pipeline's 4.0 relevance threshold).
    """

    scores = popularity_scores(
        interactions, window=window, as_of=as_of, min_rating=min_rating
    )
    return PopularityRecommender(tuple(scores.index))


def popularity_scores(
    interactions: pd.DataFrame,
    *,
    window: timedelta | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    min_rating: float | None = None,
) -> pd.DataFrame:
    """Per-item popularity, most popular first; see `fit_popularity`.

    Returns a frame indexed by movie ID with `primary` (the count used for
    ranking: windowed if `window` is given) and `total` (all-time count).
    """

    frame = _prepare(interactions, min_rating)
    as_of = _resolve_as_of(frame, as_of)

    scores = pd.DataFrame({"total": frame.groupby("movie_id").size()})
    if window is None:
        scores["primary"] = scores["total"]
    else:
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        # Inclusive start: pipeline cutoffs fall on midnight, so a window
        # of N days before the cutoff should include events at its first
        # midnight.
        recent = frame[frame["timestamp_utc"] >= as_of - window]
        scores["primary"] = recent.groupby("movie_id").size()
        scores["primary"] = scores["primary"].fillna(0)

    return scores.loc[list(_rank(scores, ["primary", "total"]))]


def fit_top_rated(
    interactions: pd.DataFrame,
    *,
    prior_weight: float | None = None,
) -> PopularityRecommender:
    """Rank items by Bayesian-average rating.

    Each item's mean rating is shrunk toward the global mean as if it had
    `prior_weight` extra ratings at that mean, so items with only a few
    perfect ratings do not dominate the ranking. Defaults to the median
    number of ratings per item.
    """

    frame = _prepare(interactions, min_rating=None)
    stats = frame.groupby("movie_id")["rating"].agg(["sum", "count"])

    if prior_weight is None:
        prior_weight = float(stats["count"].median())
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative")

    global_mean = frame["rating"].mean()
    stats["primary"] = (stats["sum"] + prior_weight * global_mean) / (
        stats["count"] + prior_weight
    )
    stats["total"] = stats["count"]

    return PopularityRecommender(_rank(stats, ["primary", "total"]))


def _prepare(interactions: pd.DataFrame, min_rating: float | None) -> pd.DataFrame:
    missing = set(_COLUMNS) - set(interactions.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")

    frame = interactions
    if min_rating is not None:
        frame = frame[frame["rating"] >= min_rating]
    if frame.empty:
        raise ValueError("no interactions remain to rank items from")
    return frame


def _resolve_as_of(
    frame: pd.DataFrame,
    as_of: datetime | pd.Timestamp | None,
) -> pd.Timestamp:
    latest = frame["timestamp_utc"].max()
    if as_of is None:
        return latest

    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware (the data is UTC)")
    # Counting events after the reference time would leak future
    # popularity into a model that claims to be anchored at `as_of`.
    if latest > as_of:
        raise ValueError(f"interactions extend past as_of ({latest} > {as_of})")
    return as_of


def _rank(scores: pd.DataFrame, by: list[str]) -> tuple[str, ...]:
    # Break remaining ties by numeric movie ID, matching the pipeline's
    # ordering convention. For decimal IDs without leading zeros,
    # (length, string) order equals numeric order, and it still yields a
    # deterministic order for non-numeric test IDs.
    scores = scores.assign(
        _id_len=scores.index.str.len(),
        _id=scores.index,
    )
    ordered = scores.sort_values(
        by + ["_id_len", "_id"],
        ascending=[False] * len(by) + [True, True],
        kind="mergesort",
    )
    return tuple(ordered.index)
