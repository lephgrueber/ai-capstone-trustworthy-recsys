"""Genre-aware personalized popularity baseline.

A simple content heuristic between plain popularity and learned models:
recommend popular movies, but favor the genres each user already likes.

How a user's ranking is built:

1. Genre profile. Each movie in the user's history adds one unit of
   weight, split evenly across its genres (a Comedy|Romance movie adds
   0.5 to each). With `recency_decay` < 1, older history counts less.
2. Smoothing. The profile is blended with the genre mix of all training
   interactions, as if the user had `prior_weight` extra history items
   of the average genre mix. This keeps a single horror movie from making
   a user look 100% horror; a user with no history gets the average mix.
3. Genre match. Each candidate movie's match is the cosine similarity
   between its genres and the profile, so movies listing many genres
   are not favored just for that. Movies with no listed genres match 0.
4. Score. `popularity * match ** alpha`, where popularity is the
   movie's training interaction count plus one. `alpha` sets how much genre
   matters: 0 reproduces plain popularity, larger values lean harder on
   genre. Equal scores fall back to popularity order, and movies already
   in the user's history are never recommended.

The model only sees each user's positively rated movies (the harness
request carries no ratings), so how much a user liked a movie does not
change their profile, and disliked genres are not penalized.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_recsys.baselines.popularity import popularity_scores


def load_movies(path: str | Path) -> pd.DataFrame:
    """Load movie metadata, such as a pipeline run's movies.parquet."""

    return pd.read_parquet(path, columns=["movie_id", "genres"])


@dataclass(frozen=True, eq=False)
class GenreRecommender:
    """Ranks popular items, boosted by how well they match a user's genres.

    Each user's genre profile is the share of their history in each genre,
    smoothed toward the training data's overall genre mix. An item's score
    is its popularity times its cosine affinity with that profile, raised
    to `alpha`: 0 recovers plain popularity, larger values lean harder on
    genre match. Build instances with `fit_genre`.
    """

    items: tuple[str, ...]  # Candidates, most popular first
    popularity: np.ndarray  # Aligned with `items`
    item_vectors: np.ndarray  # (items, genres), unit-length rows
    genre_shares: Mapping[str, np.ndarray]  # Any movie's genres, summing to 1
    prior: np.ndarray  # Training genre mix, summing to 1
    alpha: float = 1.0
    prior_weight: float = 1.0
    recency_decay: float = 1.0

    def profile(self, history_items: Sequence[str]) -> np.ndarray:
        """Return the user's smoothed genre distribution."""

        # History is oldest first; with decay < 1, the newest item has
        # weight 1 and each older one `recency_decay` times the next.
        weights = self.recency_decay ** np.arange(len(history_items) - 1, -1, -1)
        counts = np.zeros_like(self.prior)
        for item, weight in zip(history_items, weights):
            shares = self.genre_shares.get(item)
            if shares is not None:
                counts += weight * shares
        return (counts + self.prior_weight * self.prior) / (
            counts.sum() + self.prior_weight
        )

    def __call__(self, request, k: int) -> Sequence[str]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        seen = set(request.history_items)
        affinity = self.item_vectors @ self.profile(request.history_items)
        scores = affinity**self.alpha * self.popularity

        # Only the top k + |seen| items can survive filtering, so avoid
        # sorting the whole catalog. Keep every item tied with the cutoff
        # score so ties still fall back to popularity order.
        needed = min(len(self.items), k + len(seen))
        cutoff = np.partition(-scores, needed - 1)[needed - 1]
        top = np.flatnonzero(-scores <= cutoff)
        # Candidates are stored most popular first, so ordering equal scores
        # by index breaks ties by popularity.
        order = top[np.lexsort((top, -scores[top]))]

        ranked = []
        for index in order:
            item = self.items[index]
            if item not in seen:
                ranked.append(item)
                if len(ranked) == k:
                    break
        return ranked


def fit_genre(
    interactions: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    alpha: float = 1.0,
    prior_weight: float = 1.0,
    recency_decay: float = 1.0,
    window: timedelta | None = None,
    as_of: datetime | pd.Timestamp | None = None,
    min_rating: float | None = None,
) -> GenreRecommender:
    """Fit genre-weighted popularity on training interactions.

    Args:
        interactions: Training interactions with the pipeline's columns.
        movies: Movie metadata with `movie_id` and list-valued `genres`.
        alpha: Strength of the genre match; 0 is plain popularity.
        prior_weight: How many history items' worth of the overall genre
            mix to blend into each profile. Users with no history get the
            overall mix.
        recency_decay: Per-item weight decay from newest to oldest
            history item, in (0, 1]; 1 weights all history equally.
        window, as_of, min_rating: Popularity options; see `fit_popularity`.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if prior_weight <= 0:
        raise ValueError("prior_weight must be positive")
    if not 0 < recency_decay <= 1:
        raise ValueError("recency_decay must be in (0, 1]")
    missing = {"movie_id", "genres"} - set(movies.columns)
    if missing:
        raise ValueError(f"movies missing columns: {sorted(missing)}")

    scores = popularity_scores(
        interactions, window=window, as_of=as_of, min_rating=min_rating
    )

    genres = sorted({genre for listed in movies["genres"] for genre in listed})
    if not genres:
        raise ValueError("movies list no genres")
    column = {genre: index for index, genre in enumerate(genres)}

    # Each movie spreads one unit of weight evenly over its genres; movies
    # without listed genres carry none and so never shift a profile.
    genre_shares = {}
    for movie_id, listed in zip(movies["movie_id"], movies["genres"]):
        if len(listed):
            shares = np.zeros(len(genres))
            shares[[column[genre] for genre in listed]] = 1 / len(listed)
            genre_shares[movie_id] = shares

    empty = np.zeros(len(genres))
    share_matrix = np.stack([genre_shares.get(item, empty) for item in scores.index])

    # Unit-length genre indicators make `item_vectors @ profile` a cosine
    # similarity (up to the per-user profile norm, which doesn't affect
    # ranking), so items aren't favored just for listing many genres.
    # Candidates without genres get a zero vector and sink below every
    # genre-matched item when alpha > 0.
    indicators = (share_matrix > 0).astype(float)
    norms = np.linalg.norm(indicators, axis=1, keepdims=True)
    item_vectors = np.divide(indicators, norms, out=np.zeros_like(indicators), where=norms > 0)

    prior = scores["total"].to_numpy() @ share_matrix
    if prior.sum() == 0:
        raise ValueError("no candidate item has listed genres")
    prior = prior / prior.sum()

    return GenreRecommender(
        items=tuple(scores.index),
        # +1 keeps items with no interactions in a recency window rankable
        # by genre match instead of all tying at zero.
        popularity=scores["primary"].to_numpy(dtype=float) + 1,
        item_vectors=item_vectors,
        genre_shares=genre_shares,
        prior=prior,
        alpha=alpha,
        prior_weight=prior_weight,
        recency_decay=recency_decay,
    )
