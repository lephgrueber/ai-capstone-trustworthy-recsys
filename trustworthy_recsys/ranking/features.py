"""Reranker features for (user, candidate) pairs, fitted on training-period data only.

Every statistic is computed from interactions before the training cutoff, so the same
object can featurize validation users (reranker training), test users (evaluation) and
live requests (serving) without consulting held-out ratings.

Feature groups:

- retrieval: two-tower cosine score and candidate position.
- item: positive/all/recent counts, Bayesian mean rating, release year, age, genre count.
- user: history length, mean training rating, training rating count, days inactive.
- cross: genre-profile match, co-like counts with the user's history, cosine
  co-occurrence with the most recent likes, and year/popularity taste gaps. The two
  towers cannot learn these user x item interactions by construction.

Missing values (unknown user, empty history, unparsable year) are NaN; LightGBM
learns a default direction for them.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

FEATURE_NAMES = (
    "retrieval_score",
    "retrieval_rank",
    "item_log_positive_count",
    "item_log_rating_count",
    "item_log_recent_positive_count",
    "item_bayesian_rating",
    "item_release_year",
    "item_age_years",
    "item_genre_count",
    "user_log_history_length",
    "user_mean_rating",
    "user_log_rating_count",
    "user_days_inactive",
    "genre_match",
    "co_like_log_sum",
    "co_like_per_liker",
    "recent_cosine_max",
    "recent_cosine_mean",
    "year_gap",
    "popularity_gap",
)

YEAR = re.compile(r"\((\d{4})\)\s*$")
SECONDS_PER_DAY = 86_400


def release_year(title: str) -> float:
    match = YEAR.search(title.strip())
    return float(match.group(1)) if match else np.nan


class RankingFeatures:
    """Training-period statistics plus per-request featurization."""

    def __init__(
        self,
        item_ids: list[str],
        item_features: np.ndarray,
        user_codes: np.ndarray,
        user_ids: list[str],
        item_indices: np.ndarray,
        ratings: np.ndarray,
        timestamps: np.ndarray,
        titles: dict[str, str],
        cutoff_seconds: int,
        threshold: float = 4.0,
        recent_days: int = 365,
        prior_weight: float = 20.0,
        max_history: int = 500,
        recent_likes: int = 5,
    ):
        """Arrays describe training interactions; timestamps are UTC epoch seconds."""
        if len(timestamps) and int(timestamps.max()) >= cutoff_seconds:
            raise ValueError("Training interactions reach the feature cutoff")
        self.item_ids = list(item_ids)
        self.item_map = {item: index for index, item in enumerate(self.item_ids)}
        self.item_features = np.asarray(item_features, dtype="float32")
        self.cutoff_seconds = int(cutoff_seconds)
        self.max_history = max_history
        self.recent_likes = recent_likes
        n_items, n_users = len(self.item_ids), len(user_ids)
        positive = ratings >= threshold
        recent = positive & (timestamps >= cutoff_seconds - recent_days * SECONDS_PER_DAY)

        counts = np.bincount(item_indices, minlength=n_items).astype("float64")
        self.positive_counts = np.bincount(item_indices[positive], minlength=n_items).astype("float64")
        recent_counts = np.bincount(item_indices[recent], minlength=n_items)
        rating_sums = np.bincount(item_indices, weights=ratings, minlength=n_items)
        global_mean = float(ratings.mean()) if len(ratings) else 0.0
        years = np.array([release_year(titles.get(item, "")) for item in self.item_ids])
        cutoff_year = pd.Timestamp(cutoff_seconds, unit="s", tz="UTC").year
        self.item_columns = np.column_stack([
            np.log1p(self.positive_counts),
            np.log1p(counts),
            np.log1p(recent_counts),
            (rating_sums + prior_weight * global_mean) / (counts + prior_weight),
            years,
            cutoff_year - years,
            (self.item_features > 0).sum(axis=1),
        ]).astype("float32")
        self.years = years

        user_counts = np.bincount(user_codes, minlength=n_users)
        user_sums = np.bincount(user_codes, weights=ratings, minlength=n_users)
        last_seen = np.full(n_users, np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(last_seen, user_codes, timestamps)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.user_columns = np.column_stack([
                user_sums / user_counts,
                np.log1p(user_counts),
                (cutoff_seconds - last_seen) / SECONDS_PER_DAY,
            ]).astype("float32")
        self.user_map = {user: index for index, user in enumerate(user_ids)}

        # Binary user x item matrix of training positives; columns are item likers.
        self.likes = sparse.csc_matrix(
            (np.ones(int(positive.sum()), dtype="float32"),
             (user_codes[positive], item_indices[positive])),
            shape=(n_users, n_items),
        )

    @classmethod
    def from_run(cls, run_dir, scenario, item_ids, item_features, cutoff_seconds, **options):
        """Load training interactions and titles from a completed pipeline run."""
        table = pq.read_table(
            f"{run_dir}/{scenario}/train.parquet",
            columns=["user_id", "movie_id", "rating", "timestamp_utc"],
        ).combine_chunks()
        users = pc.dictionary_encode(table["user_id"]).combine_chunks()
        movies = pc.dictionary_encode(table["movie_id"]).combine_chunks()
        item_map = {item: index for index, item in enumerate(item_ids)}
        lookup = np.array([item_map.get(item, -1) for item in movies.dictionary.to_pylist()], dtype=np.int64)
        item_indices = lookup[movies.indices.to_numpy()]
        if (item_indices < 0).any():
            raise ValueError("Training interaction outside the retrieval catalog")
        timestamps = table["timestamp_utc"].cast(pa.int64()).to_numpy() // 1000
        movie_table = pq.read_table(f"{run_dir}/movies.parquet", columns=["movie_id", "title"])
        titles = dict(zip(movie_table["movie_id"].to_pylist(), movie_table["title"].to_pylist()))
        return cls(
            item_ids, item_features,
            users.indices.to_numpy().astype(np.int64), users.dictionary.to_pylist(),
            item_indices, table["rating"].to_numpy().astype("float64"), timestamps,
            titles, cutoff_seconds, **options,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def history_indices(self, history_items) -> np.ndarray:
        # Deduplicate, keep order (oldest first), and ignore unknown IDs, matching retrieval.
        known = dict.fromkeys(self.item_map[item] for item in history_items if item in self.item_map)
        return np.fromiter(known, dtype=np.int64, count=len(known))

    def build(self, history_items, candidates, retrieval_scores=None, user_id=None) -> np.ndarray:
        """Return a float32 [len(candidates), len(FEATURE_NAMES)] matrix.

        `candidates` are catalog row indices in retrieval order; `retrieval_scores`
        aligns with them or is None (popularity fallback).
        """
        candidates = np.asarray(candidates, dtype=np.int64)
        n = len(candidates)
        out = np.full((n, len(FEATURE_NAMES)), np.nan, dtype="float32")
        if not n:
            return out
        if retrieval_scores is not None:
            out[:, 0] = retrieval_scores
        out[:, 1] = np.arange(n)
        out[:, 2:9] = self.item_columns[candidates]

        history = self.history_indices(history_items)
        out[:, 9] = np.log1p(len(history))
        row = self.user_map.get(user_id) if user_id is not None else None
        if row is not None:
            out[:, 10:13] = self.user_columns[row]
        if not len(history):
            return out

        profile = self.item_features[history].mean(axis=0)
        out[:, 13] = self.item_features[candidates] @ profile

        # w[u] = how many of this user's (most recent) likes training user u also liked;
        # co[c] = total overlap among c's likers, i.e. sum over h of co-likes(h, c).
        capped = history[-self.max_history:]
        overlap = np.asarray(self.likes[:, capped].sum(axis=1)).ravel()
        candidate_likers = self.likes[:, candidates]
        co = candidate_likers.T @ overlap
        popularity = self.positive_counts[candidates]
        out[:, 14] = np.log1p(co)
        out[:, 15] = np.divide(co, popularity, out=np.zeros(n), where=popularity > 0)

        recent = history[-self.recent_likes:]
        pair_counts = (candidate_likers.T @ self.likes[:, recent]).toarray()
        norms = np.sqrt(np.outer(popularity, self.positive_counts[recent]))
        cosine = np.divide(pair_counts, norms, out=np.zeros_like(pair_counts, dtype="float64"), where=norms > 0)
        out[:, 16] = cosine.max(axis=1)
        out[:, 17] = cosine.mean(axis=1)

        history_years = self.years[history]
        if np.isfinite(history_years).any():
            out[:, 18] = np.abs(self.years[candidates] - np.nanmean(history_years))
        out[:, 19] = np.log1p(popularity) - np.log1p(self.positive_counts[history]).mean()
        return out
