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

- tower: max / mean cosine between the candidate's frozen movie-tower vector and the
  vector of each earlier favorite (one score per favorite, then aggregated).
- dislikes: log1p(number of explicit dislikes), and the tower and audience cosines
  against disliked movies, kept separate from the favorite features.
- metadata (attach_metadata): per credit role, the number of distinct people the
  candidate shares with earlier favorites and the share of favorites (with any credit in
  that role) that share at least one person; max / mean cosine between the candidate's
  tag summary and each tagged favorite's, the number of tagged favorites compared, and a
  phrase-level best-match score against the most recent tagged favorites; absolute gap
  between candidate runtime and mean favorite runtime; director share and max tag cosine
  against disliked movies. Credit matches use IMDb person IDs, so a person unseen in
  training still matches. Unknown is NaN, never zero: a candidate without credits in a
  role, without tags or without runtime gets NaN for those comparisons.

Missing values (unknown user, empty history, unparsable year, no movie vectors attached,
no dislike information) are NaN; LightGBM learns a default direction for them. A request
that carries ratings but no dislikes has dislike count 0 and NaN dislike similarities.
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
    # M3 feedback §2: learned similarity and explicit dislikes (appended; older models
    # select their own columns by name).
    "tower_cosine_max",
    "tower_cosine_mean",
    "dislike_count_log",
    "dislike_tower_cosine_max",
    "dislike_tower_cosine_mean",
    "dislike_audience_cosine_max",
    "dislike_audience_cosine_mean",
    # Movie metadata comparisons (feedback §2 table; IMDb credits/runtime and tags, §4-§5).
    "director_shared_count",
    "director_share",
    "writer_shared_count",
    "writer_share",
    "cast_shared_count",
    "cast_share",
    "tag_cosine_max",
    "tag_cosine_mean",
    "tag_comparable_favorites",
    "tag_phrase_match_max",
    "runtime_gap",
    "dislike_director_share",
    "dislike_tag_cosine_max",
)
COL = {name: index for index, name in enumerate(FEATURE_NAMES)}
CREDIT_ROLES = (("directors", "director"), ("writers", "writer"), ("cast", "cast"))
PHRASE_FAVORITES = 5
PHRASES_PER_MOVIE = 20

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
        """Arrays describe training interactions; timestamps are UTC epoch seconds.

        An item index of -1 marks a rating of a movie outside the catalog (e.g. first
        rated after the towers' fitting window): it counts toward the viewer's rating
        statistics but not toward any movie statistic or co-like signal.
        """
        if len(timestamps) and int(timestamps.max()) >= cutoff_seconds:
            raise ValueError("Training interactions reach the feature cutoff")
        item_indices = np.asarray(item_indices, dtype=np.int64)
        known = item_indices >= 0
        self.item_ids = list(item_ids)
        self.item_map = {item: index for index, item in enumerate(self.item_ids)}
        self.item_features = np.asarray(item_features, dtype="float32")
        self.cutoff_seconds = int(cutoff_seconds)
        self.max_history = max_history
        self.recent_likes = recent_likes
        n_items, n_users = len(self.item_ids), len(user_ids)
        positive = (ratings >= threshold) & known
        recent = positive & (timestamps >= cutoff_seconds - recent_days * SECONDS_PER_DAY)

        counts = np.bincount(item_indices[known], minlength=n_items).astype("float64")
        self.positive_counts = np.bincount(item_indices[positive], minlength=n_items).astype("float64")
        recent_counts = np.bincount(item_indices[recent], minlength=n_items)
        rating_sums = np.bincount(item_indices[known], weights=ratings[known], minlength=n_items)
        global_mean = float(ratings[known].mean()) if known.any() else 0.0
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
        self.item_vectors = None  # frozen movie-tower vectors; attach with attach_item_vectors

        # Binary user x item matrix of training positives; columns are item likers.
        self.likes = sparse.csc_matrix(
            (np.ones(int(positive.sum()), dtype="float32"),
             (user_codes[positive], item_indices[positive])),
            shape=(n_users, n_items),
        )

    @classmethod
    def from_run(cls, run_dir, scenario, item_ids, item_features, cutoff_seconds,
                 partitions=("train",), **options):
        """Load interactions before `cutoff_seconds` and titles from a completed pipeline run.

        With the default partition every training interaction must lie inside the
        catalog (M3 behavior). Passing partitions explicitly (fitting windows) keeps only
        events before the cutoff and allows movies outside the catalog (see __init__).
        """
        windowed = tuple(partitions) != ("train",)
        table = pa.concat_tables([
            pq.read_table(f"{run_dir}/{scenario}/{part}.parquet",
                          columns=["user_id", "movie_id", "rating", "timestamp_utc"])
            for part in partitions
        ])
        stamps = table["timestamp_utc"].cast(pa.int64()).to_numpy() // 1000
        if windowed or (len(stamps) and stamps.max() >= cutoff_seconds):
            table = table.filter(pa.array(stamps < cutoff_seconds))
            windowed = True
        table = table.combine_chunks()
        users = pc.dictionary_encode(table["user_id"]).combine_chunks()
        movies = pc.dictionary_encode(table["movie_id"]).combine_chunks()
        item_map = {item: index for index, item in enumerate(item_ids)}
        lookup = np.array([item_map.get(item, -1) for item in movies.dictionary.to_pylist()], dtype=np.int64)
        item_indices = lookup[movies.indices.to_numpy()]
        if not windowed and (item_indices < 0).any():
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

    def attach_item_vectors(self, vectors) -> "RankingFeatures":
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.shape[0] != len(self.item_ids):
            raise ValueError("Movie vectors do not match the feature catalog")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.item_vectors = vectors / np.maximum(norms, 1e-12)
        return self

    def attach_metadata(self, metadata_dir=None, tags_dir=None) -> "RankingFeatures":
        """Credits/runtime (metadata.py output) and tags (tags.py output) as of the cutoff."""
        self.metadata_sources = {}
        if metadata_dir is not None:
            from trustworthy_recsys.data.metadata import load
            frame = load(metadata_dir).reindex(self.item_ids)
            self.credit_matrices = {}
            for role, _ in CREDIT_ROLES:
                lists = [list(v) if isinstance(v, (list, np.ndarray)) else [] for v in frame[role]]
                people = {p: i for i, p in enumerate(sorted({p for v in lists for p in v}))}
                rows = np.repeat(np.arange(len(lists)), [len(v) for v in lists])
                cols = np.array([people[p] for v in lists for p in v], dtype=np.int64)
                self.credit_matrices[role] = sparse.csr_matrix(
                    (np.ones(len(cols), dtype="float32"), (rows, cols)), shape=(len(lists), max(1, len(people))))
            self.runtime = frame["runtime_minutes"].to_numpy(dtype="float64")
            self.metadata_sources["metadata"] = str(metadata_dir)
        if tags_dir is not None:
            from trustworthy_recsys.data.tags import TagIndex
            index = TagIndex(tags_dir, self.item_ids)
            self.tag_vectors, self.has_tags = index.movie_vectors(self.cutoff_seconds)
            phrase_lists = index.phrase_lists(self.cutoff_seconds)
            self.tag_phrase_vectors = index.vectors
            self.movie_phrases = [ids[-PHRASES_PER_MOVIE:] for ids in phrase_lists]
            self.metadata_sources["tags"] = str(tags_dir)
        return self

    def credit_share(self, role, candidates, others):
        """(shared-person count, share of comparable others sharing anyone); NaN when unknown."""
        matrix = self.credit_matrices[role]
        cand, other = matrix[candidates], matrix[others]
        known_other = np.asarray(other.sum(axis=1)).ravel() > 0
        has = np.asarray(cand.sum(axis=1)).ravel() > 0
        union = np.asarray(other.sum(axis=0)).ravel() > 0
        count = np.asarray(cand.multiply(union[None, :]).sum(axis=1)).ravel()
        overlap = (cand @ other[known_other].T).toarray() > 0 if known_other.any() else np.zeros((len(candidates), 0))
        share = overlap.mean(axis=1) if known_other.any() else np.full(len(candidates), np.nan)
        count = np.where(has, count, np.nan)
        share = np.where(has, share, np.nan)
        return count, share

    def tag_cosines(self, candidates, others):
        tagged = others[self.has_tags[others]]
        if not len(tagged):
            return None, tagged
        cos = self.tag_vectors[candidates] @ self.tag_vectors[tagged].T
        cos[~self.has_tags[candidates]] = np.nan
        return cos, tagged

    def phrase_match(self, candidates, favorites):
        """Per candidate: max over favorites of mean best-match cosine of its phrases."""
        out = np.full(len(candidates), np.nan)
        fav_lists = [self.movie_phrases[f] for f in favorites if len(self.movie_phrases[f])]
        if not fav_lists:
            return out
        fav_ids = np.concatenate(fav_lists)
        fav_bounds = np.cumsum([0] + [len(x) for x in fav_lists])
        cand_lists = [self.movie_phrases[c] for c in candidates]
        lengths = np.array([len(x) for x in cand_lists])
        if not lengths.sum():
            return out
        cand_ids = np.concatenate([x for x in cand_lists if len(x)])
        sims = self.tag_phrase_vectors[cand_ids] @ self.tag_phrase_vectors[fav_ids].T
        best = np.maximum.reduceat(sims, fav_bounds[:-1], axis=1)  # [candidate phrases, favorites]
        owners = np.repeat(np.flatnonzero(lengths), lengths[lengths > 0])
        sums = np.zeros((len(candidates), best.shape[1]))
        np.add.at(sums, owners, best)
        means = sums[lengths > 0] / lengths[lengths > 0, None]
        out[lengths > 0] = means.max(axis=1)
        return out

    def audience_cosine(self, candidates, others):
        """Cosine between binary liker vectors: shared likers / sqrt(likers A x likers B)."""
        pair_counts = (self.likes[:, candidates].T @ self.likes[:, others]).toarray()
        norms = np.sqrt(np.outer(self.positive_counts[candidates], self.positive_counts[others]))
        return np.divide(pair_counts, norms, out=np.zeros_like(pair_counts, dtype="float64"), where=norms > 0)

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def history_indices(self, history_items) -> np.ndarray:
        # Deduplicate, keep order (oldest first), and ignore unknown IDs, matching retrieval.
        known = dict.fromkeys(self.item_map[item] for item in history_items if item in self.item_map)
        return np.fromiter(known, dtype=np.int64, count=len(known))

    def build(self, history_items, candidates, retrieval_scores=None, user_id=None, rating_statistics=None,
              disliked_items=None) -> np.ndarray:
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
        if rating_statistics is not None:
            # A supplied browser profile overrides only request-scoped user statistics.
            # None remains NaN; a genuine count of zero remains log1p(0) == 0.
            values = (
                rating_statistics.mean_rating,
                None if rating_statistics.rating_count is None else np.log1p(rating_statistics.rating_count),
                rating_statistics.days_inactive,
            )
            out[:, 10:13] = [np.nan if value is None else value for value in values]
        if disliked_items is not None:
            disliked = self.history_indices(disliked_items)
            out[:, 22] = np.log1p(len(disliked))
            if len(disliked):
                disliked = disliked[-self.max_history:]
                if self.item_vectors is not None:
                    tower = self.item_vectors[candidates] @ self.item_vectors[disliked].T
                    out[:, 23], out[:, 24] = tower.max(axis=1), tower.mean(axis=1)
                audience = self.audience_cosine(candidates, disliked)
                out[:, 25], out[:, 26] = audience.max(axis=1), audience.mean(axis=1)
                if getattr(self, "credit_matrices", None) is not None:
                    out[:, COL["dislike_director_share"]] = self.credit_share("directors", candidates, disliked)[1]
                if getattr(self, "tag_vectors", None) is not None:
                    cos, _ = self.tag_cosines(candidates, disliked)
                    if cos is not None:
                        out[:, COL["dislike_tag_cosine_max"]] = cos.max(axis=1)
        if not len(history):
            return out
        self._metadata_features(out, candidates, history[-self.max_history:])
        if self.item_vectors is not None:
            tower = self.item_vectors[candidates] @ self.item_vectors[history[-self.max_history:]].T
            out[:, 20], out[:, 21] = tower.max(axis=1), tower.mean(axis=1)

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

        cosine = self.audience_cosine(candidates, history[-self.recent_likes:])
        out[:, 16] = cosine.max(axis=1)
        out[:, 17] = cosine.mean(axis=1)

        history_years = self.years[history]
        if np.isfinite(history_years).any():
            out[:, 18] = np.abs(self.years[candidates] - np.nanmean(history_years))
        out[:, 19] = np.log1p(popularity) - np.log1p(self.positive_counts[history]).mean()
        return out

    def _metadata_features(self, out, candidates, favorites):
        if getattr(self, "credit_matrices", None) is not None:
            for role, name in CREDIT_ROLES:
                out[:, COL[f"{name}_shared_count"]], out[:, COL[f"{name}_share"]] = \
                    self.credit_share(role, candidates, favorites)
            minutes = self.runtime[favorites]
            if np.isfinite(minutes).any():
                out[:, COL["runtime_gap"]] = np.abs(self.runtime[candidates] - np.nanmean(minutes))
        if getattr(self, "tag_vectors", None) is not None:
            cos, tagged = self.tag_cosines(candidates, favorites)
            out[:, COL["tag_comparable_favorites"]] = len(tagged)
            if cos is not None:
                out[:, COL["tag_cosine_max"]] = cos.max(axis=1)
                out[:, COL["tag_cosine_mean"]] = cos.mean(axis=1)
                out[:, COL["tag_phrase_match_max"]] = self.phrase_match(candidates, tagged[-PHRASE_FAVORITES:])
