"""Stable, model-agnostic contracts at the application/model boundary."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MovieId = Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[^\s]+$")]
Genre = Annotated[str, Field(min_length=1, max_length=64)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unique(values: list[str], name: str) -> list[str]:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain distinct values")
    return values


class RatingRecord(StrictModel):
    movie_id: MovieId
    rating: float = Field(ge=0.5, le=5.0, multiple_of=0.5)
    rated_at: datetime

    @field_validator("rating")
    @classmethod
    def finite_rating(cls, value: float):
        if not math.isfinite(value):
            raise ValueError("rating must be finite")
        return value

    @field_validator("rated_at")
    @classmethod
    def utc_timestamp(cls, value: datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("rated_at must include a UTC offset")
        if value.utcoffset() != timedelta(0):
            raise ValueError("rated_at must be UTC")
        return value


class PreferenceState(StrictModel):
    history_items: list[MovieId] = Field(default_factory=list, max_length=500)
    excluded_movie_ids: list[MovieId] = Field(default_factory=list, max_length=500)
    include_genres: list[Genre] = Field(default_factory=list, max_length=20)
    exclude_genres: list[Genre] = Field(default_factory=list, max_length=20)
    # None means the caller supplied no numeric profile. [] is a complete empty profile.
    ratings: list[RatingRecord] | None = Field(default=None, max_length=500)

    @field_validator("history_items", "excluded_movie_ids", "include_genres", "exclude_genres")
    @classmethod
    def distinct(cls, values: list[str], info):
        return _unique(values, info.field_name)

    @model_validator(mode="after")
    def consistent(self):
        overlap = set(self.history_items) & set(self.excluded_movie_ids)
        if overlap:
            raise ValueError(f"liked and excluded movie IDs overlap: {sorted(overlap)}")
        genre_overlap = set(self.include_genres) & set(self.exclude_genres)
        if genre_overlap:
            raise ValueError(f"included and excluded genres overlap: {sorted(genre_overlap)}")
        if self.ratings is not None:
            _unique([record.movie_id for record in self.ratings], "rating movie IDs")
        return self


class RatingStatistics(StrictModel):
    mean_rating: float | None = None
    rating_count: int | None = Field(default=None, ge=0)
    days_inactive: float | None = Field(default=None, ge=0)
    complete: bool


class ModelQuery(StrictModel):
    history_items: list[MovieId] = Field(max_length=500)
    known_user_id: str | None = Field(default=None, min_length=1, max_length=64)
    rated_movie_ids: list[MovieId] = Field(default_factory=list, max_length=500)
    rating_statistics: RatingStatistics | None = None
    # Optional model context derived from saved browser ratings (feedback §2-§3). When
    # present, ratings/times align with history_items; None marks an unknown value
    # (e.g. a legacy favorite). Times and as_of are Unix seconds.
    history_ratings: list[float | None] = Field(default_factory=list, max_length=500)
    history_times: list[float | None] = Field(default_factory=list, max_length=500)
    disliked_items: list[MovieId] = Field(default_factory=list, max_length=500)
    as_of: float | None = None

    @field_validator("history_items", "rated_movie_ids", "disliked_items")
    @classmethod
    def unique_movie_ids(cls, values: list[str], info):
        return _unique(values, info.field_name)

    @model_validator(mode="after")
    def aligned_history_context(self):
        for name in ("history_ratings", "history_times"):
            values = getattr(self, name)
            if values and len(values) != len(self.history_items):
                raise ValueError(f"{name} must align with history_items")
        if set(self.disliked_items) & set(self.history_items):
            raise ValueError("a movie cannot be both a favorite and a dislike")
        return self


class ComponentInfo(StrictModel):
    name: str
    version: str
    mode: Literal["fixture", "baseline", "learned", "pass_through", "external"]


class ScoredMovie(StrictModel):
    movie_id: MovieId
    score: float | None = None

    @field_validator("score")
    @classmethod
    def finite_score(cls, value: float | None):
        if value is not None and not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class CandidateBatch(StrictModel):
    candidates: list[ScoredMovie] = Field(max_length=500)
    component: ComponentInfo
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_ids(self):
        _unique([item.movie_id for item in self.candidates], "candidate movie IDs")
        return self


class RankingResult(StrictModel):
    items: list[ScoredMovie] = Field(max_length=500)
    component: ComponentInfo
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_ids(self):
        _unique([item.movie_id for item in self.items], "ranked movie IDs")
        return self


class CatalogMovie(StrictModel):
    movie_id: MovieId
    title: str = Field(min_length=1, max_length=500)
    genres: list[str] = Field(default_factory=list, max_length=30)
    poster_url: str | None = None


class CatalogSearchResponse(StrictModel):
    items: list[CatalogMovie]
    total: int
    backend_mode: str


class PosterRequest(StrictModel):
    movie_ids: list[MovieId] = Field(min_length=1, max_length=50)

    @field_validator("movie_ids")
    @classmethod
    def unique_movie_ids(cls, values: list[str]):
        return _unique(values, "movie_ids")


class PosterMetadata(StrictModel):
    movie_id: MovieId
    tmdb_id: str | None = None
    poster_url: str | None = None
    backdrop_url: str | None = None
    backdrop_url_small: str | None = None
    status: Literal["available", "missing", "unavailable"]


class PosterResponse(StrictModel):
    items: list[PosterMetadata]
    provider: Literal["tmdb"] = "tmdb"
    configured: bool


class ApplicationStatus(StrictModel):
    backend_mode: str
    model_label: str
    data_source_label: str
    refinement_available: bool
    poster_provider: Literal["tmdb"] = "tmdb"
    posters_configured: bool
    available_genres: list[str]
    feedback_mode: Literal['off', 'observe', 'randomized'] = 'off'


class RecommendationRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=128)
    preference_revision: int = Field(ge=0)
    preferences: PreferenceState
    result_count: int = Field(default=12, ge=1, le=50)
    known_user_id: str | None = Field(default=None, min_length=1, max_length=64)
    assignment_key: str | None = Field(default=None, min_length=16, max_length=64, pattern=r'^[a-zA-Z0-9-]+$')
    # Ephemeral recommendation seeds, never persisted as favorites or ratings.
    temporary_history_items: list[MovieId] = Field(default_factory=list, max_length=20)

    @field_validator("temporary_history_items")
    @classmethod
    def unique_temporary_history(cls, values: list[str]):
        return _unique(values, "temporary_history_items")

    @model_validator(mode="after")
    def unambiguous_identity(self):
        if self.known_user_id is not None and self.preferences.ratings is not None:
            raise ValueError("known_user_id cannot be combined with a browser numeric profile")
        return self


class Timings(StrictModel):
    retrieval_ms: float
    filtering_ms: float
    reranking_ms: float
    total_ms: float


class RecommendationItem(StrictModel):
    rank: int = Field(ge=1)
    movie: CatalogMovie
    retrieval_score: float | None = None
    ranking_score: float | None = None


class FeedbackAssignment(StrictModel):
    decision_id: str
    token: str
    assigned_at: str
    movie_id: MovieId
    policy: str
    propensity: float = Field(gt=0, le=1)
    eligible_count: int = Field(ge=1)
    age_ms: int = Field(default=0, ge=0)
    outcome_window_seconds: int = 1800
    synthetic: bool


class FeedbackEvent(StrictModel):
    event_id: str = Field(min_length=16, max_length=64, pattern=r'^[a-zA-Z0-9-]+$')
    decision_id: str = Field(min_length=16, max_length=64)
    token: str = Field(min_length=32, max_length=128)
    movie_id: MovieId
    kind: Literal['display', 'details_open', 'favorite_add', 'hide', 'heartbeat', 'window_end']
    elapsed_ms: int = Field(ge=0, le=1805000)
    surface: Literal['featured'] = 'featured'
    stream_ok: bool = True


class FeedbackAck(StrictModel):
    accepted: bool = True
    duplicate: bool


class RecommendationResponse(StrictModel):
    request_id: str
    preference_revision: int
    recommendations: list[RecommendationItem]
    applied_preferences: PreferenceState
    retrieval: ComponentInfo
    reranker: ComponentInfo
    backend_mode: str
    partial: bool
    warnings: list[str]
    timings: Timings
    feedback: FeedbackAssignment | None = None


class RefinementRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=128)
    preference_revision: int = Field(ge=0)
    preferences: PreferenceState
    text: str = Field(min_length=1, max_length=500)


class AppliedRefinement(StrictModel):
    outcome: Literal["applied"] = "applied"
    preference_revision: int
    preferences: PreferenceState
    summary: str
    resolved_movies: list[CatalogMovie] = Field(default_factory=list)


class ClarificationRefinement(StrictModel):
    outcome: Literal["clarification"] = "clarification"
    question: str


class UnsupportedRefinement(StrictModel):
    outcome: Literal["unsupported"] = "unsupported"
    message: str


class UnavailableRefinement(StrictModel):
    outcome: Literal["unavailable"] = "unavailable"
    message: str


RefinementResponse = Annotated[
    AppliedRefinement | ClarificationRefinement | UnsupportedRefinement | UnavailableRefinement,
    Field(discriminator="outcome"),
]


class StatusResponse(StrictModel):
    status: Literal["ok", "not_ready"]
    backend_mode: str
    detail: str | None = None


class ErrorDetail(StrictModel):
    code: str
    message: str
    request_id: str
    issues: list[dict] = Field(default_factory=list)


class ErrorResponse(StrictModel):
    error: ErrorDetail
