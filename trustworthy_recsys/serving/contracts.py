"""Stable, model-agnostic contracts at the application/model boundary."""

from __future__ import annotations

import math
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


class PreferenceState(StrictModel):
    history_items: list[MovieId] = Field(default_factory=list, max_length=100)
    excluded_movie_ids: list[MovieId] = Field(default_factory=list, max_length=100)
    include_genres: list[Genre] = Field(default_factory=list, max_length=20)
    exclude_genres: list[Genre] = Field(default_factory=list, max_length=20)

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
        return self


class ModelQuery(StrictModel):
    history_items: list[MovieId] = Field(max_length=100)
    known_user_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("history_items")
    @classmethod
    def unique_history(cls, values: list[str]):
        return _unique(values, "history_items")


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


class RecommendationRequest(StrictModel):
    session_id: str = Field(min_length=1, max_length=128)
    preference_revision: int = Field(ge=0)
    preferences: PreferenceState
    result_count: int = Field(default=12, ge=1, le=50)
    known_user_id: str | None = Field(default=None, min_length=1, max_length=64)


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
