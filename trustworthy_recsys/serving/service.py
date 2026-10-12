"""Recommendation orchestration, hard constraints, and output validation."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from trustworthy_recsys.serving.catalog import Catalog
from trustworthy_recsys.serving.components import RetrievalComponent, RerankingComponent
from trustworthy_recsys.serving.contracts import (
    CandidateBatch,
    ModelQuery,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RankingResult,
    ScoredMovie,
    Timings,
)


class ServiceInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ComponentOutputError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecommendationService:
    catalog: Catalog
    retrieval: RetrievalComponent
    reranker: RerankingComponent
    backend_mode: str
    candidate_budget: int = 100

    def validate_preferences(self, request: RecommendationRequest) -> None:
        supplied = set(request.preferences.history_items) | set(request.preferences.excluded_movie_ids)
        unknown = sorted(supplied - self.catalog.supported_ids)
        if unknown:
            raise ServiceInputError("unknown_movie_ids", f"Unknown or unsupported movie IDs: {unknown}")
        known_genres = self.catalog.genres
        genres = set(request.preferences.include_genres) | set(request.preferences.exclude_genres)
        unknown_genres = sorted(genres - known_genres)
        if unknown_genres:
            raise ServiceInputError("unknown_genres", f"Unknown genres: {unknown_genres}")

    def recommend(self, request: RecommendationRequest, request_id: str | None = None) -> RecommendationResponse:
        started = time.perf_counter()
        self.validate_preferences(request)
        query = ModelQuery(history_items=request.preferences.history_items, known_user_id=request.known_user_id)
        retrieval_started = time.perf_counter()
        budget = min(self.candidate_budget, max(request.result_count * 4, request.result_count))
        try:
            raw_candidates = self.retrieval.retrieve(query, budget)
            candidates = (
                raw_candidates
                if isinstance(raw_candidates, CandidateBatch)
                else CandidateBatch.model_validate(raw_candidates)
            )
        except (ValueError, TypeError) as error:
            raise ComponentOutputError(f"retrieval component returned invalid output: {error}") from error
        retrieval_done = time.perf_counter()
        if candidates.component != self.retrieval.info:
            raise ComponentOutputError("retrieval component metadata does not match loaded component")
        if len(candidates.candidates) > budget:
            raise ComponentOutputError("retrieval component exceeded the requested candidate limit")
        invalid = [item.movie_id for item in candidates.candidates if item.movie_id not in self.catalog.supported_ids]
        if invalid:
            raise ComponentOutputError(f"retrieval returned IDs outside the supported catalog: {invalid}")

        score_by_id = {item.movie_id: item.score for item in candidates.candidates}
        filtered = [item for item in candidates.candidates if self.catalog.allows(item.movie_id, request.preferences)]
        filtering_done = time.perf_counter()
        filtered_batch = CandidateBatch(
            candidates=filtered,
            component=candidates.component,
            warnings=candidates.warnings,
        )
        try:
            raw_ranked = self.reranker.rerank(query, filtered_batch, request.result_count)
            ranked = (
                raw_ranked
                if isinstance(raw_ranked, RankingResult)
                else RankingResult.model_validate(raw_ranked)
            )
        except (ValueError, TypeError) as error:
            raise ComponentOutputError(f"reranking component returned invalid output: {error}") from error
        reranking_done = time.perf_counter()
        if ranked.component != self.reranker.info:
            raise ComponentOutputError("reranking component metadata does not match loaded component")
        allowed_ids = {item.movie_id for item in filtered}
        ranked_ids = [item.movie_id for item in ranked.items]
        if any(movie_id not in allowed_ids for movie_id in ranked_ids):
            raise ComponentOutputError("reranker introduced or restored a filtered candidate")
        if len(ranked.items) > request.result_count:
            raise ComponentOutputError("reranker exceeded the requested result count")
        # Defense in depth: reapply all hard constraints after reranking.
        if any(not self.catalog.allows(movie_id, request.preferences) for movie_id in ranked_ids):
            raise ComponentOutputError("final results violate hard constraints")

        recommendations = [
            RecommendationItem(
                rank=index,
                movie=self.catalog.movies[item.movie_id],
                retrieval_score=score_by_id[item.movie_id],
                ranking_score=item.score,
            )
            for index, item in enumerate(ranked.items, start=1)
        ]
        warnings = [*candidates.warnings, *ranked.warnings]
        partial = len(recommendations) < request.result_count
        if partial:
            warnings.append(
                "The bounded candidate pool was exhausted after hard filtering; this does not prove the full catalog has no matches."
            )
        finished = time.perf_counter()
        return RecommendationResponse(
            request_id=request_id or str(uuid.uuid4()),
            preference_revision=request.preference_revision,
            recommendations=recommendations,
            applied_preferences=request.preferences,
            retrieval=candidates.component,
            reranker=ranked.component,
            backend_mode=self.backend_mode,
            partial=partial,
            warnings=warnings,
            timings=Timings(
                retrieval_ms=(retrieval_done - retrieval_started) * 1000,
                filtering_ms=(filtering_done - retrieval_done) * 1000,
                reranking_ms=(reranking_done - filtering_done) * 1000,
                total_ms=(finished - started) * 1000,
            ),
        )
