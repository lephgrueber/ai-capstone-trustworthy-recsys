"""Development components and adapters for future learned components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from eval.harness import RecommendationRequest as BaselineRequest
from trustworthy_recsys.baselines.genre import fit_genre
from trustworthy_recsys.baselines.predict import build_recommender
from trustworthy_recsys.serving.catalog import Catalog
from trustworthy_recsys.serving.contracts import (
    CandidateBatch,
    ComponentInfo,
    ModelQuery,
    RankingResult,
    ScoredMovie,
)


class RetrievalComponent(Protocol):
    info: ComponentInfo

    def retrieve(self, query: ModelQuery, limit: int) -> CandidateBatch: ...


class RerankingComponent(Protocol):
    info: ComponentInfo

    def rerank(self, query: ModelQuery, candidates: CandidateBatch, limit: int) -> RankingResult: ...


@dataclass(frozen=True)
class BaselineRetrievalAdapter:
    """Adapt the repository's score-less recommender callable to retrieval."""

    recommender: Callable
    info: ComponentInfo
    supported_ids: frozenset[str]

    def retrieve(self, query: ModelQuery, limit: int) -> CandidateBatch:
        supported_history = [item for item in query.history_items if item in self.supported_ids]
        warnings = []
        if len(supported_history) != len(query.history_items):
            warnings.append("Unsupported history items were ignored by retrieval.")
        request = BaselineRequest(user_id=query.known_user_id or "new-visitor", history_items=supported_history)
        ids = list(self.recommender(request, limit))
        # The baseline exposes IDs only. Missing scores intentionally remain null.
        return CandidateBatch(
            candidates=[ScoredMovie(movie_id=movie_id, score=None) for movie_id in ids],
            component=self.info,
            warnings=warnings,
        )


class FixtureRetrieval(BaselineRetrievalAdapter):
    @classmethod
    def build(cls, catalog: Catalog) -> "FixtureRetrieval":
        rows = []
        # Deterministic synthetic popularity counts; these are fixture weights, not learned scores.
        for position, movie_id in enumerate(catalog.movies):
            for user in range(max(1, len(catalog.movies) - position)):
                rows.append(
                    {"user_id": f"fixture-{user}", "movie_id": movie_id, "rating": 4.0, "timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC")}
                )
        recommender = fit_genre(pd.DataFrame(rows), catalog.as_frame(), alpha=1.0)
        return cls(
            recommender=recommender,
            info=ComponentInfo(name="fixture-genre-retrieval", version="1.0.0", mode="fixture"),
            supported_ids=catalog.supported_ids,
        )


@dataclass(frozen=True)
class LearnedRetrievalAdapter:
    """Expose the verified two-tower/Weaviate recommender at the serving boundary."""

    recommender: Any
    info: ComponentInfo
    supported_ids: frozenset[str]

    @classmethod
    def load(cls, inputs_dir, model_dir) -> "LearnedRetrievalAdapter":
        # Keep retrieval dependencies optional for fixture/baseline installations.
        try:
            from trustworthy_recsys.retrieval.recommender import RetrievalRecommender

            recommender = RetrievalRecommender.load(inputs_dir, model_dir)
        except Exception as error:
            raise ValueError(f"could not load learned retrieval artifacts: {error}") from error
        descriptor = recommender.index.descriptor
        fingerprint = descriptor.get("fingerprint", "unknown")
        return cls(
            recommender=recommender,
            info=ComponentInfo(
                name="two-tower-weaviate",
                version=f"{descriptor.get('collection', 'unknown')}:{fingerprint[:12]}",
                mode="learned",
            ),
            supported_ids=frozenset(recommender.inputs.item_ids),
        )

    def retrieve(self, query: ModelQuery, limit: int) -> CandidateBatch:
        supported_history = [item for item in query.history_items if item in self.supported_ids]
        warnings = []
        if len(supported_history) != len(query.history_items):
            warnings.append("Unsupported history items were ignored by learned retrieval.")
        request = BaselineRequest(
            user_id=query.known_user_id or "__new_visitor__",
            history_items=supported_history,
        )
        indices, fallback = self.recommender.recommend_indices(request, limit)
        if fallback:
            warnings.append("Popularity fallback used because no supported history items were supplied.")
        return CandidateBatch(
            candidates=[
                ScoredMovie(movie_id=self.recommender.inputs.item_ids[index], score=None)
                for index in indices
            ],
            component=self.info,
            warnings=warnings,
        )

    def close(self) -> None:
        self.recommender.close()


@dataclass(frozen=True)
class PassThroughReranker:
    info: ComponentInfo = field(
        default_factory=lambda: ComponentInfo(
            name="pass-through", version="1.0.0", mode="pass_through"
        )
    )

    def rerank(self, query: ModelQuery, candidates: CandidateBatch, limit: int) -> RankingResult:
        return RankingResult(
            items=candidates.candidates[:limit],
            component=self.info,
            warnings=["No learned reranking is active."],
        )


@dataclass(frozen=True)
class ExternalRetrievalAdapter:
    """Sample adapter for Tolu's callable; validation occurs at this boundary."""

    callable: Callable[[ModelQuery, int], CandidateBatch | dict]
    info: ComponentInfo

    def retrieve(self, query: ModelQuery, limit: int) -> CandidateBatch:
        raw = self.callable(query, limit)
        return raw if isinstance(raw, CandidateBatch) else CandidateBatch.model_validate(raw)


@dataclass(frozen=True)
class ExternalRerankingAdapter:
    """Sample adapter for Navin's callable; it may only reorder candidates."""

    callable: Callable[[ModelQuery, CandidateBatch, int], RankingResult | dict]
    info: ComponentInfo

    def rerank(self, query: ModelQuery, candidates: CandidateBatch, limit: int) -> RankingResult:
        raw = self.callable(query, candidates, limit)
        result = raw if isinstance(raw, RankingResult) else RankingResult.model_validate(raw)
        candidate_ids = {item.movie_id for item in candidates.candidates}
        introduced = [item.movie_id for item in result.items if item.movie_id not in candidate_ids]
        if introduced:
            raise ValueError(f"reranker introduced candidate IDs: {introduced}")
        if len(result.items) > limit:
            raise ValueError("reranker returned more than the requested limit")
        return result


def build_baseline_retrieval(run_dir, scenario: str, catalog: Catalog) -> BaselineRetrievalAdapter:
    recommender = build_recommender("genre", run_dir, scenario)
    return BaselineRetrievalAdapter(
        recommender=recommender,
        info=ComponentInfo(name="genre-baseline", version="repository-0.1.0", mode="baseline"),
        supported_ids=frozenset(recommender.items),
    )


def build_learned_retrieval(inputs_dir, model_dir) -> LearnedRetrievalAdapter:
    return LearnedRetrievalAdapter.load(inputs_dir, model_dir)
