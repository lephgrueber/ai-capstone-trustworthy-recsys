from __future__ import annotations

import pytest

from trustworthy_recsys.serving.catalog import fixture_catalog
from trustworthy_recsys.serving.components import FixtureRetrieval, PassThroughReranker
from trustworthy_recsys.serving.contracts import CandidateBatch, ComponentInfo, RankingResult, ScoredMovie
from trustworthy_recsys.serving.service import ComponentOutputError, RecommendationService
from trustworthy_recsys.serving.contracts import RecommendationRequest


def payload(**preferences):
    state = {"history_items": ["101"], "excluded_movie_ids": [], "include_genres": [], "exclude_genres": []}
    state.update(preferences)
    return {"session_id": "s", "preference_revision": 0, "preferences": state, "result_count": 6}


class IntroducesItem:
    info = ComponentInfo(name="corrupt-reranker", version="1", mode="external")

    def rerank(self, query, candidates, limit):
        return RankingResult(items=[ScoredMovie(movie_id="115")], component=self.info)


class UnknownRetrieval:
    info = ComponentInfo(name="corrupt-retrieval", version="1", mode="external")

    def retrieve(self, query, limit):
        return CandidateBatch(candidates=[ScoredMovie(movie_id="999")], component=self.info)


def service(retrieval=None, reranker=None):
    catalog = fixture_catalog()
    return RecommendationService(
        catalog,
        retrieval or FixtureRetrieval.build(catalog),
        reranker or PassThroughReranker(),
        "fixture",
    )


def test_corrupt_backend_ids_are_502_class_failures_not_user_input_errors():
    request = RecommendationRequest.model_validate(payload(history_items=[]))
    with pytest.raises(ComponentOutputError, match="outside"):
        service(retrieval=UnknownRetrieval()).recommend(request)


def test_reranker_cannot_restore_filtered_or_new_item():
    request = RecommendationRequest.model_validate(payload(history_items=[], excluded_movie_ids=["115"]))
    with pytest.raises(ComponentOutputError, match="introduced"):
        service(reranker=IntroducesItem()).recommend(request)
