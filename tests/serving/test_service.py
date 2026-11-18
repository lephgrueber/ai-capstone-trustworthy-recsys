from __future__ import annotations

import pytest

from trustworthy_recsys.serving.catalog import fixture_catalog
from trustworthy_recsys.serving.components import FixtureRetrieval, PassThroughReranker
from trustworthy_recsys.serving.contracts import CandidateBatch, ComponentInfo, RankingResult, ScoredMovie
from trustworthy_recsys.serving.service import ComponentOutputError, RecommendationService
from trustworthy_recsys.serving.contracts import RecommendationRequest
from datetime import datetime, timezone


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


class RecordsLimit:
    info = ComponentInfo(name="records-limit", version="1", mode="external")

    def __init__(self):
        self.limits = []

    def retrieve(self, query, limit):
        self.limits.append(limit)
        return CandidateBatch(candidates=[ScoredMovie(movie_id="102", score=0.5)], component=self.info)


def test_reranker_receives_full_configured_candidate_budget():
    retrieval = RecordsLimit()
    catalog = fixture_catalog()
    request = RecommendationRequest.model_validate(payload())  # result_count=6
    RecommendationService(catalog, retrieval, PassThroughReranker(), "fixture", candidate_budget=500).recommend(request)
    assert retrieval.limits == [500]


class RecordsQueryAndCandidates:
    info = ComponentInfo(name="records-query", version="1", mode="external")

    def __init__(self):
        self.query = None

    def retrieve(self, query, limit):
        self.query = query
        return CandidateBatch(
            candidates=[ScoredMovie(movie_id=movie_id, score=.5) for movie_id in ["101", "102", "103"]],
            component=self.info,
        )


def test_ratings_drive_query_statistics_history_and_all_rated_exclusions():
    retrieval = RecordsQueryAndCandidates()
    request = RecommendationRequest.model_validate(payload(
        history_items=[],
        ratings=[
            {"movie_id": "101", "rating": 5, "rated_at": "2026-11-15T12:00:00Z"},
            {"movie_id": "102", "rating": 2.5, "rated_at": "2026-11-16T12:00:00Z"},
        ],
    ))
    fixed_now = lambda: datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    response = RecommendationService(
        fixture_catalog(), retrieval, PassThroughReranker(), "fixture", clock=fixed_now
    ).recommend(request)
    assert retrieval.query.history_items == ["101"]
    assert retrieval.query.rating_statistics.mean_rating == pytest.approx(3.75)
    assert retrieval.query.rating_statistics.rating_count == 2
    assert retrieval.query.rating_statistics.days_inactive == pytest.approx(1)
    assert [item.movie.movie_id for item in response.recommendations] == ["103"]


def test_temporary_title_context_reaches_models_without_mutating_preferences():
    retrieval = RecordsQueryAndCandidates()
    request = RecommendationRequest.model_validate({
        **payload(history_items=[], ratings=[]),
        "temporary_history_items": ["102"],
    })
    response = RecommendationService(
        fixture_catalog(), retrieval, PassThroughReranker(), "fixture"
    ).recommend(request)
    assert retrieval.query.history_items == ["102"]
    assert response.applied_preferences == request.preferences
    assert response.applied_preferences.history_items == []
    assert [item.movie.movie_id for item in response.recommendations] == ["101", "103"]
