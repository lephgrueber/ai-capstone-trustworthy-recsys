from __future__ import annotations

import pytest
from pydantic import ValidationError

from types import SimpleNamespace

from trustworthy_recsys.serving.components import (
    ExternalRerankingAdapter,
    ExternalRetrievalAdapter,
    LearnedRetrievalAdapter,
)
from trustworthy_recsys.serving.contracts import (
    CandidateBatch,
    ComponentInfo,
    ModelQuery,
    PreferenceState,
    RankingResult,
    ScoredMovie,
)


RETRIEVAL_INFO = ComponentInfo(name="sample-retrieval", version="artifact-1", mode="external")
RANKING_INFO = ComponentInfo(name="sample-reranker", version="artifact-2", mode="external")


def test_new_visitor_query_needs_history_but_not_known_user_id():
    query = ModelQuery(history_items=["101", "102"])
    assert query.known_user_id is None


@pytest.mark.parametrize("field", ["history_items", "excluded_movie_ids"])
def test_preference_movie_ids_must_be_distinct(field):
    with pytest.raises(ValidationError, match="distinct"):
        PreferenceState(**{field: ["101", "101"]})


def test_contradictory_genres_are_rejected():
    with pytest.raises(ValidationError, match="overlap"):
        PreferenceState(include_genres=["Drama"], exclude_genres=["Drama"])


def test_candidate_ids_and_scores_are_validated():
    with pytest.raises(ValidationError, match="distinct"):
        CandidateBatch(
            candidates=[ScoredMovie(movie_id="1"), ScoredMovie(movie_id="1")],
            component=RETRIEVAL_INFO,
        )
    with pytest.raises(ValidationError, match="finite"):
        ScoredMovie(movie_id="1", score=float("nan"))


def test_sample_external_adapters_validate_dict_outputs():
    retrieval = ExternalRetrievalAdapter(
        lambda query, limit: {
            "candidates": [{"movie_id": "2", "score": 0.5}],
            "component": RETRIEVAL_INFO.model_dump(),
        },
        RETRIEVAL_INFO,
    )
    batch = retrieval.retrieve(ModelQuery(history_items=["1"]), 10)
    assert batch.candidates[0].movie_id == "2"


def test_external_reranker_cannot_introduce_items():
    candidates = CandidateBatch(candidates=[ScoredMovie(movie_id="2")], component=RETRIEVAL_INFO)
    adapter = ExternalRerankingAdapter(
        lambda query, batch, limit: RankingResult(
            items=[ScoredMovie(movie_id="999")], component=RANKING_INFO
        ),
        RANKING_INFO,
    )
    with pytest.raises(ValueError, match="introduced"):
        adapter.rerank(ModelQuery(history_items=["1"]), candidates, 10)


def test_short_and_empty_rankings_are_valid_contract_outputs():
    assert RankingResult(items=[], component=RANKING_INFO).items == []


class FakeLearnedRecommender:
    def __init__(self):
        self.inputs = SimpleNamespace(item_ids=["1", "2", "3"])
        self.requests = []
        self.closed = False

    def recommend_indices(self, request, limit):
        self.requests.append(request)
        return ([1, 2][:limit], not request.history_items)

    def close(self):
        self.closed = True


def test_learned_adapter_maps_original_ids_and_reports_fallback():
    recommender = FakeLearnedRecommender()
    info = ComponentInfo(name="two-tower-weaviate", version="test", mode="learned")
    adapter = LearnedRetrievalAdapter(recommender, info, frozenset(recommender.inputs.item_ids))

    learned = adapter.retrieve(ModelQuery(history_items=["1"], known_user_id="known"), 2)
    assert [item.movie_id for item in learned.candidates] == ["2", "3"]
    assert learned.warnings == []
    assert recommender.requests[-1].user_id == "known"

    fallback = adapter.retrieve(ModelQuery(history_items=["unknown"]), 2)
    assert "Unsupported history items" in fallback.warnings[0]
    assert "Popularity fallback" in fallback.warnings[1]
    assert recommender.requests[-1].user_id == "__new_visitor__"
    adapter.close()
    assert recommender.closed
