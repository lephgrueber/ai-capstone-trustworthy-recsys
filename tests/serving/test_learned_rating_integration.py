"""Opt-in check that browser ratings reach the saved learned serving stack."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from trustworthy_recsys.serving.app import _load_runtime
from trustworthy_recsys.serving.config import Settings
from trustworthy_recsys.serving.contracts import RecommendationRequest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("LEARNED_RATING_INTEGRATION") != "1",
    reason="Set LEARNED_RATING_INTEGRATION=1 with saved artifacts and Weaviate running",
)
def test_saved_two_tower_and_lambdarank_receive_live_rating_features(monkeypatch):
    settings = Settings.from_env()
    assert settings.backend_mode == "learned"
    runtime = _load_runtime(settings)
    assert runtime.readiness_error is None, runtime.readiness_error
    assert runtime.service is not None
    assert runtime.service.retrieval.info.name == "two-tower-weaviate"
    assert runtime.service.reranker.info.name == "lambdarank"

    as_of = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    object.__setattr__(runtime.service, "clock", lambda: as_of)
    ids = list(runtime.catalog.movies)[:3]
    original_build = runtime.service.reranker.features.build
    captured = {}

    def capture_build(history_items, candidates, retrieval_scores=None, user_id=None, rating_statistics=None):
        matrix = original_build(history_items, candidates, retrieval_scores, user_id, rating_statistics)
        captured.update(history=list(history_items), statistics=rating_statistics, matrix=matrix.copy())
        return matrix

    monkeypatch.setattr(runtime.service.reranker.features, "build", capture_build)
    request = RecommendationRequest.model_validate({
        "session_id": "saved-model-rating-check",
        "preference_revision": 1,
        "known_user_id": None,
        "preferences": {
            "history_items": [], "excluded_movie_ids": [],
            "include_genres": [], "exclude_genres": [],
            "ratings": [
                {"movie_id": ids[0], "rating": 5.0, "rated_at": (as_of - timedelta(days=1)).isoformat()},
                {"movie_id": ids[1], "rating": 3.0, "rated_at": (as_of - timedelta(days=2)).isoformat()},
                {"movie_id": ids[2], "rating": 2.5, "rated_at": (as_of - timedelta(days=3)).isoformat()},
            ],
        },
        "result_count": 12,
    })
    try:
        response = runtime.service.recommend(request, "saved-model-rating-check")
    finally:
        runtime.service.retrieval.close()

    assert captured["history"] == [ids[0]]
    stats = captured["statistics"]
    assert stats.mean_rating == pytest.approx(3.5)
    assert stats.rating_count == 3
    assert stats.days_inactive == pytest.approx(1.0)
    matrix = captured["matrix"]
    assert matrix.shape[0] >= 12
    assert np.allclose(matrix[:, 10], 3.5)
    assert np.allclose(matrix[:, 11], math.log1p(3))
    assert np.allclose(matrix[:, 12], 1.0)
    assert not set(ids) & {item.movie.movie_id for item in response.recommendations}
    assert response.retrieval.name == "two-tower-weaviate"
    assert response.reranker.name == "lambdarank"
    print(json.dumps({
        "retrieval": response.retrieval.model_dump(),
        "reranker": response.reranker.model_dump(),
        "rated_movie_ids": ids,
        "positive_history": captured["history"],
        "matrix_user_features": {
            "user_mean_rating": float(matrix[0, 10]),
            "user_log_rating_count": float(matrix[0, 11]),
            "user_days_inactive": float(matrix[0, 12]),
        },
        "displayed_count": len(response.recommendations),
        "rated_overlap": [],
    }, sort_keys=True))
