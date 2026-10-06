from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

import trustworthy_recsys.serving.app as app_module
from trustworthy_recsys.data.reporting import hashes
from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.catalog import Catalog
from trustworthy_recsys.serving.components import LearnedRetrievalAdapter
from trustworthy_recsys.serving.config import Settings
from trustworthy_recsys.serving.contracts import AppliedRefinement, CatalogMovie, ComponentInfo


def payload(**preferences):
    state = {
        "history_items": ["101"],
        "excluded_movie_ids": [],
        "include_genres": [],
        "exclude_genres": [],
    }
    state.update(preferences)
    return {"session_id": "browser-session", "preference_revision": 3, "preferences": state, "result_count": 6}


def test_health_readiness_and_search():
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").status_code == 200
        response = client.get("/api/v1/catalog/search", params={"q": "signal", "limit": 5})
        assert response.status_code == 200
        assert response.json()["items"][0]["movie_id"] == "101"


def test_catalog_search_keeps_punctuation_prefixed_titles():
    catalog = Catalog({"1": CatalogMovie(movie_id="1", title="'Til There Was You (1997)", genres=["Drama"])})
    assert [movie.movie_id for movie in catalog.search("'Til", 5)] == ["1"]


def test_new_visitor_recommendations_exclude_history_and_preserve_null_scores():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/recommendations", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["preference_revision"] == 3
    assert body["applied_preferences"]["history_items"] == ["101"]
    assert "101" not in [item["movie"]["movie_id"] for item in body["recommendations"]]
    assert all(item["retrieval_score"] is None for item in body["recommendations"])
    assert body["backend_mode"] == "fixture"
    assert body["reranker"]["mode"] == "pass_through"


def test_empty_history_is_defined_and_returns_fixture_results():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/recommendations", json=payload(history_items=[]))
    assert response.status_code == 200
    assert response.json()["recommendations"]


def test_hard_genre_filters_and_exclusions_are_never_relaxed():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/recommendations",
            json=payload(history_items=[], excluded_movie_ids=["103"], include_genres=["Comedy"], exclude_genres=["Romance"]),
        )
    assert response.status_code == 200
    movies = [item["movie"] for item in response.json()["recommendations"]]
    assert all("Comedy" in movie["genres"] and "Romance" not in movie["genres"] for movie in movies)
    assert all(movie["movie_id"] != "103" for movie in movies)
    assert response.json()["partial"] is True


def test_unknown_duplicate_and_contradictory_inputs_are_distinct_422_errors():
    with TestClient(create_app()) as client:
        unknown = client.post("/api/v1/recommendations", json=payload(history_items=["999"]))
        duplicate = client.post("/api/v1/recommendations", json=payload(history_items=["101", "101"]))
        contradiction = client.post(
            "/api/v1/recommendations", json=payload(include_genres=["Drama"], exclude_genres=["Drama"])
        )
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_movie_ids"
    assert duplicate.status_code == 422 and duplicate.json()["error"]["code"] == "validation_error"
    assert contradiction.status_code == 422 and contradiction.json()["error"]["code"] == "validation_error"


def test_unconfigured_refinement_is_truthfully_unavailable():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/refinements",
            json={
                "session_id": "s",
                "preference_revision": 0,
                "preferences": payload()["preferences"],
                "text": "more comedy",
            },
        )
    assert response.status_code == 200
    assert response.json()["outcome"] == "unavailable"


class FakeInterpreter:
    name = "test-fake"
    version = "1"

    def interpret(self, request):
        updated = request.preferences.model_copy(update={"include_genres": ["Comedy"]})
        return AppliedRefinement(
            preference_revision=request.preference_revision + 1,
            preferences=updated,
            summary="Included Comedy.",
        )


def test_refinement_interpreter_boundary_accepts_deterministic_fake():
    with TestClient(create_app(interpreter=FakeInterpreter())) as client:
        response = client.post(
            "/api/v1/refinements",
            json={"session_id": "s", "preference_revision": 0, "preferences": payload()["preferences"], "text": "comedy"},
        )
    assert response.json() == {
        "outcome": "applied",
        "preference_revision": 1,
        "preferences": {**payload()["preferences"], "include_genres": ["Comedy"]},
        "summary": "Included Comedy.",
    }


def test_unavailable_modes_keep_liveness_but_fail_readiness_without_fallback(tmp_path):
    settings = Settings(backend_mode="learned")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        ready = client.get("/ready")
        status = client.get("/api/v1/application/status")
        recommendations = client.post("/api/v1/recommendations", json=payload())
    assert ready.status_code == 503
    assert "artifacts" in ready.json()["detail"]
    assert recommendations.status_code == 503
    assert recommendations.json()["error"]["code"] == "backend_not_ready"
    assert status.json()["model_label"] == "Model unavailable"


def test_baseline_mode_requires_completed_ratio_run(tmp_path):
    settings = Settings(backend_mode="baseline", run_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert "completed pipeline run" in response.json()["detail"]


def test_baseline_mode_loads_only_completed_manifest_declared_training_artifacts(tmp_path):
    scenario = tmp_path / "ratio_80_10_10"
    scenario.mkdir()
    movies = pd.DataFrame(
        {
            "movie_id": ["1", "2", "3", "4"],
            "title": ["One", "Two", "Three", "Four"],
            "genres": [["Drama"], ["Comedy"], ["Drama"], ["Comedy"]],
        }
    )
    train = pd.DataFrame(
        {
            "user_id": ["a", "a", "b", "c", "d", "e"],
            "movie_id": ["1", "2", "2", "3", "3", "4"],
            "rating": [5.0] * 6,
            "timestamp_utc": pd.to_datetime(["2025-01-01"] * 6, utc=True),
        }
    )
    movies.to_parquet(tmp_path / "movies.parquet", index=False)
    train.to_parquet(scenario / "train.parquet", index=False)
    (scenario / "train_item_ids.json").write_text('["1","2","3","4"]\n', encoding="utf-8")
    outputs = {
        "movies.parquet": {},
        "ratio_80_10_10/train.parquet": {},
        "ratio_80_10_10/train_item_ids.json": {},
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "config": {"requested_ratios": {"ratio_80_10_10": [0.8, 0.1, 0.1]}},
                "validation": {"schemas": True, "temporal_boundaries": True},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(backend_mode="baseline", run_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/api/v1/recommendations",
            json={**payload(history_items=["1"]), "result_count": 2},
        )
    assert response.status_code == 200
    assert response.json()["retrieval"]["mode"] == "baseline"
    assert response.json()["reranker"]["mode"] == "pass_through"


class FakeLearnedRecommender:
    def __init__(self, movies_hash):
        self.inputs = SimpleNamespace(
            item_ids=["1", "2", "3"],
            meta={
                "scenario": "ratio_80_10_10",
                "source_hashes": {"movies.parquet": movies_hash},
            },
        )
        self.closed = False

    def recommend_indices(self, request, limit):
        ranked = [index for index in [1, 2, 0] if self.inputs.item_ids[index] not in request.history_items]
        return ranked[:limit], not request.history_items

    def close(self):
        self.closed = True


def test_learned_mode_loads_verified_adapter_and_serves_new_visitor(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    inputs_dir = tmp_path / "inputs"
    model_dir = tmp_path / "model"
    run_dir.mkdir()
    inputs_dir.mkdir()
    model_dir.mkdir()
    movies = pd.DataFrame(
        {
            "movie_id": ["1", "2", "3"],
            "title": ["One", "Two", "Three"],
            "genres": [["Drama"], ["Comedy"], ["Action"]],
        }
    )
    movies.to_parquet(run_dir / "movies.parquet", index=False)
    for root in (run_dir, inputs_dir, model_dir):
        (root / "manifest.json").write_text('{"outputs": {}}\n', encoding="utf-8")
    (model_dir / "weaviate.json").write_text('{}\n', encoding="utf-8")
    recommender = FakeLearnedRecommender(hashes(run_dir / "movies.parquet")["sha256"])
    adapter = LearnedRetrievalAdapter(
        recommender,
        ComponentInfo(name="two-tower-weaviate", version="test", mode="learned"),
        frozenset(recommender.inputs.item_ids),
    )
    monkeypatch.setattr(app_module, "build_learned_retrieval", lambda *_: adapter)
    settings = Settings(
        backend_mode="learned",
        run_dir=run_dir,
        retrieval_inputs_dir=inputs_dir,
        retrieval_model_dir=model_dir,
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/ready").status_code == 200
        status = client.get("/api/v1/application/status").json()
        response = client.post(
            "/api/v1/recommendations",
            json={**payload(history_items=["1"]), "result_count": 2},
        )
    assert response.status_code == 200
    assert response.json()["backend_mode"] == "learned"
    assert response.json()["retrieval"]["mode"] == "learned"
    assert [item["movie"]["movie_id"] for item in response.json()["recommendations"]] == ["2", "3"]
    assert recommender.closed


def test_learned_mode_rejects_catalog_hash_mismatch_and_closes_adapter(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    inputs_dir = tmp_path / "inputs"
    model_dir = tmp_path / "model"
    for root in (run_dir, inputs_dir, model_dir):
        root.mkdir()
        (root / "manifest.json").write_text('{"outputs": {}}\n', encoding="utf-8")
    pd.DataFrame(
        {"movie_id": ["1", "2", "3"], "title": ["One", "Two", "Three"], "genres": [[], [], []]}
    ).to_parquet(run_dir / "movies.parquet", index=False)
    (model_dir / "weaviate.json").write_text('{}\n', encoding="utf-8")
    recommender = FakeLearnedRecommender("not-the-catalog-hash")
    adapter = LearnedRetrievalAdapter(
        recommender,
        ComponentInfo(name="two-tower-weaviate", version="test", mode="learned"),
        frozenset(recommender.inputs.item_ids),
    )
    monkeypatch.setattr(app_module, "build_learned_retrieval", lambda *_: adapter)
    settings = Settings(
        backend_mode="learned",
        run_dir=run_dir,
        retrieval_inputs_dir=inputs_dir,
        retrieval_model_dir=model_dir,
    )
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        recommendations = client.post("/api/v1/recommendations", json=payload())
    assert ready.status_code == 503
    assert "does not match" in ready.json()["detail"]
    assert recommendations.status_code == 503
    assert recommender.closed


def test_search_bounds_are_strict():
    with TestClient(create_app()) as client:
        response = client.get("/api/v1/catalog/search", params={"limit": 500})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
