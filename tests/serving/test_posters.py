from __future__ import annotations

import asyncio

import httpx
import pandas as pd
from fastapi.testclient import TestClient

from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.posters import (
    FetchResult,
    PosterService,
    TmdbPosterProvider,
    load_tmdb_mappings,
)


def run(awaitable):
    return asyncio.run(awaitable)


def test_exact_movielens_tmdb_mapping_and_missing_links(tmp_path):
    path = tmp_path / "links.parquet"
    pd.DataFrame(
        {"movie_id": ["1", "2", "3"], "tmdb_id": ["862", None, "15602"], "imdb_id": ["a", "b", "c"]}
    ).to_parquet(path)
    assert load_tmdb_mappings(path) == {"1": "862", "3": "15602"}


def provider(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TmdbPosterProvider("test-secret", client=client), client


def test_provider_builds_only_valid_https_tmdb_image_urls():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.url.params["append_to_response"] == "images"
        assert request.url.params["include_image_language"] == "en,null"
        return httpx.Response(200, json={"poster_path": "/abc_123.jpg", "backdrop_path": "/wide-123.webp"})

    tmdb, client = provider(handler)
    result = run(tmdb.fetch("862"))
    run(client.aclose())
    assert result.poster_url == "https://image.tmdb.org/t/p/w500/abc_123.jpg"
    assert result.backdrop_url == "https://image.tmdb.org/t/p/w1280/wide-123.webp"
    assert result.backdrop_url_small == "https://image.tmdb.org/t/p/w780/wide-123.webp"
    assert "test-secret" not in result.poster_url


def test_no_poster_and_invalid_path_are_isolated():
    for value, expected in [(None, "missing"), ("https://evil.example/poster.jpg", "unavailable")]:
        tmdb, client = provider(lambda request, value=value: httpx.Response(200, json={"poster_path": value}))
        assert run(tmdb.fetch("1")).status == expected
        run(client.aclose())

    tmdb, client = provider(lambda request: httpx.Response(200, json={"poster_path": None, "backdrop_path": "/wide.jpg"}))
    result = run(tmdb.fetch("1"))
    assert result.status == "available" and result.poster_url is None
    assert result.backdrop_url == "https://image.tmdb.org/t/p/w1280/wide.jpg"
    run(client.aclose())


def test_provider_uses_deterministic_language_neutral_image_fallback():
    payload = {
        "poster_path": "/poster.jpg",
        "backdrop_path": None,
        "images": {
            "backdrops": [
                {"file_path": "/english.jpg", "iso_639_1": "en", "vote_average": 9, "vote_count": 50, "width": 1920},
                {"file_path": "/neutral-small.jpg", "iso_639_1": None, "vote_average": 8, "vote_count": 5, "width": 1280},
                {"file_path": "/neutral-best.jpg", "iso_639_1": None, "vote_average": 8, "vote_count": 5, "width": 1920},
                {"file_path": "/excluded.jpg", "iso_639_1": "fr", "vote_average": 10, "vote_count": 100, "width": 3840},
            ]
        },
    }
    tmdb, client = provider(lambda request: httpx.Response(200, json=payload))
    result = run(tmdb.fetch("1"))
    assert result.poster_url == "https://image.tmdb.org/t/p/w500/poster.jpg"
    assert result.backdrop_url == "https://image.tmdb.org/t/p/w1280/neutral-best.jpg"
    assert result.backdrop_url_small == "https://image.tmdb.org/t/p/w780/neutral-best.jpg"
    run(client.aclose())


def test_timeout_authentication_and_rate_limit_are_bounded():
    calls = 0

    def timeout_handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("slow", request=request)

    tmdb, client = provider(timeout_handler)
    assert run(tmdb.fetch("1")).status == "unavailable"
    assert calls == 2
    run(client.aclose())

    calls = 0

    def auth_handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    tmdb, client = provider(auth_handler)
    assert run(tmdb.fetch("1")).status == "unavailable"
    assert calls == 1
    run(client.aclose())

    responses = iter([httpx.Response(429, headers={"Retry-After": "0"}), httpx.Response(200, json={"poster_path": "/ok.png"})])
    tmdb, client = provider(lambda request: next(responses))
    assert run(tmdb.fetch("1")).status == "available"
    run(client.aclose())


class CountingProvider:
    def __init__(self, result=FetchResult("available", "https://image.tmdb.org/t/p/w500/x.jpg", 100)):
        self.calls = 0
        self.result = result

    async def fetch(self, tmdb_id):
        self.calls += 1
        await asyncio.sleep(0)
        return self.result

    async def close(self):
        return None


def test_cache_and_inflight_deduplicate_shared_tmdb_ids():
    upstream = CountingProvider()
    service = PosterService({"101": "500", "102": "500"}, upstream, cache_size=2)
    first = run(service.resolve(["101", "102"]))
    second = run(service.resolve(["101"]))
    assert upstream.calls == 1
    assert [item.status for item in first] == ["available", "available"]
    assert second[0].poster_url == first[0].poster_url


def test_legacy_poster_only_cache_entry_does_not_block_backdrop_refresh():
    upstream = CountingProvider(FetchResult("available", "https://image.tmdb.org/t/p/w500/new.jpg", 100, "https://image.tmdb.org/t/p/w1280/wide.jpg"))
    service = PosterService({"101": "500"}, upstream)
    service._cache["500"] = (10_000, FetchResult("available", "https://image.tmdb.org/t/p/w500/old.jpg", 100))
    item = run(service.resolve(["101"]))[0]
    assert upstream.calls == 1
    assert item.backdrop_url == "https://image.tmdb.org/t/p/w1280/wide.jpg"


def test_missing_token_and_mapping_degrade_per_movie_without_network():
    service = PosterService({"101": "500"}, None)
    items = run(service.resolve(["101", "102"]))
    assert [(item.movie_id, item.status) for item in items] == [("101", "unavailable"), ("102", "missing")]
    assert service.configured is False


def test_poster_endpoint_never_exposes_token_or_changes_recommendations():
    upstream = CountingProvider()
    posters = PosterService({"101": "500", "102": "501"}, upstream)
    request = {
        "session_id": "s",
        "preference_revision": 0,
        "preferences": {"history_items": ["101"], "excluded_movie_ids": [], "include_genres": ["Drama"], "exclude_genres": ["Mystery"]},
        "result_count": 5,
    }
    with TestClient(create_app(poster_service=posters)) as client:
        before = client.post("/api/v1/recommendations", json=request).json()
        metadata = client.post("/api/v1/catalog/posters", json={"movie_ids": ["102"]})
        after = client.post("/api/v1/recommendations", json=request).json()
        status = client.get("/api/v1/application/status").json()
    assert metadata.status_code == 200
    assert "test-secret" not in metadata.text
    assert [item["movie"]["movie_id"] for item in before["recommendations"]] == [
        item["movie"]["movie_id"] for item in after["recommendations"]
    ]
    assert status["model_label"] == "Synthetic demo"
    assert status["posters_configured"] is True

    expected = [item["movie"]["movie_id"] for item in before["recommendations"]]
    for isolated_posters in (
        PosterService({"101": "500", "102": "501"}, None),
        PosterService(
            {"101": "500", "102": "501"},
            CountingProvider(FetchResult("unavailable", None, 30)),
        ),
    ):
        with TestClient(create_app(poster_service=isolated_posters)) as client:
            client.post("/api/v1/catalog/posters", json={"movie_ids": ["102"]})
            comparison = client.post("/api/v1/recommendations", json=request).json()
        assert [item["movie"]["movie_id"] for item in comparison["recommendations"]] == expected
