"""Optional TMDb poster display metadata, isolated from recommendation logic."""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from trustworthy_recsys.serving.contracts import PosterMetadata

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"
TMDB_BACKDROP_SMALL_BASE = "https://image.tmdb.org/t/p/w780"
POSTER_PATH = re.compile(r"^/[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
ARTWORK_CACHE_VERSION = "artwork-v2"


def load_tmdb_mappings(path: Path) -> dict[str, str]:
    """Load exact MovieLens-to-TMDb identities from the pipeline artifact."""
    frame = pd.read_parquet(path, columns=["movie_id", "tmdb_id"])
    return {
        str(row.movie_id): str(row.tmdb_id)
        for row in frame.itertuples(index=False)
        if row.tmdb_id is not None and str(row.tmdb_id).strip()
    }


@dataclass(frozen=True)
class FetchResult:
    status: str
    poster_url: str | None
    ttl_seconds: float
    backdrop_url: str | None = None
    backdrop_url_small: str | None = None


class TmdbPosterProvider:
    """Small async TMDb details client using application Bearer authentication."""

    def __init__(self, token: str, timeout_seconds: float = 2.0, client: httpx.AsyncClient | None = None):
        self._token = token
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
        )

    async def fetch(self, tmdb_id: str) -> FetchResult:
        for attempt in range(2):
            try:
                response = await self.client.get(
                    f"{TMDB_API_BASE}/movie/{tmdb_id}",
                    headers={"Authorization": f"Bearer {self._token}", "Accept": "application/json"},
                    params={
                        "append_to_response": "images",
                        # TMDb image filtering excludes the overwhelmingly common
                        # language-neutral backdrops unless `null` is explicit.
                        "include_image_language": "en,null",
                    },
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 0:
                    await asyncio.sleep(0)
                    continue
                return FetchResult("unavailable", None, 30)
            if response.status_code == 200:
                try:
                    payload = response.json()
                    poster_path = payload.get("poster_path")
                    backdrop_path = payload.get("backdrop_path")
                    if backdrop_path is None:
                        images = payload.get("images")
                        candidates = images.get("backdrops", []) if isinstance(images, dict) else []
                        if isinstance(candidates, list):
                            usable = [
                                image for image in candidates
                                if isinstance(image, dict)
                                and isinstance(image.get("file_path"), str)
                                and POSTER_PATH.fullmatch(image["file_path"])
                                and image.get("iso_639_1") in {None, "en"}
                            ]
                            # Stable quality choice: neutral artwork first, then
                            # provider votes, dimensions, and finally path.
                            usable.sort(key=lambda image: (
                                image.get("iso_639_1") is None,
                                float(image.get("vote_average") or 0),
                                int(image.get("vote_count") or 0),
                                int(image.get("width") or 0),
                                image["file_path"],
                            ), reverse=True)
                            backdrop_path = usable[0]["file_path"] if usable else None
                except (ValueError, AttributeError):
                    return FetchResult("unavailable", None, 30)
                valid_poster = isinstance(poster_path, str) and POSTER_PATH.fullmatch(poster_path)
                valid_backdrop = isinstance(backdrop_path, str) and POSTER_PATH.fullmatch(backdrop_path)
                if poster_path is not None and not valid_poster:
                    return FetchResult("unavailable", None, 30)
                if backdrop_path is not None and not valid_backdrop:
                    return FetchResult("unavailable", None, 30)
                if not valid_poster and not valid_backdrop:
                    return FetchResult("missing", None, 3600)
                return FetchResult(
                    "available",
                    f"{TMDB_POSTER_BASE}{poster_path}" if valid_poster else None,
                    21600,
                    f"{TMDB_BACKDROP_BASE}{backdrop_path}" if valid_backdrop else None,
                    f"{TMDB_BACKDROP_SMALL_BASE}{backdrop_path}" if valid_backdrop else None,
                )
            if response.status_code in {401, 403, 404}:
                return FetchResult("missing" if response.status_code == 404 else "unavailable", None, 3600)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 0:
                    retry_after = response.headers.get("Retry-After", "0")
                    try:
                        delay = min(max(float(retry_after), 0), 0.25)
                    except ValueError:
                        delay = 0
                    await asyncio.sleep(delay)
                    continue
            return FetchResult("unavailable", None, 30)
        return FetchResult("unavailable", None, 30)

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()


class PosterService:
    def __init__(
        self,
        mappings: dict[str, str],
        provider: TmdbPosterProvider | None,
        *,
        cache_size: int = 512,
        concurrency: int = 4,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.mappings = mappings
        self.provider = provider
        self.cache_size = cache_size
        self.clock = clock
        self._cache: OrderedDict[str, tuple[float, FetchResult]] = OrderedDict()
        self._inflight: dict[str, asyncio.Task[FetchResult]] = {}
        self._semaphore = asyncio.Semaphore(concurrency)

    @property
    def configured(self) -> bool:
        return self.provider is not None and bool(self.mappings)

    async def _fetch(self, tmdb_id: str) -> FetchResult:
        # Versioned keys prevent poster-only entries created by the former
        # resolver from suppressing backdrop discovery after a rolling reload.
        cache_key = f"{ARTWORK_CACHE_VERSION}:{tmdb_id}"
        cached = self._cache.get(cache_key)
        now = self.clock()
        if cached and cached[0] > now:
            self._cache.move_to_end(cache_key)
            return cached[1]
        if cached:
            self._cache.pop(cache_key, None)
        task = self._inflight.get(cache_key)
        if task is None:
            async def perform() -> FetchResult:
                assert self.provider is not None
                async with self._semaphore:
                    return await self.provider.fetch(tmdb_id)
            task = asyncio.create_task(perform())
            self._inflight[cache_key] = task
        try:
            result = await task
        finally:
            if self._inflight.get(cache_key) is task:
                self._inflight.pop(cache_key, None)
        self._cache[cache_key] = (now + result.ttl_seconds, result)
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    async def resolve(self, movie_ids: Sequence[str]) -> list[PosterMetadata]:
        async def one(movie_id: str) -> PosterMetadata:
            tmdb_id = self.mappings.get(movie_id)
            if tmdb_id is None:
                return PosterMetadata(movie_id=movie_id, status="missing")
            if self.provider is None:
                return PosterMetadata(movie_id=movie_id, tmdb_id=tmdb_id, status="unavailable")
            result = await self._fetch(tmdb_id)
            return PosterMetadata(
                movie_id=movie_id,
                tmdb_id=tmdb_id,
                poster_url=result.poster_url,
                backdrop_url=result.backdrop_url,
                backdrop_url_small=result.backdrop_url_small,
                status=result.status,
            )
        return list(await asyncio.gather(*(one(movie_id) for movie_id in movie_ids)))

    async def close(self) -> None:
        if self.provider is not None:
            await self.provider.close()


def build_poster_service(links_path: Path | None, token: str | None, timeout: float, cache_size: int) -> PosterService:
    mappings = load_tmdb_mappings(links_path) if links_path and links_path.is_file() else {}
    provider = TmdbPosterProvider(token, timeout) if token and mappings else None
    return PosterService(mappings, provider, cache_size=cache_size)
