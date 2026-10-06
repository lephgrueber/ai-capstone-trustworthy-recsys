"""Versioned FastAPI application for movie discovery."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from trustworthy_recsys.serving.catalog import Catalog, catalog_from_parquet, fixture_catalog
from trustworthy_recsys.serving.components import (
    FixtureRetrieval,
    PassThroughReranker,
    build_baseline_retrieval,
    build_learned_retrieval,
)
from trustworthy_recsys.serving.config import Settings
from trustworthy_recsys.data.reporting import hashes
from trustworthy_recsys.serving.contracts import (
    CatalogSearchResponse,
    ApplicationStatus,
    ErrorDetail,
    ErrorResponse,
    RecommendationRequest,
    RecommendationResponse,
    RefinementRequest,
    RefinementResponse,
    PosterRequest,
    PosterResponse,
    StatusResponse,
)
from trustworthy_recsys.serving.refinement import RefinementInterpreter, UnavailableInterpreter
from trustworthy_recsys.serving.posters import PosterService, build_poster_service
from trustworthy_recsys.serving.service import ComponentOutputError, RecommendationService, ServiceInputError

logger = logging.getLogger("trustworthy_recsys.serving")


@dataclass
class Runtime:
    settings: Settings
    service: RecommendationService | None = None
    catalog: Catalog | None = None
    interpreter: RefinementInterpreter = field(default_factory=UnavailableInterpreter)
    readiness_error: str | None = None
    posters: PosterService | None = None


def _load_runtime(
    settings: Settings,
    interpreter: RefinementInterpreter | None = None,
    poster_service: PosterService | None = None,
) -> Runtime:
    runtime = Runtime(settings=settings, interpreter=interpreter or UnavailableInterpreter())
    retrieval = None
    try:
        if settings.backend_mode == "fixture":
            catalog = fixture_catalog()
            retrieval = FixtureRetrieval.build(catalog)
            links_path = None
        elif settings.backend_mode == "baseline":
            settings.validate_baseline_run()
            assert settings.run_dir is not None
            full_catalog = catalog_from_parquet(settings.run_dir / "movies.parquet")
            retrieval = build_baseline_retrieval(settings.run_dir, settings.scenario, full_catalog)
            # The public catalog is the active model vocabulary, not every metadata row.
            catalog = Catalog(
                {movie_id: movie for movie_id, movie in full_catalog.movies.items() if movie_id in retrieval.supported_ids}
            )
            links_path = settings.run_dir / "links.parquet"
        elif settings.backend_mode == "learned":
            settings.validate_learned_artifacts()
            assert settings.run_dir is not None
            assert settings.retrieval_inputs_dir is not None
            assert settings.retrieval_model_dir is not None
            retrieval = build_learned_retrieval(
                settings.retrieval_inputs_dir, settings.retrieval_model_dir
            )
            try:
                if retrieval.recommender.inputs.meta["scenario"] != settings.scenario:
                    raise ValueError("learned model scenario does not match configured scenario")
                expected_movies_hash = retrieval.recommender.inputs.meta["source_hashes"]["movies.parquet"]
                if hashes(settings.run_dir / "movies.parquet")["sha256"] != expected_movies_hash:
                    raise ValueError("catalog movies.parquet does not match learned retrieval inputs")
                full_catalog = catalog_from_parquet(settings.run_dir / "movies.parquet")
                missing = retrieval.supported_ids - full_catalog.supported_ids
                if missing:
                    raise ValueError(
                        f"learned catalog metadata is missing {len(missing)} model movie IDs"
                    )
                catalog = Catalog(
                    {
                        movie_id: full_catalog.movies[movie_id]
                        for movie_id in retrieval.recommender.inputs.item_ids
                    }
                )
                links_path = settings.run_dir / "links.parquet"
            except (KeyError, TypeError) as error:
                raise ValueError(f"learned artifact compatibility metadata is invalid: {error}") from error
        else:
            raise ValueError("TRUSTWORTHY_RECSYS_BACKEND must be fixture, baseline, or learned")
        runtime.catalog = catalog
        runtime.posters = poster_service or build_poster_service(
            links_path,
            settings.tmdb_read_access_token,
            settings.tmdb_timeout_seconds,
            settings.poster_cache_size,
        )
        runtime.service = RecommendationService(
            catalog=catalog,
            retrieval=retrieval,
            reranker=PassThroughReranker(),
            backend_mode=settings.backend_mode,
            candidate_budget=settings.candidate_budget,
        )
    except (ValueError, OSError) as error:
        if retrieval is not None:
            close = getattr(retrieval, "close", None)
            if close is not None:
                close()
        runtime.readiness_error = str(error)
        logger.error("backend_not_ready mode=%s reason=%s", settings.backend_mode, error)
    return runtime


def create_app(
    settings: Settings | None = None,
    *,
    interpreter: RefinementInterpreter | None = None,
    poster_service: PosterService | None = None,
    load_components: bool = True,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_components:
            app.state.runtime = _load_runtime(configured, interpreter, poster_service)
        try:
            yield
        finally:
            runtime: Runtime = app.state.runtime
            if runtime.service is not None:
                close = getattr(runtime.service.retrieval, "close", None)
                if close is not None:
                    close()
            if runtime.posters is not None:
                await runtime.posters.close()

    app = FastAPI(
        title="Trustworthy Recommendation System API",
        version="1.0.0",
        description="M3 movie-discovery application and model integration contracts.",
        lifespan=lifespan,
    )
    app.state.runtime = Runtime(settings=configured, interpreter=interpreter or UnavailableInterpreter())

    def request_id(request: Request) -> str:
        return getattr(request.state, "request_id", str(uuid.uuid4()))

    def error_response(request: Request, status: int, code: str, message: str, issues=None):
        body = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                request_id=request_id(request),
                issues=issues or [],
            )
        )
        return JSONResponse(status_code=status, content=body.model_dump(mode="json"))

    @app.middleware("http")
    async def diagnostics(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "request_complete id=%s method=%s path=%s status=%s duration_ms=%.3f backend=%s",
            request.state.request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            configured.backend_mode,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        issues = [
            {"location": [str(part) for part in error["loc"]], "message": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return error_response(request, 422, "validation_error", "Request validation failed.", issues)

    @app.exception_handler(ServiceInputError)
    async def input_error(request: Request, exc: ServiceInputError):
        return error_response(request, 422, exc.code, str(exc))

    @app.exception_handler(ComponentOutputError)
    async def component_error(request: Request, exc: ComponentOutputError):
        logger.error("component_output_error id=%s detail=%s", request_id(request), exc)
        return error_response(request, 502, "invalid_component_output", "A recommendation component returned invalid output.")

    @app.get("/health", response_model=StatusResponse)
    def health():
        return StatusResponse(status="ok", backend_mode=configured.backend_mode)

    @app.get("/ready", response_model=StatusResponse, responses={503: {"model": StatusResponse}})
    def ready(request: Request):
        runtime: Runtime = request.app.state.runtime
        if runtime.service is None:
            body = StatusResponse(
                status="not_ready",
                backend_mode=configured.backend_mode,
                detail=runtime.readiness_error or "components have not been loaded",
            )
            return JSONResponse(status_code=503, content=body.model_dump())
        return StatusResponse(status="ok", backend_mode=configured.backend_mode)

    @app.get("/api/v1/catalog/search", response_model=CatalogSearchResponse)
    def search_catalog(request: Request, q: str = Query(default="", max_length=100), limit: int = Query(default=20, ge=1, le=50)):
        runtime: Runtime = request.app.state.runtime
        if runtime.catalog is None:
            return error_response(request, 503, "backend_not_ready", runtime.readiness_error or "Backend is not ready.")
        items = runtime.catalog.search(q, limit)
        return CatalogSearchResponse(items=items, total=len(items), backend_mode=configured.backend_mode)

    @app.get("/api/v1/application/status", response_model=ApplicationStatus)
    def application_status(request: Request):
        runtime: Runtime = request.app.state.runtime
        if runtime.service is None:
            model_label = "Model unavailable"
            data_source = "Unavailable"
        elif configured.backend_mode == "baseline":
            model_label = "Baseline model"
            data_source = "MovieLens 32M · ratio_80_10_10"
        elif configured.backend_mode == "fixture":
            model_label = "Synthetic demo"
            data_source = "Synthetic fixture catalog"
        elif configured.backend_mode == "learned":
            model_label = "Learned model"
            data_source = "Configured learned artifacts"
        else:
            model_label = "Model unavailable"
            data_source = "Unavailable"
        return ApplicationStatus(
            backend_mode=configured.backend_mode,
            model_label=model_label,
            data_source_label=data_source,
            refinement_available=runtime.interpreter.name != "unavailable",
            posters_configured=bool(runtime.posters and runtime.posters.configured),
            available_genres=sorted(runtime.catalog.genres) if runtime.catalog else [],
        )

    @app.post("/api/v1/catalog/posters", response_model=PosterResponse)
    async def catalog_posters(payload: PosterRequest, request: Request):
        runtime: Runtime = request.app.state.runtime
        if runtime.catalog is None or runtime.posters is None:
            return error_response(request, 503, "backend_not_ready", runtime.readiness_error or "Backend is not ready.")
        unknown = sorted(set(payload.movie_ids) - runtime.catalog.supported_ids)
        if unknown:
            raise ServiceInputError("unknown_movie_ids", f"Unknown or unsupported movie IDs: {unknown}")
        items = await runtime.posters.resolve(payload.movie_ids)
        return PosterResponse(items=items, configured=runtime.posters.configured)

    @app.post(
        "/api/v1/recommendations",
        response_model=RecommendationResponse,
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def recommendations(payload: RecommendationRequest, request: Request):
        runtime: Runtime = request.app.state.runtime
        if runtime.service is None:
            return error_response(request, 503, "backend_not_ready", runtime.readiness_error or "Backend is not ready.")
        result = runtime.service.recommend(payload, request_id(request))
        logger.info(
            "recommendation_complete id=%s backend=%s candidates_displayed=%s partial=%s retrieval=%s reranker=%s total_ms=%.3f",
            result.request_id,
            result.backend_mode,
            len(result.recommendations),
            result.partial,
            result.retrieval.version,
            result.reranker.version,
            result.timings.total_ms,
        )
        return result

    @app.post("/api/v1/refinements", response_model=RefinementResponse)
    def refinements(payload: RefinementRequest, request: Request):
        runtime: Runtime = request.app.state.runtime
        # Prompts are intentionally not logged.
        return runtime.interpreter.interpret(payload)

    return app


app = create_app()


def export_openapi(path) -> None:
    """Export schema without entering lifespan or loading any model/data artifacts."""
    schema = create_app(load_components=False).openapi()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
