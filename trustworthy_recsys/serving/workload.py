"""Repeatable in-process or live-HTTP application workload measurement."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.config import Settings

LEARNED_WORKLOADS = {
    "learned_favorites": {
        "history_items": ["1", "260", "1196"],
        "excluded_movie_ids": [],
        "include_genres": [],
        "exclude_genres": [],
    },
    "learned_genre_constraints": {
        "history_items": ["1", "260", "1196"],
        "excluded_movie_ids": [],
        "include_genres": ["Adventure"],
        "exclude_genres": ["Horror"],
    },
}


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank percentile for this bounded release sample."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


WORKLOADS = {
    "initial_recommendations": {
        "history_items": ["101", "103"],
        "excluded_movie_ids": [],
        "include_genres": [],
        "exclude_genres": [],
    },
    "structured_refinement": {
        "history_items": ["101", "103"],
        "excluded_movie_ids": ["107"],
        "include_genres": ["Drama"],
        "exclude_genres": ["War"],
    },
}


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p50_ms": round(percentile(values, 0.50), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "p99_ms": round(percentile(values, 0.99), 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def prepare_output(path: Path) -> None:
    """Create the parent directory and reject an existing evidence target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace existing workload report: {path}")


def write_report_exclusive(path: Path, report: dict) -> None:
    """Write once, protecting against a target created after the initial guard."""
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2) + "\n")


def measure(client, workloads: dict, repetitions: int, warmups: int) -> dict:
    records = {}
    for name, preferences in workloads.items():
        client_ms, server_ms, raw = [], [], []
        warmup_errors = measurement_errors = 0
        last = None
        for index in range(warmups + repetitions):
            payload = {
                "session_id": "workload-session",
                "preference_revision": 0 if name in {"initial_recommendations", "learned_favorites"} else 1,
                "preferences": preferences,
                "result_count": 10,
            }
            started = time.perf_counter()
            response = client.post("/api/v1/recommendations", json=payload)
            elapsed = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                if index < warmups:
                    warmup_errors += 1
                else:
                    measurement_errors += 1
                    raw.append({"sample": index - warmups + 1, "status": response.status_code, "client_ms": round(elapsed, 4)})
                continue
            last = response.json()
            if index >= warmups:
                client_ms.append(elapsed)
                server_ms.append(last["timings"]["total_ms"])
                raw.append(
                    {
                        "sample": index - warmups + 1,
                        "status": response.status_code,
                        "request_id": last["request_id"],
                        "client_ms": round(elapsed, 4),
                        "server_total_ms": round(last["timings"]["total_ms"], 4),
                        "retrieval_ms": round(last["timings"]["retrieval_ms"], 4),
                        "filtering_ms": round(last["timings"]["filtering_ms"], 4),
                        "reranking_ms": round(last["timings"]["reranking_ms"], 4),
                    }
                )
        if not client_ms or last is None:
            raise RuntimeError(f"no successful samples for {name}")
        records[name] = {
            "samples": len(client_ms),
            "warmup_errors": warmup_errors,
            "measurement_errors": measurement_errors,
            "client_round_trip": summary(client_ms),
            "server_orchestration": summary(server_ms),
            "displayed_count": len(last["recommendations"]),
            "partial": last["partial"],
            "backend": last["backend_mode"],
            "retrieval": last["retrieval"],
            "reranker": last["reranker"],
            "raw_samples": raw,
        }
    return records


def run(repetitions: int, warmups: int, settings: Settings) -> dict:
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"backend not ready: {ready.json()}")
        records = measure(client, WORKLOADS, repetitions, warmups)
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "process": "in-process FastAPI TestClient; components warm-loaded once",
        },
        "conditions": {"warmups_per_workload": warmups, "repetitions_per_workload": repetitions},
        "scope_warning": "Fixture/baseline timing is not evidence for the future learned system or a deployed network path.",
        "workloads": records,
        "natural_language_refinement": {"status": "pending", "measured": False},
    }


def run_http(base_url: str, repetitions: int, warmups: int, profile: str) -> dict:
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        ready = client.get("/ready")
        status = client.get("/api/v1/application/status")
        if ready.status_code != 200:
            raise RuntimeError(f"backend not ready: {ready.json()}")
        if status.status_code != 200:
            raise RuntimeError(f"application status unavailable: HTTP {status.status_code}")
        workloads = LEARNED_WORKLOADS if profile == "learned" else WORKLOADS
        records = measure(client, workloads, repetitions, warmups)
    if profile == "learned":
        if status.json().get("backend_mode") != "learned":
            raise RuntimeError("learned profile requires a learned backend")
        for name, record in records.items():
            if record["retrieval"].get("name") != "two-tower-weaviate":
                raise RuntimeError(f"{name} did not use two-tower Weaviate retrieval")
            if record["reranker"].get("name") != "lambdarank":
                raise RuntimeError(f"{name} did not use LambdaRank")
    return {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "process": f"HTTP client to warm-running FastAPI at {base_url}",
        },
        "conditions": {
            "profile": profile,
            "warmups_per_workload": warmups,
            "repetitions_per_workload": repetitions,
            "concurrency": 1,
        },
        "timing_boundary": "Client POST start through complete backend HTTP response body; excludes browser rendering and artwork requests.",
        "application_status": status.json(),
        "workloads": records,
    }


async def run_concurrent_http(base_url: str, concurrency: int) -> dict:
    gate = asyncio.Event()

    async def request_one(client: httpx.AsyncClient, index: int) -> dict:
        payload = {
            "session_id": f"concurrent-{index}",
            "preference_revision": 0,
            "preferences": LEARNED_WORKLOADS["learned_favorites"],
            "result_count": 10,
        }
        await gate.wait()
        started = time.perf_counter()
        response = await client.post("/api/v1/recommendations", json=payload)
        elapsed = (time.perf_counter() - started) * 1000
        try:
            body = response.json()
        except ValueError:
            body = {}
        learned_identity = (
            body.get("backend_mode") == "learned"
            and body.get("retrieval", {}).get("name") == "two-tower-weaviate"
            and body.get("reranker", {}).get("name") == "lambdarank"
        )
        return {
            "sample": index,
            "status": response.status_code,
            "client_ms": round(elapsed, 4),
            "request_id": body.get("request_id"),
            "backend": body.get("backend_mode"),
            "retrieval": body.get("retrieval"),
            "reranker": body.get("reranker"),
            "learned_identity": learned_identity,
        }

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        tasks = [asyncio.create_task(request_one(client, index + 1)) for index in range(concurrency)]
        gate.set()
        raw = await asyncio.gather(*tasks)
    values = [sample["client_ms"] for sample in raw]
    return {
        "concurrency": concurrency,
        "errors": sum(sample["status"] != 200 or not sample["learned_identity"] for sample in raw),
        "client_round_trip": summary(values),
        "raw_samples": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/application_workload_fixture.json"))
    parser.add_argument("--base-url", help="Measure an already-running backend over HTTP instead of in-process.")
    parser.add_argument("--profile", choices=("fixture", "learned"), default="fixture")
    parser.add_argument("--concurrency-check", type=int, default=0)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0 or args.concurrency_check < 0:
        parser.error("repetitions must be positive; warmups and concurrency-check must be non-negative")
    try:
        prepare_output(args.output)
    except FileExistsError as error:
        parser.error(str(error))
    report = (
        run_http(args.base_url, args.repetitions, args.warmups, args.profile)
        if args.base_url
        else run(args.repetitions, args.warmups, Settings.from_env())
    )
    if args.concurrency_check:
        if not args.base_url or args.profile != "learned":
            parser.error("concurrency-check requires --base-url and --profile learned")
        report["concurrent_check"] = asyncio.run(run_concurrent_http(args.base_url, args.concurrency_check))
    try:
        write_report_exclusive(args.output, report)
    except FileExistsError:
        parser.error(f"refusing to replace workload report created during the run: {args.output}")
    console = {
        name: {key: value for key, value in record.items() if key != "raw_samples"}
        for name, record in report["workloads"].items()
    }
    print(json.dumps(console, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
