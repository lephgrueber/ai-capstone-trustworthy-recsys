"""Repeatable local fixture/baseline application workload measurement."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from trustworthy_recsys.serving.app import create_app
from trustworthy_recsys.serving.config import Settings


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
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "mean_ms": round(statistics.mean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "min_ms": round(ordered[0], 4),
        "max_ms": round(ordered[-1], 4),
    }


def run(repetitions: int, warmups: int, settings: Settings) -> dict:
    records = {}
    with TestClient(create_app(settings)) as client:
        ready = client.get("/ready")
        if ready.status_code != 200:
            raise RuntimeError(f"backend not ready: {ready.json()}")
        for name, preferences in WORKLOADS.items():
            client_ms, server_ms = [], []
            errors = 0
            last = None
            for index in range(warmups + repetitions):
                payload = {
                    "session_id": "workload-session",
                    "preference_revision": 0 if name == "initial_recommendations" else 1,
                    "preferences": preferences,
                    "result_count": 10,
                }
                started = time.perf_counter()
                response = client.post("/api/v1/recommendations", json=payload)
                elapsed = (time.perf_counter() - started) * 1000
                if response.status_code != 200:
                    errors += 1
                    continue
                last = response.json()
                if index >= warmups:
                    client_ms.append(elapsed)
                    server_ms.append(last["timings"]["total_ms"])
            if not client_ms or last is None:
                raise RuntimeError(f"no successful samples for {name}")
            records[name] = {
                "samples": len(client_ms),
                "errors": errors,
                "client_round_trip": summary(client_ms),
                "server_orchestration": summary(server_ms),
                "displayed_count": len(last["recommendations"]),
                "partial": last["partial"],
                "backend": last["backend_mode"],
                "retrieval": last["retrieval"],
                "reranker": last["reranker"],
                "sample_request_id": last["request_id"],
            }
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/application_workload_fixture.json"))
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups non-negative")
    report = run(args.repetitions, args.warmups, Settings.from_env())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["workloads"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
