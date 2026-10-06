"""Environment-backed service configuration and startup validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    backend_mode: str = "fixture"
    run_dir: Path | None = None
    retrieval_inputs_dir: Path | None = None
    retrieval_model_dir: Path | None = None
    scenario: str = "ratio_80_10_10"
    candidate_budget: int = 100
    tmdb_read_access_token: str | None = None
    tmdb_timeout_seconds: float = 2.0
    poster_cache_size: int = 512

    def __post_init__(self) -> None:
        for name in ("run_dir", "retrieval_inputs_dir", "retrieval_model_dir"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

    @classmethod
    def from_env(cls) -> "Settings":
        run = os.getenv("TRUSTWORTHY_RECSYS_RUN_DIR")
        retrieval_inputs = os.getenv("TRUSTWORTHY_RECSYS_RETRIEVAL_INPUTS_DIR")
        retrieval_model = os.getenv("TRUSTWORTHY_RECSYS_RETRIEVAL_MODEL_DIR")
        try:
            budget = int(os.getenv("TRUSTWORTHY_RECSYS_CANDIDATE_BUDGET", "100"))
        except ValueError as error:
            raise ValueError("TRUSTWORTHY_RECSYS_CANDIDATE_BUDGET must be an integer") from error
        if not 10 <= budget <= 500:
            raise ValueError("TRUSTWORTHY_RECSYS_CANDIDATE_BUDGET must be between 10 and 500")
        return cls(
            backend_mode=os.getenv("TRUSTWORTHY_RECSYS_BACKEND", "fixture"),
            run_dir=Path(run) if run else None,
            retrieval_inputs_dir=Path(retrieval_inputs) if retrieval_inputs else None,
            retrieval_model_dir=Path(retrieval_model) if retrieval_model else None,
            scenario=os.getenv("TRUSTWORTHY_RECSYS_SCENARIO", "ratio_80_10_10"),
            candidate_budget=budget,
            tmdb_read_access_token=os.getenv("TMDB_READ_ACCESS_TOKEN") or None,
        )

    def validate_baseline_run(self) -> None:
        if self.run_dir is None:
            raise ValueError("baseline mode requires TRUSTWORTHY_RECSYS_RUN_DIR")
        if (self.run_dir / "INCOMPLETE").exists() or not (self.run_dir / "manifest.json").is_file():
            raise ValueError(f"{self.run_dir} is not a completed pipeline run")
        required = [
            self.run_dir / "movies.parquet",
            self.run_dir / self.scenario / "train.parquet",
            self.run_dir / self.scenario / "train_item_ids.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError(
                f"baseline mode requires completed {self.scenario} training artifacts; missing: {missing}"
            )
        manifest = json.loads((self.run_dir / "manifest.json").read_text(encoding="utf-8"))
        requested = manifest.get("config", {}).get("requested_ratios", {}).get(self.scenario)
        if self.scenario != "ratio_80_10_10" or requested != [0.8, 0.1, 0.1]:
            raise ValueError("baseline mode requires the manifest-declared ratio_80_10_10 split [0.8, 0.1, 0.1]")
        if not manifest.get("validation") or not all(manifest["validation"].values()):
            raise ValueError("baseline run manifest does not report all pipeline validations complete")
        output_names = set(manifest.get("outputs", {}))
        required_names = {"movies.parquet", f"{self.scenario}/train.parquet", f"{self.scenario}/train_item_ids.json"}
        if not required_names.issubset(output_names):
            raise ValueError("baseline run manifest does not identify all required training artifacts")

    def validate_learned_artifacts(self) -> None:
        if self.run_dir is None:
            raise ValueError(
                "learned mode requires TRUSTWORTHY_RECSYS_RUN_DIR for catalog metadata "
                "and configured retrieval artifacts"
            )
        if self.retrieval_inputs_dir is None or self.retrieval_model_dir is None:
            raise ValueError(
                "learned mode requires TRUSTWORTHY_RECSYS_RETRIEVAL_INPUTS_DIR and "
                "TRUSTWORTHY_RECSYS_RETRIEVAL_MODEL_DIR artifacts"
            )
        for label, root in (
            ("catalog run", self.run_dir),
            ("retrieval inputs", self.retrieval_inputs_dir),
            ("published retrieval model", self.retrieval_model_dir),
        ):
            if (root / "INCOMPLETE").exists() or not (root / "manifest.json").is_file():
                raise ValueError(f"learned mode requires completed {label}: {root}")
        if not (self.run_dir / "movies.parquet").is_file():
            raise ValueError("learned mode catalog run is missing movies.parquet")
        if not (self.retrieval_model_dir / "weaviate.json").is_file():
            raise ValueError("learned mode requires a published model with weaviate.json")
