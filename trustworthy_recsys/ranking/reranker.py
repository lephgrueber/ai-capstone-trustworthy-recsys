"""Serving-time LambdaRank reranker over two-tower candidates.

Features come from the same RankingFeatures code used to build training tables, fitted
on the same training-period interactions, so training and serving cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import hashes
from trustworthy_recsys.serving.contracts import CandidateBatch, ComponentInfo, ModelQuery, RankingResult, ScoredMovie
from .features import FEATURE_NAMES, RankingFeatures
from .train import lightgbm_threads, retrieval_fingerprint


class LambdaRankReranker:
    def __init__(self, booster, features: RankingFeatures, feature_names, info: ComponentInfo):
        unknown = set(feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"reranker uses unknown features: {sorted(unknown)}")
        self.booster = booster
        self.features = features
        self.columns = [FEATURE_NAMES.index(name) for name in feature_names]
        self.info = info

    @classmethod
    def load(cls, model_dir, run_dir, scenario, retrieval_inputs, retrieval_model_dir) -> "LambdaRankReranker":
        """Load a verified reranker that matches the active retrieval model and catalog run."""
        root = Path(model_dir)
        if (root / "INCOMPLETE").exists():
            raise ValueError("reranker artifacts are incomplete")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["outputs"].items():
            if hashes(root / name)["sha256"] != digest:
                raise ValueError(f"reranker artifact hash mismatch: {name}")
        training = json.loads((root / "training.json").read_text(encoding="utf-8"))
        expected = training.get("retrieval")
        if not expected:
            raise ValueError("reranker does not record its retrieval model; retrain with --retrieval-model-dir")
        if expected["input_manifest_sha256"] != hashes(Path(retrieval_inputs.path) / "manifest.json")["sha256"]:
            raise ValueError("reranker was trained on different retrieval inputs")
        if {k: expected[k] for k in retrieval_fingerprint(retrieval_model_dir)} != retrieval_fingerprint(retrieval_model_dir):
            raise ValueError("reranker was trained with a different retrieval model")
        if training["feature_cutoff_utc"] != retrieval_inputs.meta["training_cutoff_utc"]:
            raise ValueError("reranker feature cutoff does not match retrieval training cutoff")
        if hashes(Path(run_dir) / scenario / "train.parquet")["sha256"] != retrieval_inputs.meta["source_hashes"][f"{scenario}/train.parquet"]:
            raise ValueError("catalog run training data does not match retrieval inputs")
        cutoff = int(pd.Timestamp(training["feature_cutoff_utc"]).timestamp())
        features = RankingFeatures.from_run(
            run_dir, scenario, retrieval_inputs.item_ids, retrieval_inputs.item_features, cutoff,
            threshold=retrieval_inputs.meta["positive_threshold"],
        )
        info = ComponentInfo(name="lambdarank", version=f"{root.name}:{manifest_fingerprint(root)}", mode="learned")
        lightgbm_threads(1)  # must precede Booster creation, which parses the model in parallel
        return cls(lgb.Booster(model_file=str(root / "model.txt")), features, training["features"], info)

    def rerank(self, query: ModelQuery, candidates: CandidateBatch, limit: int) -> RankingResult:
        items = candidates.candidates
        if not items:
            return RankingResult(items=[], component=self.info)
        try:
            indices = [self.features.item_map[item.movie_id] for item in items]
        except KeyError as error:
            raise ValueError(f"candidate outside the reranker catalog: {error}") from error
        scores = None
        if any(item.score is not None for item in items):
            scores = np.array([np.nan if item.score is None else item.score for item in items], dtype="float32")
        # Position is the filtered retrieval order; hard filters only remove candidates.
        matrix = self.features.build(query.history_items, indices, scores, query.known_user_id)
        predicted = self.booster.predict(matrix[:, self.columns], num_threads=lightgbm_threads(1))
        order = np.argsort(-predicted, kind="stable")[:limit]
        warnings = [] if scores is not None else ["Reranked popularity-fallback candidates without retrieval scores."]
        return RankingResult(
            items=[ScoredMovie(movie_id=items[i].movie_id, score=float(predicted[i])) for i in order],
            component=self.info,
            warnings=warnings,
        )


def manifest_fingerprint(root) -> str:
    return hashes(Path(root) / "manifest.json")["sha256"][:12]
