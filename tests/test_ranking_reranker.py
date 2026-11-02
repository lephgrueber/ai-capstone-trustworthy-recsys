"""Serving reranker: ordering, limits, fallback handling and artifact checks."""

import json

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")
pytest.importorskip("scipy")

from trustworthy_recsys.data.reporting import hashes, write_json
from trustworthy_recsys.ranking.features import FEATURE_NAMES
from trustworthy_recsys.ranking.reranker import LambdaRankReranker
from trustworthy_recsys.serving.contracts import CandidateBatch, ComponentInfo, ModelQuery, ScoredMovie
from test_ranking_features import fitted

INFO = ComponentInfo(name="lambdarank", version="test", mode="learned")
RETRIEVAL = ComponentInfo(name="two-tower-weaviate", version="test", mode="learned")


def booster(feature_names=FEATURE_NAMES):
    rng = np.random.default_rng(0)
    x = rng.random((400, len(feature_names)))
    y = (x[:, 0] > .5).astype(int)
    return lgb.train({"objective": "lambdarank", "verbosity": -1, "num_threads": 1, "min_data_in_leaf": 5},
                     lgb.Dataset(x, y, group=[20] * 20), 10)


def batch(ids, scores):
    return CandidateBatch(candidates=[ScoredMovie(movie_id=m, score=s) for m, s in zip(ids, scores)], component=RETRIEVAL)


def test_rerank_orders_by_model_score_and_respects_limit():
    features = fitted()
    model = booster()
    reranker = LambdaRankReranker(model, features, FEATURE_NAMES, INFO)
    query = ModelQuery(history_items=["a"], known_user_id="u1")
    result = reranker.rerank(query, batch(["b", "c", "d"], [.9, .5, .1]), limit=2)
    expected = model.predict(features.build(["a"], [1, 2, 3], np.array([.9, .5, .1], dtype="float32"), "u1"))
    order = np.argsort(-expected, kind="stable")[:2]
    assert [item.movie_id for item in result.items] == [["b", "c", "d"][i] for i in order]
    assert [item.score for item in result.items] == pytest.approx(expected[order])
    assert result.component == INFO and result.warnings == []


def test_ablated_models_receive_only_their_feature_columns():
    names = [n for n in FEATURE_NAMES if n not in ("retrieval_score", "retrieval_rank")]
    reranker = LambdaRankReranker(booster(names), fitted(), names, INFO)
    result = reranker.rerank(ModelQuery(history_items=["a"]), batch(["b", "c"], [.2, .1]), limit=5)
    assert len(result.items) == 2


def test_fallback_candidates_and_empty_batches():
    reranker = LambdaRankReranker(booster(), fitted(), FEATURE_NAMES, INFO)
    fallback = reranker.rerank(ModelQuery(history_items=[]), batch(["a", "b"], [None, None]), limit=2)
    assert {item.movie_id for item in fallback.items} == {"a", "b"}
    assert "popularity-fallback" in fallback.warnings[0]
    assert reranker.rerank(ModelQuery(history_items=[]), batch([], []), limit=3).items == []


def test_unknown_candidates_and_features_are_rejected():
    reranker = LambdaRankReranker(booster(), fitted(), FEATURE_NAMES, INFO)
    with pytest.raises(ValueError, match="outside the reranker catalog"):
        reranker.rerank(ModelQuery(history_items=[]), batch(["zz"], [.1]), limit=1)
    with pytest.raises(ValueError, match="unknown features"):
        LambdaRankReranker(booster(), fitted(), ["not_a_feature"], INFO)


def saved_model(root, retrieval):
    root.mkdir()
    booster().save_model(str(root / "model.txt"))
    write_json(root / "training.json", {"features": list(FEATURE_NAMES), "retrieval": retrieval,
                                        "feature_cutoff_utc": "2018-10-04T00:00:00+00:00"})
    write_json(root / "manifest.json", {"outputs": {n: hashes(root / n)["sha256"] for n in ("model.txt", "training.json")}})


def test_load_refuses_tampered_or_unpaired_models(tmp_path):
    saved_model(tmp_path / "unpaired", None)
    with pytest.raises(ValueError, match="does not record its retrieval model"):
        LambdaRankReranker.load(tmp_path / "unpaired", None, None, None, None)
    saved_model(tmp_path / "tampered", {"input_manifest_sha256": "x"})
    (tmp_path / "tampered" / "model.txt").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        LambdaRankReranker.load(tmp_path / "tampered", None, None, None, None)
