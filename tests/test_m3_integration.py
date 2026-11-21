"""Regression checks for safeguards retained while adopting PR #8."""

import json
from unittest.mock import Mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from trustworthy_recsys.data.reporting import hashes, write_json
from trustworthy_recsys.ranking.compare import compare
from trustworthy_recsys.ranking.evaluate import evaluate
from trustworthy_recsys.ranking.train import select_rounds
from trustworthy_recsys.ranking.population import viewers_for
from trustworthy_recsys.workflow import guard_request, verified_manifest


def candidate_table(path, empty=False):
    path.mkdir()
    data = {"user_id": np.array([] if empty else ["one"], dtype=str),
            "movie_id": np.array([] if empty else ["movie"], dtype=str),
            "label": np.array([] if empty else [1], dtype=np.int8),
            "fallback": np.array([] if empty else [True], dtype=bool),
            "user_relevant_count": np.array([] if empty else [1], dtype=np.int32),
            "retrieval_rank": np.array([] if empty else [0], dtype=float),
            "user_log_history_length": np.array([] if empty else [0], dtype=float)}
    pq.write_table(pa.table(data), path / "rows.parquet")
    pq.write_table(pa.table({"user_id": ["one", "empty"], "relevant_count": [1, 1],
                            "history_bucket": ["history_0", "history_0"],
                            "fallback": [True, True], "popularity_labels": [[1], []]}), path / "users.parquet")
    write_json(path / "manifest.json", {
        "rows_sha256": hashes(path / "rows.parquet")["sha256"],
        "users_sha256": hashes(path / "users.parquet")["sha256"],
        "feature_cutoff_utc": "2020-01-01"})
    return data


@pytest.mark.parametrize("empty", [False, True])
def test_empty_candidate_viewers_in_compare_selection_and_evaluation(tmp_path, monkeypatch, empty):
    table = tmp_path / "table"
    data = candidate_table(table, empty)
    expected = 0.0 if empty else 0.5
    result = compare(table, ["pass_through"], ["pass_through"], samples=10)
    assert result["scores"]["recall_at_12"]["a"] == expected
    assert result["difference"]["recall_at_12"]["all"]["viewers"] == 2
    booster = Mock()
    booster.predict.return_value = np.ones(len(data["user_id"]))
    _, curve, baseline, count = select_rounds(booster, data, ["retrieval_rank"], 12,
                                            [1, 2], viewers_for(table))
    assert count == 2 and baseline == expected and curve[1] == expected
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "manifest.json", {"outputs": {}})
    write_json(model / "training.json", {"features": ["retrieval_rank"],
                                          "feature_cutoff_utc": "2020-01-01",
                                          "table_rows_sha256": "a-different-fitting-table"})
    monkeypatch.setattr("trustworthy_recsys.ranking.evaluate.lgb.Booster", lambda **kwargs: booster)
    result = evaluate(table, model, tmp_path / "report.json", bootstrap=10)
    assert result["users"] == 2 and result["empty_candidate_users"] == (2 if empty else 1)
    assert result["candidate_ceiling_recall"] == expected
    assert result["metrics"]["lambdarank/all"]["recall_at_12"] == expected
    assert result["metrics"]["popularity/all"]["recall_at_12"] == 0.5


def test_viewer_manifest_tampering_rejected(tmp_path):
    candidate_table(tmp_path / "table")
    with (tmp_path / "table" / "users.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(tmp_path / "table", ["pass_through"], ["pass_through"], samples=10)


def test_request_changes_and_unfingerprinted_outputs_rejected(tmp_path):
    out = tmp_path / "stage"
    request = {"settings": {"depth": 500}, "source_manifest": "original"}
    guard_request(out, request)
    guard_request(out, request)
    for changed in ({**request, "settings": {"depth": 1000}}, {**request, "source_manifest": "new"}):
        with pytest.raises(ValueError, match="request changed"):
            guard_request(out, changed)
    assert json.loads((out / "request.json").read_text()) == request
    (out / "request.json").unlink()
    (out / "old-model.pt").write_text("old")
    with pytest.raises(ValueError, match="no request fingerprint"):
        guard_request(out, request)


def test_completed_artifact_hashes_checked(tmp_path):
    (tmp_path / "weights").write_text("original")
    write_json(tmp_path / "manifest.json", {"outputs": {"weights": hashes(tmp_path / "weights")["sha256"]}})
    verified_manifest(tmp_path)
    (tmp_path / "weights").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verified_manifest(tmp_path)
