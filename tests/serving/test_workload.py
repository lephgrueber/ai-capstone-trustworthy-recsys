import sys

import pytest

from trustworthy_recsys.serving import workload
from trustworthy_recsys.serving.workload import percentile, prepare_output, summary, write_report_exclusive


def test_percentiles_use_nearest_rank():
    values = list(range(1, 101))

    assert percentile(values, 0.50) == 50
    assert summary(values)["p50_ms"] == 50
    assert summary(values)["p95_ms"] == 95
    assert summary(values)["p99_ms"] == 99


def test_small_sample_p99_is_the_maximum():
    values = [211.2369, 216.8990, 218.1435, 217.4430, 218.0710]

    assert summary(values)["p99_ms"] == max(values)


def test_existing_output_is_rejected_before_http_requests(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("preserved\n", encoding="utf-8")

    def unexpected_http(*_args, **_kwargs):
        pytest.fail("HTTP workload must not start when the output already exists")

    monkeypatch.setattr(workload, "run_http", unexpected_http)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "workload",
            "--base-url",
            "http://127.0.0.1:8000",
            "--profile",
            "learned",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit):
        workload.main()

    assert output.read_text(encoding="utf-8") == "preserved\n"


def test_exclusive_write_rejects_target_created_after_guard(tmp_path):
    output = tmp_path / "new" / "report.json"
    prepare_output(output)
    output.write_text("created by another process\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_report_exclusive(output, {"workloads": {}})

    assert output.read_text(encoding="utf-8") == "created by another process\n"

