import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


GOLDEN_SET_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "golden_set_50.jsonl"
)

REQUIRED_FIELD_TYPES = {
    "example_id": str,
    "source": str,
    "bucket": str,
    "evaluation_use": str,
    "history_items": list,
    "history_display": list,
    "reference_items": list,
    "reference_display": list,
    "expected_behavior": str,
    "selection_reason": str,
}


@pytest.fixture(scope="module")
def golden_records():
    assert GOLDEN_SET_PATH.is_file(), (
        f"Golden-set file is missing: {GOLDEN_SET_PATH}"
    )

    records = []
    for line_number, line in enumerate(
        GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            pytest.fail(f"Invalid JSON on line {line_number}: {error}")

        assert isinstance(record, dict), (
            f"Line {line_number} must contain a JSON object"
        )
        records.append(record)

    return records


def test_golden_set_records_are_internally_consistent(golden_records):
    case_ids = []

    for record in golden_records:
        for field, expected_type in REQUIRED_FIELD_TYPES.items():
            assert field in record, f"Record is missing required field {field!r}"
            assert isinstance(record[field], expected_type), (
                f"{record.get('example_id', '<unknown>')} field {field!r} "
                f"must be {expected_type.__name__}"
            )

        case_id = record["example_id"]
        case_ids.append(case_id)

        assert case_id.strip(), "example_id must be nonempty"
        assert record["expected_behavior"].strip(), (
            f"{case_id} expected_behavior must be nonempty"
        )
        assert record["selection_reason"].strip(), (
            f"{case_id} selection_reason must be nonempty"
        )
        assert record["source"] == "synthetic_catalog_grounded"
        assert record["evaluation_use"] == "qualitative_regression_only"

        assert all(
            isinstance(movie_id, int) and not isinstance(movie_id, bool)
            for movie_id in record["history_items"]
        ), f"{case_id} history_items must contain integer movie IDs"
        assert all(
            isinstance(movie_id, int) and not isinstance(movie_id, bool)
            for movie_id in record["reference_items"]
        ), f"{case_id} reference_items must contain integer movie IDs"

        for field in ("history_display", "reference_display"):
            assert all(isinstance(item, dict) for item in record[field]), (
                f"{case_id} {field} entries must be JSON objects"
            )
            assert all(
                isinstance(item.get("movie_id"), int)
                and not isinstance(item.get("movie_id"), bool)
                for item in record[field]
            ), f"{case_id} {field} entries must have integer movie_id values"

        assert record["history_items"] == [
            item["movie_id"] for item in record["history_display"]
        ], f"{case_id} history_items do not match history_display"
        assert record["reference_items"] == [
            item["movie_id"] for item in record["reference_display"]
        ], f"{case_id} reference_items do not match reference_display"

    assert len(case_ids) == len(set(case_ids)), "example_id values must be unique"


def test_golden_set_current_snapshot(golden_records):
    """Guard the current artifact; review these values for intentional revisions."""

    # These expectations describe the current curated snapshot, not permanent
    # course requirements. Review them whenever the golden set changes on purpose.
    assert len(golden_records) == 50
    assert Counter(record["bucket"] for record in golden_records) == {
        "golden_path": 15,
        "representative": 20,
        "hard_edge": 15,
    }

    histories = defaultdict(list)
    for record in golden_records:
        histories[tuple(record["history_items"])].append(record["example_id"])

    duplicate_history_groups = {
        frozenset(case_ids)
        for case_ids in histories.values()
        if len(case_ids) > 1
    }

    assert len(histories) == 49
    assert duplicate_history_groups == {frozenset({"gs_023", "gs_047"})}
