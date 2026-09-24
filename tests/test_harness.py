import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from eval.harness import (
    EvalExample,
    SavedPredictionsRecommender,
    evaluate,
    load_evaluation_examples,
    load_saved_predictions,
    validate_prediction_users,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def run_harness(*arguments):
    return subprocess.run(
        [sys.executable, "eval/harness.py", *map(str, arguments)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_synthetic_harness_smoke_run(tmp_path):
    completed = run_harness("--output-dir", tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "recall@10" in completed.stdout.lower()
    assert "ndcg@10" in completed.stdout.lower()

    result_paths = list(tmp_path.glob("*.json"))
    assert len(result_paths) == 1

    payload = json.loads(result_paths[0].read_text(encoding="utf-8"))

    assert payload["synthetic"] is True
    assert payload["dataset"] == "synthetic"
    assert payload["split"] == "smoke"
    assert payload["recommender"] == "synthetic_recommender"
    assert payload["recall_k"] == 10
    assert payload["ndcg_k"] == 10
    assert payload["number_of_examples"] == 3
    assert payload["execution_mode"] == "synthetic_smoke"
    assert payload["examples_file"] is None
    assert payload["predictions_file"] is None
    assert "recall@10" in payload["aggregate"]
    assert "ndcg@10" in payload["aggregate"]
    assert len(payload["per_example"]) == 3


def test_file_based_run_joins_users_by_id_and_records_metadata(tmp_path):
    examples_path = tmp_path / "examples.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "results"
    write_jsonl(
        examples_path,
        [
            {
                "user_id": "user_1",
                "history_items": ["item_a", "item_b"],
                "relevant_items": ["item_c"],
            },
            {
                "user_id": "user_2",
                "history_items": [],
                "relevant_items": ["item_p", "item_q"],
            },
        ],
    )
    # Intentionally reverse the row order to prove matching is by user ID.
    write_jsonl(
        predictions_path,
        [
            {
                "user_id": "user_2",
                "ranked_items": ["item_q", "item_z", "item_p"],
            },
            {
                "user_id": "user_1",
                "ranked_items": ["item_x", "item_c"],
            },
        ],
    )

    completed = run_harness(
        "--examples-file",
        examples_path,
        "--predictions-file",
        predictions_path,
        "--dataset",
        "tiny-fixture",
        "--split",
        "test",
        "--recommender-name",
        "saved-baseline",
        "--data-origin",
        "synthetic",
        "--recall-k",
        "2",
        "--ndcg-k",
        "3",
        "--output-dir",
        output_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert "rankings shorter than max(recall_k, ndcg_k)=3: 1" in (
        completed.stdout.lower()
    )
    [result_path] = list(output_path.glob("*.json"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload["aggregate"]["recall@2"] == pytest.approx(0.75)
    user_1_ndcg = 1 / math.log2(3)
    user_2_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert payload["aggregate"]["ndcg@3"] == pytest.approx(
        (user_1_ndcg + user_2_ndcg) / 2
    )
    assert [result["user_id"] for result in payload["per_example"]] == [
        "user_1",
        "user_2",
    ]
    assert payload["per_example"][0]["ranked_items"] == [
        "item_x",
        "item_c",
    ]
    assert payload["dataset"] == "tiny-fixture"
    assert payload["split"] == "test"
    assert payload["recommender"] == "saved-baseline"
    assert payload["synthetic"] is True
    assert payload["execution_mode"] == "saved_predictions"
    assert payload["examples_file"] == str(examples_path)
    assert payload["predictions_file"] == str(predictions_path)
    assert payload["number_of_short_rankings"] == 1


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([{"user_id": "u", "history_items": [], "relevant_items": []}],
         "'relevant_items' must not be empty"),
        ([{"user_id": "u", "history_items": "item_a", "relevant_items": ["i"]}],
         "'history_items' must be a list"),
        ([{"user_id": 7, "history_items": [], "relevant_items": ["i"]}],
         "'user_id' must be a nonempty string"),
        ([{"user_id": "u", "history_items": [""], "relevant_items": ["i"]}],
         "'history_items' item 0 must be a nonempty string"),
    ],
)
def test_evaluation_loader_rejects_wrong_types_and_empty_labels(
    tmp_path, records, message
):
    path = tmp_path / "examples.jsonl"
    write_jsonl(path, records)

    with pytest.raises(ValueError, match=message):
        load_evaluation_examples(path)


@pytest.mark.parametrize("contents", ["not json\n", "[]\n", "\n"])
def test_evaluation_loader_rejects_malformed_records(tmp_path, contents):
    path = tmp_path / "examples.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        load_evaluation_examples(path)

    assert str(path) in str(error.value)
    assert "line 1" in str(error.value)


def test_evaluation_loader_requires_an_example_and_unique_users(tmp_path):
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one evaluation example"):
        load_evaluation_examples(empty_path)

    duplicate_path = tmp_path / "duplicate.jsonl"
    write_jsonl(
        duplicate_path,
        [
            {"user_id": "u", "history_items": [], "relevant_items": ["a"]},
            {"user_id": "u", "history_items": ["a"], "relevant_items": ["b"]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate user_id 'u'"):
        load_evaluation_examples(duplicate_path)


def test_evaluation_loader_preserves_history_order_and_duplicates(tmp_path):
    path = tmp_path / "examples.jsonl"
    write_jsonl(
        path,
        [
            {
                "user_id": "u",
                "history_items": ["item_b", "item_a", "item_b"],
                "relevant_items": ["item_c"],
            }
        ],
    )

    [example] = load_evaluation_examples(path)

    assert example.history_items == ["item_b", "item_a", "item_b"]


def test_prediction_loader_rejects_duplicate_users_and_items_beyond_cutoff(
    tmp_path,
):
    duplicate_users = tmp_path / "duplicate_users.jsonl"
    write_jsonl(
        duplicate_users,
        [
            {"user_id": "u", "ranked_items": ["a"]},
            {"user_id": "u", "ranked_items": ["b"]},
        ],
    )
    with pytest.raises(ValueError, match="duplicate user_id 'u'"):
        load_saved_predictions(duplicate_users)

    duplicate_items = tmp_path / "duplicate_items.jsonl"
    write_jsonl(
        duplicate_items,
        [{"user_id": "u", "ranked_items": ["a", "b", "c", "c"]}],
    )
    with pytest.raises(ValueError, match="duplicate item IDs"):
        load_saved_predictions(duplicate_items)


def test_prediction_loader_rejects_wrong_field_types(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [{"user_id": "u", "ranked_items": "item_a"}])

    with pytest.raises(ValueError, match="'ranked_items' must be a list"):
        load_saved_predictions(path)


def test_prediction_users_must_match_examples_exactly():
    examples = [
        EvalExample("user_1", [], ["a"]),
        EvalExample("user_2", [], ["b"]),
    ]

    with pytest.raises(ValueError) as error:
        validate_prediction_users(
            examples,
            {"user_1": ["a"], "user_3": ["c"]},
        )

    assert "missing users: ['user_2']" in str(error.value)
    assert "unexpected users: ['user_3']" in str(error.value)


def test_short_and_empty_rankings_are_scored_without_padding():
    examples = [
        EvalExample("empty", [], ["a"]),
        EvalExample("short", ["x"], ["b", "c"]),
    ]
    recommender = SavedPredictionsRecommender(
        {"empty": [], "short": ["b"]}
    )

    aggregate, per_example = evaluate(
        examples,
        recommender,
        recall_k=5,
        ndcg_k=2,
    )

    assert [result.ranked_items for result in per_example] == [[], ["b"]]
    assert aggregate["recall@5"] == pytest.approx(0.25)
    expected_short_ndcg = 1 / (1 + 1 / math.log2(3))
    assert aggregate["ndcg@2"] == pytest.approx(expected_short_ndcg / 2)


@pytest.mark.parametrize(
    "arguments",
    [
        ("--examples-file", "examples.jsonl"),
        ("--dataset", "metadata-only"),
        ("--examples-file", "e", "--predictions-file", "p"),
        ("--recall-k", "0"),
        ("--ndcg-k", "-1"),
    ],
)
def test_cli_rejects_incomplete_modes_and_nonpositive_k(arguments):
    completed = run_harness(*arguments)

    assert completed.returncode != 0
    assert "error:" in completed.stderr.lower()


def test_golden_set_reference_items_are_not_relevance_labels(tmp_path):
    path = tmp_path / "golden.jsonl"
    write_jsonl(
        path,
        [
            {
                "user_id": "golden_1",
                "history_items": ["item_a"],
                "reference_items": ["item_b"],
            }
        ],
    )

    with pytest.raises(ValueError, match="reference_items.*relevance labels"):
        load_evaluation_examples(path)
