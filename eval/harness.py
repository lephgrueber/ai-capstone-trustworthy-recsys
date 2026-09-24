"""Runnable recommendation evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence


# Allow `python eval/harness.py` to import the project package.
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from trustworthy_recsys.evaluation.metrics import ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class EvalExample:
    """One recommendation evaluation example."""

    user_id: str
    history_items: list[str]
    relevant_items: list[str]


@dataclass(frozen=True)
class RecommendationRequest:
    """Inputs available to a recommender for one evaluation example."""

    user_id: str
    history_items: list[str]


@dataclass(frozen=True)
class ExampleResult:
    """Evaluation result for one user."""

    user_id: str
    relevant_items: list[str]
    ranked_items: list[str]
    recall: float
    ndcg: float


Recommender = Callable[[RecommendationRequest, int], Sequence[str]]


def _record_context(path: Path, line_number: int) -> str:
    """Describe a JSONL record location for validation errors."""

    return f"{path}: line {line_number}"


def _load_json_object(line: str, path: Path, line_number: int) -> dict:
    """Parse one JSONL line and require an object record."""

    context = _record_context(path, line_number)

    if not line.strip():
        raise ValueError(f"{context}: blank lines are not valid records")

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{context}: invalid JSON ({error.msg})"
        ) from error

    if not isinstance(record, dict):
        raise ValueError(f"{context}: record must be a JSON object")

    return record


def _require_nonempty_string(
    record: dict,
    field_name: str,
    context: str,
) -> str:
    """Read a required, nonblank string field."""

    if field_name not in record:
        raise ValueError(f"{context}: missing required field '{field_name}'")

    value = record[field_name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{context}: '{field_name}' must be a nonempty string"
        )

    return value


def _require_string_list(
    record: dict,
    field_name: str,
    context: str,
    *,
    allow_empty: bool,
) -> list[str]:
    """Read a required list containing only nonblank string IDs."""

    if field_name not in record:
        if field_name == "relevant_items" and "reference_items" in record:
            raise ValueError(
                f"{context}: missing required field 'relevant_items'; "
                "qualitative 'reference_items' cannot be used as relevance labels"
            )
        raise ValueError(f"{context}: missing required field '{field_name}'")

    value = record[field_name]
    if not isinstance(value, list):
        raise ValueError(f"{context}: '{field_name}' must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{context}: '{field_name}' must not be empty")

    for item_index, item_id in enumerate(value):
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError(
                f"{context}: '{field_name}' item {item_index} must be "
                "a nonempty string"
            )

    return value


def load_evaluation_examples(path: Path) -> list[EvalExample]:
    """Load and validate quantitative evaluation examples from JSONL."""

    examples: list[EvalExample] = []
    user_lines: dict[str, int] = {}

    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                context = _record_context(path, line_number)
                record = _load_json_object(line, path, line_number)
                user_id = _require_nonempty_string(
                    record, "user_id", context
                )

                if user_id in user_lines:
                    raise ValueError(
                        f"{context}: duplicate user_id {user_id!r}; first "
                        f"appeared on line {user_lines[user_id]}"
                    )

                history_items = _require_string_list(
                    record,
                    "history_items",
                    context,
                    allow_empty=True,
                )
                relevant_items = _require_string_list(
                    record,
                    "relevant_items",
                    context,
                    allow_empty=False,
                )
                user_lines[user_id] = line_number
                examples.append(
                    EvalExample(
                        user_id=user_id,
                        history_items=history_items,
                        relevant_items=relevant_items,
                    )
                )
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{path}: unable to read file ({error})") from error

    if not examples:
        raise ValueError(f"{path}: must contain at least one evaluation example")

    return examples


def load_saved_predictions(path: Path) -> dict[str, list[str]]:
    """Load and validate saved, highest-ranked-first predictions from JSONL."""

    predictions: dict[str, list[str]] = {}
    user_lines: dict[str, int] = {}

    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                context = _record_context(path, line_number)
                record = _load_json_object(line, path, line_number)
                user_id = _require_nonempty_string(
                    record, "user_id", context
                )

                if user_id in user_lines:
                    raise ValueError(
                        f"{context}: duplicate user_id {user_id!r}; first "
                        f"appeared on line {user_lines[user_id]}"
                    )

                ranked_items = _require_string_list(
                    record,
                    "ranked_items",
                    context,
                    allow_empty=True,
                )
                if len(ranked_items) != len(set(ranked_items)):
                    raise ValueError(
                        f"{context}: user {user_id!r} has duplicate item IDs "
                        "in 'ranked_items'"
                    )

                user_lines[user_id] = line_number
                predictions[user_id] = ranked_items
    except (OSError, UnicodeError) as error:
        raise ValueError(f"{path}: unable to read file ({error})") from error

    return predictions


def validate_prediction_users(
    examples: Sequence[EvalExample],
    predictions: Mapping[str, Sequence[str]],
) -> None:
    """Require predictions for exactly the users in the evaluation file."""

    evaluation_users = {example.user_id for example in examples}
    prediction_users = set(predictions)
    missing_users = sorted(evaluation_users - prediction_users)
    unexpected_users = sorted(prediction_users - evaluation_users)

    if missing_users or unexpected_users:
        details = []
        if missing_users:
            details.append(f"missing users: {missing_users}")
        if unexpected_users:
            details.append(f"unexpected users: {unexpected_users}")
        raise ValueError(
            "prediction user IDs do not match evaluation user IDs ("
            + "; ".join(details)
            + ")"
        )


@dataclass(frozen=True)
class SavedPredictionsRecommender:
    """Adapt a saved prediction mapping to the recommender callable API."""

    predictions: Mapping[str, Sequence[str]]

    def __call__(
        self,
        request: RecommendationRequest,
        k: int,
    ) -> Sequence[str]:
        return self.predictions[request.user_id][:k]


def load_synthetic_examples() -> list[EvalExample]:
    """Create small synthetic examples for harness development."""

    return [
        EvalExample(
            user_id="user_1",
            history_items=["item_a", "item_b"],
            relevant_items=["item_c"],
        ),
        EvalExample(
            user_id="user_2",
            history_items=["item_d", "item_e"],
            relevant_items=["item_g"],
        ),
        EvalExample(
            user_id="user_3",
            history_items=["item_h"],
            relevant_items=["item_i", "item_j"],
        ),
    ]


def synthetic_recommender(
    request: RecommendationRequest,
    k: int,
) -> Sequence[str]:
    """
    Return deterministic synthetic recommendations.

    This exists only to verify that the evaluation harness works end-to-end.
    It will later be replaced by real baselines and recommendation models.
    """

    predictions = {
        "user_1": [
            "item_x",
            "item_c",
            "item_y",
            "item_z",
        ],
        "user_2": [
            "item_g",
            "item_x",
            "item_y",
            "item_z",
        ],
        "user_3": [
            "item_x",
            "item_i",
            "item_y",
            "item_j",
        ],
    }

    return predictions[request.user_id][:k]


def evaluate(
    examples: Sequence[EvalExample],
    recommender: Recommender,
    recall_k: int,
    ndcg_k: int,
) -> tuple[dict[str, float], list[ExampleResult]]:
    """Run the recommender and compute aggregate and per-example metrics."""

    if not examples:
        raise ValueError("examples must contain at least one evaluation case")
    if recall_k <= 0:
        raise ValueError("recall_k must be greater than 0")
    if ndcg_k <= 0:
        raise ValueError("ndcg_k must be greater than 0")

    max_k = max(recall_k, ndcg_k)

    per_example: list[ExampleResult] = []

    for example in examples:
        request = RecommendationRequest(
            user_id=example.user_id,
            history_items=example.history_items,
        )
        ranked_items = list(recommender(request, max_k))

        result = ExampleResult(
            user_id=example.user_id,
            relevant_items=example.relevant_items,
            ranked_items=ranked_items,
            recall=recall_at_k(
                example.relevant_items,
                ranked_items,
                recall_k,
            ),
            ndcg=ndcg_at_k(
                example.relevant_items,
                ranked_items,
                ndcg_k,
            ),
        )

        per_example.append(result)

    aggregate = {
        f"recall@{recall_k}": mean(
            result.recall for result in per_example
        ),
        f"ndcg@{ndcg_k}": mean(
            result.ndcg for result in per_example
        ),
    }

    return aggregate, per_example


def print_results(results: dict[str, float]) -> None:
    """Print aggregate evaluation metrics."""

    print()
    print("Evaluation Results")
    print("------------------------------")
    print(f"{'Metric':<20} {'Value':>8}")
    print("------------------------------")

    for metric, value in results.items():
        print(f"{metric:<20} {value:>8.4f}")

    print("------------------------------")


def save_results(
    aggregate: dict[str, float],
    per_example: Sequence[ExampleResult],
    output_dir: Path,
    dataset: str,
    split: str,
    recommender_name: str,
    synthetic: bool,
    recall_k: int,
    ndcg_k: int,
    execution_mode: str = "synthetic_smoke",
    examples_file: Path | None = None,
    predictions_file: Path | None = None,
    number_of_short_rankings: int | None = None,
) -> Path:
    """Save aggregate and per-example results as JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"evaluation_{timestamp}.json"

    payload = {
        "created_at_utc": timestamp,
        "dataset": dataset,
        "split": split,
        "recommender": recommender_name,
        "synthetic": synthetic,
        "recall_k": recall_k,
        "ndcg_k": ndcg_k,
        "execution_mode": execution_mode,
        "examples_file": (
            str(examples_file) if examples_file is not None else None
        ),
        "predictions_file": (
            str(predictions_file) if predictions_file is not None else None
        ),
        "number_of_short_rankings": number_of_short_rankings,
        "number_of_examples": len(per_example),
        "aggregate": aggregate,
        "per_example": [
            asdict(result)
            for result in per_example
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Run recommendation evaluation."
    )

    parser.add_argument(
        "--recall-k",
        type=int,
        default=10,
        help="K used for Recall@K.",
    )

    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=10,
        help="K used for NDCG@K.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/smoke"),
        help="Directory used for saved results.",
    )

    parser.add_argument(
        "--examples-file",
        type=Path,
        help="JSONL file containing quantitative evaluation examples.",
    )

    parser.add_argument(
        "--predictions-file",
        type=Path,
        help="JSONL file containing saved ranked predictions.",
    )

    parser.add_argument(
        "--dataset",
        help="Dataset name recorded in file-based result metadata.",
    )

    parser.add_argument(
        "--split",
        help="Dataset split recorded in file-based result metadata.",
    )

    parser.add_argument(
        "--recommender-name",
        help="Recommender name recorded in file-based result metadata.",
    )

    parser.add_argument(
        "--data-origin",
        choices=("synthetic", "observed"),
        help="Caller-declared origin of the file-based evaluation data.",
    )

    return parser


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    return build_parser().parse_args(argv)


def _validate_cli_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> bool:
    """Validate CLI mode selection and return whether it is file-based."""

    if args.recall_k <= 0:
        parser.error("--recall-k must be greater than 0")
    if args.ndcg_k <= 0:
        parser.error("--ndcg-k must be greater than 0")

    file_values = (args.examples_file, args.predictions_file)
    metadata_values = (
        args.dataset,
        args.split,
        args.recommender_name,
        args.data_origin,
    )
    supplied_values = file_values + metadata_values

    if not any(value is not None for value in supplied_values):
        return False

    required_names = (
        "--examples-file",
        "--predictions-file",
        "--dataset",
        "--split",
        "--recommender-name",
        "--data-origin",
    )
    missing_names = [
        name
        for name, value in zip(required_names, supplied_values)
        if value is None
    ]
    if missing_names:
        parser.error(
            "file-based mode requires all of --examples-file, "
            "--predictions-file, --dataset, --split, --recommender-name, "
            "and --data-origin; missing: " + ", ".join(missing_names)
        )

    for name, value in zip(required_names[2:5], metadata_values[:3]):
        if not value.strip():
            parser.error(f"{name} must be a nonempty string")

    return True


def main() -> None:
    """Run the evaluation harness."""

    parser = build_parser()
    args = parser.parse_args()
    file_based = _validate_cli_args(args, parser)

    if file_based:
        try:
            examples = load_evaluation_examples(args.examples_file)
            predictions = load_saved_predictions(args.predictions_file)
            validate_prediction_users(examples, predictions)
        except ValueError as error:
            parser.error(str(error))

        recommender: Recommender = SavedPredictionsRecommender(predictions)
        dataset = args.dataset
        split = args.split
        recommender_name = args.recommender_name
        synthetic = args.data_origin == "synthetic"
        execution_mode = "saved_predictions"
    else:
        examples = load_synthetic_examples()
        recommender = synthetic_recommender
        dataset = "synthetic"
        split = "smoke"
        recommender_name = "synthetic_recommender"
        synthetic = True
        execution_mode = "synthetic_smoke"

    aggregate, per_example = evaluate(
        examples=examples,
        recommender=recommender,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
    )

    short_ranking_count = (
        sum(
            len(result.ranked_items) < max(args.recall_k, args.ndcg_k)
            for result in per_example
        )
        if file_based
        else None
    )

    print_results(aggregate)

    output_path = save_results(
        aggregate=aggregate,
        per_example=per_example,
        output_dir=args.output_dir,
        dataset=dataset,
        split=split,
        recommender_name=recommender_name,
        synthetic=synthetic,
        recall_k=args.recall_k,
        ndcg_k=args.ndcg_k,
        execution_mode=execution_mode,
        examples_file=args.examples_file,
        predictions_file=args.predictions_file,
        number_of_short_rankings=short_ranking_count,
    )

    if file_based:
        print(
            "Rankings shorter than "
            f"max(recall_k, ndcg_k)={max(args.recall_k, args.ndcg_k)}: "
            f"{short_ranking_count}"
        )

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
