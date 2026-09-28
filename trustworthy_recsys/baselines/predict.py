"""Write baseline predictions for a data-pipeline run in the harness format.

Example:
    python -m trustworthy_recsys.baselines.predict \\
        --run-dir data/processed/movielens_phase1 --scenario ratio_80_10_10 \\
        --population test_all --model popularity_recent --window-days 90 \\
        --output results/predictions/popularity_recent_90d_test_all.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

from eval.harness import (
    RecommendationRequest,
    Recommender,
    load_evaluation_examples,
)
from trustworthy_recsys.baselines.genre import fit_genre, load_movies
from trustworthy_recsys.baselines.popularity import (
    fit_popularity,
    fit_top_rated,
    load_interactions,
)
from trustworthy_recsys.baselines.random_baseline import RandomRecommender, load_catalog

MODELS = ("random", "popularity", "popularity_recent", "top_rated", "genre")
POPULATIONS = ("validation_all", "validation_warm", "test_all", "test_warm")


def training_cutoff(run_dir: Path, scenario: str) -> pd.Timestamp:
    """Return the scenario's validation start, i.e. the end of training data."""

    statistics = json.loads((run_dir / "statistics.json").read_text(encoding="utf-8"))
    try:
        return pd.Timestamp(statistics["splits"][scenario]["validation_start_utc"])
    except KeyError as error:
        raise ValueError(f"{run_dir}: no split named {scenario!r} in statistics.json") from error


def build_recommender(
    model: str,
    run_dir: Path,
    scenario: str,
    *,
    window_days: int = 90,
    min_rating: float | None = None,
    seed: int = 0,
    genre_alpha: float = 1.0,
    prior_weight: float = 1.0,
    recency_decay: float = 1.0,
) -> Recommender:
    """Fit the named baseline on the scenario's training data only."""

    scenario_dir = run_dir / scenario
    if model == "random":
        return RandomRecommender(load_catalog(scenario_dir / "train_item_ids.json"), seed=seed)

    train = load_interactions(scenario_dir / "train.parquet")
    if model == "popularity":
        return fit_popularity(train, min_rating=min_rating)
    if model == "popularity_recent":
        return fit_popularity(
            train,
            window=timedelta(days=window_days),
            as_of=training_cutoff(run_dir, scenario),
            min_rating=min_rating,
        )
    if model == "top_rated":
        return fit_top_rated(train)
    if model == "genre":
        return fit_genre(
            train,
            load_movies(run_dir / "movies.parquet"),
            alpha=genre_alpha,
            prior_weight=prior_weight,
            recency_decay=recency_decay,
            min_rating=min_rating,
        )
    raise ValueError(f"unknown model {model!r}")


def write_predictions(
    examples_file: Path,
    recommender: Recommender,
    k: int,
    output: Path,
) -> int:
    """Rank k items for every example user and write harness JSONL."""

    examples = load_evaluation_examples(examples_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for example in examples:
            request = RecommendationRequest(
                user_id=example.user_id,
                history_items=example.history_items,
            )
            ranked = list(recommender(request, k))
            file.write(json.dumps({"user_id": example.user_id, "ranked_items": ranked}) + "\n")
    return len(examples)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed pipeline output directory")
    parser.add_argument("--scenario", required=True, help="Split folder, e.g. ratio_80_10_10")
    parser.add_argument("--population", choices=POPULATIONS, default="test_all")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--k", type=int, default=100, help="Items to rank per user")
    parser.add_argument("--window-days", type=int, default=90, help="popularity_recent window")
    parser.add_argument("--min-rating", type=float, help="Count only ratings >= this (popularity models)")
    parser.add_argument("--seed", type=int, default=0, help="random model seed")
    parser.add_argument("--genre-alpha", type=float, default=1.0, help="genre match strength; 0 is plain popularity")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="genre profile smoothing, in history items")
    parser.add_argument("--recency-decay", type=float, default=1.0, help="genre profile per-item decay in (0, 1]")
    parser.add_argument("--output", type=Path, required=True, help="Predictions JSONL to write")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be greater than 0")
    if args.window_days <= 0:
        parser.error("--window-days must be greater than 0")
    if (args.run_dir / "INCOMPLETE").exists() or not (args.run_dir / "manifest.json").exists():
        parser.error(f"{args.run_dir} is not a completed pipeline run")

    try:
        recommender = build_recommender(
            args.model,
            args.run_dir,
            args.scenario,
            window_days=args.window_days,
            min_rating=args.min_rating,
            seed=args.seed,
            genre_alpha=args.genre_alpha,
            prior_weight=args.prior_weight,
            recency_decay=args.recency_decay,
        )
        examples_file = args.run_dir / args.scenario / f"{args.population}.jsonl"
        count = write_predictions(examples_file, recommender, args.k, args.output)
    except ValueError as error:
        parser.error(str(error))

    print(f"Wrote {count} users' predictions to {args.output}")


if __name__ == "__main__":
    main()
