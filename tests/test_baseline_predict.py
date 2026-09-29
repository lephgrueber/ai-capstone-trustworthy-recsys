import hashlib
import json

import pandas as pd
import pytest

from eval.harness import (
    SavedPredictionsRecommender,
    evaluate,
    load_evaluation_examples,
    load_saved_predictions,
    validate_prediction_users,
)
from trustworthy_recsys.baselines.predict import MODELS, main, training_cutoff
from trustworthy_recsys.data.pipeline import run_pipeline


def _stamp(day):
    return int(pd.Timestamp(f"2020-01-{day:02d}", tz="UTC").timestamp())


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A real pipeline run over a tiny MovieLens-shaped fixture."""

    root = tmp_path_factory.mktemp("pipeline")
    raw = root / "raw"
    raw.mkdir()
    # (user, movie, rating, day): training before day 3, validation days 3-4,
    # test from day 5.
    ratings = [
        (1, 1, 5, 1), (1, 2, 2, 2), (1, 3, 4, 3), (1, 4, 5, 5),
        (2, 1, 4, 1), (2, 3, 2, 2), (2, 4, 4, 4), (2, 5, 2, 6),
        (3, 1, 5, 2), (3, 2, 4, 2), (3, 3, 5, 5), (4, 1, 2, 1), (4, 3, 5, 5),
    ]
    (raw / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        + "".join(f"{u},{i},{r},{_stamp(d)}\n" for u, i, r, d in ratings),
        encoding="utf-8",
    )
    (raw / "movies.csv").write_text(
        "movieId,title,genres\n" + "".join(f"{i},Movie {i} (2000),Drama\n" for i in range(1, 7)),
        encoding="utf-8",
    )
    (raw / "links.csv").write_text(
        "movieId,imdbId,tmdbId\n" + "".join(f"{i},00{i},{i}\n" for i in range(1, 7)),
        encoding="utf-8",
    )
    (raw / "tags.csv").write_text(
        f"userId,movieId,tag,timestamp\n1,1,thoughtful,{_stamp(1)}\n", encoding="utf-8"
    )
    checksums = root / "checksums.txt"
    checksums.write_text(
        "".join(
            f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(raw.glob("*.csv"))
        ),
        encoding="utf-8",
    )
    readme = root / "README.txt"
    readme.write_text("Synthetic MovieLens-shaped test fixture; no real users.\n", encoding="utf-8")

    output = root / "run"
    run_pipeline(
        raw, output, checksums, readme,
        explicit=("2020-01-03", "2020-01-05"), chunk_size=2, dataset_name="test-fixture",
    )
    return output


def test_training_cutoff_reads_validation_start(run_dir):
    assert training_cutoff(run_dir, "calendar") == pd.Timestamp("2020-01-03", tz="UTC")


def test_training_cutoff_rejects_unknown_scenario(run_dir):
    with pytest.raises(ValueError, match="no split named"):
        training_cutoff(run_dir, "missing")


@pytest.mark.parametrize("model", MODELS)
def test_predictions_score_through_the_harness(run_dir, tmp_path, model):
    output = tmp_path / f"{model}.jsonl"

    main([
        "--run-dir", str(run_dir), "--scenario", "calendar", "--population", "test_all",
        "--model", model, "--k", "5", "--window-days", "1", "--output", str(output),
    ])

    examples = load_evaluation_examples(run_dir / "calendar" / "test_all.jsonl")
    predictions = load_saved_predictions(output)
    validate_prediction_users(examples, predictions)
    aggregate, _ = evaluate(examples, SavedPredictionsRecommender(predictions), recall_k=5, ndcg_k=5)
    assert set(aggregate) >= {"recall@5", "ndcg@5"}

    train_items = set(json.loads((run_dir / "calendar" / "train_item_ids.json").read_text()))
    histories = {example.user_id: set(example.history_items) for example in examples}
    for user_id, ranked in predictions.items():
        assert ranked, f"{model} gave {user_id} no recommendations"
        assert set(ranked) <= train_items
        assert not set(ranked) & histories[user_id]


def test_recent_popularity_uses_only_the_window_before_cutoff(run_dir, tmp_path):
    output = tmp_path / "recent.jsonl"

    main([
        "--run-dir", str(run_dir), "--scenario", "calendar", "--model", "popularity_recent",
        "--k", "5", "--window-days", "1", "--output", str(output),
    ])

    # The one-day window [2020-01-02, 2020-01-03) holds movie 2 twice and
    # movies 1 and 3 once each, so movie 2 leads even though movie 1 is the
    # most-rated training movie; 1 beats 3 on all-time count. User 4 has no
    # positive history, so nothing is filtered for them.
    predictions = load_saved_predictions(output)
    assert predictions["4"][:3] == ["2", "1", "3"]


def test_rejects_incomplete_run(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "--run-dir", str(tmp_path), "--scenario", "calendar", "--model", "popularity",
            "--output", str(tmp_path / "out.jsonl"),
        ])
