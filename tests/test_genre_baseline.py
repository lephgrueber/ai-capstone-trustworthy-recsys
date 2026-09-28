import numpy as np
import pandas as pd
import pytest

from eval.harness import RecommendationRequest, evaluate, load_synthetic_examples
from trustworthy_recsys.baselines.genre import fit_genre, load_movies
from trustworthy_recsys.baselines.popularity import fit_popularity

MOVIES = pd.DataFrame({
    "movie_id": ["1", "2", "3", "4", "5", "6"],
    "genres": [["Horror"], ["Comedy"], ["Horror"], ["Comedy"], [], ["Horror", "Comedy"]],
})


def _interactions(counts):
    """One 5-star rating per (movie, n) pair, giving each movie `n` ratings."""

    rows = [
        (f"u{n}", movie, 5.0, pd.Timestamp("2024-01-01", tz="UTC"))
        for movie, count in counts.items()
        for n in range(count)
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp_utc"])


# Genre-less "5" is the most popular; comedies "2" and "4" outrank horror
# films "3" and "1"; "6" is both.
INTERACTIONS = _interactions({"5": 12, "2": 10, "4": 8, "3": 6, "1": 4, "6": 2})


def _request(*history):
    return RecommendationRequest(user_id="user", history_items=list(history))


def test_alpha_zero_is_plain_popularity():
    genre = fit_genre(INTERACTIONS, MOVIES, alpha=0)

    assert tuple(genre(_request(), 6)) == fit_popularity(INTERACTIONS).ranking


def test_horror_history_lifts_horror_over_more_popular_comedy():
    recommender = fit_genre(INTERACTIONS, MOVIES, alpha=2)

    ranked = recommender(_request("1"), 3)

    assert ranked[0] == "3"
    assert "1" not in ranked


def test_comedy_history_keeps_comedy_on_top():
    recommender = fit_genre(INTERACTIONS, MOVIES, alpha=2)

    assert recommender(_request("4"), 2) == ["2", "6"]


def test_empty_history_profile_is_training_genre_mix():
    recommender = fit_genre(INTERACTIONS, MOVIES)

    # Genre-less "5" carries no weight; "6" splits its 2 ratings evenly.
    comedy, horror = 10 + 8 + 1, 6 + 4 + 1
    expected = np.array([comedy, horror]) / (comedy + horror)
    np.testing.assert_allclose(recommender.profile([]), expected)


def test_genreless_items_sink_below_genre_matches():
    recommender = fit_genre(INTERACTIONS, MOVIES, alpha=1)

    assert recommender(_request(), 6)[-1] == "5"


def test_recency_decay_favors_newest_history():
    flat = fit_genre(INTERACTIONS, MOVIES, recency_decay=1.0)
    decayed = fit_genre(INTERACTIONS, MOVIES, recency_decay=0.1)
    history = ["1", "3", "2"]  # Two horror films, then a comedy most recently

    comedy = 0  # Genre columns are sorted: Comedy, Horror
    assert flat.profile(history)[comedy] < 0.5 < decayed.profile(history)[comedy]


def test_prior_weight_controls_smoothing():
    light = fit_genre(INTERACTIONS, MOVIES, prior_weight=0.01)
    heavy = fit_genre(INTERACTIONS, MOVIES, prior_weight=100)
    horror = 1

    assert light.profile(["1"])[horror] > 0.99
    np.testing.assert_allclose(heavy.profile(["1"]), heavy.profile([]), atol=0.01)


def test_unknown_history_items_are_ignored():
    recommender = fit_genre(INTERACTIONS, MOVIES)

    np.testing.assert_allclose(recommender.profile(["999"]), recommender.profile([]))


def test_top_k_matches_a_full_sort_with_ties():
    rng = np.random.default_rng(0)
    genres = ["A", "B", "C"]
    movie_ids = [str(i) for i in range(1, 201)]
    movies = pd.DataFrame({
        "movie_id": movie_ids,
        "genres": [list(rng.choice(genres, size=rng.integers(0, 3), replace=False)) for _ in movie_ids],
    })
    # Few distinct counts guarantees many score ties at the top-k cutoff.
    interactions = _interactions({m: int(rng.integers(1, 4)) for m in movie_ids})
    recommender = fit_genre(interactions, movies, alpha=1.5)

    for user in range(20):
        history = list(rng.choice(movie_ids, size=user, replace=False))
        affinity = recommender.item_vectors @ recommender.profile(history)
        scores = affinity**recommender.alpha * recommender.popularity
        full = [
            recommender.items[i]
            for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            if recommender.items[i] not in history
        ]

        assert recommender(_request(*history), 10) == full[:10]


def test_respects_k_and_rejects_non_positive_k():
    recommender = fit_genre(INTERACTIONS, MOVIES)

    assert len(recommender(_request("1"), 2)) == 2
    assert len(recommender(_request("1"), 50)) == 5
    with pytest.raises(ValueError):
        recommender(_request(), 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"alpha": -1}, {"prior_weight": 0}, {"recency_decay": 0}, {"recency_decay": 1.5}],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fit_genre(INTERACTIONS, MOVIES, **kwargs)


def test_rejects_movies_without_any_genres():
    movies = MOVIES.assign(genres=[[] for _ in range(len(MOVIES))])

    with pytest.raises(ValueError):
        fit_genre(INTERACTIONS, movies)


def test_load_movies_round_trips_parquet(tmp_path):
    path = tmp_path / "movies.parquet"
    MOVIES.to_parquet(path)

    recommender = fit_genre(INTERACTIONS, load_movies(path), alpha=2)

    assert recommender(_request("1"), 3) == fit_genre(INTERACTIONS, MOVIES, alpha=2)(_request("1"), 3)


def test_runs_end_to_end_through_the_harness():
    examples = load_synthetic_examples()
    items = sorted({i for e in examples for i in (*e.history_items, *e.relevant_items)})
    movies = pd.DataFrame({"movie_id": items, "genres": [["Drama"] for _ in items]})
    recommender = fit_genre(_interactions({item: 1 for item in items}), movies)

    aggregate, per_example = evaluate(examples, recommender, recall_k=2, ndcg_k=2)

    assert {"recall@2", "ndcg@2"} <= set(aggregate)
    assert len(per_example) == len(examples)
