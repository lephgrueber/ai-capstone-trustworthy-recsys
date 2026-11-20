"""History-aware viewer tower inputs (M3 feedback §2)."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.harness import RecommendationRequest
from trustworthy_recsys.retrieval.index import ExactIndex
from trustworthy_recsys.retrieval.model import HistoryConfig, TwoTower, history_batch
from trustworthy_recsys.retrieval.recommender import RetrievalRecommender, request_from_record
from trustworthy_recsys.retrieval.train import PairHistory

DAY = 86_400
# Movies 0 and 1 share every genre (the Need for Speed / Dark Knight case).
FEATURES = np.array([[.5, .5, 0], [.5, .5, 0], [0, 0, 1]], dtype="float32")


def vectors(history):
    torch.manual_seed(0)
    model = TwoTower(FEATURES, embedding_dim=4, hidden_dim=8, identity_dim=2, history=history).eval()
    inputs = SimpleNamespace(item_ids=["a", "b", "c"], item_map={"a": 0, "b": 1, "c": 2}, user_map={},
                             item_features=FEATURES, positive_counts=np.ones(3))
    rec = RetrievalRecommender(inputs, model, ExactIndex(model.all_item_vectors()))
    return [rec.query_vector([item], RecommendationRequest("x", [item], as_of=0)) for item in ("a", "b")]


def test_learned_favorites_separate_movies_with_identical_genres():
    first, second = vectors(None)
    assert np.allclose(first, second)
    first, second = vectors({"favorites": True})
    assert not np.allclose(first, second)


@pytest.mark.parametrize("pooling,expected", [
    ("mean", [.5, .5]),
    ("rating", [.5 / 2, 1.5 / 2]),            # 4.0 -> 0.5, 5.0 -> 1.5
    ("recency", [.5 / 1.5, 1 / 1.5]),         # one half-life old vs. today
])
def test_pooling_weights_follow_stated_formulas(pooling, expected):
    config = HistoryConfig(favorites=True, pooling=pooling, half_life_days=10)
    model = TwoTower(FEATURES, history=config)
    batch = history_batch(config, 3, [[0, 1]], [[4.0, 5.0]], [[0, 10 * DAY]], [10 * DAY], [[]])
    weights = model.favorite_weights(batch)[0]
    assert weights[:2].tolist() == pytest.approx(expected) and weights[2:].sum() == 0


def test_learned_pooling_starts_uniform_and_missing_values_have_defaults():
    config = HistoryConfig(favorites=True, pooling="learned")
    batch = history_batch(config, 3, [[0, 1]], [[np.nan, 5.0]], [[np.nan, 0]], [5 * DAY], [[2]])
    assert batch["ratings"][0, :2].tolist() == [4.0, 5.0] and batch["age_days"][0, 0] == 0
    assert TwoTower(FEATURES, history=config).favorite_weights(batch)[0, :2].tolist() == pytest.approx([.5, .5])


def test_pair_dislikes_are_strictly_earlier_than_each_target():
    inputs = SimpleNamespace(
        has_preference_history=True, item_ids=["a", "b", "c", "d"],
        pair_users=np.array([0, 0, 1]), context_ends=np.array([1, 2, 3]),
        target_times=np.array([5, 9, 5]), positive_offsets=np.array([0, 2, 4]),
        positive_items=np.array([0, 1, 0, 2]), positive_ratings=np.array([4, 5, 4, 4.5], dtype="float32"),
        positive_times=np.array([1, 4, 1, 2]),
        negative_offsets=np.array([0, 2, 2]), negative_items=np.array([3, 2]), negative_times=np.array([5, 7]))
    history = PairHistory(inputs, HistoryConfig(dislikes=True))
    batch = history.batch(np.array([0, 1, 2]))
    # Pair 0 targets t=5: the dislike at t=5 is not strictly earlier.
    assert batch["dislike_mask"].sum(1).tolist() == [0, 2, 0]
    assert history.latest_dislike(np.array([0, 1, 2])).tolist() == [-1, 2, -1]
    assert batch["mask"].sum(1).tolist() == [1, 2, 1]


def test_request_from_record_uses_inputs_only():
    record = {"user_id": "1", "history_items": ["a"], "history_ratings": [5.0], "history_times": [3],
              "other_rated_items": ["b", "c"], "other_rated_ratings": [1.5, 3.0], "relevant_items": ["d"]}
    request = request_from_record(record, as_of=10)
    assert request.disliked_items == ("b",) and request.exclude_items == ("b", "c")
    assert "d" not in request.history_items + list(request.exclude_items)


def test_interest_groups_separate_distinct_genres():
    config = HistoryConfig(favorites=True, interests=2)
    features = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype="float32")
    model = TwoTower(features, history=config)
    batch = history_batch(config, 4, [[0, 2, 1, 3]], [[4, 4, 4, 4]], [[0, 0, 0, 0]], [0], [[]])
    groups = model.interest_groups(batch)[0, :4]
    # Movies 0/1 (first genre) and 2/3 (second genre) land in different interests.
    assert groups[0].argmax() == groups[2].argmax() and groups[1].argmax() == groups[3].argmax()
    assert groups[0].argmax() != groups[1].argmax()
    single = history_batch(config, 4, [[0]], [[4]], [[0]], [0], [[]])
    out = model.encode_user(torch.zeros(1, 3), single)
    assert torch.isfinite(out).all()


def test_target_ratings_found_in_positive_history():
    from trustworthy_recsys.retrieval.train import target_ratings
    inputs = SimpleNamespace(has_preference_history=True, positive_offsets=np.array([0, 2, 4]),
                             positive_items=np.array([5, 7, 7, 1]), positive_ratings=np.array([4, 5, 4.5, 4.0]),
                             pair_users=np.array([0, 1, 1]), target_items=np.array([7, 7, 1]))
    assert target_ratings(inputs).tolist() == [5, 4.5, 4.0]
