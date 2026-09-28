import pytest

from eval.harness import RecommendationRequest, evaluate, load_synthetic_examples
from trustworthy_recsys.baselines.random_baseline import RandomRecommender


CATALOG = tuple(f"item_{letter}" for letter in "abcdefghij")


def test_rejects_empty_catalog():
    with pytest.raises(ValueError):
        RandomRecommender(catalog=())


def test_rejects_duplicate_catalog_items():
    with pytest.raises(ValueError):
        RandomRecommender(catalog=("item_a", "item_a"))


def test_rejects_non_positive_k():
    recommender = RandomRecommender(catalog=CATALOG)
    request = RecommendationRequest(user_id="user_1", history_items=[])

    with pytest.raises(ValueError):
        recommender(request, 0)


def test_excludes_history_items_and_respects_k():
    recommender = RandomRecommender(catalog=CATALOG, seed=7)
    request = RecommendationRequest(
        user_id="user_1",
        history_items=["item_a", "item_b", "item_c"],
    )

    ranked = recommender(request, 5)

    assert len(ranked) == 5
    assert len(set(ranked)) == 5
    assert not set(ranked) & {"item_a", "item_b", "item_c"}
    assert set(ranked) <= set(CATALOG)


def test_deterministic_for_same_seed_and_user():
    first = RandomRecommender(catalog=CATALOG, seed=42)
    second = RandomRecommender(catalog=CATALOG, seed=42)
    request = RecommendationRequest(user_id="user_5", history_items=[])

    assert first(request, 10) == second(request, 10)


def test_different_users_get_different_orderings():
    recommender = RandomRecommender(catalog=CATALOG, seed=42)

    ranked_1 = recommender(RecommendationRequest(user_id="user_1", history_items=[]), 10)
    ranked_2 = recommender(RecommendationRequest(user_id="user_2", history_items=[]), 10)

    assert ranked_1 != ranked_2


def test_ordering_independent_of_other_calls():
    recommender = RandomRecommender(catalog=CATALOG, seed=42)
    request = RecommendationRequest(user_id="user_1", history_items=[])

    baseline = recommender(request, 10)

    # Scoring other users in between must not perturb this user's ranking.
    recommender(RecommendationRequest(user_id="user_2", history_items=[]), 10)
    recommender(RecommendationRequest(user_id="user_3", history_items=[]), 10)

    assert recommender(request, 10) == baseline


def test_runs_end_to_end_through_the_harness():
    examples = load_synthetic_examples()
    catalog = sorted({item for example in examples for item in (*example.history_items, *example.relevant_items)})
    recommender = RandomRecommender(catalog=tuple(catalog), seed=0)

    aggregate, per_example = evaluate(examples, recommender, recall_k=2, ndcg_k=2)

    assert "recall@2" in aggregate
    assert "ndcg@2" in aggregate
    assert len(per_example) == len(examples)
