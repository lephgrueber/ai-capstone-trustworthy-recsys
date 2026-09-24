import math

import pytest

from trustworthy_recsys.evaluation.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k_when_relevant_item_is_retrieved():
    relevant = {"item_3"}
    ranked = ["item_1", "item_3", "item_5"]

    assert recall_at_k(relevant, ranked, k=3) == 1.0


def test_recall_at_k_when_relevant_item_is_outside_k():
    relevant = {"item_4"}
    ranked = ["item_1", "item_2", "item_3", "item_4"]

    assert recall_at_k(relevant, ranked, k=3) == 0.0


def test_recall_at_k_with_multiple_relevant_items():
    relevant = {"item_2", "item_4"}
    ranked = ["item_1", "item_2", "item_3"]

    assert recall_at_k(relevant, ranked, k=3) == 0.5


def test_ndcg_at_k_when_relevant_item_is_ranked_first():
    relevant = {"item_3"}
    ranked = ["item_3", "item_1", "item_2"]

    assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_when_relevant_item_is_ranked_second():
    relevant = {"item_3"}
    ranked = ["item_1", "item_3", "item_2"]

    expected = 1.0 / math.log2(3)

    assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(expected)


def test_ndcg_rewards_better_ranking():
    relevant = {"item_3"}

    better_ranking = ["item_3", "item_1", "item_2"]
    worse_ranking = ["item_1", "item_2", "item_3"]

    assert ndcg_at_k(
        relevant,
        better_ranking,
        k=3,
    ) > ndcg_at_k(
        relevant,
        worse_ranking,
        k=3,
    )


def test_ndcg_at_k_is_perfect_with_multiple_relevant_items():
    relevant = {"item_1", "item_2"}
    ranked = ["item_1", "item_2", "item_3"]

    assert ndcg_at_k(relevant, ranked, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_when_k_is_smaller_than_number_of_relevant_items():
    relevant = {"item_1", "item_2", "item_3"}
    ranked = ["item_1", "item_2", "item_3"]

    assert ndcg_at_k(relevant, ranked, k=2) == pytest.approx(1.0)


def test_ndcg_at_k_when_relevant_item_is_outside_k():
    relevant = {"item_3"}
    ranked = ["item_1", "item_2", "item_3"]

    assert ndcg_at_k(relevant, ranked, k=2) == 0.0


def test_metrics_return_zero_for_empty_ranked_list():
    relevant = {"item_1"}

    assert recall_at_k(relevant, [], k=3) == 0.0
    assert ndcg_at_k(relevant, [], k=3) == 0.0


def test_ranking_shorter_than_k_is_scored_at_its_actual_length():
    relevant = {"item_1", "item_2"}
    ranked = ["item_1"]
    expected_ndcg = 1.0 / (1.0 + 1.0 / math.log2(3))

    assert recall_at_k(relevant, ranked, k=5) == 0.5
    assert ndcg_at_k(relevant, ranked, k=5) == pytest.approx(expected_ndcg)


def test_metrics_return_zero_when_there_are_no_hits():
    relevant = {"item_3"}
    ranked = ["item_1", "item_2"]

    assert recall_at_k(relevant, ranked, k=2) == 0.0
    assert ndcg_at_k(relevant, ranked, k=2) == 0.0


def test_recall_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k({"item_1"}, ["item_1"], k=0)


def test_recall_at_k_rejects_empty_relevant_items():
    with pytest.raises(ValueError):
        recall_at_k(set(), ["item_1"], k=10)


def test_ndcg_at_k_rejects_duplicate_ranked_items():
    ranked = ["item_1", "item_1", "item_2"]

    with pytest.raises(ValueError):
        ndcg_at_k({"item_2"}, ranked, k=3)


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_at_k_rejects_non_positive_k(k):
    with pytest.raises(ValueError):
        ndcg_at_k({"item_1"}, ["item_1"], k=k)


def test_ndcg_at_k_rejects_empty_relevant_items():
    with pytest.raises(ValueError):
        ndcg_at_k(set(), ["item_1"], k=10)
