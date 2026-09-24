"""Ranking metrics for recommendation evaluation."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence


ItemId = Hashable


def _validate_inputs(
    relevant_items: Iterable[ItemId],
    ranked_items: Sequence[ItemId],
    k: int,
) -> set[ItemId]:
    """Validate common inputs used by ranking metrics."""

    if k <= 0:
        raise ValueError("k must be greater than 0")

    relevant = set(relevant_items)

    if not relevant:
        raise ValueError("relevant_items must contain at least one item")

    if len(ranked_items) != len(set(ranked_items)):
        raise ValueError("ranked_items must not contain duplicate item IDs")

    return relevant


def recall_at_k(
    relevant_items: Iterable[ItemId],
    ranked_items: Sequence[ItemId],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Recall@K is the fraction of relevant items that appear in the first
    K recommendations.

    Example:
        relevant_items = {"item_3", "item_8"}
        ranked_items = ["item_1", "item_3", "item_4"]

        Recall@3 = 1 / 2 = 0.5
    """

    relevant = _validate_inputs(
        relevant_items=relevant_items,
        ranked_items=ranked_items,
        k=k,
    )

    top_k = set(ranked_items[:k])
    hits = len(relevant.intersection(top_k))

    return hits / len(relevant)


def ndcg_at_k(
    relevant_items: Iterable[ItemId],
    ranked_items: Sequence[ItemId],
    k: int,
) -> float:
    """
    Compute binary NDCG@K.

    Relevant items receive relevance 1.
    Non-relevant items receive relevance 0.

    NDCG rewards relevant items for appearing earlier in the ranking.
    """

    relevant = _validate_inputs(
        relevant_items=relevant_items,
        ranked_items=ranked_items,
        k=k,
    )

    dcg = 0.0

    for rank, item_id in enumerate(ranked_items[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg