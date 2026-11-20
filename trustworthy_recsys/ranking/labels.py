"""Star ratings behind binary labels: graded NDCG and five-star weighting (feedback §4).

A held-out like (rating >= 4) earns the same binary label at 4, 4.5 or 5 stars. Graded
relevance keeps that difference: gain = rating - 3, so 4 -> 1, 4.5 -> 1.5 and 5 -> 2.
Graded NDCG@K uses those gains, with the ideal ordering built from *all* of the viewer's
later likes (including retrieval misses), mirroring the binary end-to-end NDCG.

The ratings come from the window example file a candidate table was built from (its
relevant_items / relevant_ratings), so existing tables need no rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trustworthy_recsys.data.windows import load_records, read_window_meta


def graded_gain(rating):
    return np.maximum(np.asarray(rating, dtype="float64") - 3.0, 0.0)


def table_ratings(table_dir, user_ids, movie_ids):
    """(row ratings aligned with the rows, 0 for non-likes; per-viewer sorted ideal gains)."""
    manifest = json.loads((Path(table_dir) / "manifest.json").read_text())
    examples = Path(manifest["examples"]) if "examples" in manifest else None
    if examples is None or not examples.exists() or read_window_meta(examples) is None:
        raise ValueError("Graded labels need a window export with relevant_ratings")
    ratings, ideal = {}, {}
    for record in load_records(examples):
        values = dict(zip(record["relevant_items"], record["relevant_ratings"]))
        ratings[record["user_id"]] = values
        ideal[record["user_id"]] = np.sort(graded_gain(list(values.values())))[::-1]
    rows = np.array([ratings.get(u, {}).get(m, 0.0) for u, m in zip(user_ids, movie_ids)], dtype="float32")
    return rows, ideal


def graded_ndcg(groups, gains, scores, ideal_by_group, k):
    """Per-group graded NDCG@k (ties keep table order)."""
    order = np.lexsort((-scores, groups))
    g = groups[order]
    starts = np.searchsorted(g, np.arange(len(ideal_by_group)))
    rank = np.arange(len(g)) - starts[g]
    top = rank < k
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.bincount(g[top], weights=gains[order][top] * discounts[rank[top]], minlength=len(ideal_by_group))
    idcg = np.array([(ideal[:k] * discounts[:min(k, len(ideal))]).sum() for ideal in ideal_by_group])
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def five_star_labels(row_ratings):
    """LambdaRank labels: 0 non-like, 1 like below five stars, 2 five-star like."""
    return np.where(row_ratings >= 5.0, 2, np.where(row_ratings >= 4.0, 1, 0)).astype("int32")
