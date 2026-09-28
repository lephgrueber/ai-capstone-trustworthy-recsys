"""Uniform-random baseline recommender.

The floor baseline: recommend movies chosen uniformly at random from a
catalog, skipping ones already in the user's history. Any model worth
deploying should beat it by a wide margin; a model that only slightly
beats it, or doesn't, points to a bug or leakage in the setup rather than
a weak model. Its score also shows how easy the metric is by chance for a
given catalog size and number of relevant items.

The catalog is normally a scenario's train_item_ids.json (see
`load_catalog`), the pipeline's candidate set of movies seen in training.
Each user's shuffle is seeded from (`seed`, user ID), so a user's
recommendations are reproducible and don't depend on which other users
are scored or in what order.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def load_catalog(path: str | Path) -> tuple[str, ...]:
    """Load a candidate item catalog from a train_item_ids.json file."""

    # train_item_ids.json is the data pipeline's documented candidate set
    # for warm evaluation (see DATA_CARD.md), so it doubles as this
    # baseline's default sampling pool.
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(items)


@dataclass(frozen=True)
class RandomRecommender:
    """Recommends uniformly random, unseen catalog items.

    A floor baseline: any model worth deploying should beat picking
    items at random. Each user's ordering is derived from `seed` and
    their own `user_id`, so results are reproducible regardless of call
    order or which other users are scored in the same run.
    """

    catalog: tuple[str, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.catalog:
            raise ValueError("catalog must contain at least one item")
        # A duplicate ID could be sampled twice, and the harness's metrics
        # reject ranked lists that contain repeated item IDs.
        if len(self.catalog) != len(set(self.catalog)):
            raise ValueError("catalog must not contain duplicate item IDs")

    def __call__(self, request, k: int) -> Sequence[str]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        # Never re-recommend items the user has already interacted with.
        seen = set(request.history_items)
        candidates = [item for item in self.catalog if item not in seen]

        # Seeding on (seed, user_id) rather than a single global RNG keeps
        # each user's ranking independent of call order or which other
        # users are scored in the same run.
        rng = random.Random(f"{self.seed}:{request.user_id}")
        rng.shuffle(candidates)

        return candidates[:k]
