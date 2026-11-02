"""Candidate-recall depth measurement with a stub retriever."""

from types import SimpleNamespace

import pytest

pytest.importorskip('torch')

from eval.harness import EvalExample
from trustworthy_recsys.retrieval.candidate_recall import format_table, measure


class StubRecommender:
    def __init__(self):
        self.inputs = SimpleNamespace(item_ids=['a', 'b', 'c', 'd', 'e'])

    def excluded(self, request):
        return {self.inputs.item_ids.index(x) for x in request.history_items}

    def popularity_indices(self, excluded, k):
        return [i for i in range(5) if i not in excluded][:k]

    def recommend_indices(self, request, k):
        if not request.history_items:
            return self.popularity_indices(set(), k), True
        return [i for i in (4, 3, 2, 1, 0) if i not in self.excluded(request)][:k], False


def test_recall_grows_with_depth_and_splits_paths():
    examples = [
        EvalExample('u1', ['a'], ['c', 'zz']),  # learned path; 'zz' is outside the catalog
        EvalExample('u2', [], ['b']),           # popularity fallback
    ]
    result = measure(StubRecommender(), examples, [1, 3, 2])
    assert result['ks'] == [1, 2, 3]
    learned = result['metrics']['two_tower/learned']
    assert learned['users'] == 1
    assert (learned['recall_at_1'], learned['recall_at_2'], learned['recall_at_3']) == (0, 0, 0.5)
    fallback = result['metrics']['two_tower/fallback']
    assert (fallback['recall_at_1'], fallback['recall_at_2']) == (0, 1)
    # Popularity uses the same exclusions as the model.
    pop = result['metrics']['popularity/history_1_20']
    assert (pop['recall_at_1'], pop['recall_at_2']) == (0, 0.5)
    assert result['metrics']['two_tower/all']['users'] == 2
    assert result['mean_in_catalog_label_fraction'] == 0.75
    assert '| two_tower/all | 2 |' in format_table(result)


def test_rejects_nonpositive_depths():
    with pytest.raises(ValueError):
        measure(StubRecommender(), [], [0, 10])
