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


def test_window_recall_with_genre_filter_and_exclusions():
    from types import SimpleNamespace
    from trustworthy_recsys.retrieval.candidate_recall import measure_window

    class Stub:
        inputs = SimpleNamespace(item_ids=["a", "b", "c", "d"])

        def recommend_indices(self, request, k):
            assert request.exclude_items == ("z",) and request.as_of == 100
            return [0, 1, 2, 3][:k], False

    record = {"user_id": "1", "history_items": ["h"], "other_rated_items": ["z"], "other_rated_ratings": [3.0],
              "relevant_items": ["b", "d", "missed"]}
    genres = {"a": {"Comedy"}, "b": {"Comedy"}, "c": {"Drama"}, "d": {"Drama"}, "missed": {"Comedy"}}
    result = measure_window(Stub(), [record], [2, 4], 100, genres, ["Comedy"])
    overall = result["metrics"]["two_tower/all"]
    assert overall["recall_at_2"] == 1 / 3 and overall["recall_at_4"] == 2 / 3  # full denominator
    comedy = result["metrics"]["two_tower/genre_Comedy"]
    # Later comedy likes: b (retrieved) and 'missed' -> half remain after the filter.
    assert comedy["recall_at_4"] == .5 and comedy["users"] == 1
    assert set(result["exact_search_latency_ms_by_depth"]) == {"2", "4"}
