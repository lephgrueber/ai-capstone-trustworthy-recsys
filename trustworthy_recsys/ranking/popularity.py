"""Recency-weighted popularity for viewers without usable favorites (feedback §8).

When a visitor has no usable favorites, retrieval falls back to popularity. Three rankings
of the catalog are compared, all built from likes (rating >= 4) before the dev start, with
age measured from the dev start:

- all_time: number of likes (current behavior).
- recent_<D>d: likes within the last D days (the repository's recent-window count).
- half_life_<H>d: each like weighted 0.5 ** (age_days / H); a like H days old counts half.

Each ranking supplies the top `depth` candidates (the viewer's rated movies excluded) for
the same dev viewers with no usable favorites; the catalog, depth and fitted reranker stay
fixed. Final-list NDCG@K is reported before and after reranking, with denominators counting
all later likes.

    python -m trustworthy_recsys.ranking.popularity --run-dir <run> --inputs-dir <Train A inputs> \\
        --reranker-dir <reranker> --examples <windows>/dev.jsonl --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from trustworthy_recsys.data.reporting import code_provenance, write_json
from trustworthy_recsys.data.retrieval_inputs import ModelInputs
from trustworthy_recsys.data.windows import load_events, load_records, read_window_meta
from .features import FEATURE_NAMES, RankingFeatures, rating_statistics_from_record
from .train import lightgbm_threads, ndcg_by_group

SECONDS_PER_DAY = 86_400


def popularity_scores(events, item_map, n_items, as_of, recent_days=(365,), half_lives=(90, 180, 365, 730),
                      threshold=4.0):
    """Score arrays (catalog order) for every popularity variant."""
    liked = events.rating >= threshold
    index = np.array([item_map.get(str(int(i)), -1) for i in events.item[liked]], dtype=np.int64)
    known = index >= 0
    index, age = index[known], (as_of - events.ts[liked][known]) / SECONDS_PER_DAY
    out = {"all_time": np.bincount(index, minlength=n_items).astype("float64")}
    for days in recent_days:
        out[f"recent_{days}d"] = np.bincount(index[age <= days], minlength=n_items).astype("float64")
    for half_life in half_lives:
        out[f"half_life_{half_life}d"] = np.bincount(index, weights=0.5 ** (age / half_life), minlength=n_items)
    return out


def ranked(scores):
    # Ties broken by catalog index, matching the retriever's popularity order.
    return np.lexsort((np.arange(len(scores)), -scores))


def run(run_dir, inputs_dir, reranker_dir, examples, output, depth=500, k=12, scenario="ratio_80_10_10",
        half_lives=(90, 180, 365, 730), max_viewers=0):
    window = read_window_meta(examples)
    if window is None:
        raise ValueError("Use a window export (e.g. dev.jsonl)")
    as_of = window["history_end"]
    inputs = ModelInputs(inputs_dir)
    training = json.loads((Path(reranker_dir) / "training.json").read_text())
    features = RankingFeatures.from_run(run_dir, scenario, inputs.item_ids, inputs.item_features, as_of,
                                        partitions=tuple(window["partitions"]))
    threads = lightgbm_threads(4)
    booster = lgb.Booster(model_file=str(Path(reranker_dir) / "model.txt"))
    columns = [FEATURE_NAMES.index(name) for name in training["features"]]
    events = load_events(run_dir, scenario, tuple(window["partitions"]), end=as_of)
    variants = popularity_scores(events, features.item_map, len(inputs.item_ids), as_of, half_lives=half_lives)
    orders = {name: ranked(scores) for name, scores in variants.items()}

    records = [r for r in load_records(examples) if not features.history_indices(r["history_items"]).size]
    if max_viewers:
        records = records[:max_viewers]
    results = {name: {"before": [], "after": []} for name in orders}
    for record in records:
        seen = {features.item_map[m] for m in record.get("other_rated_items", []) if m in features.item_map}
        relevant = set(record["relevant_items"])
        stats = rating_statistics_from_record(record, as_of)
        for name, order in orders.items():
            candidates = [int(i) for i in order[: depth + len(seen)] if int(i) not in seen][:depth]
            labels = np.array([features.item_ids[i] in relevant for i in candidates], dtype=float)
            matrix = features.build([], candidates, None, record["user_id"], stats)
            scores = booster.predict(matrix[:, columns], num_threads=threads)
            groups = np.zeros(len(candidates), dtype=np.int64)
            count = np.array([len(relevant)])
            results[name]["before"].append(ndcg_by_group(groups, labels, -np.arange(len(candidates), dtype=float), count, k)[0])
            results[name]["after"].append(ndcg_by_group(groups, labels, scores, count, k)[0])
    report = {
        "viewers": len(records), "population": "dev viewers with no usable earlier favorites",
        "as_of_utc": window["history_end_utc"], "depth": depth, "k": k, "reranker": str(reranker_dir),
        "variants": {name: {f"ndcg_at_{k}_popularity_order": float(np.mean(v["before"])),
                            f"ndcg_at_{k}_reranked": float(np.mean(v["after"]))} for name, v in results.items()},
        "code": code_provenance(),
    }
    write_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--reranker-dir", type=Path, required=True)
    p.add_argument("--examples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth", type=int, default=500)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--half-lives", type=int, nargs="+", default=[90, 180, 365, 730])
    p.add_argument("--max-viewers", type=int, default=0)
    a = p.parse_args()
    report = run(a.run_dir, a.inputs_dir, a.reranker_dir, a.examples, a.output, a.depth, a.k,
                 half_lives=tuple(a.half_lives), max_viewers=a.max_viewers)
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
