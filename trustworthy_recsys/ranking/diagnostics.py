"""Which movies and viewers benefit from a model (feedback §5, §8).

Uses the same personalized dev requests (viewers with at least one earlier favorite) for
every model. For each viewer and group it measures the fraction of their relevant later
likes found (a) anywhere in the candidate pool, (b) in the first K by two-tower order and
(c) in the first K after reranking. A movie group counts only the viewer's likes in that
group; a viewer group counts all their likes. Scores average over viewers with at least
one such like (zero hits included), and every row reports viewer and like counts.

Movie groups: popularity (top 10% of catalog movies by Train A + B likes, i.e. likes
before the feature cutoff, vs the rest), each listed genre (a comedy-drama counts in both),
tagged vs untagged and runtime known vs unknown at the cutoff.
Viewer groups: usable-favorite history length; decade of the median release year of their
earlier favorites (viewers without known years are counted separately); interest range,
the share of their genre profile taken by their most common genre.

Also reported: catalog coverage of the displayed lists, and for dislike experiments the
share of displayed movies the viewer later rated low (<= the dislike threshold) next to the
share they rated at all. If both fall, the model may simply be showing movies whose
reception is unknown (incomplete feedback).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from trustworthy_recsys.data.reporting import code_provenance, write_json
from trustworthy_recsys.data.windows import history_bucket, load_records
from .features import release_year
from .train import lightgbm_threads


def interest_bucket(share):
    return "single_interest_ge_0.5" if share >= .5 else "mixed_0.3_0.5" if share >= .3 else "broad_lt_0.3"


def era(years):
    known = [y for y in years if np.isfinite(y)]
    return "no_known_years" if not known else f"{int(np.median(known)) // 10 * 10}s"


def diagnose(table_dir, model_dir, run_dir, k=12, metadata_dir=None, tags_dir=None, dislike_threshold=2.0,
             scenario="ratio_80_10_10"):
    table_dir, model_dir = Path(table_dir), Path(model_dir)
    manifest = json.loads((table_dir / "manifest.json").read_text())
    training = json.loads((model_dir / "training.json").read_text())
    features = training["features"]
    table = pq.read_table(table_dir / "rows.parquet",
                          columns=list(dict.fromkeys(["user_id", "movie_id", "retrieval_rank", *features])))
    data = {c: table[c].to_numpy(zero_copy_only=False) for c in table.column_names}
    threads = lightgbm_threads(4)  # must precede Booster creation (macOS OpenMP conflict with torch)
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    scores = booster.predict(np.column_stack([data[f] for f in features]), num_threads=threads)

    movies = pq.read_table(Path(run_dir) / "movies.parquet", columns=["movie_id", "title", "genres"]).to_pandas()
    genres = dict(zip(movies["movie_id"], map(list, movies["genres"])))
    years = {m: release_year(t) for m, t in zip(movies["movie_id"], movies["title"])}
    cutoff = int(__import__("pandas").Timestamp(manifest["feature_cutoff_utc"]).timestamp())

    # Popularity of every movie from likes before the feature cutoff (Train A + B for dev).
    from trustworthy_recsys.data.windows import load_events
    events = load_events(run_dir, scenario, tuple(manifest.get("feature_partitions", ["train"])), end=cutoff)
    liked = events.item[events.rating >= 4.0]
    ids, counts = np.unique(liked, return_counts=True)
    likes = dict(zip(map(str, ids), counts))
    ranked = sorted(likes, key=likes.get, reverse=True)
    top = set(ranked[: max(1, len(ranked) // 10)])

    tagged = runtime_known = None
    if tags_dir is not None:
        import pandas as pd
        pairs = pd.read_parquet(Path(tags_dir) / "movie_phrases.parquet")
        tagged = set(pairs.loc[pairs["first_used"] < cutoff, "movie_id"])
    if metadata_dir is not None:
        from trustworthy_recsys.data.metadata import load
        frame = load(metadata_dir)
        runtime_known = set(frame.index[frame["runtime_minutes"].notna()])

    def movie_groups(movie):
        out = ["popular_top10pct" if movie in top else "less_popular"]
        out += [f"genre_{g}" for g in genres.get(movie, [])] or ["genre_(none)"]
        if tagged is not None:
            out.append("tagged" if movie in tagged else "untagged")
        if runtime_known is not None:
            out.append("runtime_known" if movie in runtime_known else "runtime_missing")
        return out

    change = np.flatnonzero(data["user_id"][1:] != data["user_id"][:-1]) + 1
    bounds = zip(np.concatenate([[0], change]), np.concatenate([change, [len(scores)]]))
    blocks = {data["user_id"][lo]: (lo, hi) for lo, hi in bounds}
    rows = defaultdict(lambda: {"viewers": 0, "likes": 0, "pool": [], "before": [], "after": []})
    displayed_after, displayed_before = set(), set()
    feedback = {"after": [], "before": []}
    for record in load_records(manifest["examples"]):
        history = record["history_items"]
        if not history or record["user_id"] not in blocks:
            continue  # personalized requests only
        lo, hi = blocks[record["user_id"]]
        movies_block = data["movie_id"][lo:hi]
        pool = set(movies_block)
        before = list(movies_block[np.argsort(data["retrieval_rank"][lo:hi], kind="stable")][:k])
        after = list(movies_block[np.argsort(-scores[lo:hi], kind="stable")][:k])
        displayed_before.update(before)
        displayed_after.update(after)
        relevant = record["relevant_items"]
        low = {m for m, r in zip(record.get("label_other_items", []), record.get("label_other_ratings", []))
               if r <= dislike_threshold}
        rated = set(relevant) | set(record.get("label_other_items", []))
        for name, shown in (("after", after), ("before", before)):
            feedback[name].append((len(low & set(shown)) / k, len(rated & set(shown)) / k))

        profile = defaultdict(float)
        for movie in history:
            for g in genres.get(movie, []):
                profile[g] += 1 / max(1, len(genres.get(movie, [])))
        share = max(profile.values()) / sum(profile.values()) if profile else 0.0
        viewer_groups = [f"viewer_{history_bucket(len(history))}", f"viewer_era_{era([years.get(m, np.nan) for m in history])}",
                         f"viewer_interest_{interest_bucket(share)}"]
        by_group = defaultdict(list)
        for movie in relevant:
            for group in movie_groups(movie):
                by_group[f"movie_{group}"].append(movie)
        for group in viewer_groups:
            by_group[group] = list(relevant)
        by_group["all_personalized"] = list(relevant)
        for group, wanted in by_group.items():
            wanted = set(wanted)
            row = rows[group]
            row["viewers"] += 1
            row["likes"] += len(wanted)
            row["pool"].append(len(wanted & pool) / len(wanted))
            row["before"].append(len(wanted & set(before)) / len(wanted))
            row["after"].append(len(wanted & set(after)) / len(wanted))

    def summary(row):
        return {"viewers": row["viewers"], "likes": row["likes"],
                "candidate_pool_recall": float(np.mean(row["pool"])),
                f"recall_at_{k}_two_tower": float(np.mean(row["before"])),
                f"recall_at_{k}_reranked": float(np.mean(row["after"]))}

    report = {
        "table": str(table_dir), "model": str(model_dir), "k": k,
        "population": "personalized dev requests (at least one earlier favorite)",
        "popular_definition": f"top 10% of {len(ranked):,} liked movies by likes before {manifest['feature_cutoff_utc']}",
        "groups": {name: summary(row) for name, row in sorted(rows.items())},
        "catalog_coverage": {"two_tower": len(displayed_before), "reranked": len(displayed_after),
                             "note": f"distinct movies in any personalized top-{k} list"},
        "incomplete_feedback": {name: {"share_displayed_later_rated_low": float(np.mean([x[0] for x in v])),
                                       "share_displayed_later_rated_at_all": float(np.mean([x[1] for x in v]))}
                                for name, v in feedback.items()},
        "dislike_threshold": dislike_threshold,
        "code": code_provenance(),
    }
    return report


def format_table(report):
    k = report["k"]
    lines = [f"| Group | Viewers | Likes | Pool recall | R@{k} two-tower | R@{k} reranked |", "|---|---:|---:|---:|---:|---:|"]
    for name, row in report["groups"].items():
        lines.append(f"| {name} | {row['viewers']:,} | {row['likes']:,} | {row['candidate_pool_recall']:.4f} | "
                     f"{row[f'recall_at_{k}_two_tower']:.4f} | {row[f'recall_at_{k}_reranked']:.4f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-dir", type=Path, required=True, help="Dev table built from a window export")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--metadata-dir", type=Path)
    p.add_argument("--tags-dir", type=Path)
    p.add_argument("--scenario", default="ratio_80_10_10")
    a = p.parse_args()
    report = diagnose(a.table_dir, a.model_dir, a.run_dir, a.k, a.metadata_dir, a.tags_dir, scenario=a.scenario)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, report)
    print(format_table(report))
    print(json.dumps({k: report[k] for k in ("catalog_coverage", "incomplete_feedback")}, indent=2))


if __name__ == "__main__":
    main()
