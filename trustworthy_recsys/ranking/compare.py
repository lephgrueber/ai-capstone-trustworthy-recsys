"""Paired comparison of two rerankers on the same candidate table (M3 feedback §8).

Both versions score the *same* viewers and candidate lists. For every viewer the
difference (B minus A) in end-to-end NDCG@K and Recall@K is computed; viewers are
resampled with replacement to give a 95% interval for the mean difference, overall and
per usable-favorite group. ``pass_through`` (two-tower order) can stand in for either
version. Repeated fits with different seeds can be passed as several directories per
side: the per-viewer score is averaged across seeds and the run-to-run spread reported.

    python -m trustworthy_recsys.ranking.compare --table-dir <dev table> \\
        --a <reranker dirs...> --b <reranker dirs...> --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json
from .evaluate import HISTORY_GROUPS, history_bucket
from .features import AVAILABILITY
from .labels import graded_gain, graded_ndcg, table_ratings
from .train import group_ids, lightgbm_threads, ndcg_by_group

PASS_THROUGH = "pass_through"


def load_table(table_dir, features):
    manifest = json.loads((Path(table_dir) / "manifest.json").read_text())
    if hashes(Path(table_dir) / "rows.parquet")["sha256"] != manifest["rows_sha256"]:
        raise ValueError("Reranker table hash mismatch")
    columns = list(dict.fromkeys(["user_id", "label", "user_relevant_count", "retrieval_rank",
                                  "user_log_history_length", *features]))
    table = pq.read_table(Path(table_dir) / "rows.parquet", columns=columns)
    return manifest, {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}


def model_features(model_dir):
    if model_dir == PASS_THROUGH:
        return []
    model_dir = Path(model_dir)
    for name, digest in json.loads((model_dir / "manifest.json").read_text())["outputs"].items():
        if hashes(model_dir / name)["sha256"] != digest:
            raise ValueError(f"Reranker artifact hash mismatch: {model_dir / name}")
    return json.loads((model_dir / "training.json").read_text())["features"]


def scores_for(model_dir, data):
    if model_dir == PASS_THROUGH:
        return -data["retrieval_rank"].astype(float)
    features = model_features(model_dir)
    threads = lightgbm_threads(4)  # must precede Booster creation (macOS OpenMP conflict with torch)
    booster = lgb.Booster(model_file=str(Path(model_dir) / "model.txt"))
    return booster.predict(np.column_stack([data[name] for name in features]), num_threads=threads)


def masked(data, hidden):
    """Copy of the columns with unavailable rating-history values set to NaN."""
    if not hidden:
        return data
    out = dict(data)
    for name in hidden:
        if name in out:
            out[name] = np.full(len(out[name]), np.nan, dtype=out[name].dtype)
    return out


def recall_by_group(groups, labels, scores, relevant, k):
    order = np.lexsort((-scores, groups))
    g = groups[order]
    starts = np.searchsorted(g, np.arange(len(relevant)))
    top = (np.arange(len(g)) - starts[g]) < k
    return np.bincount(g[top], weights=labels[order][top], minlength=len(relevant)) / relevant


def per_viewer(model_dirs, data, groups, relevant, k, graded=None):
    labels = data["label"].astype(float)
    runs = []
    for model_dir in model_dirs:
        scores = scores_for(model_dir, data)
        run = {"ndcg": ndcg_by_group(groups, labels, scores, relevant, k),
               "recall": recall_by_group(groups, labels, scores, relevant, k)}
        if graded is not None:
            run["graded_ndcg"] = graded_ndcg(groups, graded[0], scores, graded[1], k)
        runs.append(run)
    return runs


def bootstrap(diff, samples, rng):
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(samples)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_difference": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "viewers": int(len(diff)), "viewers_better": int((diff > 0).sum()),
            "viewers_worse": int((diff < 0).sum())}


def compare(table_dir, a_dirs, b_dirs, k=12, samples=2000, seed=0, output=None, availability=None,
            availability_a=None):
    """`availability`: AVAILABILITY profile for side B (and A unless `availability_a` is given);
    each side sees only the rating-history values its app version could calculate."""
    a_dirs = [str(x) for x in a_dirs]
    b_dirs = [str(x) for x in b_dirs]
    features = sorted({f for d in a_dirs + b_dirs for f in model_features(d)})
    manifest, data = load_table(table_dir, features)
    hidden = AVAILABILITY[availability] if availability else ()
    availability_a = availability if availability_a is None else availability_a
    hidden_a = AVAILABILITY[availability_a] if availability_a else ()
    groups, sizes = group_ids(data["user_id"])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    relevant = data["user_relevant_count"][starts].astype(np.int64)
    users_path = Path(table_dir) / "users.parquet"
    if users_path.exists():
        viewers = pq.read_table(users_path, columns=["user_id", "history_bucket"]).to_pandas().set_index("user_id")
        buckets = viewers.loc[data["user_id"][starts], "history_bucket"].to_numpy()
    else:
        buckets = history_bucket(data["user_log_history_length"][starts])
    graded = None
    try:
        movies = pq.read_table(Path(table_dir) / "rows.parquet", columns=["movie_id"])["movie_id"].to_numpy(zero_copy_only=False)
        row_ratings, ideal = table_ratings(table_dir, data["user_id"], movies)
        graded = (graded_gain(row_ratings) * (row_ratings >= 4.0), [ideal.get(u, np.zeros(0)) for u in data["user_id"][starts]])
    except ValueError as error:
        if "window export" not in str(error):
            raise
    runs = {"a": per_viewer(a_dirs, masked(data, hidden_a), groups, relevant, k, graded),
            "b": per_viewer(b_dirs, masked(data, hidden), groups, relevant, k, graded)}
    rng = np.random.default_rng(seed)
    report = {"table": str(table_dir), "table_rows_sha256": manifest["rows_sha256"], "k": k,
              "a": a_dirs, "b": b_dirs, "bootstrap_samples": samples,
              "availability": {"a": availability_a, "b": availability},
              "missing_at_evaluation": {"a": list(hidden_a), "b": list(hidden)},
              "method": "per-viewer B minus A, viewers resampled with replacement; seed repeats averaged per viewer",
              "scores": {}, "difference": {}, "seed_spread": {}}
    for metric in ("ndcg", "recall", "graded_ndcg"):
        if metric not in runs["a"][0]:
            continue
        mean = {side: np.mean([run[metric] for run in runs[side]], axis=0) for side in runs}
        report["scores"][f"{metric}_at_{k}"] = {side: float(mean[side].mean()) for side in mean}
        report["seed_spread"][f"{metric}_at_{k}"] = {
            side: {"runs": [float(run[metric].mean()) for run in runs[side]],
                   "std": float(np.std([run[metric].mean() for run in runs[side]], ddof=1)) if len(runs[side]) > 1 else None}
            for side in runs}
        diff = mean["b"] - mean["a"]
        report["difference"][f"{metric}_at_{k}"] = {"all": bootstrap(diff, samples, rng), **{
            name: bootstrap(diff[buckets == name], samples, rng) for name in HISTORY_GROUPS if (buckets == name).any()}}
    report["code"] = code_provenance()
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    return report


def summary_line(report):
    k = report["k"]
    row = report["difference"][f"ndcg_at_{k}"]["all"]
    lo, hi = row["ci95"]
    scores = report["scores"][f"ndcg_at_{k}"]
    return (f"NDCG@{k}: A {scores['a']:.4f} -> B {scores['b']:.4f}; "
            f"diff {row['mean_difference']:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}] over {row['viewers']} viewers")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--a", nargs="+", required=True, help=f"Reranker dir(s) or '{PASS_THROUGH}'")
    p.add_argument("--b", nargs="+", required=True, help=f"Reranker dir(s) or '{PASS_THROUGH}'")
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--output", type=Path)
    p.add_argument("--availability", choices=list(AVAILABILITY), help="App version whose missing values to blank")
    a = p.parse_args()
    print(summary_line(compare(a.table_dir, a.a, a.b, a.k, a.bootstrap, output=a.output, availability=a.availability)))


if __name__ == "__main__":
    main()
