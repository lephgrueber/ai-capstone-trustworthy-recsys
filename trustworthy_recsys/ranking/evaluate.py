"""Compare the LambdaRank reranker with pass-through (two-tower order) on a candidate table.

Metrics follow trustworthy_recsys.evaluation.metrics: macro user means of Recall@K and
binary NDCG@K, where the denominators count *all* of a user's relevant items, including
those retrieval missed. Run on the reranker's own validation table, only its early-stopping
holdout users are scored; run on a test table, every user is scored.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json
from .train import group_sizes, holdout_mask, lightgbm_threads

KS = (10, 20, 50)


def user_metrics(labels, order_scores, sizes, relevant_counts, ks=KS):
    """Per-user Recall@K and NDCG@K for one scoring of every user's candidates.

    Within each user, candidates are sorted by descending score; ties keep table order.
    """
    discounts = 1 / np.log2(np.arange(2, max(ks) + 2))
    ideal = np.cumsum(discounts)
    out = {f"{metric}_at_{k}": np.empty(len(sizes)) for k in ks for metric in ("recall", "ndcg")}
    start = 0
    for user, size in enumerate(sizes):
        block = slice(start, start + size)
        start += size
        ranked = labels[block][np.argsort(-order_scores[block], kind="stable")]
        for k in ks:
            hits = ranked[:k]
            out[f"recall_at_{k}"][user] = hits.sum() / relevant_counts[user]
            out[f"ndcg_at_{k}"][user] = (hits * discounts[:len(hits)]).sum() / ideal[min(relevant_counts[user], k) - 1]
    return out


def paired_bootstrap(new, old, samples=1000, seed=0):
    """95% interval for the mean per-user difference, resampling users."""
    rng = np.random.default_rng(seed)
    diff = new - old
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(samples)])
    return {"mean_difference": float(diff.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "users_improved": int((diff > 0).sum()), "users_worse": int((diff < 0).sum())}


def history_bucket(log_length):
    length = np.rint(np.expm1(log_length))
    return np.where(length <= 20, "history_1_20", np.where(length <= 100, "history_21_100", "history_101_plus"))


def evaluate(table_dir, model_dir, output, ks=KS, bootstrap=1000):
    output = Path(output)
    if output.exists():
        raise ValueError("Output report already exists; choose a new path")
    model_dir, table_dir = Path(model_dir), Path(table_dir)
    training = json.loads((model_dir / "training.json").read_text())
    for name, digest in json.loads((model_dir / "manifest.json").read_text())["outputs"].items():
        if hashes(model_dir / name)["sha256"] != digest:
            raise ValueError(f"Reranker artifact hash mismatch: {name}")
    manifest = json.loads((table_dir / "manifest.json").read_text())
    if hashes(table_dir / "rows.parquet")["sha256"] != manifest["rows_sha256"]:
        raise ValueError("Reranker table hash mismatch")
    if manifest["feature_cutoff_utc"] != training["feature_cutoff_utc"]:
        raise ValueError("Table and model use different feature cutoffs")
    features = training["features"]
    columns = ["user_id", "label", "fallback", "user_relevant_count", "retrieval_rank", "user_log_history_length", *features]
    table = pq.read_table(table_dir / "rows.parquet", columns=list(dict.fromkeys(columns)))
    data = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}

    same_table = manifest["rows_sha256"] == training["table_rows_sha256"]
    if same_table:
        mask = holdout_mask(data["user_id"], training["holdout"]["fraction"], training["holdout"]["seed"])
        data = {name: values[mask] for name, values in data.items()}
    population = "validation early-stopping holdout users" if same_table else "all table users"

    sizes = group_sizes(data["user_id"])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    scores = booster.predict(np.column_stack([data[name] for name in features]), num_threads=lightgbm_threads(4))
    labels = data["label"].astype(float)
    relevant = data["user_relevant_count"][starts]
    methods = {
        "pass_through": user_metrics(labels, -data["retrieval_rank"], sizes, relevant, ks),
        "lambdarank": user_metrics(labels, scores, sizes, relevant, ks),
    }
    slices = {"all": np.ones(len(sizes), dtype=bool),
              "fallback": data["fallback"][starts],
              **{name: history_bucket(data["user_log_history_length"][starts]) == name
                 for name in ("history_1_20", "history_21_100", "history_101_plus")}}
    metrics = {f"{method}/{name}": {"users": int(mask.sum()),
                                    **{metric: float(values[mask].mean()) for metric, values in per_user.items()}}
               for method, per_user in methods.items() for name, mask in slices.items() if mask.any()}
    comparison = {metric: paired_bootstrap(methods["lambdarank"][metric], methods["pass_through"][metric], bootstrap)
                  for metric in methods["lambdarank"]}
    report = {
        "population": population, "users": int(len(sizes)), "rows": int(len(labels)),
        "candidate_ceiling_recall": float((np.add.reduceat(labels, starts) / relevant).mean()),
        "metrics": metrics, "lambdarank_minus_pass_through": comparison,
        "table": str(table_dir), "table_rows_sha256": manifest["rows_sha256"],
        "model": str(model_dir), "model_manifest_sha256": hashes(model_dir / "manifest.json")["sha256"],
        "code": code_provenance(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def format_table(report, ks=KS):
    names = [f"{metric}_at_{k}" for k in ks for metric in ("ndcg", "recall")]
    lines = ["| Method / users | Users | " + " | ".join(n.replace("_at_", "@") for n in names) + " |",
             "|---|---:|" + "---:|" * len(names)]
    for key, row in report["metrics"].items():
        lines.append(f"| {key} | {row['users']} | " + " | ".join(f"{row[n]:.4f}" for n in names) + " |")
    lines.append("")
    lines.append("| LambdaRank − pass-through | mean | 95% CI | users better / worse |")
    lines.append("|---|---:|---|---|")
    for metric, row in report["lambdarank_minus_pass_through"].items():
        lo, hi = row["ci95"]
        lines.append(f"| {metric.replace('_at_', '@')} | {row['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{row['users_improved']} / {row['users_worse']} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args()
    report = evaluate(a.table_dir, a.model_dir, a.output, bootstrap=a.bootstrap)
    print(f"Population: {report['population']}; candidate ceiling Recall: {report['candidate_ceiling_recall']:.4f}")
    print(format_table(report))


if __name__ == "__main__":
    main()
