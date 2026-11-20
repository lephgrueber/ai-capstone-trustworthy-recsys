"""Reranker feature and setting ablations with paired uncertainty (feedback §2, §3, §7, §8).

Every variant is fitted on the same Train B table and chooses its boosting rounds on the
same dev table, so variants differ only in the named change. Each is compared with a
reference variant on the same dev viewers and candidate lists: per-viewer differences in
end-to-end NDCG@K (and Recall@K, graded NDCG@K) with a viewer-bootstrap 95% interval,
overall and per usable-favorite group. Variants may also:

- drop features ("drop"), keep only listed feature groups ("keep_groups"),
- change LightGBM params ("params", e.g. lambdarank_truncation_level),
- use five-star graded fitting labels ("five_star_weight"),
- mask features during fitting ("mask_features") and evaluate under an app
  availability profile ("availability"; features.AVAILABILITY),
- repeat with several seeds ("seeds") to report run-to-run spread.

    python -m trustworthy_recsys.ranking.ablations --train-table <Train B table> \\
        --dev-table <dev table> --retrieval-model-dir <towers> --variants v.json --output-dir <dir>

Completed variants are skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from trustworthy_recsys.data.reporting import write_json
from .compare import compare
from .features import AVAILABILITY, FEATURE_NAMES

M3_FEATURES = FEATURE_NAMES[:20]
GROUPS = {
    "m3": M3_FEATURES,
    "tower": ("tower_cosine_max", "tower_cosine_mean"),
    "dislikes": tuple(n for n in FEATURE_NAMES if n.startswith("dislike_")),
    "credits": tuple(f"{r}_{s}" for r in ("director", "writer", "cast") for s in ("shared_count", "share")),
    "tags": ("tag_cosine_max", "tag_cosine_mean", "tag_comparable_favorites", "tag_phrase_match_max"),
    "runtime": ("runtime_gap",),
    "since_like": ("user_days_since_like",),
}


def dropped(spec):
    drop = set(spec.get("drop", []))
    if "keep_groups" in spec:
        keep = {name for group in spec["keep_groups"] for name in GROUPS[group]}
        drop |= set(FEATURE_NAMES) - keep
    unknown = drop - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    return sorted(drop)


def fit(name, spec, train_table, dev_table, retrieval_model_dir, root, rounds, threads, k):
    from .train import train
    dirs = []
    for seed in spec.get("seeds", [42]):
        target = root / (name if seed == 42 else f"{name}_seed{seed}")
        if not (target / "training.json").exists() or (target / "INCOMPLETE").exists():
            shutil.rmtree(target, ignore_errors=True)
            train(train_table, target, dropped(spec), holdout=0, seed=seed, rounds=rounds, threads=threads,
                  params=spec.get("params"), retrieval_model_dir=retrieval_model_dir, valid_table_dir=dev_table,
                  display_k=k, mask_features=spec.get("mask_features"),
                  valid_mask=AVAILABILITY[spec["availability"]] if spec.get("availability") else (),
                  five_star_weight=spec.get("five_star_weight"))
        dirs.append(target)
    return dirs


def run(train_table, dev_table, retrieval_model_dir, variants, output_dir, reference, rounds=1000, threads=6, k=12,
        samples=2000):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fitted = {name: fit(name, spec, train_table, dev_table, retrieval_model_dir, root, rounds, threads, k)
              for name, spec in variants.items()}
    rows = []
    for name, spec in variants.items():
        report = compare(dev_table, fitted[reference], fitted[name], k=k, samples=samples,
                         availability=spec.get("availability") or variants[reference].get("availability"))
        training = json.loads((fitted[name][0] / "training.json").read_text())
        diff = report["difference"][f"ndcg_at_{k}"]
        rows.append({"variant": name, "spec": spec, "features": len(training["features"]),
                     "rounds": training["best_iteration"], "scores": report["scores"],
                     "difference_vs_reference": diff, "seed_spread": report["seed_spread"],
                     "graded": report["difference"].get(f"graded_ndcg_at_{k}", {}).get("all")})
        write_json(root / "summary.json", {"reference": reference, "k": k, "rows": rows})
        (root / "summary.md").write_text(table(rows, reference, k), encoding="utf-8")
        print(json.dumps({"variant": name, f"ndcg_at_{k}": report["scores"][f"ndcg_at_{k}"]["b"],
                          "diff": diff["all"]["mean_difference"], "ci95": diff["all"]["ci95"]}), flush=True)
    return rows


def table(rows, reference, k):
    lines = [f"Reference: `{reference}`. Dev set, end-to-end NDCG@{k}; differences are variant minus reference "
             "with viewer-bootstrap 95% intervals.", "",
             f"| Variant | Features | Rounds | NDCG@{k} | Diff (all) | 95% CI | Diff history 0 | Diff history 101+ | Graded diff |",
             "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        d = r["difference_vs_reference"]
        lo, hi = d["all"]["ci95"]
        graded = f"{r['graded']['mean_difference']:+.4f}" if r.get("graded") else "-"
        lines.append(f"| {r['variant']} | {r['features']} | {r['rounds']} | {r['scores'][f'ndcg_at_{k}']['b']:.4f} | "
                     f"{d['all']['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                     f"{d.get('history_0', {}).get('mean_difference', float('nan')):+.4f} | "
                     f"{d.get('history_101_plus', {}).get('mean_difference', float('nan')):+.4f} | {graded} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-table", type=Path, required=True)
    p.add_argument("--dev-table", type=Path, required=True)
    p.add_argument("--retrieval-model-dir", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    run(a.train_table, a.dev_table, a.retrieval_model_dir, json.loads(a.variants.read_text()), a.output_dir,
        a.reference, a.rounds, a.threads, a.k, a.bootstrap)


if __name__ == "__main__":
    main()
