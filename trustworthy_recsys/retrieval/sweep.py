"""Compare two-tower settings on the full dev set (M3 feedback §6).

Each variant changes named settings of ``retrieval.train.train`` (target sampling, epochs,
learning rate, batch size, temperature, vector size, ...) and is trained on the same Train
A inputs, with checkpoints chosen by Recall@500 over every dev viewer. The summary records
the best Recall@500, the epoch and number of weight updates that reached it, training
time and target coverage, so one setting can be varied at a time against the baseline.

    python -m trustworthy_recsys.retrieval.sweep --inputs-dir <Train A inputs> \\
        --selection-examples <windows>/dev.jsonl --output-dir <dir> --variants variants.json

variants.json: {"baseline": {}, "lr_0.004": {"learning_rate": 0.004}, ...}. Completed
variants are skipped, so an interrupted sweep resumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from trustworthy_recsys.data.reporting import write_json

BASE = {"epochs": 8, "batch_size": 256, "learning_rate": 0.002, "embedding_dim": 32, "temperature": 0.1,
        "seed": 42, "threads": 4}


def summarize(name, overrides, log):
    best_epoch = log["selected_epoch"]
    row = next(r for r in log["epochs_log"] if r["epoch"] == best_epoch)
    metric = next(k for k in row if k.startswith("validation_recall_at_"))
    return {"variant": name, "overrides": overrides, "best_recall_at_500": log[f"best_{metric}"],
            "selected_epoch": best_epoch, "updates_to_best": row["updates"],
            "training_seconds": log["training_seconds"],
            "curve": [r[metric] for r in log["epochs_log"]],
            "target_sampling": log.get("target_sampling"), "target_coverage": log.get("target_coverage")}


def run(inputs_dir, selection_examples, output_dir, variants, base=None):
    from .train import train

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, overrides in variants.items():
        target = root / name
        if not (target / "training.json").exists() or (target / "INCOMPLETE").exists():
            shutil.rmtree(target, ignore_errors=True)
            settings = {**BASE, **(base or {}), **overrides}
            train(inputs_dir, target, selection_examples=selection_examples, **settings)
        rows.append(summarize(name, overrides, json.loads((target / "training.json").read_text())))
        write_json(root / "summary.json", rows)
        (root / "summary.md").write_text(table(rows), encoding="utf-8")
        print(json.dumps({k: rows[-1][k] for k in ("variant", "best_recall_at_500", "selected_epoch")}), flush=True)
    return rows


def table(rows):
    lines = ["| Variant | Change | Dev Recall@500 | Epoch | Updates to best | Train s | Examples | Distinct targets | Distinct movies |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        cov = r.get("target_coverage") or {}
        lines.append(f"| {r['variant']} | {json.dumps(r['overrides']) if r['overrides'] else 'baseline'} | "
                     f"{r['best_recall_at_500']:.4f} | {r['selected_epoch']} | {r['updates_to_best']:,} | "
                     f"{r['training_seconds']:.0f} | {cov.get('examples_processed', 0):,} | "
                     f"{cov.get('distinct_viewer_movie_targets', 0):,} | {cov.get('distinct_target_movies', 0):,} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-dir", type=Path, required=True)
    p.add_argument("--selection-examples", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--base", type=json.loads, default=None, help="Settings shared by every variant")
    a = p.parse_args()
    run(a.inputs_dir, a.selection_examples, a.output_dir, json.loads(a.variants.read_text()), a.base)


if __name__ == "__main__":
    main()
