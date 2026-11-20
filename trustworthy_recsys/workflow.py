"""Development -> final refit -> deployment fit (M3 feedback §1).

    windows      Train A (towers) | Train B (reranker) | dev (selection) | test (final evaluation)

1. ``development``: towers on Train A (checkpoint chosen by dev Recall@500), reranker on
   Train B candidates from the frozen towers (rounds chosen by dev end-to-end NDCG@K), then
   popularity / two-tower / reranked scores on every dev viewer. Writes ``locked.json``.
2. ``final``: repartition A + B + dev into Final A / Final B with the same A:B ratio, refit
   both stages with the locked settings (no selection), and evaluate once on test.
3. ``deployment``: the same procedure on all observations (Deployment A / B) for the
   browser release; no held-out score exists for this fit.

Each fit is packaged as one version (tower inputs + towers + reranker + manifest) under
``<root>/<stage>/pipeline``. Stages skip steps whose completed output already exists, so an
interrupted run resumes. Run ``python -m trustworthy_recsys.workflow --help``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json

DEFAULT_SETTINGS = {
    "ab_ratio": 7.0,
    "depth": 500,
    "tower": {"max_targets_per_user": 4, "epochs": 8, "batch_size": 256, "learning_rate": 0.002,
              "embedding_dim": 32, "temperature": 0.1, "seed": 42, "threads": 4,
              "history": {}, "dislike_penalty": 0.0, "movie_side": {}},
    "reranker": {"rounds": 1000, "display_k": 12, "drop_features": [], "params": {}, "seed": 42,
                 "metadata_dir": None, "tags_dir": None},
}

STAGES = {
    # stage: (fitting partitions, A window, B example file, evaluation example file)
    "development": (("train",), "train_a", "train_b", "dev"),
    "final": (("train", "validation"), "final_a", "final_b", "test"),
    "deployment": (("train", "validation", "test"), "deployment_a", "deployment_b", None),
}


def complete(path: Path) -> bool:
    return path.exists() and not (path / "INCOMPLETE").exists()


def merged(base, override):
    out = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        out[key] = merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def run_stage(stage, run_dir, root, settings=None, scenario="ratio_80_10_10", windows_dir=None):
    """Run one stage; returns its summary. ``final``/``deployment`` require ``locked.json``."""
    from trustworthy_recsys.data.retrieval_inputs import prepare_inputs
    from trustworthy_recsys.ranking.dataset import build
    from trustworthy_recsys.ranking.evaluate import evaluate
    from trustworthy_recsys.ranking.train import train as train_reranker
    from trustworthy_recsys.retrieval.train import train as train_towers

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    locked_path = root / "locked.json"
    if stage == "development":
        settings = merged(DEFAULT_SETTINGS, settings)
    else:
        if not locked_path.exists():
            raise ValueError("Run the development stage first; final fits reuse its locked settings")
        if settings:
            raise ValueError("Final and deployment fits use the locked settings; do not pass overrides")
        settings = json.loads(locked_path.read_text())
    windows_dir = Path(windows_dir) if windows_dir else root / f"windows_ab{settings['ab_ratio']:g}"
    if not complete(windows_dir):
        from trustworthy_recsys.data.windows import prepare
        prepare(run_dir, scenario, windows_dir, settings["ab_ratio"])
    windows = json.loads((windows_dir / "windows.json").read_text())
    if windows["ab_ratio"] != settings["ab_ratio"]:
        raise ValueError("Window directory uses a different A:B ratio")

    partitions, a_window, b_name, eval_name = STAGES[stage]
    tower, rerank = settings["tower"], settings["reranker"]
    out = root / stage
    out.mkdir(exist_ok=True)
    inputs, towers = out / "inputs_a", out / "towers"
    if not complete(inputs):
        shutil.rmtree(inputs, ignore_errors=True)
        prepare_inputs(run_dir, scenario, inputs, tower["max_targets_per_user"], partitions,
                       windows["windows"][a_window]["end"])
    select = stage == "development"
    if not complete(towers):
        shutil.rmtree(towers, ignore_errors=True)
        train_towers(inputs, towers, epochs=tower["epochs"] if select else tower["selected_epoch"],
                     batch_size=tower["batch_size"], learning_rate=tower["learning_rate"], seed=tower["seed"],
                     embedding_dim=tower["embedding_dim"], threads=tower["threads"],
                     temperature=tower["temperature"], select=select,
                     history=tower["history"] or None, dislike_penalty=tower["dislike_penalty"],
                     movie_side=tower["movie_side"] or None,
                     # Final/deployment vectors use everything fitted on (end of the B window).
                     export_as_of=None if select else windows["windows"][b_name]["end"],
                     selection_examples=windows_dir / "dev.jsonl" if select else None)
    tables = {}
    for name in (b_name, eval_name):
        if name is None:
            continue
        tables[name] = out / f"table_{name}"
        if not complete(tables[name]):
            shutil.rmtree(tables[name], ignore_errors=True)
            build(run_dir, inputs, towers, windows_dir / f"{name}.jsonl", tables[name],
                  depth=settings["depth"], threads=tower["threads"],
                  metadata_dir=rerank.get("metadata_dir"), tags_dir=rerank.get("tags_dir"))
    reranker = out / "reranker"
    if not complete(reranker):
        shutil.rmtree(reranker, ignore_errors=True)
        train_reranker(tables[b_name], reranker, rerank["drop_features"], holdout=0, seed=rerank["seed"],
                       rounds=rerank["rounds"] if select else rerank["selected_rounds"],
                       params=rerank["params"] or None, retrieval_model_dir=towers,
                       valid_table_dir=tables["dev"] if select else None, display_k=rerank["display_k"])
    report = None
    if eval_name is not None:
        report_path = out / f"{eval_name}_report.json"
        if not report_path.exists():
            evaluate(tables[eval_name], reranker, report_path)
        report = json.loads(report_path.read_text())
    if select:
        tower_log = json.loads((towers / "training.json").read_text())
        reranker_log = json.loads((reranker / "training.json").read_text())
        settings["tower"]["selected_epoch"] = tower_log["selected_epoch"]
        settings["reranker"]["selected_rounds"] = reranker_log["best_iteration"]
        settings["selected_on"] = {"towers": tower_log["selection_metric"],
                                   "reranker": reranker_log["selection"]["metric"],
                                   "dev_examples_sha256": windows["exports"]["dev"]["sha256"]}
        write_json(locked_path, settings)
    package(out, stage, settings, windows, inputs, towers, reranker)
    return {"stage": stage, "report": None if report is None else {
        k: report["metrics"][k]["ndcg_at_12"] for k in report["metrics"] if k.endswith("/all")},
        "viewer_groups": None if report is None else report["viewer_groups"]}


def package(out, stage, settings, windows, inputs, towers, reranker):
    """One version: every artifact the pipeline needs, with hashes and the fitting windows."""
    files = {}
    for label, folder in (("inputs", inputs), ("towers", towers), ("reranker", reranker)):
        for name, digest in json.loads((folder / "manifest.json").read_text())["outputs"].items():
            files[f"{label}/{name}"] = digest if isinstance(digest, str) else digest["sha256"]
    write_json(out / "pipeline.json", {
        "stage": stage, "settings": settings,
        "windows": {k: windows["windows"][k] for k in windows["windows"]},
        "artifacts": {"inputs": str(inputs), "towers": str(towers), "reranker": str(reranker)},
        "files_sha256": files,
        "search_index": "exact cosine over towers/item_embeddings.npy; publish to Weaviate with "
                        "python -m trustworthy_recsys.retrieval.publish before learned serving",
        "evaluation": {"development": "dev (held out)", "final": "test (held out)",
                       "deployment": "none: assess with new observations or an online comparison"}[stage],
        "code": code_provenance(),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="Workflow directory (shared by all stages)")
    p.add_argument("--scenario", default="ratio_80_10_10")
    p.add_argument("--windows-dir", type=Path, help="Existing window export to reuse")
    p.add_argument("--settings", type=Path, help="Development only: JSON overrides of DEFAULT_SETTINGS")
    a = p.parse_args()
    overrides = json.loads(a.settings.read_text()) if a.settings else None
    print(json.dumps(run_stage(a.stage, a.run_dir, a.root, overrides, a.scenario, a.windows_dir), indent=2))


if __name__ == "__main__":
    main()
