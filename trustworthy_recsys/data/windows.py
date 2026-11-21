"""Chronological fitting windows: Train A -> Train B -> dev -> test.

The existing pipeline run fixes the dev (``validation``) and test boundaries. This module
splits the pre-dev training partition into Train A (tower fitting and historical feature
statistics) and Train B (reranker labels) with one *global* timestamp cutoff, so every
viewer shares the same boundaries. Equal timestamps always stay in the same window.

The same rule repartitions A + B + dev into Final A / Final B for the final refit, and all
observations into Deployment A / Deployment B for the browser release.

Each exported example file has a sidecar ``<name>.json`` with its boundaries so later
stages (feature fitting, candidate tables) cannot silently use a different cutoff.

Run with ``python -m trustworthy_recsys.data.windows --help``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .reporting import code_provenance, hashes, utc, write_json

PARTITIONS = ("train", "validation", "test")
DISLIKE_THRESHOLD = 2.0


@dataclass
class Events:
    """Rating events sorted by (user, timestamp, movie); IDs are MovieLens integers."""

    user: np.ndarray
    item: np.ndarray
    rating: np.ndarray
    ts: np.ndarray

    def __len__(self):
        return len(self.ts)

    def select(self, mask):
        return Events(self.user[mask], self.item[mask], self.rating[mask], self.ts[mask])


def to_seconds(value) -> int:
    return int(pd.Timestamp(value).timestamp())


def boundaries(run_dir, scenario):
    """Dev and test start (Unix seconds) recorded by the pipeline run."""
    stats = json.loads((Path(run_dir) / "statistics.json").read_text(encoding="utf-8"))
    split = stats["splits"][scenario]
    return to_seconds(split["validation_start_utc"]), to_seconds(split["test_start_utc"])


def load_events(run_dir, scenario, partitions=("train",), start=None, end=None) -> Events:
    """Load rating events from pipeline partitions, keeping start <= ts < end."""
    unknown = set(partitions) - set(PARTITIONS)
    if unknown or not partitions:
        raise ValueError(f"Unknown partitions: {sorted(unknown)}")
    columns = {"user": [], "item": [], "rating": [], "ts": []}
    for part in partitions:
        table = pq.read_table(Path(run_dir) / scenario / f"{part}.parquet",
                              columns=["user_id", "movie_id", "rating", "timestamp_utc"])
        columns["user"].append(pc.cast(table["user_id"], pa.int64()).to_numpy())
        columns["item"].append(pc.cast(table["movie_id"], pa.int64()).to_numpy())
        columns["rating"].append(table["rating"].to_numpy().astype("float32"))
        columns["ts"].append(table["timestamp_utc"].cast(pa.int64()).to_numpy() // 1000)
    user, item, rating, ts = (np.concatenate(columns[k]) for k in ("user", "item", "rating", "ts"))
    keep = np.ones(len(ts), dtype=bool)
    if start is not None:
        keep &= ts >= start
    if end is not None:
        keep &= ts < end
    user, item, rating, ts = user[keep], item[keep], rating[keep], ts[keep]
    order = np.lexsort((item, ts, user))
    return Events(user[order], item[order], rating[order], ts[order])


def share_cutoff(timestamps, share) -> int:
    """Smallest timestamp c such that events with ts < c hold about `share` of events.

    Events with ts == c all fall after the cutoff, so equal timestamps stay together.
    """
    if not 0 < share < 1:
        raise ValueError("share must be strictly between 0 and 1")
    ordered = np.sort(np.asarray(timestamps))
    if len(ordered) < 2:
        raise ValueError("Need at least two events to place a cutoff")
    cutoff = int(ordered[min(len(ordered) - 1, int(round(share * len(ordered))))])
    if cutoff <= ordered[0]:
        raise ValueError("Cutoff would leave the earlier window empty")
    return cutoff


def ab_share(ab_ratio) -> float:
    """A:B ratio r -> fraction of the combined A+B window assigned to A."""
    if ab_ratio <= 0:
        raise ValueError("A:B ratio must be positive")
    return ab_ratio / (ab_ratio + 1)


def plan(run_dir, scenario, ab_ratio=7.0) -> dict:
    """Development, final-refit and deployment boundaries (Unix seconds)."""
    dev_start, test_start = boundaries(run_dir, scenario)
    share = ab_share(ab_ratio)
    train_ts = load_events(run_dir, scenario, ("train",)).ts
    pre_test_ts = np.concatenate([train_ts, load_events(run_dir, scenario, ("validation",)).ts])
    all_ts = np.concatenate([pre_test_ts, load_events(run_dir, scenario, ("test",)).ts])
    a_end = share_cutoff(train_ts, share)
    final_a_end = share_cutoff(pre_test_ts, share)
    deploy_a_end = share_cutoff(all_ts, share)
    last = int(all_ts.max()) + 1

    def count(ts, lo, hi):
        return int(((ts >= lo) & (ts < hi)).sum())

    total = len(all_ts)
    windows = {
        "train_a": (None, a_end), "train_b": (a_end, dev_start),
        "dev": (dev_start, test_start), "test": (test_start, last),
        "final_a": (None, final_a_end), "final_b": (final_a_end, test_start),
        "deployment_a": (None, deploy_a_end), "deployment_b": (deploy_a_end, last),
    }
    first = int(all_ts.min())
    return {
        "ab_ratio": ab_ratio, "a_share_of_fitting_window": share,
        "windows": {
            name: {"start_utc": utc(lo if lo is not None else first), "end_utc": utc(hi),
                   "start": lo, "end": hi,
                   "events": count(all_ts, lo if lo is not None else first, hi),
                   "share_of_all_events": count(all_ts, lo if lo is not None else first, hi) / total}
            for name, (lo, hi) in windows.items()
        },
    }


def _user_bounds(users):
    change = np.flatnonzero(users[1:] != users[:-1]) + 1
    return np.concatenate([[0], change]), np.concatenate([change, [len(users)]])


def export_examples(events: Events, history_end, label_start, label_end, path, threshold=4.0,
                    dislike_threshold=DISLIKE_THRESHOLD) -> dict:
    """Write one JSONL example per viewer with at least one like in [label_start, label_end).

    Inputs come only from events before `history_end`; labels from the label window.
    Fields (all chronological, ties broken by numeric movie ID):

    - history_items / history_ratings / history_times: likes before history_end.
    - other_rated_items / other_rated_ratings / other_rated_times: non-likes before
      history_end (used for exclusion, rating statistics and dislike histories).
    - relevant_items / relevant_ratings: label-window likes (answers; never inputs).
    - label_other_items / label_other_ratings: label-window ratings below the like threshold.
    """
    if history_end > label_start or label_start >= label_end:
        raise ValueError("Labels must start at or after the history end and be nonempty")
    path = Path(path)
    stats = {"viewers_with_label_events": 0, "excluded_no_label_likes": 0, "examples": 0,
             "label_likes": 0, "empty_history_examples": 0, "history_buckets": {}}
    buckets = {"history_0": 0, "history_1_20": 0, "history_21_100": 0, "history_101_plus": 0}
    starts, ends = _user_bounds(events.user)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for lo, hi in zip(starts, ends):
            ts = events.ts[lo:hi]
            label = (ts >= label_start) & (ts < label_end)
            if not label.any():
                continue
            stats["viewers_with_label_events"] += 1
            items, ratings = events.item[lo:hi], events.rating[lo:hi]
            liked = ratings >= threshold
            if not (label & liked).any():
                stats["excluded_no_label_likes"] += 1
                continue
            before = ts < history_end
            hist, other = before & liked, before & ~liked
            lab_pos, lab_other = label & liked, label & ~liked
            record = {
                "user_id": str(int(events.user[lo])),
                "history_items": [str(int(x)) for x in items[hist]],
                "relevant_items": [str(int(x)) for x in items[lab_pos]],
                "history_ratings": [float(x) for x in ratings[hist]],
                "history_times": [int(x) for x in ts[hist]],
                "other_rated_items": [str(int(x)) for x in items[other]],
                "other_rated_ratings": [float(x) for x in ratings[other]],
                "other_rated_times": [int(x) for x in ts[other]],
                "relevant_ratings": [float(x) for x in ratings[lab_pos]],
                "label_other_items": [str(int(x)) for x in items[lab_other]],
                "label_other_ratings": [float(x) for x in ratings[lab_other]],
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            n = int(hist.sum())
            stats["examples"] += 1
            stats["label_likes"] += int(lab_pos.sum())
            stats["empty_history_examples"] += int(n == 0)
            buckets[history_bucket(n)] += 1
    stats["history_buckets"] = buckets
    meta = {"examples_file": path.name, "history_end": int(history_end), "history_end_utc": utc(history_end),
            "label_start": int(label_start), "label_start_utc": utc(label_start),
            "label_end": int(label_end), "label_end_utc": utc(label_end),
            "positive_threshold": threshold, "dislike_threshold": dislike_threshold,
            "sha256": hashes(path)["sha256"], "statistics": stats}
    write_json(sidecar(path), meta)
    return meta


def history_bucket(length: int) -> str:
    """Usable-favorite groups used in every report (feedback §8)."""
    if length == 0:
        return "history_0"
    return "history_1_20" if length <= 20 else "history_21_100" if length <= 100 else "history_101_plus"


def sidecar(path) -> Path:
    return Path(path).with_suffix(".json")


def read_window_meta(examples_path) -> dict | None:
    """Boundaries for an exported example file, verified against its hash; None for legacy files."""
    meta_path = sidecar(examples_path)
    if not meta_path.exists() or Path(examples_path).suffix != ".jsonl":
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "history_end" not in meta:
        return None
    if hashes(examples_path)["sha256"] != meta["sha256"]:
        raise ValueError(f"Window example hash mismatch: {examples_path}")
    return meta


def load_records(path, limit=0):
    with Path(path).open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            yield json.loads(line)
            if limit and count >= limit:
                return


EXPORTS = {
    # name: (events partitions, history_end window key, label window)
    "train_b": (("train",), "train_a", "train_b"),
    "dev": (("train", "validation"), "train_b", "dev"),
    "final_b": (("train", "validation"), "final_a", "final_b"),
    "test": (PARTITIONS, "final_b", "test"),
    "deployment_b": (PARTITIONS, "deployment_a", "deployment_b"),
}


def prepare(run_dir, scenario, output_dir, ab_ratio=7.0, threshold=None, names=tuple(EXPORTS)) -> dict:
    """Plan the windows and export example files for each requested stage."""
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise ValueError("Window directory already exists; choose a new directory")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    threshold = manifest["config"]["positive_threshold"] if threshold is None else threshold
    layout = plan(run, scenario, ab_ratio)
    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("Window export in progress.\n")
    exports = {}
    cache = {}
    for name in names:
        partitions, history_key, label_key = EXPORTS[name]
        if partitions not in cache:
            cache.clear()
            cache[partitions] = load_events(run, scenario, partitions)
        history_end = layout["windows"][history_key]["end"]
        label = layout["windows"][label_key]
        exports[name] = export_examples(cache[partitions], history_end, label["start"], label["end"],
                                        out / f"{name}.jsonl", threshold)
        exports[name]["partitions"] = list(partitions)
        write_json(sidecar(out / f"{name}.jsonl"), exports[name])
        print(json.dumps({name: exports[name]["statistics"]}), flush=True)
    sources = {f"{scenario}/{p}.parquet": hashes(run / scenario / f"{p}.parquet")["sha256"] for p in PARTITIONS}
    result = {"run_dir": str(run), "scenario": scenario, "positive_threshold": threshold,
              "source_hashes": sources, **layout, "exports": exports, "code": code_provenance()}
    write_json(out / "windows.json", result)
    (out / "INCOMPLETE").unlink()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--scenario", default="ratio_80_10_10")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ab-ratio", type=float, default=7.0, help="Train A : Train B event ratio (70:10 -> 7)")
    p.add_argument("--exports", nargs="+", default=list(EXPORTS), choices=list(EXPORTS))
    a = p.parse_args()
    result = prepare(a.run_dir, a.scenario, a.output_dir, a.ab_ratio, names=tuple(a.exports))
    print(json.dumps({k: {x: v[x] for x in ("start_utc", "end_utc", "events", "share_of_all_events")}
                      for k, v in result["windows"].items()}, indent=2))


if __name__ == "__main__":
    main()
