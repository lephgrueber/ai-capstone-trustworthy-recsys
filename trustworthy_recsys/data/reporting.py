"""Deterministic statistics and artifact provenance."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def hashes(path):
    digest = {name: hashlib.new(name) for name in ("md5", "sha256")}
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            for h in digest.values():
                h.update(block)
    return {k: v.hexdigest() for k, v in digest.items()}


def utc(seconds):
    return pd.Timestamp(int(seconds), unit="s", tz="UTC").isoformat()


def distribution(counts):
    a = np.asarray(list(counts), dtype=np.int64)
    if not len(a):
        return {"min": None, "median": None, "p90": None, "max": None}
    return {"min": int(a.min()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, .9)), "max": int(a.max())}


class PartitionStats:
    def __init__(self):
        self.rows = 0
        self.users, self.items, self.ratings = Counter(), Counter(), Counter()
        self.first, self.last = None, None
        self.cold_users = self.cold_items = self.both_cold = self.positives = 0

    def add(self, frame, known_users, known_items, threshold):
        if frame.empty:
            return
        self.rows += len(frame)
        self.users.update(frame.userId.value_counts().to_dict())
        self.items.update(frame.movieId.value_counts().to_dict())
        self.ratings.update(frame.rating.value_counts().to_dict())
        lo, hi = int(frame.timestamp.min()), int(frame.timestamp.max())
        self.first = lo if self.first is None else min(self.first, lo)
        self.last = hi if self.last is None else max(self.last, hi)
        cu = ~frame.userId.isin(known_users)
        ci = ~frame.movieId.isin(known_items)
        self.cold_users += int(cu.sum())
        self.cold_items += int(ci.sum())
        self.both_cold += int((cu & ci).sum())
        self.positives += int(frame.rating.ge(threshold).sum())

    def summary(self):
        warm = self.rows - self.cold_users - self.cold_items + self.both_cold
        return {"rows": self.rows, "users": len(self.users), "movies": len(self.items),
                "first_timestamp_utc": utc(self.first) if self.first is not None else None,
                "last_timestamp_utc": utc(self.last) if self.last is not None else None,
                "rating_distribution": {str(k): v for k, v in sorted(self.ratings.items())},
                "ratings_per_user": distribution(self.users.values()),
                "ratings_per_movie": distribution(self.items.values()),
                "positive_rows": self.positives, "cold_user_rows": self.cold_users,
                "cold_movie_rows": self.cold_items, "both_cold_rows": self.both_cold,
                "warm_rows": warm, "warm_rows_pct": 100 * warm / self.rows if self.rows else None}


def code_provenance():
    root = Path(__file__).resolve().parents[2]
    source_files = sorted([*(root/"trustworthy_recsys").rglob("*.py"), *(root/"eval").rglob("*.py"), root/"pyproject.toml"])
    source_files = [p for p in source_files if p.is_file()]
    source_hashes = {p.relative_to(root).as_posix(): hashes(p)["sha256"] for p in source_files}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    return {"git_revision": revision, "working_tree_dirty": bool(status) if status is not None else None,
            "package_source_sha256": source_hashes}
