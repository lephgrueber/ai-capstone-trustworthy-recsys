"""Read-only validation of the MovieLens release before any artifacts are written."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .reporting import hashes, distribution, utc

FILES = ("ratings.csv", "movies.csv", "tags.csv", "links.csv")
COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def rating_batches(path, chunk_size):
    with pd.read_csv(path, chunksize=chunk_size,
                     dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"}) as reader:
        yield from reader


def verify_sources(raw_dir, manifest):
    expected = {}
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split()
            expected[name.lstrip("*")] = digest.lower()
    result = {}
    for name in FILES:
        path = Path(raw_dir) / name
        actual = hashes(path)
        if actual["md5"] != expected.get(name):
            raise ValueError(f"{name}: checksum differs from the supplied manifest")
        result[name] = {**actual, "bytes": path.stat().st_size}
    return result


@dataclass
class Audit:
    summary: dict
    user_first: dict
    item_first: dict
    days: Counter


def audit_ratings(path, movie_ids, chunk_size):
    users, items, ratings, days = Counter(), Counter(), Counter(), Counter()
    user_first, item_first = {}, {}
    previous = None
    total, earliest, latest = 0, None, None
    release_day_rows = 0
    for frame in rating_batches(path, chunk_size):
        if list(frame.columns) != COLUMNS or frame.isna().any().any():
            raise ValueError("ratings.csv: wrong columns or missing required fields")
        if not frame.userId.gt(0).all() or not frame.movieId.gt(0).all():
            raise ValueError("Ratings IDs must be positive integers")
        if not frame.rating.isin(np.arange(.5, 5.1, .5)).all():
            raise ValueError("Ratings must be half-star values from 0.5 through 5.0")
        if not frame.movieId.isin(movie_ids).all():
            raise ValueError("A rating references a movie absent from movies.csv")
        pd.to_datetime(frame.timestamp, unit="s", utc=True, errors="raise")
        if not frame.timestamp.ge(0).all():
            raise ValueError("Negative Unix timestamps are unsupported")
        u, i = frame.userId.to_numpy(), frame.movieId.to_numpy()
        # The documented ordering makes duplicates detectable across batch boundaries.
        if np.any((u[1:] < u[:-1]) | ((u[1:] == u[:-1]) & (i[1:] <= i[:-1]))):
            raise ValueError("Ratings must be sorted by userId/movieId with no repeated user-movie pairs")
        if previous is not None and (int(u[0]), int(i[0])) <= previous:
            raise ValueError("Repeated or out-of-order user-movie pair across batches")
        previous = int(u[-1]), int(i[-1])
        users.update(frame.userId.value_counts().to_dict())
        items.update(frame.movieId.value_counts().to_dict())
        ratings.update(frame.rating.value_counts().to_dict())
        days.update((frame.timestamp // 86400).value_counts().to_dict())
        for col, target in [("userId", user_first), ("movieId", item_first)]:
            for key, stamp in frame.groupby(col).timestamp.min().items():
                target[int(key)] = min(target.get(int(key), int(stamp)), int(stamp))
        lo, hi = int(frame.timestamp.min()), int(frame.timestamp.max())
        earliest = lo if earliest is None else min(earliest, lo)
        latest = hi if latest is None else max(latest, hi)
        release_day_rows += int(frame.timestamp.ge(1697155200).sum())
        total += len(frame)
    if not total:
        raise ValueError("ratings.csv is empty")
    popular = sorted(items.values(), reverse=True)
    return Audit({"rows": total, "users": len(users), "rated_movies": len(items),
                  "catalog_movies": len(movie_ids), "catalog_movies_without_ratings": len(movie_ids - set(items)),
                  "first_timestamp_utc": utc(earliest), "last_timestamp_utc": utc(latest),
                  "rating_distribution": {str(k): v for k, v in sorted(ratings.items())},
                  "ratings_per_user": distribution(users.values()),
                  "ratings_per_rated_movie": distribution(items.values()),
                  "top_one_percent_rated_movies_share_pct": 100 * sum(popular[:max(1, int(np.ceil(len(items)*.01)))]) / total,
                  "ratings_on_or_after_2023_10_13_utc": release_day_rows,
                  "missing_required_fields": 0, "invalid_ratings": 0,
                  "unknown_movie_references": 0, "duplicate_user_movie_pairs": 0,
                  "global_user_movie_order_verified": True}, user_first, item_first, days)
