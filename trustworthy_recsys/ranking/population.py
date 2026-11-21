"""Keep eligible viewers with empty candidate lists in macro metrics."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from trustworthy_recsys.data.reporting import hashes


def viewers_for(table_dir):
    path = Path(table_dir) / "users.parquet"
    if not path.exists():
        return None
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if hashes(path)["sha256"] != manifest.get("users_sha256"):
        raise ValueError("Reranker users table hash mismatch")
    viewers = pq.read_table(path).to_pandas().set_index("user_id")
    if not viewers.index.is_unique or not len(viewers):
        raise ValueError("Evaluation population must contain unique, nonempty viewers")
    return viewers


def expand(values, row_users, viewers):
    """Align per-row-group scores to the full viewer population, filling misses with zero."""
    if viewers is None:
        return values
    positions = viewers.index.get_indexer(row_users)
    if (positions < 0).any():
        raise ValueError("Candidate viewer missing from users table")
    result = np.zeros(len(viewers), dtype=float)
    result[positions] = values
    return result
