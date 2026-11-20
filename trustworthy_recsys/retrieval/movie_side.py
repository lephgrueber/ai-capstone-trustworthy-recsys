"""Movie metadata arrays for the two towers: credits, runtime and tags (feedback §2, §4, §5).

Built once per model from the verified metadata and tag artifacts and the model's
catalog, then rebuilt identically at serving time from the paths and hashes recorded in
the model config (the large arrays are not stored in model.pt).

- Credits: person IDs per role (directors / writers / cast), at most ROLE_WIDTH per movie.
  People credited on at least ``min_movies`` catalog movies get their own index; everyone
  else, and every person first seen after fitting, maps to the role's shared UNKNOWN index
  (1). Index 0 is padding. A movie with no listed credit in a role has an all-padding row,
  which the tower turns into a zero vector plus a missing flag.
- Runtime: (minutes - median) / std with median and std from the fitting catalog; a missing
  runtime uses the median (scaled value 0) and sets a missing flag.
- Tags: the 32-dim projected mean of tag phrases first used before an as-of time, plus a
  has-tags flag; zero vector and flag 0 for untagged movies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from trustworthy_recsys.data.reporting import hashes

ROLES = ("directors", "writers", "cast")
ROLE_WIDTH = {"directors": 3, "writers": 5, "cast": 10}
PAD, UNKNOWN = 0, 1


@dataclass(frozen=True)
class MovieSideConfig:
    metadata_dir: str | None = None
    tags_dir: str | None = None
    credits: bool = False
    runtime: bool = False
    tags: bool = False
    person_dim: int = 8
    min_movies: int = 2
    unknown_mask_rate: float = 0.1

    @property
    def active(self):
        return self.credits or self.runtime or self.tags



class MovieSide:
    """Per-movie metadata tensors aligned with a catalog (index = model item index)."""

    def __init__(self, config: MovieSideConfig, item_ids, vocab=None, runtime_scale=None):
        self.config = config
        self.item_ids = list(item_ids)
        self.n_items = len(self.item_ids)
        self.vocab = vocab
        self.runtime_scale = runtime_scale
        self.sources = {}
        if config.credits or config.runtime:
            from trustworthy_recsys.data.metadata import load
            frame = load(config.metadata_dir).reindex(self.item_ids)
            self.sources["metadata"] = hashes(Path(config.metadata_dir) / "movie_metadata.parquet")["sha256"]
        if config.credits:
            if self.vocab is None:
                self.vocab = {}
                for role in ROLES:
                    counts = {}
                    for people in frame[role]:
                        for person in (people if isinstance(people, (list, np.ndarray)) else []):
                            counts[person] = counts.get(person, 0) + 1
                    keep = sorted(p for p, c in counts.items() if c >= config.min_movies)
                    self.vocab[role] = {p: i + 2 for i, p in enumerate(keep)}
            self.credit_ids = {}
            for role in ROLES:
                ids = np.zeros((self.n_items, ROLE_WIDTH[role]), dtype=np.int64)
                mapping = self.vocab[role]
                for row, people in enumerate(frame[role]):
                    people = list(people) if isinstance(people, (list, np.ndarray)) else []
                    for col, person in enumerate(people[:ROLE_WIDTH[role]]):
                        ids[row, col] = mapping.get(person, UNKNOWN)
                self.credit_ids[role] = ids
        if config.runtime:
            minutes = frame["runtime_minutes"].to_numpy(dtype="float64")
            if self.runtime_scale is None:
                known = minutes[np.isfinite(minutes)]
                self.runtime_scale = {"median": float(np.median(known)), "std": float(known.std() or 1.0)}
            missing = ~np.isfinite(minutes)
            filled = np.where(missing, self.runtime_scale["median"], minutes)
            self.runtime = np.column_stack([(filled - self.runtime_scale["median"]) / self.runtime_scale["std"],
                                            missing]).astype("float32")
            self.runtime_minutes = minutes
        if config.tags:
            from trustworthy_recsys.data.tags import TagIndex
            self.tag_index = TagIndex(config.tags_dir, self.item_ids)
            self.sources["tags"] = self.tag_index.manifest["outputs"]["phrase_vectors.npy"]

    @property
    def tag_width(self):
        """Projected tag dimensions plus the has-tags flag."""
        return self.tag_index.projection.shape[1] + 1

    def vocab_sizes(self):
        return {role: len(self.vocab[role]) + 2 for role in ROLES} if self.config.credits else {}

    def state(self):
        """Small, persistent part saved next to the model."""
        return {"config": asdict(self.config), "vocab": self.vocab, "runtime_scale": self.runtime_scale,
                "sources": self.sources}

    def tag_matrix(self, as_of):
        vectors, has = self.tag_index.movie_vectors(as_of, projected=True)
        return np.column_stack([vectors, has]).astype("float32")

    def tensors(self, as_of=None):
        """Dict of tensors for the whole catalog; tags as of `as_of` (Unix seconds)."""
        out = {}
        if self.config.credits:
            out.update({role: torch.from_numpy(self.credit_ids[role]) for role in ROLES})
        if self.config.runtime:
            out["runtime"] = torch.from_numpy(self.runtime)
        if self.config.tags:
            if as_of is None:
                raise ValueError("Tag inputs need an as-of time")
            out["tags"] = torch.from_numpy(self.tag_matrix(as_of))
        return out

    def tag_snapshots(self, first_year, last_year):
        stamps, vectors, has = self.tag_index.snapshots(range(first_year, last_year + 1), projected=True)
        return stamps, torch.from_numpy(np.concatenate([vectors, has[..., None].astype("float32")], axis=-1))


def verify_sources(side: MovieSide, expected):
    if expected and side.sources != expected:
        raise ValueError("Movie metadata or tag artifacts differ from the ones the towers were fitted on")
