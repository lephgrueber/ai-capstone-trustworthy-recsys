"""Encoded MovieLens tag phrases with as-of-time movie summaries (feedback §4).

Phrases are trimmed, lower-cased and whitespace-collapsed; each distinct phrase is encoded
once with a fixed local text encoder (default sentence-transformers/all-MiniLM-L6-v2, 384
dimensions, unit-normalized), so "hilarious" and "funny" get similar vectors and no tag
text leaves the machine. For every (movie, phrase) the first time anyone used it is kept.

Timing: a movie's tags "as of" time t are the distinct phrases first used strictly before t.
Train A examples use snapshots before their target's year (see TagIndex.snapshots);
Train B, dev and test use the tags available before their window begins.

Summaries:
- movie_vectors(as_of): mean of the movie's distinct phrase vectors (unit-normalized),
  optionally projected to a fixed 32-dimension PCA basis for the towers, plus a has-tags
  flag. Movies without tags get a zero vector and flag 0 (feedback §5).
- clusters(as_of, k): for movies with enough tags, up to k k-means centroids of the
  phrase vectors ("several themes"); the single mean otherwise.
- phrase_match(a, b): mean over a's phrases of the best cosine with any of b's phrases.

The encoder name and version, vector size, normalization and averaging rule are saved in
manifest.json and must be reused to encode new phrases at serving time.

Run: python -m trustworthy_recsys.data.tags --run-dir <run> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .reporting import code_provenance, hashes, write_json

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
PROJECTED_DIM = 32
SPACES = re.compile(r"\s+")


def normalize(phrase: str) -> str:
    return SPACES.sub(" ", str(phrase).strip().lower())


def load_encoder(name=DEFAULT_ENCODER):
    from sentence_transformers import SentenceTransformer  # optional dependency
    return SentenceTransformer(name, device="cpu")


def encode(encoder, phrases, batch_size=512):
    return np.asarray(encoder.encode(list(phrases), batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=False), dtype="float32")


def build(run_dir, output_dir, encoder_name=DEFAULT_ENCODER, encoder=None) -> dict:
    out = Path(output_dir)
    if out.exists():
        raise ValueError("Tag directory already exists; choose a new directory")
    tags = pq.read_table(Path(run_dir) / "tags.parquet", columns=["movie_id", "tag", "timestamp_utc"]).to_pandas()
    tags["phrase"] = tags["tag"].map(normalize)
    tags = tags[tags["phrase"] != ""]
    tags["ts"] = tags["timestamp_utc"].astype("int64") // 10**9 if tags["timestamp_utc"].dtype.kind == "M" \
        else pd.to_datetime(tags["timestamp_utc"], utc=True).astype("int64") // 10**9
    pairs = tags.groupby(["movie_id", "phrase"], as_index=False)["ts"].min()
    phrases = np.array(sorted(pairs["phrase"].unique()))
    phrase_id = {p: i for i, p in enumerate(phrases)}
    pairs["phrase_id"] = pairs["phrase"].map(phrase_id).astype("int32")
    encoder = encoder or load_encoder(encoder_name)
    vectors = encode(encoder, phrases)
    # Fixed linear projection for tower inputs: PCA of all phrase vectors, computed once.
    centered = vectors - vectors.mean(axis=0)
    _, _, components = np.linalg.svd(centered[:: max(1, len(centered) // 50_000)], full_matrices=False)
    projection = components[:PROJECTED_DIM].T.astype("float32")
    out.mkdir(parents=True)
    np.save(out / "phrase_vectors.npy", vectors.astype("float16"))
    np.save(out / "projection.npy", projection)
    np.save(out / "phrase_mean.npy", vectors.mean(axis=0).astype("float32"))
    pd.DataFrame({"phrase": phrases}).to_parquet(out / "phrases.parquet", index=False)
    pairs[["movie_id", "phrase_id", "ts"]].rename(columns={"ts": "first_used"}).to_parquet(
        out / "movie_phrases.parquet", index=False)
    version = getattr(__import__("sentence_transformers"), "__version__", "unknown") if encoder_name else "custom"
    manifest = {
        "encoder": encoder_name, "encoder_library_version": version, "vector_dim": int(vectors.shape[1]),
        "normalization": "phrase trimmed, lower-cased, whitespace collapsed; vectors unit-normalized",
        "movie_summary": "mean of distinct phrase vectors first used strictly before the as-of time, renormalized",
        "tower_projection": f"PCA to {PROJECTED_DIM} dims of mean-centered phrase vectors",
        "phrases": int(len(phrases)), "movie_phrase_pairs": int(len(pairs)), "movies": int(pairs["movie_id"].nunique()),
        "tags_sha256": hashes(Path(run_dir) / "tags.parquet")["sha256"],
        "outputs": {p.name: hashes(p)["sha256"] for p in out.iterdir() if p.is_file()},
        "code": code_provenance(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


class TagIndex:
    """As-of-time tag summaries for a fixed movie catalog."""

    def __init__(self, path, item_ids, verify=True):
        path = Path(path)
        self.manifest = json.loads((path / "manifest.json").read_text())
        if verify:
            for name, digest in self.manifest["outputs"].items():
                if hashes(path / name)["sha256"] != digest:
                    raise ValueError(f"Tag artifact hash mismatch: {name}")
        self.vectors = np.load(path / "phrase_vectors.npy").astype("float32")
        self.projection = np.load(path / "projection.npy")
        self.mean = np.load(path / "phrase_mean.npy")
        self.phrases = pd.read_parquet(path / "phrases.parquet")["phrase"].tolist()
        self.phrase_id = {p: i for i, p in enumerate(self.phrases)}
        pairs = pd.read_parquet(path / "movie_phrases.parquet")
        index = {item: i for i, item in enumerate(item_ids)}
        pairs["item"] = pairs["movie_id"].map(index)
        pairs = pairs.dropna(subset=["item"]).sort_values(["item", "first_used"])
        self.item = pairs["item"].to_numpy(np.int64)
        self.phrase = pairs["phrase_id"].to_numpy(np.int64)
        self.first_used = pairs["first_used"].to_numpy(np.int64)
        self.n_items = len(item_ids)
        self.dim = self.vectors.shape[1]
        self._cache = {}

    def _visible(self, as_of):
        return self.first_used < as_of

    def movie_vectors(self, as_of, projected=False):
        """(vectors [items, dim or 32], has_tags [items]) from tags first used before as_of."""
        key = (int(as_of), projected)
        if key not in self._cache:
            keep = self._visible(as_of)
            from scipy import sparse
            membership = sparse.csr_matrix((np.ones(int(keep.sum()), dtype="float32"),
                                            (self.item[keep], self.phrase[keep])),
                                           shape=(self.n_items, len(self.vectors)))
            sums = np.asarray(membership @ self.vectors, dtype="float64")
            has = np.bincount(self.item[keep], minlength=self.n_items) > 0
            norms = np.linalg.norm(sums, axis=1, keepdims=True)
            mean = np.divide(sums, norms, out=np.zeros_like(sums), where=norms > 0).astype("float32")
            if projected:
                mean = np.where(has[:, None], (mean - self.mean) @ self.projection, 0).astype("float32")
            self._cache[key] = (mean, has)
        return self._cache[key]

    def snapshots(self, years, projected=True):
        """Stacked movie_vectors at Jan 1 of each year (for Train A target-time lookups)."""
        stamps = [int(pd.Timestamp(f"{year}-01-01", tz="UTC").timestamp()) for year in years]
        stacked = [self.movie_vectors(stamp, projected) for stamp in stamps]
        return np.array(stamps), np.stack([v for v, _ in stacked]), np.stack([h for _, h in stacked])

    def phrase_lists(self, as_of):
        """Per catalog movie: array of phrase IDs first used before as_of."""
        keep = self._visible(as_of)
        items, phrases = self.item[keep], self.phrase[keep]
        bounds = np.searchsorted(items, np.arange(self.n_items + 1))
        return [phrases[bounds[i]:bounds[i + 1]] for i in range(self.n_items)]

    def clusters(self, phrase_ids, k=3, min_tags=6, iterations=10):
        """Up to k unit centroids of a movie's phrase vectors (single mean when few tags)."""
        vectors = self.vectors[phrase_ids]
        if len(vectors) < min_tags:
            mean = vectors.mean(axis=0, keepdims=True)
            return mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
        k = min(k, len(vectors))
        rng = np.random.default_rng(0)
        centers = vectors[rng.choice(len(vectors), k, replace=False)]
        for _ in range(iterations):
            assign = np.argmax(vectors @ centers.T, axis=1)
            centers = np.stack([vectors[assign == c].mean(axis=0) if (assign == c).any() else centers[c] for c in range(k)])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        return centers

    def phrase_match(self, a_ids, b_ids):
        """Mean over a's phrases of the best cosine with b's phrases; NaN when either is empty."""
        if not len(a_ids) or not len(b_ids):
            return np.nan
        return float((self.vectors[a_ids] @ self.vectors[b_ids].T).max(axis=1).mean())

    def encode_new(self, phrases, encoder=None):
        """Vectors for phrases at serving time: stored vectors, else the same encoder."""
        known = [self.phrase_id.get(normalize(p)) for p in phrases]
        missing = [normalize(p) for p, i in zip(phrases, known) if i is None]
        fresh = iter(encode(encoder or _encoder(self.manifest["encoder"]), missing)) if missing else iter(())
        return np.stack([self.vectors[i] if i is not None else next(fresh) for i in known])


@lru_cache(maxsize=2)
def _encoder(name):
    return load_encoder(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    a = p.parse_args()
    print(json.dumps({k: v for k, v in build(a.run_dir, a.output_dir, a.encoder).items()
                      if k not in ("outputs", "code")}, indent=2))


if __name__ == "__main__":
    main()
