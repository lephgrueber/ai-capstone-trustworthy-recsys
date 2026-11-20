"""Movie credits and runtime from an IMDb snapshot, keyed by MovieLens movie ID (feedback §4).

MovieLens links.csv supplies each movie's IMDb ID. From IMDb's non-commercial dataset files:

- title.basics.tsv.gz: runtimeMinutes (kept only when positive).
- title.crew.tsv.gz: directors and writers (person IDs).
- title.principals.tsv.gz: principal cast, the rows whose category is actor or actress.
  IMDb's row order does not establish billing order, so the cast is an unordered set and
  downstream summaries weight cast members equally.

Missing values stay missing (null runtime, empty credit lists) so later stages can tell
"unknown" from "no match" (feedback §5).

Timing caveat: one snapshot (recorded in the manifest) is used for every historical
example, which assumes those credits and runtimes were known when each prediction would
have been made.

Run: python -m trustworthy_recsys.data.metadata --run-dir <run> --imdb-dir <dir>
     --snapshot-date 2026-11-20 --output-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .reporting import code_provenance, hashes, write_json

FILES = ("title.basics.tsv.gz", "title.crew.tsv.gz", "title.principals.tsv.gz")
CAST_CATEGORIES = ("actor", "actress")
SCHEMA = pa.schema([
    pa.field("movie_id", pa.string(), nullable=False),
    pa.field("imdb_id", pa.string()),
    pa.field("runtime_minutes", pa.float32()),
    pa.field("directors", pa.list_(pa.string()), nullable=False),
    pa.field("writers", pa.list_(pa.string()), nullable=False),
    pa.field("cast", pa.list_(pa.string()), nullable=False),
])


def imdb_tconst(imdb_id) -> str | None:
    if imdb_id is None or (isinstance(imdb_id, float) and np.isnan(imdb_id)) or not str(imdb_id).strip():
        return None
    return f"tt{int(imdb_id):07d}"


def read_tsv(path, usecols, chunksize=None):
    return pd.read_csv(path, sep="\t", usecols=usecols, dtype=str, na_values="\\N", keep_default_na=False,
                       quoting=3, chunksize=chunksize)


def split_people(value):
    return [] if not isinstance(value, str) or not value else value.split(",")


def build(run_dir, imdb_dir, output_dir, snapshot_date, catalog_ids=None) -> dict:
    run, imdb, out = Path(run_dir), Path(imdb_dir), Path(output_dir)
    if out.exists():
        raise ValueError("Metadata directory already exists; choose a new directory")
    links = pq.read_table(run / "links.parquet").to_pandas()
    links["tconst"] = links["imdb_id"].map(imdb_tconst)
    wanted = set(links["tconst"].dropna())

    basics = read_tsv(imdb / FILES[0], ["tconst", "runtimeMinutes"])
    basics = basics[basics["tconst"].isin(wanted)]
    runtime = pd.to_numeric(basics.set_index("tconst")["runtimeMinutes"], errors="coerce")
    runtime = runtime.where(runtime > 0)

    crew = read_tsv(imdb / FILES[1], ["tconst", "directors", "writers"])
    crew = crew[crew["tconst"].isin(wanted)].set_index("tconst")

    cast: dict[str, list[str]] = {}
    for chunk in read_tsv(imdb / FILES[2], ["tconst", "nconst", "category"], chunksize=2_000_000):
        chunk = chunk[chunk["tconst"].isin(wanted) & chunk["category"].isin(CAST_CATEGORIES)]
        for tconst, person in zip(chunk["tconst"], chunk["nconst"]):
            cast.setdefault(tconst, []).append(person)

    rows = []
    for movie_id, tconst in zip(links["movie_id"], links["tconst"]):
        value = runtime.get(tconst) if tconst else None
        rows.append({
            "movie_id": str(movie_id), "imdb_id": tconst,
            "runtime_minutes": None if value is None or pd.isna(value) else float(value),
            "directors": sorted(set(split_people(crew["directors"].get(tconst)))) if tconst in crew.index else [],
            "writers": sorted(set(split_people(crew["writers"].get(tconst)))) if tconst in crew.index else [],
            "cast": sorted(set(cast.get(tconst, []))),
        })
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    out.mkdir(parents=True)
    pq.write_table(table, out / "movie_metadata.parquet", compression="zstd")
    frame = table.to_pandas()

    def coverage(ids):
        part = frame[frame["movie_id"].isin(ids)] if ids is not None else frame
        n = len(part)
        return {"movies": n,
                "positive_runtime_pct": 100 * float(part["runtime_minutes"].notna().mean()) if n else None,
                "director_pct": 100 * float((part["directors"].map(len) > 0).mean()) if n else None,
                "writer_pct": 100 * float((part["writers"].map(len) > 0).mean()) if n else None,
                "cast_pct": 100 * float((part["cast"].map(len) > 0).mean()) if n else None}

    manifest = {
        "snapshot_date": snapshot_date,
        "timing_assumption": "credits and runtimes from this snapshot are treated as known at every "
                             "historical prediction time",
        "sources": {name: hashes(imdb / name)["sha256"] for name in FILES},
        "links_sha256": hashes(run / "links.parquet")["sha256"],
        "cast_rule": f"principals with category in {list(CAST_CATEGORIES)}; unordered, equal weights",
        "coverage": {"all_movielens": coverage(None),
                     **({"catalog": coverage(set(catalog_ids))} if catalog_ids is not None else {})},
        "people": {role: int(len({p for people in frame[role] for p in people})) for role in ("directors", "writers", "cast")},
        "output_sha256": hashes(out / "movie_metadata.parquet")["sha256"],
        "code": code_provenance(),
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def load(path) -> pd.DataFrame:
    """Verified metadata indexed by MovieLens movie ID."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if hashes(path / "movie_metadata.parquet")["sha256"] != manifest["output_sha256"]:
        raise ValueError("Movie metadata hash mismatch")
    return pq.read_table(path / "movie_metadata.parquet").to_pandas().set_index("movie_id")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--imdb-dir", type=Path, required=True)
    p.add_argument("--snapshot-date", required=True, help="Date the IMDb files were downloaded (YYYY-MM-DD)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--catalog", type=Path, help="JSON list of catalog movie IDs for a coverage report")
    a = p.parse_args()
    catalog = json.loads(a.catalog.read_text()) if a.catalog else None
    print(json.dumps(build(a.run_dir, a.imdb_dir, a.output_dir, a.snapshot_date, catalog)["coverage"], indent=2))


if __name__ == "__main__":
    main()
