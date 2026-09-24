"""Run with python -m trustworthy_recsys.data.pipeline --help."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import platform
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from .audit import audit_ratings, rating_batches, verify_sources
from .data_card import render_data_card
from .evaluation import export_examples, validate_exports
from .preprocess import interaction_table, load_metadata, write_metadata, write_tags
from .reporting import PartitionStats, code_provenance, hashes, utc, write_json
from .schema import INTERACTIONS, SCHEMA_VERSION, describe
from .split import choose_cutoffs, masks


def run_pipeline(raw_dir, output_dir, checksum_file, source_readme, *, ratios=None,
                 explicit=None, positive_threshold=4.0, chunk_size=50_000,
                 dataset_name="movielens-32m"):
    raw, output = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if chunk_size < 1 or positive_threshold not in np.arange(.5, 5.1, .5):
        raise ValueError("Chunk size must be positive; relevance threshold must be a valid half-star rating")
    if explicit is not None and ratios is not None:
        raise ValueError("Use ratios or explicit cutoffs, not both")
    source_readme = Path(source_readme)
    if not source_readme.is_file():
        raise ValueError("Dataset README is required for provenance and license documentation")
    execution_code = code_provenance()
    print("Verifying source checksums and auditing all ratings...", flush=True)
    inputs = verify_sources(raw, checksum_file)
    movies, links = load_metadata(raw)
    movie_ids = set(movies.movieId)
    audit = audit_ratings(raw/"ratings.csv", movie_ids, chunk_size)
    if dataset_name == "movielens-32m" and (audit.summary["rows"], audit.summary["users"], len(movies)) != (32_000_204, 200_948, 87_585):
        raise ValueError("Source counts do not match MovieLens 32M; specify an accurate dataset name for other data")
    ratios = ratios if ratios is not None else [(0.8, 0.1, 0.1), (0.7, 0.2, 0.1)]
    configs = [("calendar", None)] if explicit is not None else [
        ("ratio_" + "_".join(f"{x*100:g}".replace(".", "p") for x in ratio), ratio) for ratio in ratios]
    if not configs or len({name for name, _ in configs}) != len(configs):
        raise ValueError("Supply distinct, nonempty split configurations")
    cutoffs = {name: choose_cutoffs(audit.days, ratio or (.8,.1,.1), explicit) for name, ratio in configs}
    for name, bounds in cutoffs.items():
        print(f"{name}: validation {utc(bounds[0])}; test {utc(bounds[1])}", flush=True)
    output.mkdir(parents=True)
    marker = output/"INCOMPLETE"
    marker.write_text("Do not use this run until manifest.json is written and this marker is removed.\n", encoding="utf-8")
    shutil.copyfile(source_readme, output/"source_README.txt")
    shutil.copyfile(checksum_file, output/"source_checksums.txt")
    for name in cutoffs:
        (output/name).mkdir()
    config = {"dataset": dataset_name, "raw_dir": str(raw), "positive_threshold": positive_threshold,
              "chunk_size": chunk_size, "requested_ratios": {name: ratio for name, ratio in configs},
              "explicit_cutoffs": list(explicit) if explicit else None,
              "history_policy": "positive training interactions only; no validation refit",
              "warm_policy": "user has any training rating; held-out positive items restricted to training-observed movies"}
    metadata = write_metadata(movies, links, output)
    known_users = {name: {user for user, first in audit.user_first.items() if first < bounds[0]} for name, bounds in cutoffs.items()}
    known_items = {name: {item for item, first in audit.item_first.items() if first < bounds[0]} for name, bounds in cutoffs.items()}
    counts = {(name, part): PartitionStats() for name in cutoffs for part in ("train", "validation", "test")}
    print("Writing temporal Parquet partitions...", flush=True)
    with ExitStack() as stack:
        writers = {key: stack.enter_context(pq.ParquetWriter(output/key[0]/f"{key[1]}.parquet", INTERACTIONS)) for key in counts}
        for frame in rating_batches(raw/"ratings.csv", chunk_size):
            for name, bounds in cutoffs.items():
                for part, selected in masks(frame.timestamp, bounds).items():
                    subset = frame.loc[selected]
                    if not subset.empty:
                        writers[name, part].write_table(interaction_table(subset))
                        counts[name, part].add(subset, known_users[name], known_items[name], positive_threshold)
    splits = {}
    for name, bounds in cutoffs.items():
        partitions = {part: counts[name, part].summary() for part in ("train", "validation", "test")}
        if any(not s["rows"] for s in partitions.values()):
            raise ValueError(f"{name}: empty temporal partition; choose different cutoffs")
        if sum(s["rows"] for s in partitions.values()) != audit.summary["rows"]:
            raise ValueError("Partition counts do not reconcile")
        if not (counts[name,"train"].last < bounds[0] <= counts[name,"validation"].first and counts[name,"validation"].last < bounds[1] <= counts[name,"test"].first):
            raise ValueError("Temporal boundary validation failed")
        for part, s in partitions.items():
            s["dataset_pct"] = 100*s["rows"]/audit.summary["rows"]
            s["cold_users"] = len(set(counts[name,part].users)-known_users[name])
            s["cold_movies"] = len(set(counts[name,part].items)-known_items[name])
            artifact = pq.ParquetFile(output/name/f"{part}.parquet")
            if artifact.metadata.num_rows != s["rows"] or not artifact.schema_arrow.equals(INTERACTIONS):
                raise ValueError("Written Parquet schema/count validation failed")
        splits[name] = {"validation_start_utc": utc(bounds[0]), "test_start_utc": utc(bounds[1]), "partitions": partitions}
        write_json(output/name/"train_item_ids.json", [str(item) for item in sorted(known_items[name])])
    del counts
    print("Preparing timestamp-filtered tags and evaluation JSONL...", flush=True)
    tags = write_tags(raw, output, movie_ids, set(audit.user_first), cutoffs, chunk_size)
    evaluation = export_examples(raw, output, cutoffs, known_items, positive_threshold)
    validate_exports(output, evaluation)
    for name in splits:
        splits[name]["evaluation"] = evaluation[name]
        for part in ("validation", "test"):
            if evaluation[name][part]["all_positive_labels"] != splits[name]["partitions"][part]["positive_rows"]:
                raise ValueError("Exported relevance labels do not reconcile with positive ratings")
    if verify_sources(raw, checksum_file) != inputs:
        raise ValueError("Source files changed during execution")
    write_json(output/"schema.json", describe())
    write_json(output/"statistics.json", {"dataset": audit.summary, "metadata": metadata, "tags": tags,
                                         "preprocessing": {"rating_rows_removed": 0, "tag_rows_removed": tags["blank_tags_removed"]}, "splits": splits})
    render_data_card(output, dataset_name, audit.summary, metadata, tags, splits, config)
    manifest = {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": config, "inputs": inputs,
                "source_documentation": {"README": hashes(output/"source_README.txt"), "checksums": hashes(output/"source_checksums.txt")},
                "code": execution_code,
                "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__},
                "validation": {"source_checksums_verified_before_and_after": True, "parquet_schema_and_counts": True,
                               "temporal_boundaries": True, "harness_loaders": True, "label_counts_reconciled": True},
                "outputs": {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": hashes(path)["sha256"]}
                            for path in sorted(output.rglob("*")) if path.is_file() and path != marker}}
    write_json(output/"manifest.json", manifest)
    marker.unlink()
    print(f"Complete: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; existing runs are never overwritten")
    parser.add_argument("--checksums", type=Path, default=Path("checksums.txt"))
    parser.add_argument("--source-readme", type=Path, default=Path("README.txt"))
    parser.add_argument("--ratios", nargs=3, type=float, action="append", help="Repeat for multiple ratios; default is both 80/10/10 and 70/20/10")
    parser.add_argument("--validation-start", help="Explicit UTC cutoff (requires --test-start)")
    parser.add_argument("--test-start", help="Explicit UTC cutoff (requires --validation-start)")
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--dataset-name", default="movielens-32m")
    args = parser.parse_args()
    if bool(args.validation_start) != bool(args.test_start):
        parser.error("Both explicit cutoffs are required")
    try:
        run_pipeline(args.raw_dir, args.output_dir, args.checksums, args.source_readme,
                     ratios=args.ratios, explicit=(args.validation_start, args.test_start) if args.validation_start else None,
                     positive_threshold=args.positive_threshold, chunk_size=args.chunk_size, dataset_name=args.dataset_name)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Pipeline failed: {error}\nPartial outputs, if any, are marked INCOMPLETE.\n")


if __name__ == "__main__":
    main()
