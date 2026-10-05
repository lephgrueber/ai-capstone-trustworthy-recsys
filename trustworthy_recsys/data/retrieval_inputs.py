"""Build bounded-memory model inputs from an immutable completed pipeline run."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .reporting import hashes, write_json, code_provenance


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def grouped_training(path):
    """Yield one numeric user and their interactions; verify global user ordering."""
    last, rows = None, []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
        users = batch.column("user_id").to_numpy(zero_copy_only=False)
        items = batch.column("movie_id").to_numpy(zero_copy_only=False)
        ratings = batch.column("rating").to_numpy(zero_copy_only=False)
        times = batch.column("timestamp_utc").cast(pa.int64()).to_numpy() // 1000
        for user, item, rating, stamp in zip(users, items, ratings, times):
            user = int(user)
            if last is not None and user != last:
                if user < last:
                    raise ValueError("Training users are not globally sorted")
                yield str(last), rows
                rows = []
            last = user
            rows.append((str(item), float(rating), int(stamp)))
    if last is not None:
        yield str(last), rows


def prepare_inputs(run_dir, scenario, output_dir, max_targets_per_user=4):
    run, out = Path(run_dir).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise ValueError("Model-input directory already exists; choose a new directory")
    if max_targets_per_user < 1:
        raise ValueError("max_targets_per_user must be positive")
    if (run/"INCOMPLETE").exists():
        raise ValueError("Source data run is incomplete")
    manifest = read_json(run/"manifest.json")
    stats = read_json(run/"statistics.json")
    if scenario not in stats["splits"]:
        raise ValueError(f"Unknown scenario: {scenario}")
    sources = ["movies.parquet", "statistics.json", f"{scenario}/train.parquet", f"{scenario}/train_item_ids.json"]
    source_hashes = {}
    for name in sources:
        actual = hashes(run/name)["sha256"]
        if manifest["outputs"][name]["sha256"] != actual:
            raise ValueError(f"Source hash mismatch: {name}")
        source_hashes[name] = actual
    cutoff = int(pd.Timestamp(stats["splits"][scenario]["validation_start_utc"]).timestamp())
    threshold = manifest["config"]["positive_threshold"]
    item_ids = sorted(read_json(run/scenario/"train_item_ids.json"), key=int)
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("Training catalog must be nonempty and unique")
    mapping = {item: i for i, item in enumerate(item_ids)}
    movies = pq.read_table(run/"movies.parquet").to_pandas().set_index("movie_id")
    if not set(item_ids).issubset(movies.index):
        raise ValueError("Training movie missing from metadata")
    vocabulary = sorted({g for item in item_ids for g in movies.loc[item, "genres"]})
    genre_map = {g: i for i, g in enumerate(vocabulary)}
    features = np.zeros((len(item_ids), len(vocabulary)), dtype="float32")
    for i, item in enumerate(item_ids):
        genres = movies.loc[item, "genres"]
        for genre in genres:
            features[i, genre_map[genre]] = 1 / max(1, len(genres))
    out.mkdir(parents=True)
    (out/"INCOMPLETE").write_text("Model inputs are not ready.\n")
    np.save(out/"item_features.npy", features)
    all_counts = np.zeros(len(item_ids), dtype=np.int64)
    positive_counts = np.zeros(len(item_ids), dtype=np.int64)
    user_ids, offsets, positive_offsets = [], [0], [0]
    counts = Counter()
    handles = {name: (out/f"{name}.bin").open("wb") for name in
               ("seen_items", "positive_items", "positive_times", "query_features", "target_items", "pair_users", "context_ends", "target_times")}
    try:
        for user, events in grouped_training(run/scenario/"train.parquet"):
            if len({item for item, _, _ in events}) != len(events):
                raise ValueError("Repeated training user-movie pair")
            if any(stamp >= cutoff for _, _, stamp in events):
                raise ValueError("Training interaction reaches validation/test time")
            if any(item not in mapping for item, _, _ in events):
                raise ValueError("Training item absent from training catalog")
            user_index = len(user_ids)
            user_ids.append(user)
            seen = np.array(sorted(mapping[item] for item, _, _ in events), dtype="int32")
            seen.tofile(handles["seen_items"])
            offsets.append(offsets[-1]+len(seen))
            np.add.at(all_counts, seen, 1)
            positive = sorted((stamp, int(item), mapping[item]) for item, rating, stamp in events if rating >= threshold)
            ids = np.array([x[2] for x in positive], dtype="int32")
            times = np.array([x[0] for x in positive], dtype="int64")
            ids.tofile(handles["positive_items"])
            times.tofile(handles["positive_times"])
            np.add.at(positive_counts, ids, 1)
            starts = np.searchsorted(times, times, side="left")
            eligible = np.flatnonzero(starts > 0)
            counts["positive_events_without_strictly_earlier_history"] += int((starts == 0).sum())
            if not len(eligible):
                counts["users_without_training_pair"] += 1
            chosen = eligible[np.unique(np.linspace(0, len(eligible)-1, min(max_targets_per_user,len(eligible)), dtype=int))] if len(eligible) else []
            cumulative = np.cumsum(features[ids], axis=0)
            for target in chosen:
                end = int(starts[target])
                # Equal-time ratings are all excluded from the context, including the target.
                if times[end-1] >= times[target] or ids[target] in ids[:end]:
                    raise ValueError("Training context leaks its target")
                genre_profile = cumulative[end-1]/end
                query = np.concatenate([genre_profile, [np.log1p(end)/10]]).astype("float32")
                query.tofile(handles["query_features"])
                np.array([ids[target]],dtype="int32").tofile(handles["target_items"])
                np.array([user_index],dtype="int32").tofile(handles["pair_users"])
                np.array([positive_offsets[-1]+end],dtype="int64").tofile(handles["context_ends"])
                np.array([times[target]],dtype="int64").tofile(handles["target_times"])
                counts["training_pairs"] += 1
            counts["training_rows"] += len(events)
            counts["positive_training_rows"] += len(positive)
            positive_offsets.append(positive_offsets[-1]+len(positive))
    finally:
        for handle in handles.values():
            handle.close()
    if not counts["training_pairs"]:
        raise ValueError("No training targets have a strictly earlier positive history")
    if counts["training_rows"] != stats["splits"][scenario]["partitions"]["train"]["rows"]:
        raise ValueError("Source training row count mismatch")
    np.save(out/"seen_offsets.npy", np.array(offsets,dtype="int64"))
    np.save(out/"positive_offsets.npy",np.array(positive_offsets,dtype="int64"))
    np.save(out/"item_counts.npy",all_counts)
    np.save(out/"positive_counts.npy",positive_counts)
    write_json(out/"item_ids.json",item_ids)
    write_json(out/"user_ids.json",user_ids)
    metadata = {"version":"1.0", "source_run":str(run), "scenario":scenario,
                "source_hashes":source_hashes, "source_manifest_sha256":hashes(run/"manifest.json")["sha256"],
                "training_cutoff_utc":stats["splits"][scenario]["validation_start_utc"],
                "positive_threshold":threshold,"max_targets_per_user":max_targets_per_user,
                "genres":vocabulary,"query_feature_dim":len(vocabulary)+1,
                "users":len(user_ids),"items":len(item_ids),"statistics":dict(counts),
                "integrity":{"training_only_ids_and_vocabulary":True,"strictly_earlier_contexts":True,
                             "no_target_in_context":True,"no_duplicate_user_movie_pairs":True,
                             "source_hashes_verified":True},"code":code_provenance()}
    write_json(out/"metadata.json",metadata)
    schemas = {"version":"1.0","byte_order":"native little-endian (run on little-endian hosts)",
               "query_features.bin":{"dtype":"float32","shape":[counts["training_pairs"],len(vocabulary)+1],
                  "meaning":"mean fractional genre weights of strictly earlier positives, then log1p(history length)/10"},
               "target_items.bin":{"dtype":"int32","shape":[counts["training_pairs"]]},
               "pair_users.bin":{"dtype":"int32","shape":[counts["training_pairs"]]},
               "context_ends.bin":{"dtype":"int64","meaning":"exclusive position in positive_items.bin; start from positive_offsets[pair_user]"},
               "target_times.bin":{"dtype":"int64","meaning":"target Unix seconds"},
               "seen_items.bin":{"dtype":"int32","meaning":"all training-rated item indices, CSR via seen_offsets.npy"},
               "positive_items.bin":{"dtype":"int32","meaning":"chronological positive item indices, CSR via positive_offsets.npy"},
               "positive_times.bin":{"dtype":"int64","meaning":"positive Unix seconds aligned with positive_items.bin"},
               "item_features.npy":"float32 [items, genres]; per-item genre weights sum to 1 or 0 if missing",
               "item_ids.json":"training-only original string movie IDs; position is contiguous model index",
               "user_ids.json":"training-only original string user IDs; used for seen filtering, not a learned user-ID feature",
               "item_counts.npy":"all training rating counts; int64", "positive_counts.npy":"positive training counts; int64"}
    write_json(out/"schema.json",schemas)
    parent_card = Path(os.path.relpath(run/"DATA_CARD.md",out)).as_posix()
    (out/"DATA_CARD.md").write_text(f"""# Data Card: model-ready retrieval inputs

## Dataset overview
Derived from {metadata['source_run']} / {scenario}: {counts['training_rows']:,} training ratings,
{counts['positive_training_rows']:,} positive ratings, {len(user_ids):,} users and {len(item_ids):,} candidate movies.
The preparer produced **{counts['training_pairs']:,} context/target examples**, with at most {max_targets_per_user} targets per user.

## Provenance and collection
These are transformations of the completed MovieLens pipeline run, not newly collected observations.
See the [source Data Card]({parent_card}) for collection and release details. metadata.json records verified input hashes.
The original split and its reports remain unchanged.

## Licensing and usage
The source MovieLens usage terms and citation requirements continue to apply to these transformations.
Refer to the source Data Card and its bundled README; the software's MIT license does not replace dataset terms.

## Data structure and splits
schema.json describes binary memory-mapped arrays, ID mappings, CSR histories, and genre features.
Only training movies and their genre vocabulary are fitted. Validation/test events and labels are not read during preparation.
For each user, up to {max_targets_per_user} eligible positive targets are evenly spaced through their training history.
Each context contains only positive events with timestamps strictly before its target: ties and the target are excluded.
The user feature is a mean genre profile plus log history length; the item feature is identity plus fractional genres.
No titles, tags, release years, or held-out interactions are model features. Known users' all-rating histories are retained for seen-item filtering.
Evaluation continues to use the source validation/test JSONL and training-only histories.
The retrieval publication stage stores training-catalog movie IDs, row indices, an embedding fingerprint,
and model-produced vectors in Weaviate. User histories and held-out labels remain local.
The published model's weaviate.json describes the collection; loading verifies remote IDs and vectors
against the saved model embeddings. This storage step does not change the temporal split or these inputs.

## Limitations and risks
- Target subsampling per user changes interaction weighting; this is not training on every positive event.
- {counts['users_without_training_pair']:,} training users have no eligible strictly earlier positive context.
- Genre metadata is a release snapshot with unknown historical availability; temporal event separation does not eliminate this metadata limitation.
- Genres discard ordering and specific item interactions in the user tower; identity embeddings cover training movies only.
- Empty/unknown histories need fallback behavior. New movies are outside the index, limiting attainable recall on the all population.
- No impression propensities are available; these inputs do not establish causal policy value or demographic fairness.

## Integrity and reproducibility
Source hashes, timestamp boundaries, uniqueness, target exclusion, and row counts were checked during preparation.
metadata.json contains statistics and schema.json describes the artifact contract. manifest.json fingerprints each output.
""",encoding="utf-8")
    for name,digest in source_hashes.items():
        if hashes(run/name)["sha256"] != digest:
            raise ValueError("Source changed during preparation")
    write_json(out/"manifest.json", {"outputs":{p.name:hashes(p)["sha256"] for p in out.iterdir() if p.is_file() and p.name!='INCOMPLETE'}})
    (out/"INCOMPLETE").unlink()
    return metadata


class ModelInputs:
    def __init__(self, path, verify=False):
        self.path = Path(path)
        if (self.path/"INCOMPLETE").exists():
            raise ValueError("Incomplete model inputs")
        if verify:
            for name,digest in read_json(self.path/"manifest.json")["outputs"].items():
                if hashes(self.path/name)["sha256"] != digest:
                    raise ValueError(f"Model input hash mismatch: {name}")
        self.meta = read_json(self.path/"metadata.json")
        self.item_ids = read_json(self.path/"item_ids.json")
        self.user_ids = read_json(self.path/"user_ids.json")
        self.item_map = {x:i for i,x in enumerate(self.item_ids)}
        self.user_map = {x:i for i,x in enumerate(self.user_ids)}
        for name in ('item_features','seen_offsets','positive_offsets','item_counts','positive_counts'):
            setattr(self,name,np.load(self.path/f'{name}.npy',mmap_mode='r'))
        for name,dtype in [('seen_items','int32'),('positive_items','int32'),('positive_times','int64'),
                           ('target_items','int32'),('pair_users','int32'),('context_ends','int64'),('target_times','int64')]:
            setattr(self,name,np.memmap(self.path/f'{name}.bin',dtype=dtype,mode='r'))
        self.query_features = np.memmap(self.path/'query_features.bin',dtype='float32',mode='r',
                                       shape=(self.meta['statistics']['training_pairs'],self.meta['query_feature_dim']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--scenario',default='ratio_80_10_10')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-targets-per-user',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(prepare_inputs(args.run_dir,args.scenario,args.output_dir,args.max_targets_per_user),indent=2))


if __name__=='__main__':
    main()
