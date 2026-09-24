# MovieLens data pipeline

This is the runnable counterpart to the EDA. It reads the four local MovieLens
files, checks their integrity, produces chronological interaction splits, exports
examples for the existing evaluator, and generates statistics and a Data Card.
It does not train a recommender or regenerate the qualitative golden set.

## Setup and source files

From the repository root, create a Python 3.11+ environment and install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

Download and extract [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
if the files are not already present. Keep `ratings.csv`, `movies.csv`, `tags.csv`,
and `links.csv` together. Retain the distribution's `README.txt` and
`checksums.txt`: the pipeline requires both, and copies them into the output for
provenance and usage terms. It does not automatically download anything.

The current local CSVs are under `trustworthy_recsys/data/`. A dedicated
`data/raw/ml-32m/` directory is also supported by passing its path. No input file
is moved, edited, or deleted.

## Run both ratio comparisons

```powershell
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.pipeline --raw-dir trustworthy_recsys/data --output-dir data/processed/movielens_phase1
```

This runs **both 80/10/10 and 70/20/10** by default. These are chronological
fractions of rating records, not random splits or fractions of users. The 70%
and 80% boundaries differ, but the shared 90% boundary gives both runs identical
test events. Ratio boundaries fall at midnight after the UTC day containing the
target cumulative count; whole days remain together, so proportions are approximate.

The documented full release produces these boundaries:

| Scenario | Validation begins, UTC | Test begins, UTC |
|---|---|---|
| `ratio_80_10_10` | 2018-10-04 | 2020-11-06 |
| `ratio_70_20_10` | 2016-10-14 | 2020-11-06 |

Choose a **new output directory for every run**. Existing directories are refused,
including incomplete runs, so successful artifacts cannot be silently replaced.
The command exits nonzero on validation failure. Do not consume a directory with
an `INCOMPLETE` marker. A successful run contains `manifest.json` and no marker.

## Other configurations

To generate only 70/20/10:

```powershell
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.pipeline --raw-dir trustworthy_recsys/data --output-dir data/processed/only_70_20_10 --ratios 0.7 0.2 0.1
```

Repeat `--ratios` to compare additional choices. To use explicit calendar dates:

```powershell
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.pipeline --raw-dir trustworthy_recsys/data --output-dir data/processed/calendar_2020_2022 --validation-start 2020-01-01 --test-start 2022-01-01
```

Explicit cutoffs and ratios cannot be combined. Dates without a timezone are
interpreted as UTC; timezone-aware dates are converted to UTC. Whole-second
cutoffs are required. The partition rule is:

- Training: timestamp **before** validation start.
- Validation: timestamp **at or after** validation start and **before** test start.
- Test: timestamp **at or after** test start.

Optional flags include `--positive-threshold 4.0`, `--chunk-size 50000`,
`--checksums path/to/checksums.txt`, and `--source-readme path/to/README.txt`.
The default dataset name enforces the documented MovieLens 32M rating/user/movie
counts. `--dataset-name` can label a smaller MovieLens-shaped research fixture;
its provenance and Data Card must be reviewed for that source. This pipeline is
not a generic importer for unrelated datasets.

Changing the chunk size controls CSV working memory, not the logical data
selection. The implementation also keeps user/item counts and the largest
individual user's history in memory. Final contract validation loads one
evaluation population at a time through the harness; this is not a constant-memory
operation. Expect several minutes and roughly gigabytes of output space for the
full two-scenario run. Exact resource use depends on the environment.

## Generated artifacts

```text
movielens_phase1/
  movies.parquet
  links.parquet
  tags.parquet
  source_README.txt
  source_checksums.txt
  schema.json
  statistics.json
  manifest.json
  DATA_CARD.md
  ratio_80_10_10/
    train.parquet
    validation.parquet
    test.parquet
    train_tags.parquet
    train_item_ids.json
    validation_all.jsonl
    validation_warm.jsonl
    test_all.jsonl
    test_warm.jsonl
  ratio_70_20_10/
    ...same scenario-specific files...
```

Read the generated **DATA_CARD.md** for the overview, provenance, usage terms,
schemas, exact splits, measured eligibility counts, limitations, and risks.
`statistics.json` contains full dataset and per-partition counts, time ranges,
rating distributions, user/item activity summaries, cold-start rates, and every
evaluation exclusion count. Empty populations are recorded and written as empty
JSONL files, which the harness intentionally cannot score.

## Shared data contract and preprocessing

The interaction tables have four non-null fields:

| Field | Arrow type | Meaning |
|---|---|---|
| `user_id` | string | Original user ID in decimal form |
| `movie_id` | string | Original movie ID in decimal form |
| `rating` | float32 | Original explicit star rating |
| `timestamp_utc` | timestamp[ms, tz=UTC] | Rating submission time; source precision is seconds |

`schema.json` describes all tables, including metadata and tags. No integer
remapping or learned feature transformation is fitted. Movie genres become
lists; `(no genres listed)` becomes an empty list. Missing external IDs remain
null, and IMDb leading zeroes are preserved.

All valid rating rows are retained. The source must have its documented ascending
user/movie ordering and unique user-movie pairs. Bad checksums, missing required
rating fields, invalid ratings/IDs, invalid timestamps, and broken movie references
fail instead of triggering unreported deletions. There is no minimum-history filter.

Tag text is whitespace-trimmed, and blank tags are removed with counts reported.
Distinct tag events are retained. The root `tags.parquet` spans the whole release;
**do not use that whole table as training features**. Each scenario's
`train_tags.parquet` contains only tags strictly before its training cutoff.
These tags can still refer to movies outside the training-rated catalog; apply
`train_item_ids.json` as well when building training-catalog features.
Movie metadata remains a release snapshot with unknown historical availability.

## How the examples connect to the evaluator

The generated JSONL records match `eval.harness.load_evaluation_examples`:

```json
{"user_id":"42","history_items":["1","17"],"relevant_items":["50"]}
```

- IDs are strings, and each user appears once per file.
- `history_items` contains only **training ratings >= the relevance threshold**.
  Lower ratings remain in the Parquet files but are not represented as positive
  preferences in this item-only interface.
- `relevant_items` contains positive ratings from the selected held-out partition.
- Both lists are ordered by timestamp, then numeric movie ID to break ties.
- Validation and test use the same training-only history. No validation interactions
  or earlier test events are added to the history.
- Users without positive held-out labels are counted and excluded because the
  harness requires nonempty relevance labels.
- The `all` files retain eligible cold users, empty positive histories, and cold
  relevant movies. The `warm` files require a user with **any** training interaction,
  and retain only relevant movies that occur in training. A warm user can still have
  an empty positive history. Remaining empty relevance sets are counted and excluded.

The warm files change the scoring population and relevance set. Do not compare
their scores with all-population scores as though they answered the same question.
For a controlled model comparison across ratios, also establish a common eligible
test population and candidate universe. If retraining on train+validation, export
a separately documented protocol instead of reusing these training-only histories.

`train_item_ids.json` lists movies with any training rating. It is an available
candidate universe, not a restriction enforced by the harness. Seen-item filtering,
fallback behavior, cold-item candidates, and ranking remain recommender responsibilities.
The full Parquet history is available when a model needs all consumed movies or low
ratings, not just the positive IDs accepted by the current request interface.

A model must separately produce predictions in the harness format:

```json
{"user_id":"42","ranked_items":["50","32","260"]}
```

Prediction users must match the chosen examples exactly and ranked item IDs must
not repeat. To score a model's real prediction file, for example:

```powershell
.\.venv\Scripts\python.exe -m eval.harness --examples-file data/processed/movielens_phase1/ratio_80_10_10/test_all.jsonl --predictions-file results/model_test_predictions.jsonl --dataset movielens-32m --split ratio_80_10_10_test_all --recommender-name your_model --data-origin observed --output-dir results/model_evaluation
```

That command requires an actual prediction file; the data pipeline does not invent
model predictions. The existing `data/golden_set_50.jsonl` stays separate and
unchanged. Its qualitative `reference_items` are not quantitative relevance labels.

## Verification and reproducibility

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The pipeline tests check exact timestamp boundaries, cold users/items, positive
label construction, low-rating-only training users, tag availability, identifier
preservation, rejection of bad checksums and duplicate pairs across batch boundaries,
logical equality across chunk sizes, and integration with the actual evaluation harness.

At runtime, every split must account for all input ratings and respect the cutoffs.
Written Parquet schemas and row counts are checked. Evaluation label totals must
match held-out positive-rating totals, and nonempty JSONL files are loaded through
the real harness. Source checksums are checked again at completion.

The manifest records inputs, configuration, actual software versions, Git revision,
dirty working-tree status, source-code hashes, and output hashes. Exact binary
Parquet hashes can vary with chunk size or library versions, while logical records
remain the same. Run timestamps also vary intentionally. The synthetic tests verify
logical equality rather than requiring identical run timestamps.

Generated data is ignored by Git. Keep the code, tests, and instructions in version
control; use the manifest and generated Data Card when sharing a particular run,
subject to the source dataset's distribution terms.
