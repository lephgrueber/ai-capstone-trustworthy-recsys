# Trustworthy Recommendation System

A recommendation-system capstone exploring retrieval, ranking, and trustworthy offline evaluation.

Developed for CIS 5980: AI Capstone at the University of Pennsylvania under the AI Engineering track.

## Project and current status

The planned system will retrieve candidate movies using a two-tower model and approximate nearest-neighbor search, rerank them with LambdaRank, and expose recommendations through a user-facing interface. The broader evaluation plan includes direct method (DM), inverse propensity scoring (IPS), self-normalized IPS (SNIPS), and doubly robust (DR) estimation.

The repository currently includes:

- A MovieLens 32M exploratory analysis notebook.
- A runnable Python pipeline for source validation, preprocessing, temporal splitting, and evaluation-example exports.
- Generated dataset/split statistics, schemas, provenance metadata, and a Data Card.
- An evaluation harness with Recall@K and binary NDCG@K, plus synthetic and saved-prediction modes.
- A curated 50-case qualitative golden set and automated tests.

The baseline model files and evaluation-slice file are still placeholders. The two-tower model, ANN index, LambdaRank reranker, user interface, and policy-value estimators have not been implemented. MovieLens does not supply the logged propensities needed for the planned IPS/SNIPS/DR experiments; that work requires a separate suitable evaluation dataset.

## Project structure

The tree below shows the main source files and current local data locations. Python package initializer files and development caches are omitted for readability.

```text
ai-capstone-trustworthy-recsys/
├── .github/
│   └── workflows/ci.yml                  # Install the project and run tests
├── data/
│   ├── golden_set_50.jsonl               # Curated qualitative regression scenarios
│   └── processed/
│       └── <run_name>/                    # Generated local run; Git-ignored
│           ├── DATA_CARD.md
│           ├── schema.json
│           ├── statistics.json
│           ├── manifest.json
│           ├── movies.parquet
│           ├── links.parquet
│           ├── tags.parquet
│           ├── source_README.txt
│           ├── source_checksums.txt
│           ├── ratio_80_10_10/           # Splits and evaluation exports
│           └── ratio_70_20_10/           # Same layout for the other ratio
├── docs/
│   ├── data_pipeline.md                 # Detailed pipeline and evaluation instructions
│   └── milestone 1/
│       ├── Trustworthy_Recommendation_System_Proposal.pdf
│       └── Trustworthy_Recommendation_System_Pitch_Deck.pdf
├── eval/
│   └── harness.py                       # Command-line evaluation and JSON reporting
├── notebooks/
│   └── eda.ipynb                        # Exploratory analysis and split comparisons
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_golden_set.py
│   ├── test_harness.py
│   ├── test_metrics.py
│   └── test_smoke.py
├── trustworthy_recsys/
│   ├── baselines/
│   │   ├── knn.py                       # Placeholder
│   │   └── popularity.py                # Placeholder
│   ├── data/
│   │   ├── audit.py                     # Source integrity and full ratings audit
│   │   ├── preprocess.py                # Metadata, tag cleaning, and Parquet conversion
│   │   ├── split.py                     # UTC calendar and ratio-based cutoffs
│   │   ├── schema.py                    # Shared table and evaluation contracts
│   │   ├── evaluation.py                # Quantitative JSONL example exports
│   │   ├── reporting.py                 # Statistics, hashes, and code provenance
│   │   ├── data_card.py                 # Generate a readable Markdown Data Card
│   │   ├── pipeline.py                  # Pipeline command-line entry point
│   │   ├── ratings.csv                  # Local source data; Git-ignored
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   └── evaluation/
│       ├── metrics.py                   # Recall@K and binary NDCG@K
│       └── slices.py                    # Placeholder
├── README.txt                           # Local MovieLens documentation and usage terms
├── checksums.txt                        # Local MovieLens source checksum manifest
├── pyproject.toml                       # Package configuration and dependencies
├── .gitignore
├── CODE_OF_CONDUCT.md
├── LICENSE                              # Software license
└── README.md
```

The source CSVs, dataset documentation, and generated run are local artifacts; a fresh clone may not contain them. There is currently no `scripts/` directory. Evaluation result directories are created when the harness runs.

## Quick start

Requires Python 3.11 or later. From the repository root, create an environment and install the project and test dependencies.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

**macOS/Linux:**

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

The project installs NumPy, pandas, and PyArrow. To work interactively with the EDA, also install `matplotlib` and `ipykernel` in the environment used by your notebook editor. Select that environment as the kernel and open [notebooks/eda.ipynb](notebooks/eda.ipynb).

## Data pipeline and Data Card

The pipeline verifies the four source files against `checksums.txt`, audits the ratings, and generates both chronological **80/10/10** and **70/20/10** splits by default. It preserves the raw files and retains all valid ratings. Blank tags are removed with counts reported; historical tag exports exclude tags recorded at or after the training cutoff.

With the current local source layout, run:

```powershell
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.pipeline --raw-dir trustworthy_recsys/data --output-dir data/processed/my_run
```

Use `.venv/bin/python` on macOS/Linux. Choose a new output directory for each run: existing directories are refused. The command expects the dataset `README.txt` and `checksums.txt` at the repository root unless their paths are supplied explicitly. Source acquisition, custom ratios, calendar cutoffs, and other options are explained in [docs/data_pipeline.md](docs/data_pipeline.md).

Each scenario directory contains:

| Artifact | Purpose |
|---|---|
| `train.parquet`, `validation.parquet`, `test.parquet` | All valid interactions assigned by UTC timestamp |
| `train_tags.parquet` | Cleaned tags strictly before the training cutoff |
| `train_item_ids.json` | Movie IDs observed in training |
| `validation_all.jsonl`, `test_all.jsonl` | All users with positive held-out labels, including cold-start cases |
| `validation_warm.jsonl`, `test_warm.jsonl` | Training-observed users with relevant movies restricted to training-observed items |

The shared interaction columns are `user_id`, `movie_id`, `rating`, and `timestamp_utc`. IDs are strings. Evaluation histories contain positive training interactions only, and held-out relevance defaults to ratings **>= 4 stars**. Both validation and test use training-only histories. Users without positive held-out labels are counted and excluded because the harness requires nonempty relevance labels.

Both ratios use the same test period, but their training histories and warm-start populations differ. Scores over different populations should not be treated as a controlled model comparison. See the run instructions and Data Card for the full eligibility rules.

Each completed local run under `data/processed/<run_name>/` contains:

- **Data Card** — `data/processed/<run_name>/DATA_CARD.md`: overview, provenance, usage terms, structure, splits, limitations, and risks.
- **Statistics** — `data/processed/<run_name>/statistics.json`: dataset quality, preprocessing counts, split sizes, cold-start coverage, and evaluation exclusions.
- **Schema** — `data/processed/<run_name>/schema.json`: field types and evaluation contracts.
- **Manifest** — `data/processed/<run_name>/manifest.json`: configuration, environment versions, source/code/output hashes, and validation results.

These generated paths are available when the local run exists. New runs produce the same core reports. A run is complete only when `manifest.json` exists and the `INCOMPLETE` marker is absent.

## Evaluation harness and golden set

`eval/harness.py` combines pipeline-generated evaluation examples with separately generated recommendation rankings, scores the rankings with Recall@K and binary NDCG@K, and writes timestamped JSON results. It does not generate recommendations. The harness is model-agnostic and split-agnostic: the supplied examples file determines the split and evaluation population being scored.

### Input and prediction format

A pipeline-generated evaluation example has this form:

```json
{"user_id":"123","history_items":["1","50"],"relevant_items":["260"]}
```

A separately generated prediction row is:

```json
{"user_id":"123","ranked_items":["318","356","527"]}
```

Prediction users must match evaluation users. Item IDs are strings, and ranked items must be unique and ordered highest-ranked first. Row order does not matter because the evaluator matches by `user_id`; short or empty rankings are accepted and scored without padding. See the Data Pipeline and Data Card section above for evaluation-population and label-construction details.

### Metrics and evaluation protocol

**Recall@K** measures the fraction of held-out relevant movies appearing in the top K recommendations. **Binary NDCG@K** gives more credit when held-out relevant movies appear nearer the top of the ranking. Their cutoffs are independently configurable; the intended project evaluation uses **Recall@100** and **NDCG@10** on one supplied ranked list.

Use a validation split for development evaluation and keep the test split held out for final evaluation. The harness does not enforce validation versus test; maintaining that separation is evaluation protocol discipline.

### Commands

Run the deterministic synthetic smoke check with:

```bash
.venv/bin/python -m eval.harness
```

Its expected output is:

```text
Recall@10 = 1.0000
NDCG@10   = 0.7606
```

For a saved-prediction evaluation, use:

```bash
.venv/bin/python -m eval.harness \
  --examples-file data/processed/<run_name>/<split_scenario>/validation_warm.jsonl \
  --predictions-file <predictions.jsonl> \
  --dataset movielens-32m \
  --split <validation_split_label> \
  --recommender-name <recommender_name> \
  --data-origin observed \
  --recall-k 100 \
  --ndcg-k 10 \
  --output-dir results/<recommender_name>
```

`validation_warm.jsonl` is one example evaluation population; another pipeline-generated validation artifact can be supplied. Replace the placeholder paths and labels with those for the evaluation run being scored.

Each run writes a timestamped JSON file containing run metadata, aggregate Recall/NDCG scores, and per-example rankings and scores.

### Golden set

[`data/golden_set_50.jsonl`](data/golden_set_50.jsonl) contains 50 curated synthetic recommendation cases: 15 `golden_path`, 20 `representative`, and 15 `hard_edge`. It is intended for qualitative and regression review and remains separate from quantitative MovieLens validation evaluation. Its `reference_items` are illustrative reviewer guidance, not quantitative relevance labels, and must not be used to calculate Recall@K or NDCG@K.

## Tests and reproducibility

The tests cover pipeline boundaries and data contracts, output consistency across batch sizes, golden-set structure, ranking metrics, and harness input/reporting behavior. GitHub Actions installs the package and runs the test suite on Python 3.12.

Pipeline manifests record the actual configuration and software versions. Repeated runs with the same settings should produce the same logical records, while processing timestamps and Parquet bytes can differ across batch sizes or library versions. Generated data and local environments are excluded from Git.

## License

The project software is licensed under the MIT License; see [LICENSE](LICENSE).

MovieLens has separate dataset usage terms and citation requirements. Consult the supplied dataset `README.txt`, the copy included with each run, and the generated Data Card. The software license does not replace the dataset terms.
