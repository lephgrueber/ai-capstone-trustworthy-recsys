# Trustworthy Recommendation System

A recommendation-system capstone exploring retrieval, ranking, and trustworthy offline evaluation.

Developed for CIS 5980: AI Capstone at the University of Pennsylvania under the AI Engineering track.

## Project and current status

The system now retrieves candidate movies using a trained two-tower model and approximate nearest-neighbor search. The planned next components are LambdaRank reranking and a user-facing interface. The broader evaluation plan includes direct method (DM), inverse propensity scoring (IPS), self-normalized IPS (SNIPS), and doubly robust (DR) estimation.

The repository currently includes:

- A MovieLens 32M exploratory analysis notebook.
- A runnable Python pipeline for source validation, preprocessing, temporal splitting, and evaluation-example exports.
- Generated dataset/split statistics, schemas, provenance metadata, and a Data Card.
- An evaluation harness with Recall@K and binary NDCG@K, plus synthetic and saved-prediction modes.
- A uniform-random baseline recommender for the harness's `Recommender` callable API.
- Popularity baselines (all-time, recent-window, and Bayesian-average top-rated) for the same API.
- A genre-aware baseline that boosts popular movies matching each user's genre history.
- A curated 50-case qualitative golden set and automated tests.
- Training-only model inputs, a trained CPU two-tower model, and Weaviate embedding storage/search.
- Retrieval quality, latency, integrity, robustness and error-analysis reports, plus a retrieval Model Card.

Uniform-random, popularity, genre-aware and two-tower retrieval are implemented. The evaluation-slice file remains a placeholder; retrieval-specific slices are implemented in the retrieval evaluator. LambdaRank, the user interface, and policy-value estimators are not implemented. MovieLens does not supply the logged propensities needed for the planned IPS/SNIPS/DR experiments; that work requires a separate suitable evaluation dataset.

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
│           └── ratio_80_10_10/           # Splits and evaluation exports
├── docs/
│   ├── data_pipeline.md                 # Detailed pipeline and evaluation instructions
│   ├── retrieval.md                     # Model inputs, training, index and evaluation commands
│   ├── retrieval_report.md              # Measured retrieval report snapshot
│   ├── model_card_retrieval.md           # Retrieval Model Card snapshot
│   ├── data_card_retrieval.md            # Model-ready Data Card snapshot
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
│   ├── test_retrieval.py                 # Temporal, training, ANN and harness integration
│   └── test_smoke.py
├── trustworthy_recsys/
│   ├── baselines/
│   │   ├── genre.py                     # Genre-weighted popularity recommender
│   │   ├── popularity.py                # Popularity and top-rated recommenders
│   │   ├── predict.py                   # Write baseline predictions for the harness
│   │   └── random_baseline.py           # Uniform-random recommender
│   ├── data/
│   │   ├── audit.py                     # Source integrity and full ratings audit
│   │   ├── preprocess.py                # Metadata, tag cleaning, and Parquet conversion
│   │   ├── split.py                     # UTC calendar and ratio-based cutoffs
│   │   ├── schema.py                    # Shared table and evaluation contracts
│   │   ├── evaluation.py                # Quantitative JSONL example exports
│   │   ├── reporting.py                 # Statistics, hashes, and code provenance
│   │   ├── data_card.py                 # Generate a readable Markdown Data Card
│   │   ├── pipeline.py                  # Pipeline command-line entry point
│   │   ├── retrieval_inputs.py          # Training-only features, pairs and CSR histories
│   │   ├── ratings.csv                  # Local source data; Git-ignored
│   │   ├── movies.csv
│   │   ├── tags.csv
│   │   └── links.csv
│   ├── retrieval/
│   │   ├── model.py                     # User and item towers
│   │   ├── train.py                     # Training and validation checkpoint selection
│   │   ├── index.py                     # Exact cosine reference for training/evaluation
│   │   ├── weaviate_store.py            # Versioned vector storage, verification and search
│   │   ├── publish.py                   # Import a trained model's vectors into Weaviate
│   │   ├── predict.py                   # Request recommendations from the saved model
│   │   ├── recommender.py               # Harness-compatible callable and fallback
│   │   ├── evaluate.py                  # Quality, timing, checks and error analysis
│   │   └── model_card.py                # Generate the retrieval Model Card
│   └── evaluation/
│       ├── metrics.py                   # Recall@K and binary NDCG@K
│       └── slices.py                    # Placeholder
├── README.txt                           # Local MovieLens documentation and usage terms
├── checksums.txt                        # Local MovieLens source checksum manifest
├── pyproject.toml                       # Package configuration and dependencies
├── docker-compose.yml                  # Persistent local Weaviate service
├── artifacts/retrieval/                 # Models and collection descriptors; Git-ignored
├── results/retrieval/                   # Local reports and predictions; Git-ignored
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

The pipeline verifies the four source files against `checksums.txt`, audits the ratings, and generates the chronological **80/10/10** split by default. It preserves the raw files and retains all valid ratings. Blank tags are removed with counts reported; historical tag exports exclude tags recorded at or after the training cutoff.

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

The project uses the 80/10/10 temporal split. All-user and warm-start scores describe different evaluation populations and relevance sets. See the run instructions and Data Card for the full eligibility rules.

Each completed local run under `data/processed/<run_name>/` contains:

- **Data Card** — `data/processed/<run_name>/DATA_CARD.md`: overview, provenance, usage terms, structure, splits, limitations, and risks.
- **Statistics** — `data/processed/<run_name>/statistics.json`: dataset quality, preprocessing counts, split sizes, cold-start coverage, and evaluation exclusions.
- **Schema** — `data/processed/<run_name>/schema.json`: field types and evaluation contracts.
- **Manifest** — `data/processed/<run_name>/manifest.json`: configuration, environment versions, source/code/output hashes, and validation results.

These generated paths are available when the local run exists. New runs produce the same core reports. A run is complete only when `manifest.json` exists and the `INCOMPLETE` marker is absent.

## Two-tower retrieval

**Teammates can use the saved model without retraining.** Download the Git LFS bundle at `models/two_tower_80_v1.zip` and follow [the team setup guide](docs/team_model_setup.md). It includes weights, movie embeddings, training configuration, mappings, manifests and the complete prepared inputs needed by the loader. After extraction, import the saved vectors into your own local Weaviate instance once.

Install the optional training/index dependencies with `python -m pip install -e ".[dev,retrieval]"` in your virtual environment. The runnable stages are:

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.retrieval_inputs --run-dir data/processed/movielens_phase1 --scenario ratio_80_10_10 --output-dir data/processed/retrieval_inputs_80
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.train --inputs-dir data/processed/retrieval_inputs_80 --output-dir artifacts/retrieval/two_tower_80
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.publish --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80 --output-dir artifacts/retrieval/two_tower_80_weaviate --collection MovieLensTwoTower80V1
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.evaluate --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80_weaviate --output-dir results/retrieval/two_tower_80_weaviate
```

These paths are used by the local run; choose fresh destination names and a new collection name to rebuild. Preparation generates its own schema, statistics, provenance and Data Card. Training selects a checkpoint on validation only and saves weights and normalized movie embeddings. Publication imports those embeddings into a versioned Weaviate HNSW collection and verifies every stored ID/vector. Evaluation reports full-test and warm retrieval quality, a matched popularity baseline, latency including database requests, ANN agreement with exact search, integrity and robustness checks, and error cases.

The existing trained model has already been published; you can run it without retraining or republishing. Start Docker Desktop, then:

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.predict --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80_weaviate --history-items 1 260 1196 --k 10
```

Weaviate runs on loopback HTTP/gRPC ports 8080/50051 and persists embeddings in a named Docker volume. The model supplies its own vectors; no embedding API or automatic vectorizer is used. An unavailable database raises a connection error. To run the live database tests, set `$env:WEAVIATE_INTEGRATION = "1"` before `python -m pytest -q`; CI starts the service and runs them automatically.

The measured run uses 80/10/10 and at most four historical targets per training user. Empty histories use popularity, and unseen movies are outside the candidate index.

Read the [run instructions](docs/retrieval.md), [retrieval report](docs/retrieval_report.md), [retrieval Model Card](docs/model_card_retrieval.md), and [model-ready Data Card](docs/data_card_retrieval.md). Binary artifacts and full prediction files remain local and Git-ignored.

The Weaviate run achieved all-user Recall@100 of **0.2335** versus **0.2227** for matched-policy popularity, and warm Recall@100 of **0.2403** versus **0.1613**. Learned-path p95 latency was **35.51 ms**, including database requests; validation top-100 agreement with exact search was **99.90%**. About 79% of test users require fallback, and long-tail recall remains a limitation. Results cover one split and seed; historical local-search timings are not Weaviate timings.

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
