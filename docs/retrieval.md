# Two-tower retrieval pipeline

This extends the completed data pipeline with model inputs, training, Weaviate embedding storage/search, and evaluation. PyTorch creates the embeddings; Weaviate stores and searches them without an automatic vectorizer. Source splits and the qualitative golden set remain unchanged.

## Install and run

To use the existing trained checkpoint, follow [the team download/setup guide](team_model_setup.md). The versioned Git LFS bundle includes the model and prepared inputs, so you can skip preparation and training. The commands below are for rebuilding artifacts.

Use Python 3.11 or later with PyTorch, the pinned Weaviate Python client, and Docker Desktop using Linux containers. From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,retrieval]"
docker compose up -d
.\.venv\Scripts\python.exe -m trustworthy_recsys.data.retrieval_inputs --run-dir data/processed/movielens_phase1 --scenario ratio_80_10_10 --output-dir data/processed/retrieval_inputs_80 --max-targets-per-user 4
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.train --inputs-dir data/processed/retrieval_inputs_80 --output-dir artifacts/retrieval/two_tower_80 --epochs 4 --batch-size 256 --validation-users 512 --threads 2
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.publish --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80 --output-dir artifacts/retrieval/two_tower_80_weaviate --collection MovieLensTwoTower80V1
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.evaluate --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80_weaviate --output-dir results/retrieval/two_tower_80_weaviate --benchmark-users 200
$env:WEAVIATE_INTEGRATION = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

Replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` on macOS/Linux. If the source run is absent, follow [data_pipeline.md](data_pipeline.md) first. Existing destination directories are refused; choose fresh names for a rerun. An `INCOMPLETE` marker means the stage did not finish and its artifacts should not be consumed.

The existing local inputs and trained checkpoint can be reused: skip preparation and training and run publication once. The documented collection and published model already exist after the completed migration, so do not publish them again. Publication refuses existing collection names and never deletes a collection. A failed import leaves an incomplete output directory and may leave a partial collection; choose a new output path and collection name after resolving the failure.

## Run the saved model

Start Docker Desktop, then run:

```powershell
docker compose up -d
.\.venv\Scripts\python.exe -m trustworthy_recsys.retrieval.predict --inputs-dir data/processed/retrieval_inputs_80 --model-dir artifacts/retrieval/two_tower_80_weaviate --user-id demo-user --history-items 1 260 1196 --k 10
```

The output contains original MovieLens IDs. History items represent prior likes; omit `--history-items` to exercise popularity fallback. Every load checks the complete remote movie mapping and vectors against the local model artifacts, so initial loading is slower than a single query. If Weaviate is unavailable, loading fails explicitly rather than silently using another search backend.

## Local database and persistence

`docker-compose.yml` pins Weaviate 1.39.8. HTTP port 8080 and gRPC port 50051 are bound only to loopback; this development configuration permits anonymous local access. Movie vectors persist in the named Docker volume `weaviate_data`. `docker compose stop` stops the service while retaining data; `docker compose up -d` starts it again. Do not remove the volume if you want to keep the published collections.

The client defaults to localhost. `WEAVIATE_HOST`, `WEAVIATE_PORT`, and `WEAVIATE_GRPC_PORT` can point it at another self-hosted instance; the current connector is for unauthenticated HTTP/gRPC, not an authenticated Weaviate Cloud deployment. Credentials are not stored in artifacts.

Each versioned collection holds `movie_id` (original string ID), `item_index` (integer row in the embedding matrix), `embedding_fingerprint` (catalog/vector digest), and a self-provided 32-dimensional vector for this model. It does not store user histories or held-out labels. `weaviate.json` records the collection, dimensions, counts, versions, vector fingerprint and index configuration. The Docker volume is the search database; the local embedding matrix supports auditing, exact-reference evaluation and rebuilding a collection.

The project uses the 80/10/10 temporal split for preparation, training, and evaluation. All-user and warm metrics use different eligibility rules and relevance sets; compare each model with its baseline within the same population.

## What each stage produces

| Stage | Outputs | Contract |
|---|---|---|
| Prepare | `schema.json`, `metadata.json`, `DATA_CARD.md`, `manifest.json`; `.npy` features/offsets/counts; `.bin` histories/targets; ID JSON mappings | Contiguous training-only IDs; original string IDs retained in mappings; float32 features; int32 item/user indices; int64 offsets and Unix seconds |
| Train | `model.pt`, `item_embeddings.npy`, `training.json`, sampled pair/user IDs, `manifest.json` | Row `i` of the normalized embedding matrix is item `i` in `item_ids.json`; validation selects a checkpoint using exact cosine search |
| Publish | A new model directory with copied weights/embeddings, updated `training.json`, `weaviate.json` and `manifest.json`; a versioned database collection | No retraining; every remote movie ID and vector is verified before the publication is marked complete |
| Evaluate | `report.json`, `REPORT.md`, `MODEL_CARD.md`, `predictions.jsonl`, `error_cases.json`, `manifest.json` | Existing harness metrics and prediction schema; full eligible test population; exact-index benchmark uses validation |

`metadata.json` contains model-input statistics: training rows, positive rows, fitted users/items, prepared pairs, users without a usable prefix, and equal-time/first-event exclusions. The model-ready Data Card supplements the immutable source Data Card with preprocessing and model-feature limitations.

All arrays are local artifacts with native NumPy numeric storage; use the generated schema to open binary arrays. `ModelInputs` memory-maps the large histories. Input, model and report manifests include SHA-256 hashes. The loader rejects incomplete or mismatched input/model artifacts.

## Training and selection

The preparer streams all training ratings, retains every rated item for serving exclusions, and selects at most four evenly spaced eligible positive targets per user. Positivity is inherited from the source pipeline (four stars for the local run). Each target's feature vector uses strictly earlier positive events; timestamp ties never enter its context. This produces bounded training cost without claiming to train on every event.

The user tower learns from mean fractional genres and log history length. The item tower learns movie identity and genre features. Both yield normalized vectors. In-batch softmax uses empirical target-frequency correction and masks duplicate targets and historical positives. Other unobserved items are sampled training alternatives, not proven dislikes.

Training defaults to all prepared pairs, four epochs, seed 42 and two CPU threads. `--max-pairs N` explicitly caps the selected pairs for a smoke experiment; the cap and selected pair indices are recorded. A deterministic hash sample of up to 512 warm validation users selects the checkpoint by Recall@100. New training runs use exact NumPy cosine search for checkpoint selection, so training does not require a running database. The publish stage stores the selected embeddings in Weaviate HNSW with max_connections=32, ef_construction=100, ef=128, cosine distance and quantization disabled. Seen items are excluded by adaptive overfetch and client-side filtering. The Compose search-result limit accommodates the complete 50,986-movie catalog.

The migrated checkpoint retains its original weights and validation selection history. It was originally selected with the previous search implementation. Publication preserves that configuration in `source_training_index` and records the source model manifest and publication code separately. It does not pretend that the old checkpoint was selected using the new exact-search training path.

The integration follows Weaviate's [self-provided vector workflow](https://docs.weaviate.io/weaviate/quickstart/local) and [vector-index configuration](https://docs.weaviate.io/weaviate/config-refs/indexing/vector-index). The movie-identity table uses sparse PyTorch embeddings with SparseAdam.

## Evaluation and interpretation

The evaluator scores every user in `test_all.jsonl`. It reports all, warm, learned-history, fallback, history-length, and head/tail-label slices. Warm eligibility matches the source pipeline: a training-observed user and at least one training-catalog relevant item, with relevance restricted to that catalog. Head movies are the top 10% of the training catalog by positive count; tail metrics use the remaining in-catalog relevant labels. Head and tail populations can overlap.

The reference baseline is positive training-count popularity under the **same** catalog and all-training-rated seen-item filter. The old baseline CLI may use a different filtering policy, so its previously saved scores need not match this comparison. All relevant test labels stay in the all-population denominator, including movies the index cannot retrieve.

The ANN benchmark measures overlap with exact NumPy filtered top-100 recommendations on a deterministic validation sample. This is separate from relevance Recall@100. Serving and search timings report p50/p95/p99 in milliseconds and include database RPC/network time. Model loading, disk I/O and the initial full remote vector audit are excluded. The validation benchmark performs ten warmups; test timings include the first test request. No production latency guarantee is implied.

Integrity checks verify fingerprints, every prepared temporal prefix, and every test ranking's catalog membership, uniqueness and seen exclusion. Behavioral robustness probes cover cold/unknown histories, duplicates, order changes and long repeated histories. They do not establish robustness to future population drift. The report classifies misses into out-of-catalog versus in-catalog retrieval failures and saves 20 worst-recall cases with deterministic tie breaking.

A validation ablation keeps every second history item while preserving the full training-rated exclusion list. Comparing its Recall@100 with the original histories tests sensitivity to incomplete profile information without changing the candidate eligibility policy.

`RetrievalRecommender` implements the existing harness callable API:

```python
from eval.harness import RecommendationRequest
from trustworthy_recsys.retrieval.recommender import RetrievalRecommender

with RetrievalRecommender.load(
    "data/processed/retrieval_inputs_80", "artifacts/retrieval/two_tower_80_weaviate"
) as recommender:
    recommendations = recommender(RecommendationRequest("example-user", ["1", "260"]), 100)
```

Generated prediction rows use `{"user_id": "...", "ranked_items": ["..."]}`, so the existing saved-predictions harness can independently rescore them. The curated golden set remains qualitative and is not used as observed relevance or as training data.

## Measured artifacts

The working model and descriptor are Git-ignored under `artifacts/retrieval/two_tower_80_weaviate`; its search index lives in the Weaviate Docker volume. The distributable model and complete prepared inputs are packaged separately in `models/two_tower_80_v1.zip`, tracked with Git LFS, with an adjacent SHA-256 checksum. Full reports and predictions remain Git-ignored under `results/retrieval/two_tower_80_weaviate`. Reviewable snapshots are in [retrieval_report.md](retrieval_report.md), [model_card_retrieval.md](model_card_retrieval.md), and [data_card_retrieval.md](data_card_retrieval.md). A fresh clone can download the trained bundle and populate a local database without the raw dataset or retraining. Previous local model/report directories are preserved as historical provenance and are not the active serving backend.
