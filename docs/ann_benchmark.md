# ANN recall, latency and artifact reload verification

Measured November 9, 2026 on the local Windows CPU client and Docker Weaviate service. The frozen HNSW configuration is M=32, efConstruction=100, ef=128, cosine distance, no quantization. No index tuning used the test set.

A deterministic seed-42 validation sample selected 200 users; 199 had usable histories. Each query was repeated three times at each K (597 observations per K), after 10 warmups per index. ANN/exact execution order alternated. Both methods used identical exclusions. Search timings include RPC and seen filtering but exclude query encoding and model loading. Requests were sequential, not a concurrent load test.

| K | Exact overlap | ANN p50 / p95 / p99 (ms) | Exact p50 / p95 / p99 (ms) |
|---|---:|---|---|
| 100 | 99.8995% | 16.89 / 31.78 / 68.77 | 7.45 / 10.26 / 11.84 |
| 500 | 99.9618% | 47.12 / 90.09 / 154.46 | 7.53 / 10.41 / 12.25 |

## Targets, failures and interpretation

The speedup expectation failed: ANN was slower than exact search at both K values. At this 50,986-item scale, local matrix scoring is inexpensive and remote query/overfetch overhead matters. This experiment does not isolate the contributions of those costs or prove behavior at larger scale. We retain Weaviate as the requested persistent search component; we do not claim an observed speedup.

For this run, new engineering acceptance thresholds were specified before the repeated benchmark: at least 99% mean exact overlap and local search p95 at most 100 ms for each K. Both passed. These are provisional local targets, not pre-existing production SLOs. Top-500 p99 was 154.46 ms; a 100 ms tail target at p99 would fail. Future concurrency, larger-catalog and ef sweeps should use validation data and report failures rather than relaxing thresholds after measurement.

## Reproducible loading

The original and separately unpacked bundle both verified all remote mappings and vectors. Item/user mappings and saved embeddings matched; regenerated tower vectors matched within 1e-6. All 12 prediction probes matched exactly, including empty and unknown histories. Full verified loading took 14.15 seconds from the original files and 9.95 seconds from the bundle. These warm-machine measurements are not cold-machine startup guarantees.

The bundle contains local embeddings and descriptors, not the Docker volume. A teammate must publish the embeddings once to a new collection; see [team setup](team_model_setup.md). The raw MovieLens source run is additionally needed for evaluation, but not for standalone prediction.

## Evidence and reproduction

Full query measurements, versions, descriptor and hashes: `results/retrieval/ann_20261109/benchmark.json`. A compact tracked snapshot is [ann_benchmark.json](ann_benchmark.json). Run commands are in [retrieval.md](retrieval.md). Initial validation was blocked because Docker Desktop was stopped; starting Docker Desktop and `docker compose up -d` restored access without rebuilding or deleting the index.
