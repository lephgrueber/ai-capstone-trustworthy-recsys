# Model Card — retrieval section

## Overview and intended use

This CPU two-tower movie retriever generates up to 100 unseen training-catalog candidates for offline recommendation research. It is a retrieval component, not a deployed recommendation policy or reranker.

## Architecture and inputs

The user tower maps a mean fractional-genre profile and log history length through a 64-unit ReLU MLP to a normalized 32-dimensional vector. The item tower combines a learned 16-dimensional movie-identity embedding with fractional genres, then uses a 64-unit ReLU MLP and the same output dimension. Scores are cosine similarities. Missing genres contribute zeros; unknown history IDs are ignored.

Weaviate stores self-provided movie embeddings in a versioned HNSW collection: cosine distance, max_connections=32, ef_construction=100, ef=128, quantization disabled. Every training-observed movie is indexed, including movies without positive targets. Adaptive overfetch removes all previously rated training items and supplied history items on the client. Empty or entirely unknown histories use positive training-count popularity with numeric-ID tie breaks.

## Training data and procedure

Scenario: `ratio_80_10_10`. The model was trained on 676,684 of 676,684 prepared pairs for 4 epochs; checkpoint 4 was selected by validation Recall@100 over 512 deterministic hash-sampled warm users.

Preparation selects at most four evenly spaced eligible positive targets per user by default. Contexts include only positive ratings strictly earlier than the target timestamp; equal-time ratings are excluded. Only training IDs and training-movie genres are fitted. Validation and test histories remain training-only, with no updates from validation events.

Training uses in-batch sampled softmax, temperature 0.1, empirical target-frequency logQ correction, and masking of duplicate targets and historical positives. AdamW trains the dense towers; SparseAdam trains movie identities. Seed 42; batch size 256; learning rate 0.002; CPU threads 2. The target frequency correction and in-batch negatives approximate a sampled objective; unrated items are not established dislikes.

The published checkpoint is reused without retraining. Publication records the source model manifest and original search configuration in training.json. Current retrieval and latency measurements come from Weaviate. New training runs use exact cosine search for validation checkpoint selection, independently of the database.

## Retrieval quality

Metrics are macro user means from the existing evaluation harness. All and warm populations have different relevance sets. The matched popularity baseline uses the identical candidate catalog and seen-item filtering.

| Method / population | Users | Recall@100 | NDCG@10 |
|---|---:|---:|---:|
| two_tower/all | 22,015 | 0.2335 | 0.2669 |
| popularity/all | 22,015 | 0.2227 | 0.2619 |
| two_tower/warm | 4,310 | 0.2403 | 0.1057 |
| popularity/warm | 4,310 | 0.1613 | 0.0785 |
| two_tower/learned | 4,548 | 0.1666 | 0.0965 |
| popularity/learned | 4,548 | 0.1145 | 0.0723 |

## Efficiency and index accuracy

Mean ANN overlap with exact filtered top-100 is 0.9989949748743719 on 199 validation users. This measures approximation fidelity, not relevance Recall@100. Detailed embedding/search and full-serving p50/p95/p99 timings are in report.json; model loading and disk I/O are excluded.

Learned-path latency (milliseconds): p50=18.487150000055408, p95=35.51438000001781, p99=45.781578000146496. Popularity fallback served 17,467/22,015 test users (79.34%). Interpret the aggregate as a hybrid retriever plus fallback system.

## Integrity, robustness and error analysis

Input, model and evaluation hashes are checked. Every remote movie ID and vector is checked against the saved catalog and embeddings at load time. Every prepared context is audited for a strictly earlier timestamp and target exclusion. Every test ranking is checked for duplicate, out-of-catalog and previously seen items. Checkpoints are selected on validation only. Tests cover a synthetic timestamp tie, artifact tampering, training, database publication and the harness contract.

Behavioral probes cover unknown users/items, empty histories, duplicate/reordered/long histories, and saved-embedding agreement. report.json records each result; these probes do not establish resistance to distribution shift. error_cases.json contains the 20 lowest-recall users, with history length, missing-label catalog coverage and example recommendations.

A validation history ablation retained every second historical item while preserving full training-rated exclusions. Recall@100 was 0.2605684661293538 with original histories versus 0.2446520326966398 with half histories, over 199 learned-path users. This measures sensitivity to missing profile information, not a retrained-model comparison.

## Limitations and risks

- User features average genres, so they discard sequence order and item-specific affinities. Items without sampled targets have untrained identity embeddings, although shared genre layers still apply.
- Cold users frequently receive the same popularity recommendations. Warm-only results exclude much of the future population. Unseen movies cannot enter this index; catalog coverage limits recall.
- Movie metadata is a release snapshot. Its availability at historical event times is unknown, even though rating events are temporally separated.
- Explicit ratings are selected feedback. Popularity and exposure bias remain; offline relevance does not establish online satisfaction, causal policy value or demographic fairness.
- These results cover the project's temporal scenario and one seed. Repeated-seed uncertainty analysis and online validation remain future work. The small validation sample introduces checkpoint-selection uncertainty.
- Latency includes Weaviate RPC/network time and client processing; model loading and the full remote vector audit are excluded. This local client/server measurement is not a production concurrency or service-level guarantee.

## Licensing, provenance and reproduction

MovieLens terms and citation requirements apply to source data and derived artifacts; see the source and model-ready Data Cards. The repository software license does not replace dataset terms. No external pretrained weights are used.

training.json records architecture, hyperparameters, selected checkpoint, package versions, code fingerprints, input hash and validation hash. Model and report manifests fingerprint generated artifacts. See docs/retrieval.md for commands and report.json for the exact evaluation protocol. Bit-for-bit reproducibility across library versions or hardware is not promised.
