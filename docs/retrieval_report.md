# Retrieval evaluation report

Metrics use the existing harness implementation. Every eligible test user is included.

This report measures the existing selected checkpoint after publishing its 50,986 movie vectors to Weaviate 1.39.8, collection `MovieLensTwoTower80V1`, using Python client 4.23.1. The model was not retrained. The 80/10/10 source split and 22,015 eligible test users are unchanged. Historical local-search reports remain in their original directories; their latency and index-accuracy figures do not describe this backend.

The original checkpoint trained on 676,684 prepared examples for four epochs; epoch 4 was selected on 512 warm validation users. Its original selection history is retained in training.json. New training runs use exact cosine validation search; publication is a separate step.

All-user Recall@100 is 0.2335 versus 0.2227 for matched-policy popularity; warm Recall@100 is 0.2403 versus 0.1613. These are descriptive single-run results, not a statistical-significance claim. Popularity fallback serves 17,467 users (79.34%); learned retrieval serves 4,548 users. Warm labels are restricted to the training catalog, while learned-path metrics retain all relevant labels.

| Method / population | Users | Recall@10 | Recall@100 | NDCG@10 |
|---|---:|---:|---:|---:|
| two_tower/all | 22015 | 0.0501 | 0.2335 | 0.2669 |
| popularity/all | 22015 | 0.0479 | 0.2227 | 0.2619 |
| two_tower/fallback | 17467 | 0.0546 | 0.2509 | 0.3112 |
| popularity/fallback | 17467 | 0.0546 | 0.2509 | 0.3112 |
| two_tower/history_0 | 17467 | 0.0546 | 0.2509 | 0.3112 |
| popularity/history_0 | 17467 | 0.0546 | 0.2509 | 0.3112 |
| two_tower/head_labels | 21507 | 0.0620 | 0.2884 | 0.2747 |
| popularity/head_labels | 21507 | 0.0578 | 0.2695 | 0.2691 |
| two_tower/tail_labels | 13979 | 0.0003 | 0.0026 | 0.0003 |
| popularity/tail_labels | 13979 | 0.0000 | 0.0000 | 0.0000 |
| two_tower/learned | 4548 | 0.0325 | 0.1666 | 0.0965 |
| popularity/learned | 4548 | 0.0219 | 0.1145 | 0.0723 |
| two_tower/warm | 4310 | 0.0486 | 0.2403 | 0.1057 |
| popularity/warm | 4310 | 0.0315 | 0.1613 | 0.0785 |
| two_tower/history_21_plus | 4277 | 0.0314 | 0.1610 | 0.0929 |
| popularity/history_21_plus | 4277 | 0.0207 | 0.1071 | 0.0680 |
| two_tower/history_1_20 | 271 | 0.0504 | 0.2542 | 0.1535 |
| popularity/history_1_20 | 271 | 0.0420 | 0.2311 | 0.1401 |

ANN agreement with exact filtered top-100 on validation: 0.9989949748743719

Serving latency in milliseconds (p50 / p95 / p99): {"fallback": {"p50": 0.058200000239594374, "p95": 0.12807000002794675, "p99": 0.19373599998289143}, "learned": {"p50": 18.487150000055408, "p95": 35.51438000001781, "p99": 45.781578000146496}}

Missed-label counts: {"outside_training_catalog": 182611, "in_catalog_not_retrieved": 1082335, "relevant_labels": 1561410, "hits_at_100": 296464}

Integrity: verified input/model/evaluation hashes, every training prefix, and every prediction. Test labels were used only for reporting.

Robustness checks: {"unknown_user_with_known_history": true, "unknown_items_ignored": true, "duplicate_history_invariant": true, "history_order_invariant": true, "empty_history_popularity_fallback": true, "all_unknown_history_fallback": true, "long_duplicate_history_invariant": true, "saved_embeddings_match_model": true}

Validation history ablation (every second history item retained; full seen filtering preserved): {"users": 199, "original_history_recall_at_100": 0.2605684661293538, "half_history_recall_at_100": 0.2446520326966398}

The two-tower user representation averages genres and loses item-level order. Empty histories use popularity; out-of-catalog positives cannot be retrieved. These are offline explicit-rating results, not causal, online, or demographic-fairness evidence.

See report.json for timings, provenance, model selection, and catalog coverage; error_cases.json contains 20 lowest-recall cases (ties prefer more labels).

## Interpretation and operating limits

Learned-path latency is 18.49 ms at p50, 35.51 ms at p95, and 45.78 ms at p99. This includes Weaviate RPC/network time and client filtering but excludes model loading and the full initial remote audit. Docker and the client run on this Windows machine; this is not a production concurrency benchmark. Fallback p95 is 0.128 ms because it does not issue a vector query.

The validation benchmark used 199 histories from a 200-user warm sample; one had no usable positive history. Top-100 agreement with exact NumPy cosine search was 99.90%. Median database search/filter time was 16.11 ms versus 7.22 ms for local exact search. Weaviate provides persistent vector storage and a service interface; it is not faster than local exact search at this catalog size in this measurement.

Of 1,561,410 relevant user/movie labels, 296,464 were retrieved, 1,082,335 were in the catalog but missed, and 182,611 were outside the training catalog. These counts weight labels, while the main metric table averages users. Recommendations covered 7.85% of the catalog. Tail-label Recall@100 was only 0.0026, so popularity concentration remains a weakness. Head movies are the top 10% of catalog movies by positive training count; head/tail user populations overlap.

On 199 validation users, keeping every second history item reduced Recall@100 from 0.2606 to 0.2447 while preserving full training-rated exclusions. This tests sensitivity to missing profile information, not a retrained-model comparison. Remote vectors and movie mappings were verified against the saved model; behavioral probes passed. The test suite additionally checks changed database mappings and local artifact tampering.

The authoritative machine-readable report is [report.json](../results/retrieval/two_tower_80_weaviate/report.json), with [error cases](../results/retrieval/two_tower_80_weaviate/error_cases.json) and [predictions](../results/retrieval/two_tower_80_weaviate/predictions.jsonl). Generated files are Git-ignored. See [retrieval.md](retrieval.md) for Docker startup, publication, prediction, evaluation and persistence instructions.
