# Retrieval evaluation report

Metrics use the existing harness implementation. Every eligible test user is included.

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

Serving latency in milliseconds (p50 / p95 / p99): {"fallback": {"p50": 0.07429998368024826, "p95": 0.1813000242691487, "p99": 0.36111001390963815}, "learned": {"p50": 22.693850012728944, "p95": 72.03073501586911, "p99": 360.2517660194988}}

Missed-label counts: {"outside_training_catalog": 182611, "in_catalog_not_retrieved": 1082335, "relevant_labels": 1561410, "hits_at_100": 296464}

Integrity: verified input/model/evaluation hashes, every training prefix, and every prediction. Test labels were used only for reporting.

Robustness checks: {"unknown_user_with_known_history": true, "unknown_items_ignored": true, "duplicate_history_invariant": true, "history_order_invariant": true, "empty_history_popularity_fallback": true, "all_unknown_history_fallback": true, "long_duplicate_history_invariant": true, "saved_embeddings_match_model": true}

Validation history ablation (every second history item retained; full seen filtering preserved): {"users": 199, "original_history_recall_at_100": 0.2605684661293538, "half_history_recall_at_100": 0.2446520326966398}

The two-tower user representation averages genres and loses item-level order. Empty histories use popularity; out-of-catalog positives cannot be retrieved. These are offline explicit-rating results, not causal, online, or demographic-fairness evidence.

See report.json for timings, provenance, model selection, and catalog coverage; error_cases.json contains 20 lowest-recall cases (ties prefer more labels).

## November 9, 2026 verification

This snapshot comes from `results/retrieval/validation_20261109`. The frozen checkpoint was evaluated without retraining or test-based tuning. See [ANN and reload comparison](ann_benchmark.md) for repeated top-100/top-500 timings and portable bundle loading. See [natural-language search](natural_language_search.md) for the bounded text/voice interface, which converts requests to existing preference inputs and does not change the model weights. These scores exclude the optional LambdaRank reranker and are not an evaluation of speech recognition or language understanding.

## Latency limitation observed in this rerun

The full learned path measured p50 22.69 ms, p95 72.03 ms and p99 360.25 ms, compared with the earlier run's p95 35.51 ms. The identical relevance results do not imply stable tail latency. The machine also performed other validation work during this session; these measurements do not identify a single cause. The search-only repeated benchmark is a different timing boundary and cannot replace this full-path result. Production concurrency and latency diagnosis remain follow-up work.
