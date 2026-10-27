# Reranking evaluation report

LightGBM LambdaRank reranks the top 500 two-tower candidates for each user. This report compares it with **pass-through** (the two-tower order, which the application currently serves) on the same users and candidates. Metrics are macro user means using the repository definitions: the Recall@K and binary NDCG@K denominators count every relevant held-out movie, including ones retrieval missed.

## Protocol

| Step | Data | Users |
|---|---|---:|
| Features | Training-period interactions only (before 2018-10-04) | — |
| Fit | `validation_warm` labels (2018-10-04 to 2020-11-05), 90% of users by stable hash | 5,374 with a positive candidate |
| Early stopping and ablations | the other 10% of validation users | 670 |
| Final evaluation, run once | `test_warm` labels (from 2020-11-06) | 4,310 |

- **Candidates:** exact top-500 two-tower retrieval over the saved embeddings, with all training-rated movies excluded.
- **Label:** 1 when the candidate is among the user's held-out ratings of 4 or more.
- **Model:** 210 boosting rounds selected by holdout NDCG@10; training takes about 24 s on 8 CPU threads.
- **Integrity:** the pipeline run is verified byte-for-byte against the retrieval model's training Parquet before tables are built. Pass-through test metrics reproduce the retrieval report's warm row (NDCG@10 0.1057, Recall@10 0.0486).
- **Missing labels:** a movie the user never rated counts as a miss even if they would have liked it.

## Test results (4,310 warm users)

| Method | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | Recall@50 |
|---|---:|---:|---:|---:|---:|
| Pass-through (two-tower order) | 0.1057 | 0.0486 | 0.1074 | 0.0851 | 0.1599 |
| **LambdaRank** | **0.1204** | **0.0657** | **0.1267** | **0.1119** | **0.1993** |

| LambdaRank − pass-through | Mean | 95% paired-bootstrap CI |
|---|---:|---|
| NDCG@10 | +0.0147 (+14%) | [+0.0104, +0.0191] |
| Recall@10 | +0.0172 (+35%) | [+0.0130, +0.0214] |
| Recall@50 | +0.0394 (+25%) | [+0.0341, +0.0445] |

The candidate ceiling (Recall@500) is 0.5156.

Gains appear in every history-length slice. They are largest for users with 1–20 liked movies: NDCG@10 0.158 → 0.205 on 270 users.

## Validation holdout versus test

On the validation holdout, NDCG@10 improved from 0.1473 to 0.2112 (+0.064), more than four times the test gain. The likely cause is temporal distance. Features are frozen at 2018-10-04. Validation labels fall 0–2 years after that date, while test labels fall 2–5 years after it, and test-user histories also stop in 2018 under the current export. Popularity and release-year signals learned for 2018–2020 transfer only partly to 2020–2023.

**Recommended next experiment:** recompute features at the test start using training plus validation interactions, and extend test-user histories the same way. The validation-period ratings are not test labels.

## Feature ablations (validation holdout, 670 users)

| Variant | NDCG@10 | Δ vs full |
|---|---:|---:|
| Full model | 0.2112 | — |
| Without user × item crosses (genre match, co-likes, recent-like cosine, year/popularity gaps) | 0.1985 | −0.0127 |
| Without retrieval score and rank | 0.2013 | −0.0099 |
| Without co-occurrence only | 0.2044 | −0.0067 |
| Without user activity features | 0.2056 | −0.0055 |
| Without recent popularity | 0.2072 | −0.0039 |
| Without release year | 0.2081 | −0.0030 |
| Pass-through | 0.1473 | −0.0638 |

The crosses, which the two towers cannot express, contribute most. Differences below about 0.005 are within noise for 670 users.

## Reproduce

```bash
pip install -e ".[retrieval,rank]"
python -m trustworthy_recsys.ranking.dataset --run-dir data/processed/movielens_phase1 \
  --inputs-dir data/downloads/two_tower_80_v1/inputs --model-dir data/downloads/two_tower_80_v1/model \
  --examples data/processed/movielens_phase1/ratio_80_10_10/validation_warm.jsonl \
  --output-dir artifacts/ranking/validation_warm_500        # repeat with test_warm.jsonl
python -m trustworthy_recsys.ranking.train --table-dir artifacts/ranking/validation_warm_500 \
  --output-dir artifacts/ranking/lambdarank_v1 --threads 8
python -m trustworthy_recsys.ranking.evaluate --table-dir artifacts/ranking/test_warm_500 \
  --model-dir artifacts/ranking/lambdarank_v1 --output results/ranking/lambdarank_v1_test_warm.json
```

Building a table takes about 34 ms per user. On macOS, LightGBM runs single-threaded in any process that has already imported PyTorch, because two OpenMP runtimes in one process can crash.

## Limits

These are offline explicit-rating results from one temporal split and one model fit. They are not online or causal evidence. Only warm users are evaluated, and new visitors use the popularity fallback. The reranker is not yet wired into the FastAPI application.
