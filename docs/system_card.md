# System Card: Trustworthy Recommendation System

**Status:** M3 local prototype with learned retrieval integrated  
**Last updated:** October 25, 2026

## Overview and intended use

The application lets visitors explore movie recommendations without creating an account. Visitors select favorite movies, include or avoid genres, and hide individual titles.

It is intended for local movie discovery, course evaluation, and integration testing, not consequential decisions or production use.

## Architecture and data flow

FastAPI supports three backends: synthetic fixtures for development, a MovieLens genre-weighted popularity baseline, and a trained two-tower retriever using Weaviate.

The personalized learned path is:

```text
React/Vite → FastAPI → two-tower/Weaviate retrieval
→ hard filtering → LambdaRank reranking (optional; pass-through otherwise)
→ final validation → recommendations with MovieLens metadata → UI
```

The browser sends the submitted preference state with each recommendation request. Liked movies guide retrieval, while genre selections and hidden titles constrain the results. An empty history uses training-popularity fallback in learned mode.

When `TRUSTWORTHY_RECSYS_RERANKER_MODEL_DIR` is set, a LightGBM LambdaRank model reorders the top 500 two-tower candidates using training-period features: popularity, rating, release year, user activity, and user × movie crosses such as co-likes with the submitted favorites. Offline, it raised test NDCG@10 from 0.1057 to 0.1204 and Recall@10 from 0.0486 to 0.0657 (docs/ranking_report.md). It can only reorder candidates that already passed hard filtering. Without it, a pass-through reranker preserves retrieval order. Preference refinement uses structured controls.

MovieLens supplies movie IDs, titles, and genres. The frontend requests optional TMDb posters and backdrops separately, allowing recommendations to appear before the images finish loading.

## Privacy and retention

Only a random session identifier is stored in `sessionStorage`. Preferences, results, and undo history remain in React memory and are lost on page reload. There is no persistent visitor-preference database.

Logs may contain request IDs, status, component versions, counts, timings, and errors. Raw histories and prompt bodies are not intentionally logged. Production log-retention and access policies have not been established.

The TMDb token stays on the backend. Artwork metadata requests send mapped movie IDs, not visitor histories or session identifiers. The browser downloads images directly from TMDb’s image service, which receives its network address and ordinary request metadata.

## Safeguards and recovery

The API rejects unknown or duplicate movie IDs, conflicting preferences, and inputs outside the defined bounds. Hard filters remove liked movies, hidden titles, and candidates that fail genre constraints. Final validation checks that returned movies are valid and that a reranker has not introduced candidates outside the supplied list.

Failed updates preserve earlier results and pending edits. Visitors can retry or reset their preferences. Request-sequence and revision checks prevent older responses from replacing newer results.

Backend labels, component information, warnings, and timing fields make the active configuration visible.

## Evaluation evidence

The published retrieval evaluation reports higher Recall@100 than matched-policy popularity, with frequent popularity fallback in the offline test population. Results cover one split and seed. The [retrieval report](retrieval_report.md) and [Retrieval Model Card](model_card_retrieval.md) provide the metrics and methodology.

Application tests passed for validation, filtering, preference handling, and recovery. Some unit and API tests used mocks. Browser tests used the genre baseline, with controlled responses for selected failure cases.

A separate browser check used the trained two-tower model and Weaviate. The returned recommendations appeared in the interface, excluded the selected favorites, and did not use popularity fallback. Recommendation responses were not mocked, and artwork access was disabled.

These are functional checks, not recommendation-quality measurements. Available application timings cover fixture and genre-baseline modes. Learned-application timings and a complete operating-cost estimate remain pending.

## Dependencies and operational constraints

Learned mode requires compatible prepared inputs, a published model, matching MovieLens catalog metadata, and Weaviate access during startup and vector search. Failed startup checks leave the backend unavailable rather than switching algorithms. `/ready` reports successful startup, not continuous database health. Recovery from an outage after startup has not been tested.

TMDb artwork is optional. Missing credentials, missing artwork, and handled provider failures use neutral artwork. A malformed local `links.parquet` can still prevent startup; this risk was identified in code review and has not been tested.

The prototype runs locally without production authentication, rate limiting, or continuous monitoring. Setup and run instructions are in the [README](../README.md).

## Known limitations and mitigations

- **Limited personalization:** Favorites and structured controls capture basic preferences, not ratings, viewing chronology, or detailed intent. Exclusions filter results rather than train the model to recognize dislikes.

- **Candidate coverage:** Filters can leave too few matching movies in the retrieved pool. The application reports partial results rather than relaxing the visitor’s constraints.

- **Data limitations:** MovieLens coverage and historical popularity can affect recommendations. These limitations are documented in the retrieval evaluation. The available data do not support conclusions about demographic fairness.