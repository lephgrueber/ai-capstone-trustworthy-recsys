# System Card: Trustworthy Recommendation System

**Status:** Local release candidate tested with two-tower retrieval and LambdaRank

**Last updated:** November 15, 2026

## Overview and intended use

The application lets visitors explore movie recommendations without creating an account. Visitors select favorite movies, include or avoid genres, and hide individual titles.

It is a local course prototype for movie discovery and integration testing, not a production service.

## Architecture and data flow

FastAPI supports three backends: synthetic fixtures for development, a MovieLens genre-weighted popularity baseline, and a trained two-tower retriever using Weaviate.

The trained-model path is:

```text
React/Vite → FastAPI → two-tower/Weaviate retrieval
→ hard filtering → LambdaRank reranking (optional; pass-through otherwise)
→ final validation → recommendations with MovieLens metadata → UI
```

The browser sends the current preferences with each recommendation request. Favorite movies guide retrieval; genre selections and hidden titles constrain the results. In learned mode, an empty history uses popularity calculated from the training data.

Setting `TRUSTWORTHY_RECSYS_RERANKER_MODEL_DIR` enables the saved LightGBM LambdaRank model. With a candidate limit of 500, it reorders the retrieved movies that pass the hard filters. Its features include popularity, rating, release year, user activity, and user–movie features such as co-likes with the submitted favorites. Without LambdaRank, the app keeps retrieval order.

In the reported offline comparison, LambdaRank increased test NDCG@10 from 0.1057 to 0.1204 and Recall@10 from 0.0486 to 0.0657. See the [ranking report](ranking_report.md) for the evaluation details.

The search bar supports a limited set of English requests about genres and exact movie titles. Optional browser speech recognition fills the search field for editing before submission. See [text and voice search](natural_language_search.md).

MovieLens supplies movie IDs, titles, and genres. The frontend requests optional TMDb posters and backdrops separately, so recommendations can appear before the images finish loading.

## Privacy and retention

A random session identifier is stored in `sessionStorage`. When feedback is enabled, the browser also keeps server-issued decision tokens and pending events for that tab so failed submissions can be retried.

Preferences, results, and undo history remain in React memory. With feedback enabled, the backend separately saves the preferences submitted with the request and a keyed hash of the session identifier (HMAC) in local SQLite. That database is used for the feedback experiment; it does not create user accounts.

At INFO level, startup logs show the backend, retrieval and reranker versions, candidate limit, and whether artwork is configured. Request logs include the request ID, method, path, status, duration, and backend. Recommendation logs also record component versions, result count, whether results are partial, and total processing time. Startup failures and handled component or storage errors are logged separately.

Detailed retrieval, filtering, and reranking timings, along with response warnings, are available in API responses and saved workload results rather than normal application logs. These records help identify which request failed and which configuration was running. Normal application logs exclude credentials, raw histories, and submitted preference text. Log-retention and access rules for a public deployment have not been defined.

Browser speech recognition may send audio to its provider. The application receives the submitted transcript, not recorded audio. The microphone only starts after an explicit click.

The TMDb token stays on the backend. Artwork metadata requests send mapped movie IDs, not visitor histories or session identifiers. The browser downloads images directly from TMDb’s image service, which receives its network address and normal request metadata.

## Safeguards and recovery

The API rejects unknown or duplicate movie IDs, conflicting preferences, and inputs outside the defined limits. Hard filters remove favorite movies, hidden titles, and candidates that fail genre constraints. Final validation checks that returned movies are valid and that the reranker has not added movies outside the supplied candidate list.

If an update fails, the app keeps the earlier results and pending edits. Visitors can retry or reset their preferences. Request-sequence and revision checks prevent an older response from replacing newer results.

Backend labels, component information, warnings, and timing fields show which configuration is active.

## Threat model

Users or scripts may submit invalid preferences, outdated revisions, duplicate feedback, or forged identifiers. Request validation, catalog checks, candidate limits, hard filtering, and final output checks restrict what the API accepts and returns. The feedback API checks server-issued decision tokens and ignores duplicate events.

The app assumes its local data, model files, and configuration are trusted. It does not protect against someone who can modify those files.

Startup checks confirm that the model files and Weaviate collection are compatible before serving recommendations. Losing access to Weaviate can still interrupt recommendations after startup. Handled TMDb failures affect artwork, not ranking.

## Evaluation evidence

The retrieval evaluation reports higher Recall@100 than matched-policy popularity, with frequent popularity fallback in the offline test population. Results cover one split and seed. The [retrieval report](retrieval_report.md) and [Retrieval Model Card](model_card_retrieval.md) provide the metrics and methodology.

Application tests passed for validation, filtering, preference handling, and recovery. Some unit and API tests use mocks. The November 9 M4 browser check used the saved two-tower model, a live Weaviate collection, and the saved LambdaRank model. It covered favorites, a hidden title, genre constraints, and text refinement without mocking recommendation responses. Live TMDb artwork worked with a backend token, and the same learned flow passed without the token using neutral artwork.

The [M4 release report](m4_release_check_20261109.md) records request IDs, p50/p99 HTTP and component timings, a five-request concurrency check, browser and artwork results, logs, and cost assumptions. Feedback collection was off during these measurements. These checks test application behavior and local latency, not recommendation quality or performance under sustained load.

The regular browser artwork tests use mocked responses; the selected release tests use the real learned backend. For the browser measurement, timing starts before text is entered and ends after the recommendation response arrives and the preference summary is visible. It does not wait for every result card or artwork request to finish.

### Failure cases and mitigations

| Case | Result | Handling or limitation |
|---|---|---|
| Favorites, a hidden title, and genre constraints | Passed in the learned browser flow with the saved models and live Weaviate. | Hard filters and final validation enforce the constraints. This checks behavior, not recommendation quality. |
| Simulated HTTP 503 followed by Retry | Retry succeeded. | The app keeps earlier results and edits so the user can retry. Recovery after a real Weaviate outage has not been tested. |
| TMDb token disabled | Recommendations worked with neutral artwork. | Artwork is optional. This does not cover every possible provider failure. |
| Malformed local `links.parquet` | Code review identified a possible startup failure. This case was not tested. | Handling malformed local metadata remains an open startup risk. |

## Dependencies and operating limits

Learned mode needs compatible prepared inputs, saved model files, MovieLens catalog metadata, and a Weaviate publication. Weaviate must be reachable at startup and during vector search. Failed startup checks leave the backend unavailable rather than switching to the genre baseline. `/ready` confirms startup success, not ongoing database availability.

Missing TMDb credentials or artwork, and provider errors handled by the app, use neutral artwork. Malformed local metadata can still prevent startup, as noted above.

The app has no production authentication, rate limiting, or automated monitoring. The local SQLite feedback store is not designed for multiple application instances. Setup and run instructions are in the [README](../README.md).

### Reproducibility checklist

Follow the [README](../README.md) for commands and the [model setup guide](team_model_setup.md) for extracting the saved model and publishing its embeddings.

- [ ] Install the Python and frontend dependencies, including the ranking dependencies.
- [ ] Obtain a compatible MovieLens data run and extract the saved model bundle. No model retraining is required.
- [ ] Start Weaviate and use a compatible publication. Use fresh output directories and collection names when publishing saved embeddings.
- [ ] Configure learned retrieval and the saved LambdaRank model, set `TRUSTWORTHY_RECSYS_FEEDBACK_MODE=off` for measurements, and start the API and frontend.
- [ ] Check `/ready` and `/api/v1/application/status`. Submit favorites `1`, `260`, and `1196`; confirm `two-tower-weaviate`, `lambdarank`, and no popularity-fallback warning.
- [ ] Run the documented application checks and learned HTTP workload. Save new measurements separately from existing results. The recorded workloads used 10 warm-ups and 100 measured requests each.

## Residual risks and testing gaps

- **Limited personalization:** Favorites and structured controls capture basic preferences, not ratings, viewing chronology, or detailed intent. Exclusions filter results; they do not train the model to recognize dislikes.
- **Candidate coverage:** Filters can leave too few matching movies in the retrieved pool. The app returns partial results rather than relaxing the visitor’s constraints.
- **Data limitations:** MovieLens coverage and historical popularity can affect recommendations. The available data do not support conclusions about demographic fairness.
- **Untested interactions and failures:** Physical microphone capture and recovery from a real Weaviate outage remain untested.
- **Reproduction and load testing:** The learned release checks reused prepared data and services on the same Mac. Independent second-machine reproduction and sustained-load testing remain unverified.

## Movie feedback experiment

Feedback collection supports `off`, `observe`, and `randomized` modes. In randomized mode, the app chooses one featured movie from a recorded candidate pool, keeps the ordinary grid fixed, and excludes duplicates. The server saves the decision before returning the response. Feedback events are tied to a server-issued decision and token, and retries are deduplicated.

The page displays a collection notice, and About explains which preferences and interaction events are stored. Raw databases and exports are ignored by Git and should not be committed to the repository.

Movie-specific off-policy evaluation (OPE) uses movie-details opens within a 30-minute outcome window, with a further 24 hours allowed for late-arriving events. Missing event streams are treated as unknown outcomes, not negative responses. Reports include bounds for missing outcomes and uncertainty estimates grouped by session.

Synthetic validation passed, but it does not establish an improvement for real users. See the [movie feedback and OPE documentation](movie_ope.md).

## Monitoring planned for future use

There is no automated monitoring yet. Before making the app publicly available, we would track failed startup and recommendation requests, response times, partial results, fallback frequency, and failures in Weaviate, artwork requests, or feedback storage.

Request IDs and model versions would help trace problems. We would distinguish expected cold-start fallback from backend failures and check model, data, and collection versions when deploying changes. Logs should continue to exclude credentials and raw user preferences, with retention and access rules defined before public use.

Recommendation quality and bias would need separate, versioned evaluations. A successful HTTP response does not show that a recommendation was useful.

## Planned response to serious failures

If recommendations include movies that the user explicitly excluded, stop the app and investigate the request and filtering steps. Fix the problem or restore a previously verified version, then rerun the affected checks before restarting.

If private feedback data is exposed, stop the app and restrict access to the affected data. Do not include credentials or raw feedback records in shared bug reports.