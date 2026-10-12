# System Card — Trustworthy Recommendation System application

**Status:** M3 local research prototype with learned retrieval integrated  
**Last updated:** October 5, 2026

## System overview and purpose

This system is a movie-discovery research prototype. A visitor selects liked MovieLens movies, may add structured genre and exclusion preferences, and receives an ordered recommendation list. It is intended for low-stakes exploration and course-project evaluation, not consequential decision-making, child-directed use, or claims about a person's psychology.

The application exposes three clearly labeled backends:

- synthetic fixtures for interface development;
- a MovieLens genre-weighted popularity baseline; and
- Learned two-tower retriever, served through Weaviate.

The learned application path is:

```text
React/Vite
  → FastAPI
  → two-tower retrieval
  → Weaviate ANN
  → hard filtering
  → pass-through reranking
  → final constraint validation
  → MovieLens metadata
  → React UI
```

LambdaRank is not integrated. The current reranker is explicitly pass-through. Natural-language preference interpretation is also not implemented; the structured preference controls remain functional without it.

## Architecture and components

- **React/Vite interface:** collects liked movies and structured preferences, displays recommendations, and reports the active backend and warnings.
- **FastAPI boundary:** validates requests, selects the configured backend, coordinates filtering and reranking, validates final output, and exposes health/readiness status.
- **Retrievers:** fixture data, a genre-weighted popularity baseline, or the learned two-tower model.
- **Weaviate:** stores and searches the learned model's saved movie embeddings. The application supplies vectors; Weaviate does not train the model or call an external embedding API.
- **Hard filter:** removes liked movies, explicit exclusions, and items that violate active structured genre constraints.
- **Pass-through reranker:** preserves candidate order after filtering. It is not LambdaRank.
- **Metadata and artwork:** MovieLens supplies canonical IDs, titles, and genres. Optional TMDb enrichment supplies display artwork only.

## Data flow

1. The browser stores an opaque session identifier and preference state for the browser session, then sends the complete state with each request.
2. FastAPI validates field bounds, duplicate and unknown IDs, supported preference values, and contradictory constraints.
3. The configured retrieval backend produces a bounded candidate list.
4. The application removes history items and candidates that violate exclusions or hard genre preferences.
5. The pass-through reranker preserves the remaining candidate order.
6. Final validation ensures output IDs are known, unique, drawn from the candidate set, and compliant with the request.
7. The service joins MovieLens metadata and may add optional TMDb artwork before returning results to the interface.

In learned mode, an empty history—or a profile containing no movies supported by the learned input catalog—uses the retriever's documented popularity fallback. A mixed profile can still use the learned path when at least one supported movie remains.

## Intended use

Appropriate uses include:

- local exploration of MovieLens recommendations;
- integration and regression testing of retrieval components;
- offline comparison of learned retrieval and matched popularity behavior; and
- demonstrating explicit user control through structured preferences.

The system is not intended to infer sensitive traits, provide high-stakes recommendations, or operate as a production consumer service.

## External dependencies

- **MovieLens:** supplies historical interactions and catalog metadata. Its dataset terms and known coverage limitations apply.
- **Weaviate:** must be reachable for learned mode. A database or artifact mismatch makes that mode not ready rather than silently changing algorithms.
- **TMDb (optional):** enrichment is available for real MovieLens-backed baseline and learned modes when the matching data run contains `links.parquet`. Exact MovieLens-to-TMDb mappings are used.

TMDb is display-only. Its availability and responses must not affect recommendation IDs, ordering, scores, filtering, or model readiness. Missing tokens, mappings, artwork, or provider availability degrade to neutral artwork without disabling recommendations. Fixture identities are not sent for enrichment.

## Privacy and retention

- The browser uses an opaque session ID; there are no user accounts.
- The application has no preferences database and does not persist preference histories or prompts.
- Browser preference state is session-local.
- Raw histories and prompt bodies are not intentionally included in application logs. Operational metadata such as request IDs, status, component versions, counts, durations, and error categories may be logged.
- `TMDB_READ_ACCESS_TOKEN` remains in the FastAPI process and is not returned to the browser or intentionally logged.
- When enabled, the backend sends exact TMDb movie IDs to TMDb. It does not send MovieLens user IDs, browser session IDs, preference histories, genre preferences, or prompts.
- Artwork metadata uses a bounded in-memory cache and is not persistent application preference storage.

A production deployment would require explicit retention, access-control, provider-review, and incident-response policies.

## Validation, safety, and recovery controls

- Request validation rejects invalid bounds, duplicate or unknown movie IDs, unsupported vocabulary, and contradictory structured preferences.
- Learned startup checks prepared-input and model compatibility, saved vectors, catalog metadata, scenario, and the remote Weaviate mapping. A failure is exposed through readiness.
- Filtering occurs before reranking and final validation repeats critical output constraints.
- The reranker may not introduce IDs outside the candidate list.
- Candidate retrieval is bounded; partial output is returned with a warning rather than padded with unchecked items.
- TMDb failures are isolated from recommendation generation.
- The interface retains preferences across request failures, supports retry/reset, and avoids applying stale responses to newer preference state.
- Health, readiness, backend mode, component identity, warnings, and server timing fields support local diagnosis.

Automated tests cover serving contracts, filtering, fallback and partial-output cases, artifact compatibility, learned retrieval integration, artwork isolation, frontend state transitions, and connected browser flows. These controls reduce known implementation risks but do not constitute production security or reliability certification.

## Evaluation evidence

The published retrieval evaluation for the documented 80/10/10 checkpoint reports:

- all-user Recall@100 of **0.2335**, compared with **0.2227** for matched-policy popularity;
- warm-user Recall@100 of **0.2403**, compared with **0.1613** for matched-policy popularity;
- learned-path p95 retrieval latency of **35.51 ms**, including Weaviate database requests;
- validation top-100 ANN/exact agreement of **99.90%**; and
- popularity fallback for about **79%** of test users under the documented eligibility rules.

These are offline retrieval-component measurements for one split and seed. They are not deployed end-to-end learned-application latency, cost, fairness, or user-study results. The [retrieval report](retrieval_report.md) and [Retrieval Model Card](model_card_retrieval.md) contain the full scope and methodology.

Application tests provide evidence for request validation, constraint enforcement, recovery behavior, backend status reporting, and artwork isolation. Fixture output is not recommendation-quality evidence. Existing local fixture or baseline timing observations must not be interpreted as deployed learned-system measurements.

No user study, deployment benchmark, accessibility audit, live natural-language refinement evaluation, or production cost measurement has been completed.

## Operational constraints

- Learned mode requires compatible prepared inputs, a published model bundle, its matching MovieLens catalog, and reachable Weaviate.
- Baseline and learned artwork require the matching optional `links.parquet`; recommendation generation does not.
- The service reads configuration from process environment variables and does not automatically load `.env`.
- The TMDb token must remain server-side and must not be placed in a frontend `VITE_*` variable.
- This prototype assumes local operation and does not include production authentication, authorization, rate limiting, durable monitoring, or autoscaling.

## Known limitations and mitigations

- **No learned reranking:** pass-through reranking is not LambdaRank. The UI and status metadata label the active component rather than claiming learned ranking.
- **Frequent fallback:** empty and unsupported learned profiles use popularity fallback. Responses expose fallback status so it is not presented as learned retrieval.
- **Partial results:** bounded retrieval plus hard filtering can return fewer movies than requested. The service reports partial output instead of weakening constraints.
- **Limited preference expression:** structured genres and exclusions do not capture ratings, dislikes, chronology, or nuanced intent. The system does not claim otherwise.
- **No natural-language interpretation:** refinement text is unavailable until an interpreter is implemented and evaluated. Structured controls remain the supported path.
- **Catalog and historical bias:** MovieLens coverage, genres, popularity, and historical interactions can propagate limitations into recommendations. No fairness or subgroup-performance claims are supported.
- **External artwork risk:** TMDb latency, quotas, policy changes, and CDN behavior may affect presentation. Provider failures are isolated from recommendation selection.
- **No live LLM evidence:** there is no LLM provider, so there is no live refinement security, hallucination, privacy, latency, or cost evidence.
- **Local-only operations:** production authentication, rate limiting, retention controls, incident response, and monitoring remain future work.
