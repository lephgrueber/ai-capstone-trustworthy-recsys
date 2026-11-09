# OPE for the movie application

The technical path is implemented: durable assignment/event collection, a bounded featured-slot experiment, mature-outcome export, chronological splits, reward-model fitting and supported-policy evaluation. It uses our movie application's newly collected feedback. Existing MovieLens ratings do not contain exposure probabilities and are not retroactively converted into OPE logs. Open Bandit remains a separate fashion benchmark.

## What changes in the application

With collection off, recommendations behave as before. `observe` records deterministic assignments; those rows cannot evaluate alternative actions. `randomized` runs `featured_epsilon_v1`:

1. Retrieve, apply all user constraints and rank the bounded pool using the configured backend.
2. Reserve ordinary ranks 2 through the requested result count as a fixed grid.
3. Make the featured pool from the ordinary top result plus remaining ranked items outside that grid.
4. Choose the first featured candidate with probability `1 - epsilon + epsilon/K`; each other candidate receives `epsilon/K`. Default epsilon is 0.1. With ten candidates, probabilities are 0.91 and 0.01.
5. Display the draw only in the featured hero; the grid excludes it and stays identical for every possible draw. No post-draw substitutions are made.

This evaluates alternate featured selections within the exact saved pool and presentation, not a different retriever, index, entire slate, or long-term policy. Hard constraints still apply. Empty responses create no assignment; singleton pools are logged but excluded from alternative-policy OPE. Fixture traffic is always synthetic. Baseline and learned data are exported separately.

## Enable the local movie pilot

First configure the learned backend using [the README](../README.md#learned-movie-serving), and start Docker/Weaviate. Set these in the same PowerShell terminal before launching the backend:

```powershell
$env:TRUSTWORTHY_RECSYS_FEEDBACK_MODE = "randomized"
$env:TRUSTWORTHY_RECSYS_FEEDBACK_DB = "data/feedback/movie.sqlite3"
$env:TRUSTWORTHY_RECSYS_FEEDBACK_EPSILON = "0.1"
$env:TRUSTWORTHY_RECSYS_FEEDBACK_TRAFFIC = "pilot"
python -m uvicorn trustworthy_recsys.serving.app:app --host 127.0.0.1 --port 8000
```

Alternatively put these values into the local `.env` and launch Uvicorn with `--env-file .env`. Variables are read from the process environment; the application does not automatically read `.env`. Keep the already configured learned-model paths and TMDB settings. Start the frontend normally, then verify `/api/v1/application/status` reports `backend_mode: learned` and `feedback_mode: randomized`. The page displays a collection notice and About explains the stored data. `off` disables new collection; `observe` is available for deterministic instrumentation checks.

The SQLite file, write-ahead log and shared-memory sidecars stay under Git-ignored `data/feedback/`. Do not share them publicly: preferences, MovieLens IDs, pseudonymous sessions and the private token-signing secret are stored locally. SQLite is suitable for this local single-service pilot; this is not a production multi-instance ingestion service. No automatic retention deletion is implemented; the research operator controls retention and access.

## Decision and event contracts

Every decision has a server-generated UUID independent of the client-controlled request ID. It stores the pre-decision preferences, session HMAC, component/artifact provenance, fixed grid, complete ordered candidate IDs/scores/hash, action, probability and policy settings. The service commits this before returning the experimental response; a database failure returns an error, not an unlogged randomized recommendation.

The browser receives a per-decision capability token, keeps it out of URLs, and sends it to `POST /api/v1/feedback/events`. The server validates the token, movie membership, event type, time and event ID. It derives probabilities and rewards itself. A stable assignment key deduplicates recommendation retries within the same preference revision; reusing a key with different context is rejected. Event retries reuse their original IDs and payloads, so duplicate delivery does not duplicate rewards.

The featured card records a display acknowledgment after it is at least 50% visible for one continuous second while the document is visible. Details opens can occur before that acknowledgment and still count. Featured details/favorite/hide controls retain their originating assignment; ordinary grid and catalog-picker clicks are not featured rewards. Responses discarded as stale by the browser never start a display tracker.

A tab-scoped sessionStorage outbox retries network failures and survives reloads in that tab. Healthy heartbeats occur every 15 seconds until the 30-minute window ends. Permanent event rejection prevents a later complete-window certification. Background throttling, queue/storage failure, offline loss and tab closure can leave outcomes unknown. Browser events are self-reported; capability binding prevents cross-decision attribution, not determined client manipulation or verified human identity. Use synthetic labeling for automated traffic.

## Outcome and missing-data rule

Primary reward: at least one featured `details_open` within 30 minutes of assignment. Repeated opens count once. Favorite/hide events are secondary diagnostics, not alternate rewards chosen after seeing results. A details open indicates interest, not watching or satisfaction.

Wait 30 minutes plus a fixed 24-hour late-arrival allowance before exporting an assignment. A positive event establishes reward 1. Reward 0 requires a healthy stream with a window-end marker and no heartbeat gap over 45 seconds from assignment through window end. All other outcomes remain null/unknown. Relative client event times are bounded against server receipt times, with a small timing tolerance; browser timing and telemetry completeness remain assumptions.

Assignments are never selected based on visibility. A failed display or missing stream is not a dislike. The report evaluates scenarios with each unknown reward set to zero or one, producing missing-outcome sensitivity endpoints and a session-bootstrap sampling envelope. It does not label those endpoints as a standard confidence interval or identify a single policy value when outcomes are missing.

## Export and evaluate

From the repository root with the environment active:

```powershell
python -m trustworthy_recsys.ope.movie prepare --db-path data/feedback/movie.sqlite3 --backend learned --output-dir data/processed/movie_feedback_run1
python -m trustworthy_recsys.ope.movie evaluate --inputs-dir data/processed/movie_feedback_run1 --output-dir results/movie_ope/run1 --bootstrap 1000
```

Use new output folders for each run. The exporter takes a consistent read transaction across decisions/events, omits secrets/tokens and fingerprints the logical snapshot. It checks candidate hashes, grid disjointness, action alignment and the complete logging distribution. It emits train/validation/test JSONL, schema, statistics, a Data Card and a manifest. Approximate 80/10/10 splits are chronological; tied timestamps stay together. Before export, training and validation assignments are purged if their 30-minute outcome plus 24-hour late-arrival window reaches the next partition’s first assignment. This prevents using outcomes that were not available at the later decision time. Actual ratios after purging differ and removed counts are reported. A short collection can have empty train/validation partitions even after its outcomes mature; elapsed time alone does not guarantee a usable evaluation dataset. Decision IDs cannot overlap; browser sessions can cross boundaries and overlap counts are reported. No session ID is used as a model feature.

Statistics include mature/immature and excluded assignments, outcomes, displays, stream completeness, source event joins, duplicate retries, candidate support, minimum logging probability and provenance groups. Empty or insufficient data produces an explicit insufficient-data report. There is no fabricated result for a newly started pilot.

The reward model is a train-only smoothed per-movie details-open rate, with global-prior fallback for unseen movies. Smoothing strength is fixed at 20. It is a nuisance model for DM/DR, not a replacement for two-tower retrieval. Observed-outcome selection can bias this model when telemetry is missing; sensitivity reporting remains necessary.

The four predefined target policies are the logged policy, original first-ranked candidate, uniform candidates, and a train-fitted movie-CTR policy with fixed 10% exploration. Each distribution assigns zero probability outside its row's saved pool. Validation IPS selects a candidate only if validation outcomes are fully observed; otherwise selection is withheld. Test results never select or train the policy. The evaluator reports DM, IPS, SNIPS and DR, session-cluster bootstrap intervals/scenarios, effective sample size, weight concentration and clipping sensitivity. Sparse support or too few sessions produces an explicit uncertainty warning. The estimators reuse the existing implementation, consistent with [Open Bandit Pipeline's estimator definitions](https://github.com/st-tech/zr-obp/blob/master/obp/ope/estimators.py).

Do not deploy a candidate because one point estimate is higher. Inspect missing outcomes, uncertainty, support, informative positive counts, provenance consistency and user experience first. These one-step estimates are conditional on logged contexts; they do not establish long-term effects or correct unidentified repeated-person dependence across browser sessions.

## Reproduce the synthetic technical check

```powershell
python -m trustworthy_recsys.ope.movie_pilot --output-dir results/movie_ope/synthetic_check --decisions 100
python -m pytest tests/test_movie_ope.py -q
```

This creates explicitly synthetic fixture assignments, events and simulated time with availability gaps between the three partitions; it does not use real people or claim real movie-policy improvement. Exporting fixture traffic normally excludes it. The pilot explicitly uses `--include-synthetic` in the preparation function and every report is labeled synthetic. See [the validation record](movie_ope_validation.md).

For the actual browser-to-store path, from `frontend`:

```powershell
$env:PLAYWRIGHT_CHANNEL = "msedge"
npx playwright test --config playwright.feedback.config.ts
```

Use installed Edge, or omit that variable after installing Playwright Chromium. The test starts isolated ports 18000/15173 and a fresh synthetic feedback database, leaving the ordinary backend and real pilot logs untouched.

## What remains after implementation

Collect real, mature movie-application observations. Then run the exporter/evaluator and examine data quality and uncertainty. No real movie-policy value or improvement is claimed yet. The existing MovieLens retrieval-quality results and the Open Bandit results remain separate evidence.
