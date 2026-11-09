# Movie application OPE implementation validation

Completed November 9, 2026. The implementation is runnable, including the real movie-serving integration. Its validation traffic is synthetic; no real-user movie-policy improvement is claimed.

## Evidence

- **100-assignment synthetic pilot:** `results/movie_ope/synthetic_20261109_v3`. Exported 80 train, 10 validation and 10 test assignments, with simulated time gaps long enough for earlier outcomes to mature before the next phase. Test outcomes were two positive, six negative and two unknown. Logging-policy missing-outcome sensitivity endpoints were 0.20 and 0.40, correctly preserving uncertainty from the two unknown outcomes. Validation selection was withheld because validation outcomes were incomplete.
- **Trained model and live Weaviate:** a labeled synthetic learned-mode request produced 12 recommendations, with 25 eligible featured choices. The selected first candidate had exact propensity 0.904 (0.9 + 0.1/25). MovieLens genre/seen-item constraints passed. Repeating the assignment and event reused the existing records. Evidence: `results/movie_ope/learned_smoke_20261109.json`.
- **Browser to SQLite:** the Microsoft Edge test passed against isolated fixture servers and a fresh synthetic database. The UI sent acknowledged heartbeat, display and featured-details events to the real API. The event carried the correct decision/movie, the hero was excluded from the grid, and clicking grid details did not add a featured reward.
- **Known-value estimators:** a balanced synthetic two-action log recovered policy value 1.0 using IPS/SNIPS/DR even with a deliberately inaccurate reward prediction. Other tests cover idempotency, token/movie validation, storage failure, fixed-grid randomization, singleton/empty pools, late events, missing-window handling, chronological export with outcome-availability purging, observed/unknown bounds and insufficient-data reports.
- **Browser delivery tests:** stable retry IDs, tab-outbox restoration, display deduplication, original-decision attribution, complete-window heartbeats, permanent-rejection handling and an in-flight callback race during collector replacement are tested.

A compact, non-event-level [JSON snapshot](movie_ope_validation.json) is tracked. Raw databases and event-level exports remain ignored. The generated input/evaluation manifests identify code and outputs. Existing Open Bandit results and MovieLens retrieval benchmarks were not reinterpreted as movie-policy OPE evidence.

## Limitations found during implementation

The observation rule is deliberately conservative: many visitors will close or background the tab before 30 minutes, leaving unknown outcomes. The exporter retains those assignments and the evaluator provides missing-outcome scenarios instead of reporting an observed-only estimate as causal. This may produce wide ranges and withhold validation-based policy selection. Measure completeness during the pilot before concluding the data can compare policies.

Browser telemetry is self-reported and session IDs do not authenticate people. SQLite, local capability tokens, tab-scoped retries and visibility checks support this research prototype; they are not a production trust or delivery guarantee. No real policy-value report exists until genuine observations have matured and been exported. Collection is off by default; [the run guide](movie_ope.md) explains enabling it.


## Test execution

- Full backend regression with live Weaviate enabled: **178 passed**, two optional LightGBM modules skipped because LightGBM is absent.
- Final movie-specific tests after adding outcome-availability purging: **8 passed**, including a deliberately short collection that must return insufficient data.
- Frontend unit tests: **23 passed**.
- Real browser/API feedback integration: **1 passed** in installed Microsoft Edge.
- Final TypeScript/production frontend build: passed.
- Final 100-assignment synthetic export/evaluation: passed with availability gaps; all three partitions remained nonempty, and missing outcomes remained explicit.

The full regression preceded the final purge-specific case; that final change was checked by the complete movie-specific test module and a fresh end-to-end pilot. Existing HTTP-client deprecation warnings remain. No optional LambdaRank retraining or production load test was part of this change.
