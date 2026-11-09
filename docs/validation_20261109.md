# Data, retrieval and search validation — November 9, 2026

## Completed work

1. **Open Bandit:** re-prepared random/all/position-1 data with a different chunk size and independently verified every selected record, temporal split, action mapping, schema, count and artifact hash. All 458,005 exposures reproduced. The existing OPE policy selection and estimates were preserved.
2. **Saved retrieval assets:** unpacked the trained Git LFS bundle, checked its hashes, loaded both original and extracted model/input artifacts against live Weaviate, and compared 12 prediction probes. Mappings, saved vectors, regenerated model vectors and predictions matched.
3. **Quality and robustness:** evaluated all 22,015 eligible MovieLens test users, checked all 676,684 training prefixes and every prediction, and passed eight behavioral robustness probes. Recall@100 was 0.233487 versus popularity 0.222729. The learned-history population scored 0.166589 versus 0.114512. About 79.34% of all users used popularity fallback.
4. **ANN comparison:** repeated exact/Weaviate comparisons at top-100 and top-500 on 199 usable validation histories. Agreement was 99.90% and 99.96%; local ANN p95 was 31.78 and 90.09 ms. Both met the new local agreement/p95 thresholds. The expected speed advantage failed: exact search was faster. Full retrieval p99 reached 360.25 ms in the separate test run.
5. **Search:** implemented bounded genre/exact-title interpretation and a main-page search bar with a microphone icon. Voice produces editable text and waits for submission. A live learned-mode request for science-fiction movies like The Matrix returned 12 scored, filtered recommendations. No retraining was needed.
6. **Documentation:** updated MovieLens/Open Bandit/model-ready Data Cards, retrieval Model Card, measured retrieval report, README and reproduction instructions.

## Evidence locations

| Evidence | Location |
|---|---|
| Fresh Open Bandit input run | `data/processed/open_bandit_20261109` |
| Independent OPE data audit | `results/ope/validated_20261109`; tracked [snapshot](ope_validation.json) |
| Full retrieval evaluation, predictions and 20 error cases | `results/retrieval/validation_20261109` |
| Repeated ANN measurements and reload probes | `results/retrieval/ann_20261109`; tracked [summary](ann_benchmark.json) |
| Learned text-search API smoke check | `results/retrieval/language_smoke_20261109.json` |
| Main-page desktop/mobile screenshots | `frontend/test-results/search-desktop.png`, `search-mobile.png` |

Large per-event artifacts remain local and ignored; the tracked cards/reports summarize the measured results. This was validation on the existing Windows machine, not a newly provisioned machine. The [earlier onboarding audit](readme_audit.md) has a different scope.

## Automated checks

- Backend: `WEAVIATE_INTEGRATION=1 python -m pytest -q -rs` — 171 passed; two optional LightGBM modules skipped because the dependency is absent.
- Frontend: `npm test` — 18 passed.
- Production frontend: `npm run build` — passed, including TypeScript checking.
- Isolated search browser checks: two passed in Microsoft Edge, including desktop/mobile layout and mock speech-to-search submission.
- `git diff --check` — passed. Dependency deprecation warnings remain.

## Failures encountered and fixes

- Docker Desktop was stopped. Starting it and the Compose service restored the persisted index; no volume was deleted.
- Bare pytest discovered an archived copy of the repository inside downloads and reported duplicate test-module imports. The project now explicitly restricts discovery to `tests`.
- One frontend test timed out while other validations were running. The next run passed all 18 tests without increasing its timeout.
- Build/test tooling needed filesystem access outside the sandbox. The final production build passed after that access was granted.
- Browser checks initially found port 8000 occupied and an incompatible Windows executable path. The new search-specific configuration uses isolated ports 18000/15173 and a quoted platform-appropriate Python path.
- Playwright's matching Chromium executable was absent and its download stalled. The download was cancelled; both browser checks passed using installed Microsoft Edge. The speech provider is mocked in the automated browser check.
- Adding resolved title metadata exposed a stale API test expectation and optional generated TypeScript fields. Both were corrected, and OpenAPI/client types were regenerated.

## Remaining limitations and next steps

A real microphone and browser-provider transcription check is still manual; no speech accuracy or arbitrary-language understanding claim is made. The parser supports genres and exact-title seeds, not actor/plot/mood search. Production concurrency and repeated-machine latency tests remain future work. Optional LightGBM reranking is outside this retrieval validation and is not installed in this environment. Open Bandit remains a separate fashion benchmark; the subsequent [movie OPE implementation](movie_ope.md) now collects randomized featured assignments and response events. Real mature movie-user data is still needed.

Restart the application backend after updating the code, then refresh the frontend to see the top-of-page search bar. Existing models and Weaviate collections can be reused.
