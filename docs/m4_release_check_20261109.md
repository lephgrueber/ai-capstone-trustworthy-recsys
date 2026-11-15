# Milestone 4 learned release check — 2026-11-09

## Scope and tested revision

The tested source is the base commit
`84366c245ea071a0b4181b9352cae43c9f449a51` (`Add movie feedback processing
and validation modules`, 2026-11-09 06:10:28 -0500) **plus the local, unstaged
changes listed below**. The release worktree was clean before those changes, so
no earlier uncommitted work was included. Nothing was committed, pushed, tagged,
or packaged as an application container.

The check ran on a Mac mini (Apple M4, 10 cores, 24 GB RAM), macOS 27.0.1
(26A434), arm64. The isolated environment used Python 3.13.7, PyTorch 2.10.0,
LightGBM 4.7.0, Weaviate client 4.23.1, FastAPI 0.162.2, Node 24.22.2, Docker
29.8.2, Docker Compose 5.5.1, and Homebrew `libomp` 23.1.5.

## Reused artifacts and services

No model was retrained and no large dataset was regenerated.

| Item | Identifier |
|---|---|
| Portable Git LFS bundle | `models/two_tower_80_v1.zip`, SHA-256 `4a93d61b8c51b198398c13bd9fd5cef3843c03844f8da7124bbda3057aad4914` |
| Retrieval inputs manifest | `ff907e2beb8d65df874d1ed13ff4ee914cfd09f2469d49059d35b0775406b5d5` |
| Two-tower weights / embeddings | `44e54105ae01e90f30a56b1b91065a7541c69cced83d78d3ae6164a7a23a281d` / `2086343bee63ea9efebdcb6944bd43526f2757c02589c014aea1c216a8b2a36f` |
| Published collection | `MovieLensTwoTower80M4Release84366c2`, 50,986 vectors, fingerprint `ab4837d3048b812cb76e043911007c4aa71ee588a15275347f31bb7b9e516512` |
| LambdaRank | `lambdarank_v1:4c078ad298a7`, 210 trees, 20 features; model SHA-256 `feb0cf496fd36f06cc0e6b38ff03fc0924c8e95863657491377d289d6e250f9b` |
| MovieLens application run | manifest `ad7a5019da5469e6e6ed84d93d5b48acb4abb648c36c47c650c8d621fa863d82` |
| Artwork mapping | `links.parquet` SHA-256 `f11a8c7d1dcde8debc291aa3852b50d08ebfe010f86bf124cd284fe105c5ce0d` |
| Weaviate | existing persistent 1.39.8 service and volume; a new collection was added, existing collections were not removed |

The previously available local collection was intentionally not used for the
final result because its retrieval fingerprints did not match LambdaRank. The
application correctly refused readiness rather than silently using a fake or
pass-through reranker.

## Reproducible setup and launch

Project metadata allows Python 3.11 or later and Node 22 or 24. This release check
specifically used macOS 27.0.1 arm64, Python 3.13.7, and Node 24.22.2. The exact
tested Python packages are captured in
`constraints/m4-macos-arm64-py313.txt`; they are tested pins, not a change to the
broader supported ranges. The `rank` extra and native OpenMP runtime are required
for **LambdaRank inference**, not model training.

```bash
brew install libomp
python3 -m venv .venv
.venv/bin/python -m pip install \
  -c constraints/m4-macos-arm64-py313.txt \
  -e ".[dev,retrieval,rank,serve,app-dev]"
cd frontend
npm ci
cd ..

git lfs install --local
git lfs pull --include=models/two_tower_80_v1.zip
.venv/bin/python -m trustworthy_recsys.retrieval.bundle unpack \
  --bundle models/two_tower_80_v1.zip \
  --output-dir data/downloads/two_tower_80_v1

docker compose up -d
.venv/bin/python -m trustworthy_recsys.retrieval.publish \
  --inputs-dir data/downloads/two_tower_80_v1/inputs \
  --model-dir data/downloads/two_tower_80_v1/model \
  --output-dir artifacts/retrieval/m4_bundle_weaviate_84366c2 \
  --collection MovieLensTwoTower80M4Release84366c2
```

The application also needs matching MovieLens catalog metadata. Obtain MovieLens
32M as described in `docs/data_pipeline.md`, keep `ratings.csv`, `movies.csv`,
`tags.csv`, `links.csv`, `README.txt`, and `checksums.txt` together, and reproduce
the compatible 80/10/10 run with:

```bash
.venv/bin/python -m trustworthy_recsys.data.pipeline \
  --raw-dir data/ml-32m \
  --output-dir data/processed/m3_ratio_80_10_10_20261002 \
  --checksums data/ml-32m/checksums.txt \
  --source-readme data/ml-32m/README.txt \
  --ratios 0.8 0.1 0.1
```

The successful release reused that existing ignored run rather than regenerating
it. Its manifest SHA-256 is `ad7a5019da5469e6e6ed84d93d5b48acb4abb648c36c47c650c8d621fa863d82`;
the compatibility-critical `movies.parquet`, `train.parquet`,
`train_item_ids.json`, and `statistics.json` hashes are respectively
`eda45b144340f2e27872a8b36aa0dbce046a81304fc1bfa1484e30c54ffb9af9`,
`3c89f096225a42270899f77839699d217590fc85a285900304de13b25ae83622`,
`69d2d06504c862cbbd077d44190d058577ee350233da997239b0610e7249d612`,
and `83942aad6e8fe32680de44aa766a6068d1cdad135d396dd2d143d8d12e9dbf76`.
The exact tested launch used:

```bash
export TRUSTWORTHY_RECSYS_BACKEND=learned
export TRUSTWORTHY_RECSYS_RUN_DIR=data/processed/m3_ratio_80_10_10_20261002
export TRUSTWORTHY_RECSYS_RETRIEVAL_INPUTS_DIR=data/downloads/two_tower_80_v1/inputs
export TRUSTWORTHY_RECSYS_RETRIEVAL_MODEL_DIR=artifacts/retrieval/m4_bundle_weaviate_84366c2
export TRUSTWORTHY_RECSYS_RERANKER_MODEL_DIR=artifacts/ranking/lambdarank_v1
export TRUSTWORTHY_RECSYS_FEEDBACK_MODE=off
.venv/bin/python -m uvicorn trustworthy_recsys.serving.app:app \
  --env-file .env --host 127.0.0.1 --port 8000 --log-level info
```

The ignored backend `.env` contained `TMDB_READ_ACCESS_TOKEN`; its value was
never printed or copied. To run without artwork, omit `--env-file .env` and make
sure `TMDB_READ_ACCESS_TOKEN` is unset. Its `.env.example` entry remains blank.
Start the frontend separately:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Verify identity, not only liveness:

```bash
curl -fsS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8000/api/v1/application/status
```

For a previously published index, inspect the local publication's
`weaviate.json` and reuse it only when its collection, item count, and fingerprint
pass the application's startup audit. To publish the saved bundle without
overwriting anything, choose a fresh `--output-dir` and `--collection` in the
publication command above; the publisher refuses an existing destination. Do not
delete old collections or the persistent Docker volume merely to reproduce this
check.

The passing startup log identifies `two-tower-weaviate`, collection fingerprint
`ab4837d3048b`, `lambdarank_v1:4c078ad298a7`, candidate budget 500, and artwork
configuration. Request logs include request ID, method, path, status, duration,
and backend. Recommendation-complete records add both component versions, result
count, partial status, and total time. Detailed stage timings and warnings are in
the response/workload evidence, not ordinary application log fields. Prompts and
secrets are not logged.

## Commands used for tests and measurements

> Historical transcript only: the commands below record the November 9 run and
> include its archival output paths. Do not rerun them against those paths. Use
> the current non-destructive commands in the [README](../README.md#learned-movie-serving),
> which select fresh outputs and refuse replacement.

```bash
.venv/bin/python -m pytest -q
cd frontend
npm test
npm run typecheck
npm run build
npm run test:e2e:release:artwork
# Stop/restart the backend without --env-file and with the token unset.
npm run test:e2e:release:no-artwork
npx playwright test e2e/search.spec.ts --grep "main-page text search" --project=chromium
npx playwright test e2e/discovery.spec.ts \
  --grep "initial visit|one drawer update|update failure" --project=chromium
cd ..

.venv/bin/python -m trustworthy_recsys.serving.workload \
  --base-url http://127.0.0.1:8000 --profile learned \
  --warmups 10 --repetitions 100 \
  --output docs/application_workload_learned_20261109.json

.venv/bin/python -m trustworthy_recsys.serving.workload \
  --base-url http://127.0.0.1:8000 --profile learned \
  --warmups 0 --repetitions 1 --concurrency-check 5 \
  --output results/m4_release_check_20261109/concurrent-check.json
```

The corrected Playwright setup now excludes both release files from the ordinary
baseline project. `npm run test:e2e:release:artwork` selects only the live-artwork
release file; `npm run test:e2e:release:no-artwork` selects only the neutral-artwork
file. The release configuration starts the frontend but requires a separately
started learned backend, so its identity/artwork assertions fail clearly against
baseline or the wrong token state. Use the exact backend export, start, stop, and
token-free restart commands in the README. The no-artwork launch intentionally
omits `--env-file` and uses `env -u TMDB_READ_ACCESS_TOKEN` while retaining the
learned artifact variables.

## Results

### Passed

- `/ready` returned `ok` in learned mode. Application status reported “Learned
  retrieval + LambdaRank reranking.” A supported history (`1`, `260`, `1196`)
  produced non-null cosine retrieval scores and distinct LambdaRank scores with
  no fallback warning.
- The browser selected Toy Story, hid a displayed movie, applied Adventure and
  Horror constraints, submitted `show comedy without thriller`, and verified
  the exact applied snapshot and displayed results. The response metadata stayed
  learned/LambdaRank throughout.
- A browser-intercepted, controlled first-request HTTP 503 displayed the failure
  and Retry then reached the real learned backend. This tests client recovery from
  an HTTP failure; it does not test recovery from an actual Weaviate outage.
- Five simultaneous learned requests all returned HTTP 200 with the intended
  learned retriever and LambdaRank identities. The retained concurrent evidence
  did not record response warnings, so it does not establish fallback absence for
  every request.
- With the private TMDb token configured, live `image.tmdb.org` posters/backdrops
  appeared in Chromium. Twenty-four artwork resource loads were recorded. The
  existing exact mapping and attribution remained unchanged.
- With the token explicitly unset, learned readiness, LambdaRank, genre filtering,
  text search, and the browser flow passed. Poster metadata was neutral
  (`configured=false`, null URLs) and the UI showed its neutral artwork state.
- Python: 189 passed, 1 intentionally skipped. Frontend: 23 unit tests passed;
  typecheck and production build passed. Seven focused browser checks passed.

### Initial failures found and resolved

1. LightGBM could not load `libomp.dylib`. `brew install libomp` fixed the missing
   native inference runtime and is now documented.
2. The original broad `torch>=2.6,<3` range installed PyTorch 2.15.0, which could
   not coexist safely with LightGBM in this macOS process. The trained bundle
   records PyTorch 2.10.0; constraining the extra to `torch>=2.10,<2.11` restored
   stable retrieval and reranking. No unsafe `KMP_DUPLICATE_LIB_OK` workaround is
   used.
3. The first local publication did not match the LambdaRank fingerprints. The
   committed LFS bundle was verified and published to a new collection. No
   compatibility check was weakened.
4. Application INFO records were hidden under the documented Uvicorn command.
   Routing them through Uvicorn's configured error logger made startup identity,
   request metadata, component versions, errors, result state,
   and total timings visible.
5. One Playwright run collided with another process in their shared trace output
   directory. The affected initial-visit test passed when rerun alone. This was
   harness-output contention, not an application defect.

### Skipped

- Real microphone capture/transcription was not exercised. The existing mocked
  speech/unit path passed, but it is not evidence for physical microphone access.
- The opt-in `WEAVIATE_INTEGRATION=1` test was left skipped because it trains a
  synthetic model and creates/tampers with a temporary collection; the release
  check instead verified the actual saved model and live collection directly.

### Failed or blocked after retest

None. The deprecation warnings for Starlette `httpx` integration and Authlib's
`httpx` integration remain non-blocking.

## Latency evidence

Both measured workloads used 10 warm-ups followed by 100 sequential HTTP samples
against one warm Uvicorn worker on loopback. There were zero warm-up or measured
errors. Feedback collection was off. Percentiles use nearest rank. Artwork was
configured on the backend but
the measurement client made no poster requests, so the table excludes artwork,
browser rendering, and network-to-TMDb time.

| Workload | Boundary | n | p50 | p99 | Errors |
|---|---|---:|---:|---:|---:|
| Favorites `1,260,1196` | HTTP round trip | 100 | 108.15 ms | 157.61 ms | 0 |
| Favorites `1,260,1196` | Server orchestration | 100 | 106.54 ms | 154.14 ms | 0 |
| Adventure, excluding Horror | HTTP round trip | 100 | 45.72 ms | 60.27 ms | 0 |
| Adventure, excluding Horror | Server orchestration | 100 | 43.90 ms | 56.71 ms | 0 |
| Five simultaneous favorites requests | HTTP round trip | 5 | 217.44 ms | 218.14 ms | 0 |
| Text entry and submission to response receipt and updated preference-summary visibility in Chromium | Browser action | 1 | 296 ms | not estimated | 0 |
| TMDb image resource duration | Browser resource timing | 24 | 292.20 ms | 441.60 ms | 0 observed |

For the five-request check, p99 is the maximum of only five observations and is
not a stable production tail estimate. The browser value is one user-flow sample,
not an API percentile. The timer actually starts before filling the search field,
then covers text entry and submission through recommendation-response receipt and
updated preference-summary visibility. It does not measure completion of all
result cards or artwork rendering. The preserved raw JSON uses the older field
`browser_action_to_displayed_results_ms` and says “Enter key”; those labels are
imprecise, but the original evidence is retained unchanged. Raw request IDs and
component timings are in
`application_workload_learned_20261109.json`; the preserved browser and
concurrency JSON records are under `results/m4_release_check_20261109/`.

The browser-selection follow-up writes corrected metadata to
`results/m4_release_handoff_20261109/browser-flow-artwork.json`. That separate
observation is verification evidence, not a replacement benchmark result.

## Cost estimate

Estimated direct metered cost for these local workloads: $0.00 under the stated
assumptions. Assumptions: an already-owned Mac, local Docker/Weaviate volume,
existing internet connection, and an existing TMDb token. There were no cloud
compute, hosted vector database, LLM, speech API, or MCP calls. Hardware purchase
or depreciation, electricity, internet service, developer time, and any external
TMDb account terms are explicitly excluded; no claim is made that those costs are
zero.

## Local unstaged changes for review

- `README.md`: include the `rank` extra, macOS `libomp` inference prerequisite,
  and link this check.
- `pyproject.toml`: constrain PyTorch to the artifact-verified 2.10 series.
- `trustworthy_recsys/serving/app.py`: make useful application INFO logs visible
  and log active component identities at startup.
- `trustworthy_recsys/serving/workload.py`: retain the existing workload tool,
  add live HTTP learned profiles, raw samples, nearest-rank percentiles, and a small
  concurrent check.
- `tests/serving/test_api.py`: verify startup component identity logging.
- `tests/serving/test_workload.py`: verify nearest-rank percentile behavior,
  including the five-sample p99 boundary.
- `constraints/m4-macos-arm64-py313.txt`: capture the exact tested Python
  environment without editable paths or credentials.
- `frontend/e2e/release.learned.spec.ts` and
  `frontend/e2e/release.no-artwork.spec.ts`: focused real learned browser checks.
- `frontend/playwright.config.ts`, `frontend/playwright.release.config.ts`, and
  `frontend/package.json`: keep release files out of default baseline discovery
  and provide explicit artwork/no-artwork projects.
- `docs/system_card.md`: replace pending M4 claims with the checked learned flow,
  observability, reproduction checklist, and retained limitations.
- This report and `docs/application_workload_learned_20261109.json`: reviewable
  release evidence. Selected sanitized browser/concurrency JSON is retained under
  `results/`; transient logs and traces remain local.

Follow-up review changed only the percentile helper/test assertion and
documentation. Targeted verification passed 21 serving/workload tests, frontend
typecheck, and both learned Playwright checks against the real backend. The
contemporaneous review materials referred to the latter retest as
`browser-flow-followup.json`; that file is not retained in this checkout or the
proposed commit and is an archival-only reference. The original 296 ms browser
evidence remains unchanged. No benchmark was rerun because the measured request
path did not change.

## Remaining reproducibility gaps

This was not an independent second-machine reproduction: it reused the existing
Mac, prepared MovieLens data, Weaviate service/volume, and trained artifacts. The
documented raw-data preparation command and a from-scratch publication under a
new name were not rerun during this documentation follow-up. Physical microphone
capture and recovery from a real Weaviate outage remain untested. The latency run
used feedback collection off.

