# README onboarding and runtime audit — November 2, 2026

Audited commit: `055fb626bcbc2b88261c8edd5da64a22ba984fbe`.

## Scope of the evidence

We exported the committed repository to `data/downloads/readme_audit_20261102/repo`
and created a new virtual environment with system site packages disabled. This
was an isolated source/environment check on the existing Windows machine, not a
new physical machine, container image, or independent stranger's checkout.
The host already has Python, Git, Node, npm, network access and package caches.
Repository authentication and a complete learned-model deployment remain unproven.
The archived model ZIP was not extracted or used. Local processed datasets and
the working project's `.env` were not copied into the audit environment.

Environment: Python 3.14.6, Node 23.10.0, npm 9.9.4. The fixture probes explicitly
remove inherited application settings and the TMDb token from child environments.
No real users or behavioral click events were involved.

## What was actually tried

1. The README's `.venv/bin/python` command fails in PowerShell: command not found.
   Substituting `.\.venv\Scripts\python.exe` is required on Windows.
2. In the new environment, `python -m pip install -e ".[dev,retrieval,serve,app-dev]"`
   completed after allowing the installer access to PyPI. The initial blocked
   socket error was an execution-sandbox restriction, not a repository defect.
3. `npm ci --no-audit --no-fund` installed 225 packages. It warned that Node 23.10.0
   is outside the declared supported ranges of several dependencies. The README
   does not state a supported Node version. Successful installation does not
   remove that compatibility warning.
4. The new environment ran `python -m pytest tests/serving tests/test_ope.py -q`:
   **48 passed**, with one upstream TestClient deprecation warning. This is a
   targeted subset, not a full clean-machine integration run.
5. A probe started actual Uvicorn processes and made HTTP requests against both
   fixture mode and learned mode without configured artifacts. Each process was
   stopped after its checks. No TestClient was used for these live probes.
6. `npm run build` passed (TypeScript plus Vite, 2,136 modules transformed).
   The initial sandboxed attempt failed on filesystem access; the permitted retry
   completed. The observed Vite build stage took 2m32s. Node engine warnings still
   apply. A successful build is not a browser walkthrough or an end-to-end test.

| Live request | Observed result |
|---|---|
| Fixture `/health` and `/ready` | Both HTTP 200 |
| Fixture recommendations, empty history | HTTP 200, 12 recommendations |
| Fixture recommendations, all genres excluded | HTTP 200, zero recommendations, `partial=true` |
| Fixture poster request | HTTP 200, `configured=false`, missing artwork and null URLs |
| Learned mode without paths: `/health` | HTTP 200 |
| Same learned mode: `/ready` | HTTP 503, explicit missing-configuration explanation |
| Same learned mode: recommendations | HTTP 503 |

Raw evidence is kept locally under `data/downloads/readme_audit_20261102/`:
`probe.py`, `http_results.json`, `fixture_server.log`, and `learned_server.log`.
The source export and installed audit environment are ignored by Git.

## What this reveals beyond relevance evaluation

Ranking metrics do not show whether configuration was loaded, posters are
available, filters leave enough candidates, or a server can serve recommendations.
In this probe, a healthy process did not imply a ready learned model. An HTTP 200
did not imply a nonempty recommendation list or usable artwork. These are
observations of deliberately exercised local scenarios, not production incidence
rates or evidence of customer behavior.

The default startup printed Uvicorn access logs but did **not** print the custom
`request_complete` and `recommendation_complete` INFO messages. Writing a logger
call is insufficient: the application logger needs effective level/handler
configuration. Existing unit tests do not establish that deployment logging is
configured correctly.

There is also a semantic gap: `candidates_displayed` is calculated from the API
response length before the browser confirms rendering. It measures returned
recommendations, not viewed recommendations. There is no implemented impression/
response event collection; see `movie_feedback_plan.md`. We cannot claim that
observability has revealed actual click behavior or movie-policy value.

## The clearest hidden setup dependency

`frontend/playwright.config.ts` starts a baseline server using the hardcoded path
`data/processed/m3_ratio_80_10_10_20261002` and `.venv/bin/python`, with Bash-style
environment assignments. That processed dataset is not in a fresh checkout.
The README acknowledges the local dependency, but `npm run test:e2e` is not a
self-contained onboarding check. Someone must know how to create that exact run,
adapt the shell command, and manage any existing server. `reuseExistingServer`
can also allow a developer's already-running service to hide missing setup.

## Remaining acceptance work

- Publish one fixture quick start for PowerShell and one for Bash, with explicitly
  tested Python/Node versions and expected readiness/UI output.
- Make browser tests launch a disposable fixture backend with portable commands;
  test them with no preexisting API/frontend process.
- Configure the application logger and verify custom request IDs, backend identity,
  timings and recommendation counts appear under the documented launch command.
- Keep returned, rendered and interacted-with recommendations distinct. Implement
  display/response recording before claiming behavior-based movie evaluation.
- Add CI coverage for frontend installation, build and fixture browser startup;
  the current workflow runs Python tests and Weaviate, not the frontend journey.
- Separately prove learned-mode onboarding, including LFS access, catalog metadata,
  model publication and readiness. The bundle-only prediction instructions do not
  by themselves demonstrate a complete movie UI installation.

This audit records evidence and gaps. It does not change serving behavior, turn on
event collection, or claim that the end-to-end browser journey is verified.
