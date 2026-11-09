# Movie text and voice search

The main page has a search bar above the movie results, with a microphone icon inside it. Personalize remains available for manual favorite, genre and hide controls. Search applies a request to the current preferences and refreshes recommendations; the visible preference summary shows the active filters. Use Reset all for a fresh start.

## Supported requests

- `show comedy without thriller`: include Comedy and exclude Thriller.
- `science fiction`: include Science Fiction.
- `movies like The Matrix`: resolve an exact catalog title and add it as a positive example. Include the year for ambiguous titles. In fixture mode, use `movies like Orbit House` instead; the fixture catalog contains synthetic movies.
- `reset preferences`: clear the saved preferences.

This is a bounded English parser, not an LLM or an arbitrary text embedding search engine. Actor, plot, release-year filters and mood requests are unsupported. A year can disambiguate a title but cannot filter results by year. Unsupported clauses apply no partial changes. Contradictory genres and ambiguous/missing titles return a message asking for clarification. Include filters match at least one included genre; excluded genres are hard exclusions. Requests update existing preferences rather than replacing all earlier choices.

## How retrieval uses the request

`serving/refinement.py` translates text into the existing `PreferenceState` contract through `/api/v1/refinements`. Applied responses include resolved movie metadata so title seeds remain visible and removable in Personalize. The frontend then calls `/api/v1/recommendations`. It ignores a refinement response if the user edited preferences while waiting.

A genre-only query filters candidates; without favorite movies it still uses popularity fallback. A resolved title supplies a positive history item to the current two-tower profile. The model averages genre features and history length; it does not embed the query text or understand movie plots. Weaviate searches the existing 32-dimensional movie vectors. No retraining or re-indexing is required. The same parser works with fixture and baseline catalogs, but those modes do not become learned retrieval.

## Microphone setup and privacy

Start the backend and frontend using the README. Use localhost for development or HTTPS for a deployed frontend, and a browser that exposes `SpeechRecognition` or `webkitSpeechRecognition`. Click the microphone and grant microphone access if prompted. The recognized English text appears in the search field. Review or edit it, then press Enter or the search button. Speech never automatically submits a request.

Browser support varies; the microphone is disabled when the API is unavailable, and typed search remains available. Permission denial and recognition failures display a message. The browser's speech service may send audio to its provider; this application receives the submitted transcript, not audio recordings. Browser speech recognition is not guaranteed to work offline. See [MDN SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition) for the browser API and its limitations.

## Validation and limits

Backend tests cover genre inclusion/exclusion, title resolution, ambiguity, unsupported clauses, hidden titles, reset and applying the result to recommendations. Frontend tests cover microphone activation, returned transcripts, permission denial, unsupported browsers, cleanup and main-page search submission. These use mocked speech events; they do not measure transcription accuracy. A real microphone/permission/browser-provider check remains a manual acceptance step: dictate a supported request, confirm the text, submit it, and verify the resulting filters. No speech-quality or general language-understanding benchmark is claimed.


Run the isolated browser checks from `frontend` after installing dependencies and Chromium:

```powershell
npx playwright install chromium
npx playwright test --config playwright.search.config.ts
```

This cross-platform configuration starts its own synthetic backend on port 18000 and frontend on 15173, so it needs no dataset or Weaviate and does not reuse an unknown running backend. The speech test injects a mock browser speech service. Desktop/mobile screenshots are written under `frontend/test-results/`.


A live November 9 API check used the trained MovieLens model and verified Weaviate collection. `science fiction movies like The Matrix (1999)` resolved MovieLens ID 2571 and the catalog's `Sci-Fi` genre, returned 12 learned-score recommendations, and excluded the seed movie. The full server request took 201.48 ms in this single smoke check; this is functional evidence, not a latency distribution. Evidence is in `results/retrieval/language_smoke_20261109.json`.

If Chromium is not installed and Microsoft Edge is available, set `$env:PLAYWRIGHT_CHANNEL = "msedge"` before the Playwright command (PowerShell). The November 9 browser check uses that installed-browser option; the Chromium download stalled and was cancelled.
