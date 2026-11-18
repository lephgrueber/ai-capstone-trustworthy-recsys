# Movie lookup and genre text refinement

## Primary movie search

The main page searches the active catalog by movie title. Results show the catalog title and release year where available so the visitor can choose the intended movie. Selection opens the explicit **Your rating** editor; it does not save a favorite or rating. Details views and typing likewise do not mutate the browser profile.

Search uses the existing `GET /api/v1/catalog/search` endpoint. It does not embed arbitrary text, call an LLM, create exposure/reward events, or change rating statistics.

## Main-page recommendation search

The main search area has separate **Rate a movie** and **Find recommendations** modes. Each mode keeps its own draft and offers optional browser voice transcription. In **Rate a movie**, speech fills the editable title field and displays catalog suggestions; the visitor must still select a movie and explicitly save a rating. In **Find recommendations**, speech fills the editable bounded request for review before explicit submission:

- `show comedy without thriller` includes Comedy and excludes Thriller;
- `science fiction` includes the catalog's Science Fiction/Sci-Fi genre;
- `reset preferences` clears genre filters while preserving ratings, visit-only hidden titles, and actual legacy favorites.

The `/api/v1/refinements` endpoint uses the deterministic catalog interpreter. Actor, plot, year-filter, and mood clauses remain unsupported; contradictory genres return clarification and unsupported clauses make no partial change. Applied genres refresh recommendations automatically while preserving numeric rating records and timestamps.

`movies like [title]` resolves an exact catalog title and sends its movie ID as `temporary_history_items` on the resulting recommendation request. The seed reaches retrieval and reranking history for that request context and is excluded from displayed results, but it is not written to the browser profile, rating statistics, favorites, or rating timestamps. A later Find recommendations submission replaces this temporary context. Structured rating and genre changes preserve the active context until it is replaced or all personalization is reset.

The application receives a transcript rather than audio; browser/provider behavior and permissions vary. Typed input remains available in both modes. A title transcript does not select a result or save feedback, and a recommendation transcript does not submit itself. Text and voice interactions do not create rating or research-reward events.

## Checks

Backend tests cover genre inclusion/exclusion, ambiguity, unsupported clauses, profile-preserving reset, and request-scoped title context. Frontend and Playwright checks cover title selection without persistence, separate drafts, explicit voice/text submission, and recommendation refinement that retains the numeric profile. Run the isolated checks from `frontend`:

```bash
npx playwright test --config playwright.search.config.ts
```

These functional checks do not measure transcription accuracy or recommendation quality.
