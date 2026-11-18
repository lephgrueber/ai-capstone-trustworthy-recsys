# Browser rating profile and live model features

## Visitor workflow and storage

The main search looks up the active MovieLens catalog. Choose the intended title and release year, select **Your rating** from 0.5 through 5 stars in half-star increments, and press **Save rating**. Searching, choosing a result, opening details, hovering, and typing do not save feedback. **Your ratings** lists the browser's current records with Edit, Remove, and the separate **Clear saved ratings** action. **Hide for now** excludes an unrated movie only for the current page visit; **Hidden this visit** in Preferences lists every active exclusion and allows individual Unhide. These exclusions are not ratings and disappear after a full page reload.

The versioned profile is stored under `trustworthy-recsys-browser-profile-v1` in browser `localStorage`. It contains ratings, actual legacy favorites, and display metadata—not visit-only hidden IDs. It is restored synchronously before the first recommendation request, so a reload restores saved ratings and favorites while starting with no hidden exclusions. If storage is unavailable or unreadable, the application warns the visitor; a rating save that cannot be persisted leaves the dialog open with its unsaved draft available to retry. The application does not accept that rating as a session-only profile or claim it was saved. There is no account, server-side personal profile, or cross-device synchronization.

Each current numeric record contains a stable `movie_id`, the actual numeric `rating`, and `rated_at`, the UTC time of the latest genuine numeric submission/change. An identical save, reload, details view, or unrelated preference change preserves that timestamp. Editing the number replaces the record and timestamp; removing it removes the record from all derived statistics. Actual migrated favorites remain `history_items` without fabricated stars or dates until explicitly rated or removed.

Rating, deletion, hiding, genre, and other effective preference changes increment the preference revision and refresh automatically. A successfully persisted rating closes its rating dialog immediately, then refreshes recommendations in the background. Earlier results remain visible during that request. If the refresh fails, the saved rating remains and a page-level Retry action resubmits the current state. Abort handling, monotonic revisions, and request-sequence checks prevent stale responses from replacing newer results.

## Request and authoritative derivation

`PreferenceState.ratings` is optional. Omission means the original favorites-only contract; `[]` means a complete numeric profile with zero ratings. A browser request uses `known_user_id: null`:

```json
{
  "session_id": "browser-session-id",
  "preference_revision": 7,
  "known_user_id": null,
  "result_count": 12,
  "preferences": {
    "history_items": ["2571"],
    "excluded_movie_ids": ["1196"],
    "include_genres": ["Drama"],
    "exclude_genres": ["Horror"],
    "ratings": [
      {"movie_id": "1", "rating": 4.5, "rated_at": "2026-11-17T20:15:00Z"},
      {"movie_id": "260", "rating": 2.5, "rated_at": "2026-11-16T18:00:00Z"}
    ]
  }
}
```

The backend validates unique supported IDs, finite half-star values, UTC timestamps, request limits, and the identity mode. A MovieLens `known_user_id` cannot be combined with a numeric browser profile. The backend—not the client—derives one `ModelQuery` at one request-scoped UTC `as_of` time:

- mean rating is the unweighted mean of every current numeric rating;
- count is the number of current numeric records, with `log1p(count)` sent as `user_log_rating_count`;
- every rating at least 4.0 enters positive retrieval/reranking history, ordered by `rated_at` and movie ID as deterministic tie-breaker;
- inactivity is `(as_of - latest numeric rated_at) / 86400`, using a rating of any value;
- every numerically rated movie is excluded before reranking and checked again in final validation, while positive ratings remain available to the model as history;
- item features and the saved model's `2018-10-04T00:00:00+00:00` training cutoff remain frozen.

Unavailable mean/count/recency values become internal `NaN` in the LightGBM matrix. An explicit empty complete profile has count 0 but missing mean and recency. If unresolved undated legacy favorites coexist with numeric records, all three profile statistics stay missing so a partial numeric count is not presented as total history. Omitted `ratings` retains the old feature-building and known-user lookup behavior. Slight client clock skew is clamped to zero inactivity; materially future timestamps are rejected.
