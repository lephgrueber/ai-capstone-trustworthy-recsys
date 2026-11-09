# Data Card: movie-application feedback

## Dataset overview
A new application-owned research dataset of featured-movie assignments and response events. The technical pipeline is implemented; no mature real-user outcome dataset or movie-policy improvement is claimed in this revision. The completed 100-assignment pilot is synthetic and is labeled separately.

## Provenance and collection
The active movie backend supplies a bounded candidate pool after hard constraints and ranking. A server-side epsilon-greedy selector assigns the featured movie, freezes the grid and commits the exact candidates, probabilities, preferences and model/artifact provenance to SQLite before responding. A server-generated decision ID and capability token bind browser events to the decision. Session identifiers are HMAC pseudonyms, not authenticated people. Source snapshots and exports have hashes; the database secret and event tokens are never exported.

## Licensing and usage
MovieLens terms continue to apply to movie metadata and derived models. Newly collected feedback is local research data, not automatically licensed under MIT or approved for public redistribution. Keep raw preferences, identifiers, tokens and database files private. The application displays a collection notice when enabled; About describes the data flow. No automatic retention schedule or production access-control service is implemented.

## Data structure and splits
The store separates decisions from unique events. Events include display, details-open, favorite, hide, heartbeat and window-end. The exporter creates nested JSONL with a per-row candidate mapping, logging distribution, action, context, outcome/status and collection diagnostics. Chronological approximate 80/10/10 splits keep equal timestamps together; sessions may cross partitions and this is counted. Training/validation rows are purged when their outcome plus late-arrival window reaches the next partition’s first assignment, preventing future-outcome availability leakage. Actual ratios after purging can differ; short runs can remain insufficient even after outcomes mature. Training uses only train outcomes, selection only validation, and reporting uses test. `schema.json`, `statistics.json`, `DATA_CARD.md` and `manifest.json` describe each export.

## Limitations and risks
The primary outcome is featured details-open within 30 minutes, followed by a 24-hour late-arrival allowance. It measures interest, not satisfaction. Incomplete outcomes stay unknown; negative outcomes require healthy complete telemetry. Missingness bounds and session-cluster uncertainty must accompany estimates. Fixture/automated traffic is synthetic and excluded by default. These logs only support alternate choices inside each saved featured pool with the same fixed grid; they cannot establish a new retriever's or full slate's value. Browser self-report, tab closure, throttling, session resets, sparse clicks and exposure bias limit inference. The local SQLite prototype is not a production multi-instance collection platform.

## Access and reproduction
See [movie OPE setup](movie_ope.md) for collection and export commands, [technical validation](movie_ope_validation.md) for evidence, and the generated per-run card for measured counts. Raw SQLite files live under ignored `data/feedback/`; prepared exports under ignored `data/processed/`; evaluations under ignored `results/movie_ope/`.
