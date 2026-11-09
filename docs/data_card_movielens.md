# Data Card: movielens-32m

## Dataset overview

This dataset supports research on movie recommendation, retrieval, and ranking. The tables below describe this pipeline run; they are generated from the files rather than copied from an expected split ratio.

| Field | Value |
|---|---|
| Rating events | 32,000,204 |
| Rating users | 200,948 |
| Catalog / rated movies | 87,585 / 84,432 |
| Raw / retained tag events | 2,000,072 / 2,000,066 |
| Observed rating period (UTC) | 1995-01-09T11:46:44+00:00 to 2023-10-13T02:29:07+00:00 |
| Labels | Explicit half-star ratings from 0.5 to 5.0 |
| Evaluation relevance | Rating >= 4.0 |

## Provenance and collection

MovieLens 32M was released by GroupLens at the University of Minnesota. The bundled README identifies a release generated October 13, 2023: ratings and free-text tags contributed through MovieLens. Users were selected at random for inclusion from MovieLens, with at least 20 ratings in the complete release. That selection does not make the dataset representative of all movie viewers. IDs are anonymized and no demographics are supplied.

Sources: [GroupLens dataset access](https://grouplens.org/datasets/movielens/32m/), [bundled dataset documentation](../data/processed/movielens_phase1/source_README.txt), and [bundled checksums](../data/processed/movielens_phase1/source_checksums.txt). The original local download date and downloading party are not recorded; this pipeline does not invent them or redownload the data. Input hashes establish consistency with the supplied manifest, not independent proof of the acquisition route. The run manifest records processing time, configuration, source hashes, code hashes, and environment versions.

## Licensing and usage

The bundled MovieLens README supplies custom research-use terms; this is not a CC0 dataset. It requires acknowledgment in resulting publications, prohibits implying endorsement, and requires redistributions (including transformations) to retain the same conditions. Commercial or revenue-bearing use requires prior permission from a GroupLens faculty member. The data are supplied without guarantees of correctness or suitability. Consult the complete terms in source_README.txt; the repository's MIT software license does not replace these dataset terms. IMDb/TMDb IDs are supplied as identifiers only; no external metadata was fetched and those services have their own terms.

Required citation: F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19. [DOI](https://doi.org/10.1145/2827872).

## Data structure and splits

See [schema.json](../data/processed/movielens_phase1/schema.json) for exact types. Interaction Parquet files contain `user_id` and `movie_id` as original decimal strings, `rating` as float32, and `timestamp_utc` as a UTC timestamp with millisecond storage precision. Source times have second precision. Movie genres are lists (empty when no genres are listed). External IDs remain strings, preserving leading zeroes; missing external IDs are null.

Raw inputs are immutable. All valid ratings are retained, including low ratings and sparse users/items. Invalid required fields, duplicates, broken rating references, or unexpected rating ordering cause failure instead of silent filtering. Tag whitespace is trimmed (6,392 affected rows); 6 blank tag rows are removed. Distinct tag events are preserved. The full tag table is descriptive only: use each scenario's train_tags.parquet for historical features, and intersect those tags with the model's training catalog as needed.

The global temporal rule is **train < validation start <= validation < test start <= test**. Equal timestamps stay together. Ratio-based boundaries fall at midnight after the UTC day containing the target cumulative event count, so actual ratios are approximate. User/item overlap is allowed and measured: later events, not whole users, are held out. This models future serving and avoids training on later interactions.

| Scenario | Validation starts (UTC) | Test starts (UTC) | Train rows | Validation rows | Test rows | Warm test events |
|---|---|---|---:|---:|---:|---:|
| ratio_80_10_10 | 2018-10-04T00:00:00+00:00 | 2020-11-06T00:00:00+00:00 | 25,602,867 | 3,198,273 | 3,199,064 | 9.79% |

### Evaluation contract and exclusions

Each scenario includes validation_all.jsonl, test_all.jsonl, and corresponding warm files. Each record has a string user_id, history_items, and nonempty relevant_items, as required by eval.harness. Histories contain training movies rated >= 4.0; relevance contains positive ratings from the held-out partition. Both validation and test use training-only histories; no validation refit or within-test history updates occur. Lists are ordered by timestamp, with numeric movie ID breaking ties. Full histories are retained without truncation. Users without positive held-out labels cannot be scored by the harness and are counted as exclusions.

The all files retain cold users, empty positive histories, and cold relevant items. The warm files require at least one training interaction for the user (not necessarily positive), then keep only relevant movies observed in training. They therefore use a narrower relevance set; scores across populations or ratios must not be interpreted as a controlled model comparison without a common evaluation population. Candidate train_item_ids.json lists every training-observed movie, including movies with only low ratings. A cold-item-capable model needs an explicitly documented wider candidate set. Exclude previously consumed items when designing a recommender; the harness itself does not enforce a candidate universe or seen-item filtering.

| Scenario / partition | Users with events | All eligible | No positive labels | Warm eligible | Empty positive history (all) |
|---|---:|---:|---:|---:|---:|
| ratio_80_10_10 / validation | 22,909 | 22,720 | 189 | 6,432 | 16,181 |
| ratio_80_10_10 / test | 22,251 | 22,015 | 236 | 4,310 | 17,467 |

The exporter is checked using the existing harness loader. Empty population files, if any, are recorded but cannot be scored. No model predictions or performance claims are generated here. The curated 50-case golden set remains untouched and separate from these observed relevance labels.

## Limitations and risks

- **Popularity concentration:** the top 1% of rated movies account for 54.88% of ratings. This motivates popularity and long-tail evaluation but is not itself a measurement of model bias.
- **Cold start:** many users enter ratings over short periods and are absent from earlier training partitions. The split table reports warm coverage; warm-only scores exclude substantial future activity.
- **Metadata gaps:** 7,080 movies lack listed genres, 124 TMDb IDs are missing, and cleaned tags cover 58.60% of catalog movies.
- **Feedback bias:** ratings reflect selected interactions, not all exposures. Unrated movies are not known dislikes. No demographic fairness conclusions are supported without demographic data.
- **Time and metadata:** timestamps record rating submission, not viewing. Movie metadata is a release snapshot; historical availability cannot be established from it. The last calendar month/year is partial.
- **Documented boundary discrepancy:** 108 ratings occur on or after October 13, 2023 UTC, while the release README describes an October 12 activity end. A timezone/reporting difference is possible but unverified; valid rows are retained.
- **Offline policy evaluation:** there are no impression logs, displayed positions, or logged action propensities. These files alone do not support the logged-propensity IPS/SNIPS/DR experiments. The separate Open Bandit benchmark supplies logged propensities for fashion recommendations; it does not establish movie-policy value.
- **Comparison scope:** the active model uses the chronological 80/10/10 scenario. Warm and all-user populations differ; compare methods within the same population.

## Reproduction and related artifacts

See manifest.json for the exact configuration, input/output hashes, software versions, and code revision/source hashes. See statistics.json for complete partition and eligibility counts and schema.json for output contracts. A successful run ends with manifest.json and no INCOMPLETE marker. Processing timestamps and Parquet binary representation may differ across environments; compare logical records using the same configuration and library versions.

## Current retrieval and application use

This maintained card describes the active 80/10/10 scenario. The original generated source artifacts remain immutable. [Model-ready inputs](data_card_retrieval.md) contain 50,986 training movies and 676,684 strictly earlier context/target examples; every example passed the November 9 retrieval audit. Weaviate stores the model-produced vectors with their original movie IDs and row mappings. Titles are used only by the application to resolve a requested positive example, not as tower features. Browser speech produces editable text; this project does not add microphone recordings to MovieLens. See [retrieval results](retrieval_report.md), [ANN verification](ann_benchmark.md), and [setup](../README.md).



## New movie-application feedback

The source ratings and temporal split are unchanged. A separate [Movie Feedback Data Card](data_card_movie_feedback.md) describes application-owned exposure/response logs collected prospectively by the optional featured-slot experiment. Those logs contain known assignment probabilities; the original MovieLens ratings still do not. The movie-specific exporter and OPE evaluator are implemented, but synthetic checks are not evidence of real-user policy value. See [run instructions](movie_ope.md).
