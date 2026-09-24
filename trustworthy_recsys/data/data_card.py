"""Render a readable Data Card from measured results and source documentation."""

from pathlib import Path


def render_data_card(output, dataset_name, audit, metadata, tags, splits, config):
    lines = [f"# Data Card: {dataset_name}", "",
             "## Dataset overview", "",
             "This dataset supports research on movie recommendation, retrieval, and ranking. "
             "The tables below describe this pipeline run; they are generated from the files rather than copied from an expected split ratio.", "",
             "| Field | Value |", "|---|---|",
             f"| Rating events | {audit['rows']:,} |",
             f"| Rating users | {audit['users']:,} |",
             f"| Catalog / rated movies | {audit['catalog_movies']:,} / {audit['rated_movies']:,} |",
             f"| Raw / retained tag events | {tags['input_rows']:,} / {tags['output_rows']:,} |",
             f"| Observed rating period (UTC) | {audit['first_timestamp_utc']} to {audit['last_timestamp_utc']} |",
             "| Labels | Explicit half-star ratings from 0.5 to 5.0 |",
             f"| Evaluation relevance | Rating >= {config['positive_threshold']} |", "",
             "## Provenance and collection", "",
             "MovieLens 32M was released by GroupLens at the University of Minnesota. "
             "The bundled README identifies a release generated October 13, 2023: ratings and free-text tags contributed through MovieLens. "
             "Users were selected at random for inclusion from MovieLens, with at least 20 ratings in the complete release. "
             "That selection does not make the dataset representative of all movie viewers. IDs are anonymized and no demographics are supplied.", "",
             "Sources: [GroupLens dataset access](https://grouplens.org/datasets/movielens/32m/), "
             "[bundled dataset documentation](source_README.txt), and [bundled checksums](source_checksums.txt). "
             "The original local download date and downloading party are not recorded; this pipeline does not invent them or redownload the data. "
             "Input hashes establish consistency with the supplied manifest, not independent proof of the acquisition route. "
             "The run manifest records processing time, configuration, source hashes, code hashes, and environment versions.", "",
             "## Licensing and usage", "",
             "The bundled MovieLens README supplies custom research-use terms; this is not a CC0 dataset. "
             "It requires acknowledgment in resulting publications, prohibits implying endorsement, and requires redistributions "
             "(including transformations) to retain the same conditions. Commercial or revenue-bearing use requires prior permission "
             "from a GroupLens faculty member. The data are supplied without guarantees of correctness or suitability. "
             "Consult the complete terms in source_README.txt; the repository's MIT software license does not replace these dataset terms. "
             "IMDb/TMDb IDs are supplied as identifiers only; no external metadata was fetched and those services have their own terms.", "",
             "Required citation: F. Maxwell Harper and Joseph A. Konstan (2015), "
             "*The MovieLens Datasets: History and Context*, ACM TiiS 5(4), Article 19. "
             "[DOI](https://doi.org/10.1145/2827872).", "",
             "## Data structure and splits", "",
             "See [schema.json](schema.json) for exact types. Interaction Parquet files contain `user_id` and `movie_id` "
             "as original decimal strings, `rating` as float32, and `timestamp_utc` as a UTC timestamp with millisecond storage precision. "
             "Source times have second precision. Movie genres are lists (empty when no genres are listed). "
             "External IDs remain strings, preserving leading zeroes; missing external IDs are null.", "",
             "Raw inputs are immutable. All valid ratings are retained, including low ratings and sparse users/items. "
             "Invalid required fields, duplicates, broken rating references, or unexpected rating ordering cause failure instead of silent filtering. "
             f"Tag whitespace is trimmed ({tags['whitespace_trimmed_rows']:,} affected rows); "
             f"{tags['blank_tags_removed']:,} blank tag rows are removed. Distinct tag events are preserved. "
             "The full tag table is descriptive only: use each scenario's train_tags.parquet for historical features, "
             "and intersect those tags with the model's training catalog as needed.", "",
             "The global temporal rule is **train < validation start <= validation < test start <= test**. "
             "Equal timestamps stay together. Ratio-based boundaries fall at midnight after the UTC day containing the target cumulative event count, "
             "so actual ratios are approximate. User/item overlap is allowed and measured: later events, not whole users, are held out. "
             "This models future serving and avoids training on later interactions.", "",
             "| Scenario | Validation starts (UTC) | Test starts (UTC) | Train rows | Validation rows | Test rows | Warm test events |",
             "|---|---|---|---:|---:|---:|---:|"]
    for name, s in splits.items():
        parts = s["partitions"]
        lines.append(f"| {name} | {s['validation_start_utc']} | {s['test_start_utc']} | "
                     f"{parts['train']['rows']:,} | {parts['validation']['rows']:,} | {parts['test']['rows']:,} | "
                     f"{parts['test']['warm_rows_pct']:.2f}% |")
    lines += ["", "### Evaluation contract and exclusions", "",
              "Each scenario includes validation_all.jsonl, test_all.jsonl, and corresponding warm files. "
              "Each record has a string user_id, history_items, and nonempty relevant_items, as required by eval.harness. "
              f"Histories contain training movies rated >= {config['positive_threshold']}; relevance contains positive ratings from the held-out partition. "
              "Both validation and test use training-only histories; no validation refit or within-test history updates occur. "
              "Lists are ordered by timestamp, with numeric movie ID breaking ties. Full histories are retained without truncation. "
              "Users without positive held-out labels cannot be scored by the harness and are counted as exclusions.", "",
              "The all files retain cold users, empty positive histories, and cold relevant items. "
              "The warm files require at least one training interaction for the user (not necessarily positive), "
              "then keep only relevant movies observed in training. They therefore use a narrower relevance set; "
              "scores across populations or ratios must not be interpreted as a controlled model comparison without a common evaluation population. "
              "Candidate train_item_ids.json lists every training-observed movie, including movies with only low ratings. "
              "A cold-item-capable model needs an explicitly documented wider candidate set. Exclude previously consumed items when designing a recommender; "
              "the harness itself does not enforce a candidate universe or seen-item filtering.", "",
              "| Scenario / partition | Users with events | All eligible | No positive labels | Warm eligible | Empty positive history (all) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, s in splits.items():
        for part, e in s["evaluation"].items():
            lines.append(f"| {name} / {part} | {e['users_with_events']:,} | {e['all_examples']:,} | "
                         f"{e['excluded_no_positive_labels']:,} | {e['warm_examples']:,} | {e['empty_positive_history_examples']:,} |")
    lines += ["", "The exporter is checked using the existing harness loader. Empty population files, if any, are recorded but cannot be scored. "
              "No model predictions or performance claims are generated here. The curated 50-case golden set remains untouched and separate from these observed relevance labels.", "",
              "## Limitations and risks", "",
              f"- **Popularity concentration:** the top 1% of rated movies account for {audit['top_one_percent_rated_movies_share_pct']:.2f}% of ratings. "
              "This motivates popularity and long-tail evaluation but is not itself a measurement of model bias.",
              "- **Cold start:** many users enter ratings over short periods and are absent from earlier training partitions. "
              "The split table reports warm coverage; warm-only scores exclude substantial future activity.",
              f"- **Metadata gaps:** {metadata['movies_without_listed_genres']:,} movies lack listed genres, "
              f"{metadata['missing_tmdb_ids']:,} TMDb IDs are missing, and cleaned tags cover {tags['catalog_tag_coverage_pct']:.2f}% of catalog movies.",
              "- **Feedback bias:** ratings reflect selected interactions, not all exposures. Unrated movies are not known dislikes. "
              "No demographic fairness conclusions are supported without demographic data.",
              "- **Time and metadata:** timestamps record rating submission, not viewing. Movie metadata is a release snapshot; "
              "historical availability cannot be established from it. The last calendar month/year is partial.",
              f"- **Documented boundary discrepancy:** {audit['ratings_on_or_after_2023_10_13_utc']:,} ratings occur on or after October 13, 2023 UTC, "
              "while the release README describes an October 12 activity end. A timezone/reporting difference is possible but unverified; valid rows are retained.",
              "- **Offline policy evaluation:** there are no impression logs, displayed positions, or logged action propensities. "
              "These files alone do not support the planned logged-propensity IPS/SNIPS/DR experiments. A separate suitable dataset is still needed.",
              "- **Comparison scope:** data coverage does not establish which split yields better recommendation quality. "
              "If both models are refit on train+validation at a shared test boundary, recompute eligibility and coverage for that protocol.", "",
              "## Reproduction and related artifacts", "",
              "See manifest.json for the exact configuration, input/output hashes, software versions, and code revision/source hashes. "
              "See statistics.json for complete partition and eligibility counts and schema.json for output contracts. "
              "A successful run ends with manifest.json and no INCOMPLETE marker. "
              "Processing timestamps and Parquet binary representation may differ across environments; compare logical records using the same configuration and library versions.", ""]
    Path(output, "DATA_CARD.md").write_text("\n".join(lines), encoding="utf-8")
