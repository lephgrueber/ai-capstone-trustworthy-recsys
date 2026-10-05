# Data Card: model-ready retrieval inputs

## Dataset overview
Derived from C:\Users\tolul\Documents\ai-capstone-trustworthy-recsys\data\processed\movielens_phase1 / ratio_80_10_10: 25,602,867 training ratings,
12,760,312 positive ratings, 170,509 users and 50,986 candidate movies.
The preparer produced **676,684 context/target examples**, with at most 4 targets per user.

## Provenance and collection
These are transformations of the completed MovieLens pipeline run, not newly collected observations.
See the [source Data Card](../data/processed/movielens_phase1/DATA_CARD.md) for collection and release details. metadata.json records verified input hashes.
The original split and its reports remain unchanged.

## Licensing and usage
The source MovieLens usage terms and citation requirements continue to apply to these transformations.
Refer to the source Data Card and its bundled README; the software's MIT license does not replace dataset terms.

## Data structure and splits
schema.json describes binary memory-mapped arrays, ID mappings, CSR histories, and genre features.
Only training movies and their genre vocabulary are fitted. Validation/test events and labels are not read during preparation.
For each user, up to 4 eligible positive targets are evenly spaced through their training history.
Each context contains only positive events with timestamps strictly before its target: ties and the target are excluded.
The user feature is a mean genre profile plus log history length; the item feature is identity plus fractional genres.
No titles, tags, release years, or held-out interactions are model features. Known users' all-rating histories are retained for seen-item filtering.
Evaluation continues to use the source validation/test JSONL and training-only histories.

## Limitations and risks
- Target subsampling per user changes interaction weighting; this is not training on every positive event.
- 592 training users have no eligible strictly earlier positive context.
- Genre metadata is a release snapshot with unknown historical availability; temporal event separation does not eliminate this metadata limitation.
- Genres discard ordering and specific item interactions in the user tower; identity embeddings cover training movies only.
- Empty/unknown histories need fallback behavior. New movies are outside the index, limiting attainable recall on the all population.
- No impression propensities are available; these inputs do not establish causal policy value or demographic fairness.

## Integrity and reproducibility

The retrieval publication stage stores only training-catalog movie IDs, their row indices, an embedding fingerprint, and model-produced vectors in Weaviate. User histories and validation/test labels remain local. Changing the storage/search backend does not change these model-ready inputs or the temporal split. The published model's `weaviate.json` describes the collection and index settings; loading verifies every remote movie ID and vector against the saved embedding matrix.
Source hashes, timestamp boundaries, uniqueness, target exclusion, and row counts were checked during preparation.
metadata.json contains statistics and schema.json describes the artifact contract. manifest.json fingerprints each output.
