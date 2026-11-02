# Open Bandit Data Card

## Overview and provenance

Open Bandit Dataset, ZOZO/ZOZOTOWN. Random policy, all campaign, position 1. 1,374,327 source rows; 458,005 selected exposures; 80 catalog actions. Local source files are immutable; source hashes, version and README are preserved. Download date and downloading party are not known. Hashes identify the local files, not an independently authenticated download.

## Licensing and usage

The supplied README states CC BY 4.0. Retain attribution, license and modification notices. Cite Saito, Aihara, Matsutani and Narita, Open Bandit Dataset and Pipeline (arXiv:2008.07146). These Parquet tables, action mappings and temporal partitions are project transformations.

## Structure and splits

Click reward, original logging propensity, categorical user context, action and source-row identity are retained. Approximate chronological 80/10/10 event splits keep equal timestamps together. Validation begins 2019-11-29 10:49:20.841960+00:00; test begins 2019-11-30 06:01:18.082529+00:00. See schema.json and statistics.json for exact types, counts and missingness-related checks. No clicks or nonclicks are selectively removed. Affinities and item features are not model inputs.

## Limitations and risks

This is a position-specific fashion click benchmark, not evaluation of the MovieLens model. Other displayed items may influence clicks: this is not joint-slate policy evaluation. Propensity checks assume a fixed uniform action catalog at the selected position. No verified user/slate identifier is available in the selected schema; anonymized features are not user IDs. Context features are treated as pre-decision covariates according to the source description; their upstream construction is not independently audited. Short time coverage, repeated-user dependence, temporal drift, and limited clicks constrain inference. Campaign names do not justify demographic fairness claims.
