# Open Bandit Data Card

## Overview and provenance

Open Bandit Dataset, ZOZO/ZOZOTOWN. Random policy, all campaign, position 1. 1,374,327 source rows; 458,005 selected exposures; 80 catalog actions. Local source files are immutable; source hashes, version and README are preserved. Download date and downloading party are not known. Hashes identify the local files, not an independently authenticated download.

## Licensing and usage

The supplied README states CC BY 4.0. Retain attribution, license and modification notices. Cite Saito, Aihara, Matsutani and Narita, Open Bandit Dataset and Pipeline (arXiv:2008.07146). These Parquet tables, action mappings and temporal partitions are project transformations.

## Structure and splits

Click reward, original logging propensity, categorical user context, action and source-row identity are retained. Approximate chronological 80/10/10 event splits keep equal timestamps together. Validation begins 2019-11-29 10:49:20.841960+00:00; test begins 2019-11-30 06:01:18.082529+00:00. See schema.json and statistics.json for exact types, counts and missingness-related checks. No clicks or nonclicks are selectively removed. Affinities and item features are not model inputs.

## Limitations and risks

This is a position-specific fashion click benchmark, not evaluation of the MovieLens model. Other displayed items may influence clicks: this is not joint-slate policy evaluation. Propensity checks assume a fixed uniform action catalog at the selected position. No verified user/slate identifier is available in the selected schema; anonymized features are not user IDs. Context features are treated as pre-decision covariates according to the source description; their upstream construction is not independently audited. Short time coverage, repeated-user dependence, temporal drift, and limited clicks constrain inference. Campaign names do not justify demographic fairness claims.

## Validation and reproducibility (November 9, 2026)

A fresh preparation with 50,000-row input chunks reproduced the existing 100,000-row-chunk run exactly at the logical record level, including all split assignments, schemas, statistics and action mappings. The independent `trustworthy_recsys.ope.validate` command checks artifact hashes, column types, action/catalog alignment, binary rewards, uniform propensities, temporal ordering, disjoint event IDs and source-row accounting.

| Partition | Exposures | Clicks |
|---|---:|---:|
| Train | 366,404 | 1,288 |
| Validation | 45,800 | 173 |
| Test | 45,801 | 161 |

All 458,005 exposures and 1,622 clicks were accounted for. Evidence: [validation snapshot](ope_validation.json); full local run: `results/ope/validated_20261109`. Fresh inputs: `data/processed/open_bandit_20261109`. This rerun validates data preparation; it does not reselect a policy on the test set. Existing OPE estimates remain in [the OPE report](ope_report.md). The scope remains random logging, the all campaign and position 1; other campaigns/positions have not received this full-data audit.

## Relationship to movie-application OPE

The application now has its own implemented feedback store, featured-slot randomization and separate movie exporter/evaluator. Open Bandit data, splits and policy results remain unchanged and are not merged with movie events. The new implementation still needs real movie-user observations before it can support a movie-policy improvement claim. See [movie OPE](movie_ope.md) and [its Data Card](data_card_movie_feedback.md).
