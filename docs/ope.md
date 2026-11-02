# Open Bandit off-policy evaluation

This is a runnable, separate fashion-click benchmark. It does not evaluate the
MovieLens two-tower model or change the Weaviate index. The first experiment uses
the `all` campaign, random logging policy, and original display position 1.
Other campaigns and positions are configurable; BTS logs are not used in this run.

## Run from the repository root

The implementation uses the existing base dependencies (NumPy, pandas, PyArrow).
No OBP installation, GPU, TMDb token, or Weaviate service is required.

```powershell
python -m trustworthy_recsys.ope.prepare --raw-dir data/downloads/open_bandit_dataset --output-dir data/processed/open_bandit_my_run
python -m trustworthy_recsys.ope.evaluate --inputs-dir data/processed/open_bandit_my_run --output-dir results/ope/my_run --bootstrap 1000 --seed 42
python -m pytest tests/test_ope.py -q
```

Activate the virtual environment first, or replace `python` with
`.\.venv\Scripts\python.exe` on Windows. The output directories must be new.
Failed runs retain `INCOMPLETE`; they cannot be consumed or silently overwritten.
Raw data and generated runs are ignored by Git. Compact measured snapshots are
tracked in `docs/ope_report.md`, `docs/data_card_ope.md`, and `docs/model_card_ope.md`.
Machine-readable snapshots are `docs/ope_report.json` and `docs/ope_statistics.json`.

The completed local run uses `data/processed/open_bandit_all_position1_v2` and
`results/ope/open_bandit_all_position1_v3`. It contains 366,404 training exposures,
45,800 validation exposures and 45,801 test exposures. Validation starts at
2019-11-29 10:49:20.841960 UTC; testing starts at 2019-11-30 06:01:18.082529 UTC.
The first preparation attempt (`open_bandit_all_position1_v1`) stopped on mixed
fractional-second timestamp formatting and retains an `INCOMPLETE` marker. The
parser now accepts ISO timestamps with and without fractional seconds. Earlier
evaluation directories are historical reports; v3 includes all candidate diagnostics.

The selected policy is uniform. Its held-out IPS/SNIPS CTR is 0.003515 (0.3515%),
with an hourly-block 95% interval of approximately [0.2823%, 0.4235%]. The three
contextual candidates have test ESS of roughly 704, 1,153 and 2,207 out of 45,801
events. Their higher test point estimates do not override validation selection.
No personalized-policy improvement has been established.

For a plain-language examination of why, see [the follow-up analysis](ope_analysis.md).
The [movie feedback plan](movie_feedback_plan.md) maps the next collection step to
the actual backend and frontend. It is a design, not a deployed logging feature.

To reproduce the descriptive analysis of the frozen run:

```powershell
python -m trustworthy_recsys.ope.analyze --inputs-dir data/processed/open_bandit_all_position1_v2 --evaluation-dir results/ope/open_bandit_all_position1_v3 --output-dir results/ope/new_analysis
```

The recorded follow-up is in `results/ope/interpretation_v1`; its summary is also
preserved as `docs/ope_analysis.json`. No model is fitted or reselected by this command.

The local source directory must contain `README`, `VERSION`, and
`random/<campaign>/<campaign>.csv` plus `item_context.csv`. To change scope, pass
`--campaign men` or `--campaign women`, and/or `--position 2` or `--position 3`.
Keep each scope in a separate run. The importer deliberately rejects random logs
whose supplied propensity does not equal the reciprocal of the complete catalog
size: investigate availability rather than fabricating replacement probabilities.

## Data contract and temporal protocol

The preparer validates required values, binary rewards, catalog references,
positions, timestamps, and propensities across the source CSV. It retains all
click and nonclick events at the selected position. It does not deduplicate
identical-looking exposures: those may be legitimate observations. Event IDs
combine source policy, campaign and row ordinal; they are not user identifiers.

Sorted events are partitioned approximately 80/10/10. The timestamp at the target
80% and 90% ranks begins the later partition; equal timestamps remain together.
Dates are not rounded to whole days because the source covers only seven days.
User-level separation cannot be claimed without a verified user identifier.

Categorical `user_feature_*` columns are the only model features. Their mappings
are learned from training only; unseen categories fall back to an action prior.
Affinity features are excluded pending temporal provenance review. Item metadata
is preserved but not modeled. Source feature semantics and availability are
assumptions based on the dataset description, not independently verified facts.

Generated inputs include:

- `train.parquet`, `validation.parquet`, `test.parquet`: exposure records with
  source row, event ID, UTC time, campaign, policy, position, original action ID,
  action index, binary reward, original propensity and categorical context.
- `action_ids.json`, `item_context.parquet`: complete campaign-local action mapping
  and source item metadata.
- `schema.json`, `statistics.json`, `DATA_CARD.md`: contract, measured counts,
  action/context coverage, clicks, CTR, date ranges and limitations.
- `source_README.txt`, `source_VERSION.txt`, `manifest.json`: attribution, version,
  source/output hashes, code fingerprints and package versions.

## Policy and reward model

The baseline is an inspectable categorical click-rate model. A smoothed overall
CTR is `(clicks + 1)/(events + 2)`. Each action's CTR shrinks toward this global
rate with strength 100. Each categorical-feature/action CTR then shrinks toward
the corresponding action prior with strength 100. Predictions average these
per-feature rates. This is not logistic regression, a neural model, or a calibrated
probability guarantee. It ignores feature interactions and item features.

An epsilon-greedy policy assigns `epsilon / K` to every action and adds
`1 - epsilon` to the highest-scoring action. Ties use the smallest mapped action
ID. Epsilon candidates are fixed at 0.1, 0.3, 0.5 and 1.0. Validation IPS selects
the candidate; ties prefer greater exploration. Epsilon 1.0 is exactly uniform.
There is no train+validation refit. The final test is first read after selection.
All predeclared candidates are reported for diagnostics, without changing the
validation-selected winner based on test estimates.

`reward_model.json` stores the fitted tables, priors and category keys;
`policy.json` records validation scores and selection. Reload with Python's JSON
reader and call `predict_rewards(model, frame)` and
`policy_probabilities(predicted_rewards, epsilon)` in `ope/policy.py`.

## Estimators and diagnostics

For logged action `a`, reward `r`, propensity `p`, target distribution `pi`, and
training-fitted reward predictions `q`, use `w = pi(a|x)/p`:

- DM: mean of `sum_a pi(a|x) q(x,a)`.
- IPS: mean of `w*r`.
- SNIPS: `sum(w*r)/sum(w)`.
- DR: mean of `sum_a pi(a|x) q(x,a) + w*(r-q(x,a))`.

Definitions can be checked against the [official OBP estimators](https://github.com/st-tech/zr-obp/blob/master/obp/ope/estimators.py).
This implementation is independent and does not pin the legacy OBP dependency tree.
Tests include exact balanced-logging examples with deliberately wrong reward
predictions, probability validation, timestamp ties, end-to-end exports, unknown
contexts and artifact corruption.

Primary intervals use 1,000 resamples of observed UTC hour blocks. The model and
policy stay fixed. SNIPS is recomputed for every draw. The report also uses
15-minute and 2-hour blocks as sensitivity checks. Blocks can differ in size.
Intervals are descriptive with a short held-out window; block resampling does
not eliminate cross-block repeated-user dependence or temporal drift and does
not account for model-fitting or policy-selection uncertainty. DM intervals in
particular condition on reward predictions and do not measure model bias.

Report ESS `(sum w)^2/sum(w^2)`, mean/max/quantiles of weights, top-1% weight mass,
and clipped-weight sensitivity at 5, 10 and 20. Clipped estimates are secondary
and biased in general; thresholds are not selected using test performance.
Unclipped estimates/intervals are not forced into [0,1].

`contributions.parquet` preserves the per-event estimator inputs and terms for
each candidate. `error_analysis.json` contains highest-weight observations and
daily/context slices. Hashed categories are not interpreted as demographics.
`report.json`, `REPORT.md`, and `MODEL_CARD.md` contain the results and scope.

## Interpretation

Uniform IPS/SNIPS must equal the observed CTR when supplied propensities are
uniform; this is an algebraic sanity check. It is not ground-truth validation of
a different target policy. Position-specific marginal probabilities do not
identify a general intervention on an entire three-item slate. Results require
appropriate support, correct logging probabilities, consistency, and a usable
pre-decision context. Other displayed items may influence the outcome.

The source README states CC BY 4.0. Preserve attribution and modifications and
cite Saito, Aihara, Matsutani and Narita, *Open Bandit Dataset and Pipeline*,
[arXiv:2008.07146](https://arxiv.org/abs/2008.07146). Dataset access and field
descriptions are on [ZOZO Research](https://research.zozo.com/data.html).
