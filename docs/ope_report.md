# Open Bandit OPE report

This is a position-specific fashion click benchmark, not evaluation of the MovieLens retriever.

Campaign: all; source position: 1; logging policy: random.
Test exposures: 45,801; clicks: 161; observed CTR: 0.003515.
Validation-selected policy: uniform (epsilon 1.0). Selection metric: validation IPS; no test tuning.
All predeclared policy candidates are reported below as diagnostics; test results do not change selection.

| Policy | Estimator | Value | 95% interval (hour-block bootstrap) |
|---|---|---:|---|
| uniform | DM | 0.003520 | [0.003510, 0.003527] |
| uniform | IPS | 0.003515 | [0.002823, 0.004235] |
| uniform | SNIPS | 0.003515 | [0.002823, 0.004235] |
| uniform | DR | 0.003502 | [0.002812, 0.004212] |
| contextual_epsilon_0.1 | DM | 0.011286 | [0.011252, 0.011315] |
| contextual_epsilon_0.1 | IPS | 0.006640 | [0.000312, 0.013431] |
| contextual_epsilon_0.1 | SNIPS | 0.006645 | [0.000328, 0.013990] |
| contextual_epsilon_0.1 | DR | 0.006718 | [0.000778, 0.013973] |
| contextual_epsilon_0.3 | DM | 0.009561 | [0.009532, 0.009584] |
| contextual_epsilon_0.3 | IPS | 0.005945 | [0.000937, 0.011348] |
| contextual_epsilon_0.3 | SNIPS | 0.005949 | [0.000969, 0.011591] |
| contextual_epsilon_0.3 | DR | 0.006003 | [0.001290, 0.011657] |
| contextual_epsilon_0.5 | DM | 0.007835 | [0.007812, 0.007853] |
| contextual_epsilon_0.5 | IPS | 0.005251 | [0.001562, 0.009241] |
| contextual_epsilon_0.5 | SNIPS | 0.005253 | [0.001597, 0.009302] |
| contextual_epsilon_0.5 | DR | 0.005289 | [0.001757, 0.009381] |

## Diagnostics and interpretation

- uniform: ESS 45801.0/45801; mean weight 1.0000; maximum weight 1.00; 18 hourly blocks.
- contextual_epsilon_0.1: ESS 704.2/45801; mean weight 0.9992; maximum weight 72.10; 18 hourly blocks.
- contextual_epsilon_0.3: ESS 1153.0/45801; mean weight 0.9994; maximum weight 56.30; 18 hourly blocks.
- contextual_epsilon_0.5: ESS 2207.3/45801; mean weight 0.9996; maximum weight 40.50; 18 hourly blocks.

The uniform-policy IPS and SNIPS estimates reproduce the observed click rate; this is an algebraic sanity check, not independent proof of estimator accuracy for the contextual candidates. The contextual candidates have no matched on-policy ground truth. If epsilon 1.0 wins validation, the selected policy is exactly uniform and no personalized improvement is demonstrated.

Intervals resample observed UTC hour blocks and condition on the fitted model and selected policy. They do not include training/selection uncertainty or ensure independence across repeated users. The held-out window is short; block-size sensitivity (15 minutes and 2 hours) is included in report.json. Intervals are not clipped to [0,1]. Clipped-weight estimates are sensitivity analyses, not primary estimates.

Inspect error_analysis.json for high-weight events and daily/context slices. contributions.parquet preserves per-event reward, target probability, weight, reward prediction and estimator contributions for every candidate. Test outcomes are used only for reporting, never selection. The policy uses four (or the discovered set of) anonymized categorical user fields; no demographic meaning or identity is inferred. Averages of smoothed feature/action rates are a simple baseline, not a two-tower or calibrated causal model.

No improvement or statistical-significance claim follows from point estimates alone. Position-wise marginal propensities do not identify arbitrary joint-slate interventions; other displayed items may affect clicks.
