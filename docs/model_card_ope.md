# Model Card: Open Bandit contextual baseline

## Purpose and scope

Position-specific fashion click OPE research. Separate from MovieLens and Weaviate.

## Model and training

Training-only smoothed click rates per categorical feature/action, averaged across features. Per-action prior shrinks toward global CTR; feature/action rates shrink toward that prior (strength 100). Unknown categories fall back to the action prior. No affinity, item metadata, user identity or test outcomes enter fitting. This is a transparent baseline, not an interaction-rich or neural model.

## Policy selection

Epsilon-greedy with candidates 0.1, 0.3, 0.5, 1.0 selected by validation IPS; selected epsilon 1.0. No train+validation refit. All target actions have uniform logging support.

## Evaluation

See REPORT.md for held-out DM/IPS/SNIPS/DR, time-block bootstrap intervals and diagnostics. policy.json records all validation candidates; reward_model.json is sufficient to reload the fitted model.

## Limitations

Sparse clicks, high-variance importance weights, short time coverage, repeated-user dependence, unknown feature semantics, position interactions and possible temporal drift constrain inference. No matched on-policy ground truth for the contextual target; no online, demographic fairness, joint-slate or movie-model conclusions. Confidence intervals condition on the fitted policy, not the full learning procedure.

## Provenance and usage

Dataset terms are CC BY 4.0 as stated in its README. Preserve attribution and source citation. Model and report manifests record exact input and code fingerprints.
