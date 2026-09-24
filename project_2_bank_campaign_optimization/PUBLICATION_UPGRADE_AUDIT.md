# Publication Upgrade Audit

## Original project

The prior pipeline removed exact duplicates, excluded `duration`, compared a dummy,
class-weighted logistic regression, and random forest by five-fold training PR-AUC,
sigmoid-calibrated the selected forest, and reported fixed-holdout results.

## Concerns identified

- The old `safe_features` mixed planning-time data with current-contact fields, raw
  `campaign`, and macro indicators. UCI defines `campaign` as including the focal contact.
- No boosting benchmark, paired model uncertainty, explicit calibration selection,
  information-set comparison, or duplicate-retention sensitivity was present.
- The holdout had already been reported and cannot be called pristine external validation.

## Changes made

1. Audited all 20 source predictors plus `campaign_prior` using official UCI definitions.
2. Made a strict planning set primary: age, job, marital, education, default, housing, loan, pdays, previous, poutcome.
3. Added planning-plus-macro, operational, and operational-plus-macro sensitivities.
4. Replaced raw `campaign` with nonnegative `campaign_prior = campaign - 1` only in
   operational sensitivity specifications.
5. Added histogram gradient boosting, nested training-only calibration selection,
   1,000 paired bootstrap comparisons, expanded targeting benchmarks, duplicate
   sensitivity, permutation importance, and environment recording.
6. Generated five publication-focused figures and made `analysis.py` the only source of
   analytical calculations used by the notebook and paper-ready results.

## Findings and remaining limitations

Verified values are generated in `PAPER_RESULTS.md`. No code can recover customer IDs,
complete timestamps, randomized treatment, external validation, contact costs, or
customer value. Exact duplicates therefore remain an explicit sensitivity analysis,
and the fixed holdout is labeled a previously analyzed development holdout.
