# Paper-Ready Results

All values below are generated from saved machine-readable outputs. This is manuscript input, not a completed paper.

## Sample and estimand

The UCI extract contains 41,188 contact records. The primary analysis removes 12 exact duplicate rows, leaving 41,176 observations and 4,639 observed subscriptions (11.3%). The estimand is out-of-sample response propensity and ranking performance, not the causal effect of a marketing contact.

## Decision-time information set

The conservative primary specification uses 10 planning-time predictors: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `pdays`, `previous`, `poutcome`. Raw `campaign`, scheduled-contact fields, contemporaneous macroeconomic fields, and post-contact `duration` are excluded. `campaign_prior = campaign - 1` is used only in operational sensitivity analysis. This implements decision C: compare multiple information sets while retaining the strict planning set as primary.

## Model development and validation

A stratified 80/20 split yields 32,940 training and 8,236 holdout observations. Five-fold training-only PR-AUC selected Random Forest; nested training-only Brier score selected isotonic calibration. The holdout was used in earlier repository analyses and is a development holdout, not a pristine prospective test.

- Random Forest: PR-AUC 0.353 (fold SD 0.015); ROC-AUC 0.704 (fold SD 0.011).
- Histogram Gradient Boosting: PR-AUC 0.353 (fold SD 0.018); ROC-AUC 0.705 (fold SD 0.010).
- Logistic Regression: PR-AUC 0.337 (fold SD 0.020); ROC-AUC 0.696 (fold SD 0.012).
- Dummy Baseline: PR-AUC 0.113 (fold SD 0.000); ROC-AUC 0.500 (fold SD 0.000).

## Fixed-holdout results

The final Random Forest (isotonic calibration) achieved PR-AUC 0.339 (95% bootstrap CI 0.307-0.372), ROC-AUC 0.706 (95% CI 0.687-0.724), Brier score 0.087 (95% CI 0.082-0.092), and log loss 0.310.

### Paired model uncertainty

- Final vs Logistic Regression, PR_AUC: delta +0.012 (95% CI -0.000 to +0.026; positive favors final).
- Final vs Logistic Regression, Top20_Capture: delta +0.012 (95% CI -0.011 to +0.030; positive favors final).
- Final vs Histogram Gradient Boosting, PR_AUC: delta -0.003 (95% CI -0.011 to +0.006; positive favors final).
- Final vs Histogram Gradient Boosting, Top20_Capture: delta +0.006 (95% CI -0.012 to +0.020; positive favors final).
- Final vs Random Forest raw, PR_AUC: delta -0.002 (95% CI -0.006 to +0.002; positive favors final).
- Final vs Random Forest raw, Top20_Capture: delta -0.009 (95% CI -0.020 to +0.001; positive favors final).

## Probability calibration

The selected training-only specification had Brier score 0.086 and log loss 0.308. On the holdout, Brier score was 0.087 and log loss was 0.310.

## Resource-constrained targeting

- Top 10%: 824 contacts; 318 observed subscribers; 38.6% conversion; 34.3% capture; 3.43x lift; 2.59 contacts per observed subscriber.
- Top 20%: 1,648 contacts; 422 observed subscribers; 25.6% conversion; 45.5% capture; 2.27x lift; 3.91 contacts per observed subscriber.
- Top 30%: 2,471 contacts; 518 observed subscribers; 21.0% conversion; 55.8% capture; 1.86x lift; 4.77 contacts per observed subscriber.
- Top 40%: 3,295 contacts; 595 observed subscribers; 18.1% conversion; 64.1% capture; 1.60x lift; 5.54 contacts per observed subscriber.

At 20%, capture was 45.5% (95% CI 42.6%-48.3%) and lift was 2.27x (95% CI 2.13-2.41). These are observational ranking quantities, not incremental causal effects.

## Information timing and leakage

- Strict planning (primary): holdout PR-AUC 0.339, ROC-AUC 0.706, top-20% capture 45.5%.
- Planning plus macro context: holdout PR-AUC 0.454, ROC-AUC 0.805, top-20% capture 65.3%.
- Operational pre-contact: holdout PR-AUC 0.423, ROC-AUC 0.781, top-20% capture 60.1%.
- Operational plus macro context: holdout PR-AUC 0.485, ROC-AUC 0.813, top-20% capture 65.1%.

Adding post-contact `duration` increased comparable holdout ROC-AUC from 0.813 to 0.946, PR-AUC from 0.485 to 0.651, and top-20% capture from 65.1% to 87.5%. The leaky model is not deployable.

## Duplicate sensitivity

Retaining all 12 exact duplicate rows changed holdout PR-AUC by +0.010. Without customer IDs, accidental duplicates cannot be distinguished from distinct identical contacts.

## Predictive associations

- `pdays`: mean PR-AUC decrease 0.0409 (SD 0.0044).
- `age`: mean PR-AUC decrease 0.0297 (SD 0.0026).
- `poutcome`: mean PR-AUC decrease 0.0280 (SD 0.0035).
- `default`: mean PR-AUC decrease 0.0173 (SD 0.0050).
- `previous`: mean PR-AUC decrease 0.0144 (SD 0.0027).

These are predictive associations, not causal effects.

## Material limitations

- No stable customer identifier is available, so repeated customers cannot be grouped during splitting.
- Month and weekday do not support a defensible row-level chronological split across May 2008-November 2010.
- The historical sample is from one Portuguese bank; validity for modern campaigns, other countries, digital channels, or other populations is not established.
- Observed subscription is not incremental response caused by calling; these are propensity models, not treatment-effect models.
- Contact cost and customer lifetime value are unavailable, so monetary ROI is not estimated.
- The repository's holdout was analyzed previously and is not a pristine prospective external validation.
- Future research should use customer IDs, true timestamps, prospective temporal validation, and randomized policy evaluation with observed costs and value.
