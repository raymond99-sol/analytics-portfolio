# Bank Marketing Campaign Optimization

## Research question

Can response modeling improve pre-call targeting under class imbalance while avoiding
information leakage and reporting uncertainty honestly?

## Research design

- Cleaned **41,188** UCI campaign contact records to **41,176** observations after
  removing 12 exact duplicates; the positive subscription rate is **11.3%**.
- Reserved a stratified **20% holdout** before model selection.
- Compared a prevalence baseline, class-weighted logistic regression, and random forest
  with **five-fold stratified cross-validation on the training set only**.
- Selected the model by mean PR-AUC, then applied sigmoid probability calibration using
  training data only.
- Evaluated the untouched holdout with ROC-AUC, PR-AUC, Brier score, budget lift,
  responder capture, and **1,000 bootstrap samples** for 95% confidence intervals.
- Refit the same random-forest architecture with post-call `duration` solely to quantify
  target leakage; it is excluded from every legitimate pre-call model.

## Verified findings

- Random forest led training-set selection with mean **0.467 PR-AUC (+/- 0.018 SD)**
  and **0.800 ROC-AUC (+/- 0.011 SD)** across five folds.
- The calibrated model achieved **0.813 ROC-AUC** (95% CI **0.797-0.830**),
  **0.485 PR-AUC** (95% CI **0.449-0.521**), and a **0.075 Brier score** on
  8,236 untouched holdout observations.
- The top-ranked 20% converted at **37.1%**, captured **65.9% of subscribers**
  (95% CI **63.2%-68.9%**), and produced **3.30x lift**
  (95% CI **3.16x-3.44x**) relative to the holdout baseline.
- Including `duration` inflated same-architecture holdout ROC-AUC from **0.814 to
  0.946** and PR-AUC from **0.487 to 0.652**, demonstrating why it is invalid for
  pre-call targeting.

## Reproduce

From the repository root:

```bash
python download_data.py --project bank
pip install -r project_2_bank_campaign_optimization/requirements.txt
python project_2_bank_campaign_optimization/analysis.py
python project_2_bank_campaign_optimization/validate_outputs.py
```

The notebook calls the same deterministic analysis pipeline and presents its key outputs.
See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the source hash and expected results.

## Files

- `bank_campaign_optimization.ipynb` - executed reader-facing analysis
- `analysis.py` - deterministic modeling and output pipeline
- `validate_outputs.py` - independent saved-output integrity tests
- `RESEARCH_BRIEF.md` - research framing, evidence, and extensions
- `sql/campaign_analysis.sql` - standalone descriptive queries
- `outputs/model_selection_summary.csv` - training-set cross-validation comparison
- `outputs/bootstrap_intervals.csv` - holdout uncertainty estimates
- `outputs/budget_metrics.csv` - lift and capture across call-budget levels
- `outputs/leakage_audit.csv` - safe versus post-call feature comparison
- `outputs/calibration_metrics.csv` - raw and calibrated probability diagnostics
- `outputs/charts/` - presentation-ready figures

## Data source

Sérgio Moro, Paulo Rita, and Paulo Cortez (2014), *Bank Marketing*, UCI Machine
Learning Repository. DOI: https://doi.org/10.24432/C5K306. Licensed under CC BY 4.0.

## Limitations

The extract provides no customer identifier, so repeated clients cannot be grouped
during splitting. It also lacks a complete year-level timestamp for defensible temporal
validation. Results estimate subscription propensity, not the causal effect of calling.
Production use requires prospective temporal testing, randomized incrementality,
fairness, consent, contact-frequency, and calibration-drift review.
