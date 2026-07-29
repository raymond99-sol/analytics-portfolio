# Bank Marketing Campaign Optimization

## Business question

Can a bank prioritize outbound calls more efficiently using information available before
the call begins?

## What I built

- Cleaned and validated 41,188 campaign observations.
- Used SQL to compare conversion across prior outcomes, occupations, and contact channels.
- Built logistic-regression and random-forest pipelines with numeric preprocessing and
  one-hot encoding.
- Explicitly removed call `duration` to prevent post-contact target leakage.
- Evaluated ranking quality using ROC-AUC, PR-AUC, lift, and responder capture.

## Key findings

- Baseline subscription rate is **11.3%**.
- The selected **Random Forest** achieved
  **0.814 ROC-AUC** and **0.487 PR-AUC**
  on a stratified holdout set.
- The highest-ranked 20% of holdout customers converted at
  **37.1%**, or
  **3.30x baseline lift**, and contained
  **65.9% of holdout subscribers**.

## Recommended actions

1. Pilot the ranked call list against the current strategy through randomized assignment.
2. Use incremental subscriptions per call and cost per incremental subscription as success metrics.
3. Validate on a later time period before deployment.
4. Complete fairness, consent, contact-frequency, calibration, and monitoring reviews.

## Files

- `bank_campaign_optimization.ipynb` - executed analysis notebook
- `sql/campaign_analysis.sql` - standalone SQL queries
- `outputs/model_metrics.csv` - holdout model metrics
- `outputs/targeting_deciles.csv` - lift and capture by score decile
- `outputs/charts/` - presentation-ready charts

## Data source

Sérgio Moro, Paulo Rita, and Paulo Cortez (2014), *Bank Marketing*,
UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5K306.
Licensed under CC BY 4.0.

## Limitations

The public file lacks a complete timestamp for temporal validation. The model ranks
likely subscribers but does not estimate the causal effect of calling a customer.
Production use would require compliance and fairness review.
