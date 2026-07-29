# E-commerce Customer Segmentation & Revenue Strategy

## Business question

Which customer groups should an online retailer prioritize for retention,
reactivation, and second-purchase campaigns?

## What I built

- Cleaned and validated 541,909 invoice-line records.
- Used SQLite queries to calculate revenue, order, customer, country, and monthly KPIs.
- Built RFM features for 4,338 customers.
- Applied standardized log-transformed K-means clustering and checked cluster separation
  with a silhouette score of 0.336.
- Produced campaign-ready customer segments and reproducible visualizations.

## Key findings

- Cleaned transactions represent **£8,911,408** in revenue
  across **18,532** orders.
- The **Champions** segment contains **720 customers**
  (16.6% of the customer base) and generates
  **64.9% of cleaned revenue**.
- The **At Risk** segment contains **1,579 customers**;
  it is a focused reactivation opportunity, but campaign impact should be tested with a holdout.
- The source begins and ends with partial months, so edge-month trend comparisons are caveated.

## Recommended actions

1. Protect Champions with loyalty benefits, early access, and replenishment reminders.
2. Use a controlled reactivation test for At Risk customers.
3. Build second-purchase journeys for Developing customers.
4. Track incremental revenue and margin, not only response rate.

## Files

- `retail_customer_segmentation.ipynb` - executed analysis notebook
- `sql/customer_analysis.sql` - standalone SQL queries
- `outputs/customer_segments.csv` - customer-level segment assignments
- `outputs/segment_summary.csv` - segment KPIs
- `outputs/charts/` - presentation-ready charts

## Data source

Daqing Chen (2015), *Online Retail*, UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5BW33. Licensed under CC BY 4.0.

## Limitation

This is observational transaction data without product cost, margin, marketing exposure,
or experimentation. Segment recommendations are hypotheses to test, not causal findings.
