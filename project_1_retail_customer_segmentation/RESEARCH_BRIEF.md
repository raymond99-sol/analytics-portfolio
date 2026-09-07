# Research Brief: Behavioral Segmentation in Online Retail

## Research question

How concentrated is customer value across purchasing-behavior segments, and which groups
should an online retailer prioritize for retention, reactivation, and second-purchase tests?

## Design

The study uses 541,909 invoice-line observations from the UCI Online Retail dataset. After
excluding cancellations, missing customer identifiers, and non-positive quantities or
prices, it constructs customer-level recency, frequency, and monetary value features.
Because these measures are strongly right-skewed, the analysis applies `log1p`
transformation and standardization before K-means clustering.

K=4 is treated as a decision-oriented segmentation choice rather than a uniquely true
number of customer types. The notebook compares K=2 through K=8 using silhouette scores
and evaluates K=4 stability across repeated random seeds with adjusted Rand index.

## Main evidence

- The cleaned data contain 4,338 customers, 18,532 orders, and GBP 8.91 million in revenue.
- The Champions segment includes 720 customers, or 16.6% of the customer base.
- Champions account for 64.9% of cleaned revenue, indicating substantial value concentration.
- K=2 produces the strongest geometric separation (silhouette 0.433). K=4 is retained
  because it yields a more actionable four-part targeting framework while remaining stable
  across 25 random seeds (mean adjusted Rand index 0.952; minimum 0.915).
- At Risk customers form a large reactivation pool, but the observational data cannot show
  whether discounts would create incremental purchases.

## Interpretation

The segmentation is useful as a targeting framework: protect high-value active customers,
test reactivation offers on inactive customers, and develop newer customers toward a second
purchase. The recommended actions are hypotheses for experiments, not estimates of causal
effects.

## Limitations and next study

The source lacks product cost, margin, marketing exposure, demographics, and randomized
treatment. A stronger follow-up would randomly assign eligible At Risk customers to a
reactivation message or holdout group, preregister primary outcomes, and estimate incremental
revenue and margin with confidence intervals.
