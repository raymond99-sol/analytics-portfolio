# Research Brief

## Question

Can a bank rank outbound campaign contacts more efficiently under an 11.3% response
rate without using post-call information or overstating model certainty?

## Design

The study uses 41,176 deduplicated observations from the UCI Bank Marketing dataset.
A stratified 20% holdout is isolated first. A prevalence baseline, class-weighted
logistic regression, and random forest are then compared through five-fold training-set
cross-validation using PR-AUC as the selection criterion. The selected random forest is
sigmoid-calibrated using training data only and evaluated once on the holdout. One
thousand bootstrap resamples quantify uncertainty around ranking and targeting metrics.

## Evidence

The random forest produced mean training-fold PR-AUC 0.467 (+/- 0.018 SD). On the
8,236-row holdout, the calibrated model reached ROC-AUC 0.813 (95% CI 0.797-0.830) and
PR-AUC 0.485 (95% CI 0.449-0.521). A 20% call budget captured 65.9% of subscribers
(95% CI 63.2%-68.9%) at 3.30x lift (95% CI 3.16x-3.44x).

## Validity checks

`duration` is known only after a call and is therefore excluded from legitimate model
features. A same-architecture audit shows that adding it raises holdout ROC-AUC from
0.814 to 0.946 and PR-AUC from 0.487 to 0.652, quantifying the resulting leakage. Model
selection never consults the holdout, and probability calibration reduces Brier score
from 0.143 to 0.075.

## Interpretation

The model supports a prospective hypothesis: score-based prioritization may reduce calls
per acquired subscriber. It does not show that calling causes subscription. The next
credible study is a later-period validation followed by randomized assignment between
the ranked list and the current targeting strategy, measuring incremental subscriptions
per call and cost per incremental subscription.

## Boundaries

The source omits customer identifiers and complete timestamps, preventing grouped-client
and temporal validation. `unknown` values are observed source categories rather than
true nulls. Production use also requires consent, contact-frequency, subgroup fairness,
and calibration-drift controls.
