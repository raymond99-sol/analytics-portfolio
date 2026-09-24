# Research Brief

## Question

Can a bank rank likely term-deposit responders under contact constraints using only
information available at the targeting decision, and how sensitive is apparent value to
information timing, calibration, model choice, and duplicate handling?

## Design

The estimand is out-of-sample response propensity, not incremental treatment effect. The
primary set contains customer attributes and prior-campaign history available before the
current campaign list is selected. A compact benchmark is selected by five-fold training
PR-AUC; calibration is selected through nested training-only Brier score. A fixed
development holdout supports prespecified model, budget, calibration, information-set,
duplicate, and leakage comparisons with 1,000 paired bootstrap resamples.

## Main methodological finding

Information timing matters at least as much as the choice among competent algorithms.
Current-contact and macro context materially improve ranking but change the operational
interpretation, while post-contact `duration` produces a much stronger invalid
benchmark. `PAPER_RESULTS.md` contains the generated numerical evidence.

## Managerial interpretation

Budget outputs report contacts, observed subscribers, conversion, lift, capture, and
contacts per observed subscriber against random targeting. They describe historical
ranking efficiency, not causal acquisition. With no cost or customer value, economic
preference is only symbolic: expected value from additional observed responders must
exceed added modeling and operating costs.

## Validity boundary

The data omit customer IDs, complete timestamps, cost, and customer lifetime value; the
holdout was examined previously. External validity to modern banking, other countries,
digital channels, and other populations is untested. Future work needs customer-level
grouping, prospective temporal validation, randomized policy evaluation, and observed
cost/value outcomes.
