# Bank Marketing Campaign Optimization

## Research question

Can response models improve resource-constrained bank marketing targeting when restricted
to information genuinely available at the targeting decision, and how much apparent
performance changes when information leakage, calibration, and uncertainty are handled?

## Empirical design

- **Primary set:** ten customer and prior-campaign fields available before constructing
  an outreach list. Raw `campaign`, current-contact fields, macro indicators, and
  post-contact `duration` are excluded.
- **Sensitivity sets:** planning plus macro context; operational pre-contact fields with
  `campaign_prior = campaign - 1`; and operational plus macro context.
- **Benchmark:** prevalence dummy, class-weighted logistic regression, random forest,
  and histogram gradient boosting, selected by five-fold training-only PR-AUC.
- **Calibration:** raw, sigmoid, and isotonic probabilities compared with nested
  training-only predictions; the final choice minimizes Brier score.
- **Evaluation:** fixed 20% stratified development holdout, 1,000 paired bootstrap
  samples, budget metrics, duplicate sensitivity, and a post-contact leakage benchmark.

Exact verified results are generated in [`PAPER_RESULTS.md`](PAPER_RESULTS.md); the
methodological change log is [`PUBLICATION_UPGRADE_AUDIT.md`](PUBLICATION_UPGRADE_AUDIT.md).

## Reproduce

```bash
python download_data.py --project bank
python -m pip install -r project_2_bank_campaign_optimization/requirements.txt
python project_2_bank_campaign_optimization/analysis.py
python project_2_bank_campaign_optimization/validate_outputs.py
```

The notebook calls `run_analysis()` from `analysis.py`; it contains no independent
modeling implementation. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Main artifacts

- `analysis.py` - authoritative empirical pipeline and manuscript inputs
- `bank_campaign_optimization.ipynb` - executed reader-facing companion
- `validate_outputs.py` - independent output and traceability tests
- `outputs/feature_availability_audit.csv` - audit of every predictor
- `outputs/model_pairwise_bootstrap.csv` - paired model uncertainty
- `outputs/information_set_comparison.csv` - timing/macro sensitivity
- `outputs/budget_metrics.csv` - model and random targeting benchmarks
- `outputs/leakage_audit.csv` - valid versus post-contact specification
- `outputs/summary.json` - headline values and version record

## Interpretation boundary

Source: Moro, Rita, and Cortez (2014), *Bank Marketing*, UCI Machine Learning
Repository, DOI: https://doi.org/10.24432/C5K306 (CC BY 4.0).

The outcome is observed subscription, not incremental treatment effect. The data omit
stable customer IDs, full row-level dates, costs, and customer value; the project claims
neither causal lift nor monetary ROI. Earlier versions used the fixed holdout, so it is
labeled a development holdout rather than pristine prospective external validation.
