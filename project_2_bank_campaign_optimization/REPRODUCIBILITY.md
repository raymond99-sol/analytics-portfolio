# Reproducibility

## Environment and source

Run the project with Python 3.12 or a compatible recent Python 3 release. Install the
pinned minimum dependencies in `requirements.txt`, then download the public source file:

```bash
python download_data.py --project bank
pip install -r project_2_bank_campaign_optimization/requirements.txt
```

Expected input:

- Path: `project_2_bank_campaign_optimization/data/bank-additional-full.csv`
- SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`
- Raw rows: 41,188
- Rows after removing 12 exact duplicates: 41,176

The downloaded `data/` directory is intentionally excluded from GitHub.

## Deterministic execution

```bash
python project_2_bank_campaign_optimization/analysis.py
python project_2_bank_campaign_optimization/validate_outputs.py
```

The analysis uses random seed 42, a stratified 80/20 train-holdout split, five
stratified training folds, and 1,000 bootstrap resamples. It refreshes every CSV, JSON,
SQLite, and PNG artifact under `outputs/`; the SQLite file is ignored by Git.

Expected headline results, allowing for small dependency-version differences:

- selected base model: Random Forest
- training five-fold mean PR-AUC: approximately 0.467
- calibrated holdout ROC-AUC: approximately 0.813
- calibrated holdout PR-AUC: approximately 0.485
- top-20% responder capture: approximately 65.9%
- top-20% lift: approximately 3.30x

`validate_outputs.py` independently checks source identity, row reconciliation, training
fold coverage, selection logic, holdout metrics, budget/decile totals, confidence
intervals, calibration improvement, leakage direction, and required chart files.
