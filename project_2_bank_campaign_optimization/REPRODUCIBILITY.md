# Reproducibility

## Source and environment

```bash
python download_data.py --project bank
python -m pip install -r project_2_bank_campaign_optimization/requirements.txt
```

Expected source:

- `project_2_bank_campaign_optimization/data/bank-additional-full.csv`
- SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`
- 41,188 source rows; 12 exact duplicates; 41,176 primary-analysis rows

`outputs/summary.json` records Python, pandas, NumPy, scikit-learn, and Matplotlib
versions. `requirements.txt` specifies tested minimum direct dependencies.

## Deterministic execution

```bash
python project_2_bank_campaign_optimization/analysis.py
python project_2_bank_campaign_optimization/validate_outputs.py
```

The pipeline uses seed 42, a fixed stratified 80/20 split, five stratified training
folds, three-fold nested calibration selection, and 1,000 paired bootstrap resamples.
The prespecified information-set contrasts reuse identical fixed-holdout row indices
within every bootstrap resample; their machine-readable intervals are saved in
`outputs/information_set_pairwise_bootstrap.csv`. Preprocessing is fitted inside each
training fold. Model and calibration choices are locked before holdout evaluation.

To refresh empirical outputs without rewriting the generated manuscript-input files:

```bash
python -c "from project_2_bank_campaign_optimization.analysis import run_analysis; run_analysis(write_manuscript_files=False)"
```

## Information timing

UCI states that `campaign` includes the focal contact and `duration` is unavailable
before the call. The primary model excludes both. Nonnegative
`campaign_prior = campaign - 1` appears only in operational sensitivity models. Macro
indicators are tested separately because they may be observable at contact time yet
encode campaign-period regimes and may not be publication-vintage values.

## Holdout status

Earlier repository versions reported this split, so it is a fixed development holdout,
not pristine prospective external validation. The file is date ordered but has only
month and weekday at row level; a complete chronological split cannot be reconstructed.

## Notebook

```bash
cd project_2_bank_campaign_optimization
python -m jupyter nbconvert --execute --to notebook --inplace bank_campaign_optimization.ipynb
```

The notebook imports and calls `run_analysis()`; calculations are not duplicated.
