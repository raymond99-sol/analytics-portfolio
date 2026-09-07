"""Independent integrity checks for saved bank campaign analysis outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_PATH = PROJECT_DIR / "data" / "bank-additional-full.csv"
EXPECTED_HASH = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    cv_summary = pd.read_csv(OUTPUT_DIR / "model_selection_summary.csv")
    folds = pd.read_csv(OUTPUT_DIR / "cross_validation_folds.csv")
    holdout = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    budget = pd.read_csv(OUTPUT_DIR / "budget_metrics.csv")
    deciles = pd.read_csv(OUTPUT_DIR / "targeting_deciles.csv")
    intervals = pd.read_csv(OUTPUT_DIR / "bootstrap_intervals.csv")
    calibration = pd.read_csv(OUTPUT_DIR / "calibration_metrics.csv")
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_audit.csv")

    assert sha256(DATA_PATH) == EXPECTED_HASH == summary["data_sha256"]
    assert summary["raw_rows"] - summary["duplicate_rows_removed"] == summary["analysis_rows"]
    assert len(folds) == 15 and set(folds["Fold"]) == {1, 2, 3, 4, 5}
    selected = cv_summary.sort_values("PR_AUC_Mean", ascending=False).iloc[0]
    assert selected["Model"] == summary["selected_base_model"] == "Random Forest"
    assert np.isclose(selected["PR_AUC_Mean"], summary["cv_pr_auc_mean"])
    assert np.isclose(holdout.iloc[0]["ROC_AUC"], summary["roc_auc"])
    assert np.isclose(holdout.iloc[0]["PR_AUC"], summary["pr_auc"])

    assert np.allclose(budget["contact_share"], np.arange(0.1, 1.01, 0.1))
    top_20 = budget.loc[np.isclose(budget["contact_share"], 0.2)].iloc[0]
    assert np.isclose(top_20["lift"], summary["top_20_lift"])
    assert np.isclose(top_20["responder_capture"], summary["top_20_capture"])
    assert np.isclose(budget.iloc[-1]["lift"], 1.0)
    assert np.isclose(budget.iloc[-1]["responder_capture"], 1.0)

    assert int(deciles["ContactRecords"].sum()) == summary["holdout_rows"]
    assert int(deciles["Subscribers"].sum()) == round(
        summary["holdout_rows"] * summary["subscription_rate"]
    )
    assert np.isclose(deciles.iloc[-1]["CumulativeCapture"], 1.0)

    assert set(intervals["Metric"]) == {
        "ROC_AUC", "PR_AUC", "Top20_Lift", "Top20_Capture"
    }
    assert (
        (intervals["CI95_Lower"] <= intervals["Estimate"])
        & (intervals["Estimate"] <= intervals["CI95_Upper"])
    ).all()
    assert (intervals["BootstrapSamples"] == 1_000).all()
    assert calibration.iloc[1]["Brier_Score"] < calibration.iloc[0]["Brier_Score"]
    assert bool(leakage.iloc[0]["Includes_Duration"]) is False
    assert bool(leakage.iloc[1]["Includes_Duration"]) is True
    assert leakage.iloc[1]["PR_AUC"] > leakage.iloc[0]["PR_AUC"]
    assert all(summary["validation_checks"].values())

    expected_charts = {
        "conversion_by_previous_outcome.png",
        "model_validation.png",
        "cumulative_gains.png",
        "calibration_curve.png",
        "logistic_coefficients.png",
    }
    assert expected_charts <= {path.name for path in (OUTPUT_DIR / "charts").glob("*.png")}
    print("All bank campaign output checks passed.")


if __name__ == "__main__":
    main()
