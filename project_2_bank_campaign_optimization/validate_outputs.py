"""Independent integrity checks for the publication-upgrade outputs."""

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

REQUIRED_OUTPUTS = {
    "data_quality.csv", "feature_availability_audit.csv",
    "model_selection_summary.csv", "cross_validation_folds.csv",
    "model_metrics.csv", "model_pairwise_bootstrap.csv",
    "budget_metrics.csv", "targeting_deciles.csv", "capture_efficiency.csv",
    "bootstrap_intervals.csv", "calibration_metrics.csv", "calibration_curve.csv",
    "information_set_comparison.csv", "duplicate_sensitivity.csv",
    "leakage_audit.csv", "logistic_coefficients.csv",
    "permutation_importance.csv", "summary.json",
}
REQUIRED_CHARTS = {
    "model_validation.png", "cumulative_gains.png", "calibration_curve.png",
    "information_set_and_leakage.png", "permutation_importance.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    assert set(columns) <= set(frame.columns)
    assert np.isfinite(frame[columns].to_numpy(dtype=float)).all()


def main() -> None:
    assert REQUIRED_OUTPUTS <= {path.name for path in OUTPUT_DIR.iterdir()}
    assert REQUIRED_CHARTS <= {path.name for path in (OUTPUT_DIR / "charts").glob("*.png")}
    assert (PROJECT_DIR / "PAPER_RESULTS.md").exists()
    assert (PROJECT_DIR / "PUBLICATION_UPGRADE_AUDIT.md").exists()

    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(OUTPUT_DIR / "feature_availability_audit.csv")
    folds = pd.read_csv(OUTPUT_DIR / "cross_validation_folds.csv")
    selection = pd.read_csv(OUTPUT_DIR / "model_selection_summary.csv")
    models = pd.read_csv(OUTPUT_DIR / "model_metrics.csv")
    pairwise = pd.read_csv(OUTPUT_DIR / "model_pairwise_bootstrap.csv")
    budget = pd.read_csv(OUTPUT_DIR / "budget_metrics.csv")
    deciles = pd.read_csv(OUTPUT_DIR / "targeting_deciles.csv")
    intervals = pd.read_csv(OUTPUT_DIR / "bootstrap_intervals.csv")
    calibration = pd.read_csv(OUTPUT_DIR / "calibration_metrics.csv")
    curve = pd.read_csv(OUTPUT_DIR / "calibration_curve.csv")
    information = pd.read_csv(OUTPUT_DIR / "information_set_comparison.csv")
    duplicates = pd.read_csv(OUTPUT_DIR / "duplicate_sensitivity.csv")
    leakage = pd.read_csv(OUTPUT_DIR / "leakage_audit.csv")
    importance = pd.read_csv(OUTPUT_DIR / "permutation_importance.csv")

    assert sha256(DATA_PATH) == EXPECTED_HASH == summary["data_sha256"]
    assert summary["raw_rows"] - summary["duplicate_rows"] == summary["analysis_rows"]
    assert len(audit) == 21 and audit["Feature"].nunique() == 21
    assert set(audit.loc[audit["Primary_Model_Allowed"], "Feature"]) == set(summary["primary_features"])
    assert not audit.loc[audit["Feature"].isin(["duration", "campaign"]), "Primary_Model_Allowed"].any()

    assert len(folds) == 20 and set(folds["Fold"]) == {1, 2, 3, 4, 5}
    assert (folds["Index_Overlap"] == 0).all()
    assert_finite(folds, ["ROC_AUC", "PR_AUC", "Brier_Score", "Log_Loss"])
    selected = selection.sort_values(["PR_AUC_Mean", "ROC_AUC_Mean"], ascending=False).iloc[0]
    assert selected["Model"] == summary["selected_base_model"]
    assert np.isclose(selected["PR_AUC_Mean"], summary["cv_pr_auc_mean"])

    final = models.loc[models["Role"] == "Final selected model"].iloc[0]
    assert np.isclose(final["ROC_AUC"], summary["roc_auc"])
    assert np.isclose(final["PR_AUC"], summary["pr_auc"])
    assert np.isclose(final["Brier_Score"], summary["brier_score"])
    assert_finite(models, ["ROC_AUC", "PR_AUC", "Brier_Score", "Log_Loss", "Top20_Capture", "Top20_Lift"])

    assert np.allclose(budget["contact_share"], np.arange(0.1, 1.01, 0.1))
    assert budget["contact_records"].is_monotonic_increasing
    assert budget["responder_capture"].is_monotonic_increasing
    assert (budget["random_lift"] == 1).all()
    assert np.allclose(budget["random_responder_capture"], budget["selected_record_share"])
    assert int(budget.iloc[-1]["contact_records"]) == summary["holdout_rows"]
    assert np.isclose(budget.iloc[-1]["responder_capture"], 1.0)
    top20 = budget.loc[np.isclose(budget["contact_share"], 0.2)].iloc[0]
    assert np.isclose(top20["lift"], summary["top_20_lift"])
    assert np.isclose(top20["responder_capture"], summary["top_20_capture"])

    assert int(deciles["ContactRecords"].sum()) == summary["holdout_rows"]
    assert int(deciles["Subscribers"].sum()) == int(budget.iloc[-1]["subscribers_captured"])
    assert deciles["CumulativeCapture"].is_monotonic_increasing
    assert np.isclose(deciles.iloc[-1]["CumulativeCapture"], 1.0)

    required_intervals = {"ROC_AUC", "PR_AUC", "Brier_Score", "Top10_Capture", "Top20_Capture", "Top30_Capture", "Top20_Lift"}
    assert set(intervals["Metric"]) == required_intervals
    assert ((intervals["CI95_Lower"] <= intervals["Estimate"]) & (intervals["Estimate"] <= intervals["CI95_Upper"])).all()
    assert (intervals["Bootstrap_Samples"] == 1_000).all()
    assert (pairwise["Bootstrap_Samples"] == 1_000).all()
    assert_finite(pairwise, ["Estimate", "CI95_Lower", "CI95_Upper"])

    training_calibration = calibration.loc[calibration["Evaluation_Split"] == "Training nested OOF"]
    chosen = training_calibration.loc[training_calibration["Selected"]].iloc[0]
    assert np.isclose(chosen["Brier_Score"], training_calibration["Brier_Score"].min())
    assert summary["calibration_method_label"] in chosen["Method"].lower()
    assert curve["Mean_Predicted_Probability"].between(0, 1).all()
    assert curve["Observed_Subscription_Rate"].between(0, 1).all()

    assert information["Primary"].sum() == 1
    primary = information.loc[information["Primary"]].iloc[0]
    assert primary["Information_Set"] == summary["primary_information_set"]
    assert set(primary["Features"].split("|")) == set(summary["primary_features"])
    assert duplicates["Rows"].tolist() == [41_176, 41_188]
    assert len(importance) == len(summary["primary_features"])
    assert_finite(importance, ["Importance_Mean", "Importance_SD"])

    assert leakage["Includes_Duration"].tolist() == [False, True]
    assert leakage["Deployable"].tolist() == [True, False]
    assert leakage.iloc[1]["PR_AUC"] > leakage.iloc[0]["PR_AUC"]
    assert summary["final_model"] not in leakage.loc[leakage["Includes_Duration"], "Feature_Set"].tolist()
    assert all(summary["validation_checks"].values())

    paper = (PROJECT_DIR / "PAPER_RESULTS.md").read_text(encoding="utf-8")
    required_snippets = [
        f"{summary['analysis_rows']:,} observations",
        f"PR-AUC {summary['pr_auc']:.3f}",
        f"ROC-AUC {summary['roc_auc']:.3f}",
        f"Brier score {summary['brier_score']:.3f}",
        f"capture was {summary['top_20_capture']:.1%}",
        f"lift was {summary['top_20_lift']:.2f}x",
        f"to {summary['leaky_model_pr_auc']:.3f}",
    ]
    assert all(snippet in paper for snippet in required_snippets)
    print("All publication-upgrade validation checks passed.")


if __name__ == "__main__":
    main()
