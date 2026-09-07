"""Reproducible analysis pipeline for the bank campaign targeting project."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
CV_FOLDS = 5
BOOTSTRAP_REPEATS = 1_000
TOP_SHARE = 0.20
EXPECTED_DATA_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

BLUE = "#2F6BFF"
BLUE_DARK = "#163B77"
BLUE_LIGHT = "#AFC9FF"
GOLD = "#D8A227"
INK = "#20242B"
GREY = "#7C848E"


def resolve_project_dir() -> Path:
    """Resolve the project folder when run from the repo root or project folder."""
    current = Path.cwd()
    if (current / "data" / "bank-additional-full.csv").exists():
        return current
    candidate = current / "project_2_bank_campaign_optimization"
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = features.select_dtypes(exclude="object").columns.tolist()
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def candidate_estimators() -> dict[str, object]:
    return {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=1_500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def make_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features)),
            ("model", clone(estimator)),
        ]
    )


def evaluate_ranking(actual: np.ndarray, scores: np.ndarray, share: float) -> dict[str, float]:
    order = np.argsort(-scores, kind="mergesort")
    ranked_actual = actual[order]
    selected_count = int(np.ceil(len(ranked_actual) * share))
    selected_actual = ranked_actual[:selected_count]
    baseline_rate = float(ranked_actual.mean())
    selected_rate = float(selected_actual.mean())
    return {
        "contact_share": float(share),
        "contact_records": selected_count,
        "conversion_rate": selected_rate,
        "lift": selected_rate / baseline_rate,
        "responder_capture": float(selected_actual.sum() / ranked_actual.sum()),
    }


def bootstrap_intervals(
    actual: np.ndarray,
    scores: np.ndarray,
    point_estimates: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_metrics = {name: [] for name in point_estimates}
    row_count = len(actual)

    for _ in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, row_count, size=row_count)
        sampled_actual = actual[indices]
        sampled_scores = scores[indices]
        if sampled_actual.min() == sampled_actual.max():
            continue
        targeting = evaluate_ranking(sampled_actual, sampled_scores, TOP_SHARE)
        values = {
            "ROC_AUC": roc_auc_score(sampled_actual, sampled_scores),
            "PR_AUC": average_precision_score(sampled_actual, sampled_scores),
            "Top20_Lift": targeting["lift"],
            "Top20_Capture": targeting["responder_capture"],
        }
        for name, value in values.items():
            sampled_metrics[name].append(float(value))

    rows = []
    for name, estimate in point_estimates.items():
        distribution = np.asarray(sampled_metrics[name])
        rows.append(
            {
                "Metric": name,
                "Estimate": estimate,
                "CI95_Lower": float(np.quantile(distribution, 0.025)),
                "CI95_Upper": float(np.quantile(distribution, 0.975)),
                "BootstrapSamples": int(len(distribution)),
            }
        )
    return pd.DataFrame(rows)


def save_charts(
    chart_dir: Path,
    previous_outcome: pd.DataFrame,
    cv_summary: pd.DataFrame,
    decile_summary: pd.DataFrame,
    calibration_table: pd.DataFrame,
    top_coefficients: pd.DataFrame,
    top_capture: float,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CDD3DA",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )

    previous_plot = previous_outcome.sort_values("conversion_rate_pct")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(previous_plot["previous_outcome"], previous_plot["conversion_rate_pct"], color=BLUE)
    ax.set_title("Subscription Rate by Previous Campaign Outcome", loc="left", pad=28)
    ax.text(
        0,
        1.02,
        "Campaign contact records, May 2008-Nov. 2010; n=41,176",
        transform=ax.transAxes,
        color=GREY,
        fontsize=9,
    )
    ax.set_xlabel("Subscription rate (%)")
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    for index, row in previous_plot.reset_index(drop=True).iterrows():
        ax.text(
            row["conversion_rate_pct"] + 0.5,
            index,
            f'{row["conversion_rate_pct"]:.1f}%  (n={int(row["contact_records"]):,})',
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(chart_dir / "conversion_by_previous_outcome.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ordered = cv_summary.sort_values("PR_AUC_Mean")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, label in [
        (axes[0], "PR_AUC", "PR-AUC"),
        (axes[1], "ROC_AUC", "ROC-AUC"),
    ]:
        ax.barh(ordered["Model"], ordered[f"{metric}_Mean"], color=BLUE)
        ax.errorbar(
            ordered[f"{metric}_Mean"],
            ordered["Model"],
            xerr=ordered[f"{metric}_Std"],
            fmt="none",
            ecolor=BLUE_DARK,
            capsize=4,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Mean {label} (error bars: +/- 1 SD)")
        ax.grid(axis="x", color="#E8EBEF", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.suptitle("Five-fold Training-set Model Validation", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.91, "Stratified folds; selection metric: mean PR-AUC", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(chart_dir / "model_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1.5, label="Random targeting")
    ax.plot(
        np.r_[0, decile_summary["ContactShare"]],
        np.r_[0, decile_summary["CumulativeCapture"]],
        color=BLUE,
        marker="o",
        linewidth=2.4,
        label="Calibrated Random Forest",
    )
    ax.axvline(TOP_SHARE, color=GOLD, linewidth=1.3, linestyle=":")
    ax.scatter([TOP_SHARE], [top_capture], s=70, color=GOLD, zorder=4)
    ax.annotate(
        f"Top 20% captures {top_capture:.1%}\nof holdout subscribers",
        xy=(TOP_SHARE, top_capture),
        xytext=(0.33, max(0.18, top_capture - 0.12)),
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax.set_title("Cumulative Gains from Pre-call Targeting", loc="left", pad=28)
    ax.text(
        0,
        1.02,
        "Stratified holdout set; n=8,236 contact records",
        transform=ax.transAxes,
        color=GREY,
        fontsize=9,
    )
    ax.set_xlabel("Share of contact records prioritized")
    ax.set_ylabel("Share of subscribers captured")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#E8EBEF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(chart_dir / "cumulative_gains.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    raw = calibration_table.loc[calibration_table["Model"] == "Raw Random Forest"]
    calibrated = calibration_table.loc[
        calibration_table["Model"] == "Calibrated Random Forest"
    ]
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, label="Perfect calibration")
    ax.plot(
        raw["MeanPredictedProbability"],
        raw["ObservedSubscriptionRate"],
        color=GOLD,
        marker="s",
        linestyle=":",
        label="Raw Random Forest",
    )
    ax.plot(
        calibrated["MeanPredictedProbability"],
        calibrated["ObservedSubscriptionRate"],
        color=BLUE,
        marker="o",
        label="Calibrated Random Forest",
    )
    ax.set_title("Holdout Probability Calibration", loc="left", pad=28)
    ax.text(
        0,
        1.02,
        "Ten equal-frequency bins; n=8,236 contact records",
        transform=ax.transAxes,
        color=GREY,
        fontsize=9,
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed subscription rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="#E8EBEF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(chart_dir / "calibration_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    colors = [BLUE if value >= 0 else GOLD for value in top_coefficients["Coefficient"]]
    ax.barh(top_coefficients["Feature"], top_coefficients["Coefficient"], color=colors)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_title("Largest Logistic Regression Coefficients", loc="left", pad=28)
    ax.text(
        0,
        1.02,
        "Training data only; standardized log-odds associations, not causal effects",
        transform=ax.transAxes,
        color=GREY,
        fontsize=9,
    )
    ax.set_xlabel("Coefficient")
    ax.grid(axis="x", color="#E8EBEF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(chart_dir / "logistic_coefficients.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(project_dir: Path | None = None) -> dict[str, object]:
    project_dir = project_dir or resolve_project_dir()
    data_path = project_dir / "data" / "bank-additional-full.csv"
    output_dir = project_dir / "outputs"
    chart_dir = output_dir / "charts"
    output_dir.mkdir(exist_ok=True)
    chart_dir.mkdir(exist_ok=True)

    data_hash = file_sha256(data_path)
    if data_hash != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected source data hash: {data_hash}")

    campaign_raw = pd.read_csv(data_path, sep=";")
    expected_columns = {
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays",
        "previous", "poutcome", "emp.var.rate", "cons.price.idx",
        "cons.conf.idx", "euribor3m", "nr.employed", "y",
    }
    if set(campaign_raw.columns) != expected_columns:
        raise ValueError("Source columns do not match the documented UCI extract.")

    duplicate_rows = int(campaign_raw.duplicated().sum())
    campaign = campaign_raw.drop_duplicates().copy()
    campaign["subscribed"] = (campaign["y"] == "yes").astype(int)
    object_data = campaign.select_dtypes(include="object")
    unknown_cells = int((object_data == "unknown").sum().sum())
    rows_with_unknown = int((object_data == "unknown").any(axis=1).sum())

    data_quality = pd.DataFrame(
        [
            {"Check": "Raw rows", "Value": len(campaign_raw), "Status": "Pass"},
            {"Check": "Exact duplicate rows", "Value": duplicate_rows, "Status": "Documented"},
            {"Check": "Analysis rows", "Value": len(campaign), "Status": "Pass"},
            {"Check": "Missing cells", "Value": int(campaign.isna().sum().sum()), "Status": "Pass"},
            {"Check": "Literal unknown cells", "Value": unknown_cells, "Status": "Documented"},
            {"Check": "Rows containing unknown", "Value": rows_with_unknown, "Status": "Documented"},
            {"Check": "Positive outcomes", "Value": int(campaign["subscribed"].sum()), "Status": "Pass"},
        ]
    )

    database_path = output_dir / "bank_campaign.sqlite"
    if database_path.exists():
        database_path.unlink()
    with sqlite3.connect(database_path) as connection:
        campaign.to_sql("campaign", connection, index=False, if_exists="replace")
        previous_outcome = pd.read_sql_query(
            """
            SELECT poutcome AS previous_outcome,
                   COUNT(*) AS contact_records,
                   SUM(subscribed) AS subscribers,
                   ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
            FROM campaign
            GROUP BY poutcome
            ORDER BY conversion_rate_pct DESC
            """,
            connection,
        )
        job_conversion = pd.read_sql_query(
            """
            SELECT job, COUNT(*) AS contact_records, SUM(subscribed) AS subscribers,
                   ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
            FROM campaign
            GROUP BY job
            HAVING COUNT(*) >= 500
            ORDER BY conversion_rate_pct DESC
            """,
            connection,
        )
        contact_conversion = pd.read_sql_query(
            """
            SELECT contact, COUNT(*) AS contact_records, SUM(subscribed) AS subscribers,
                   ROUND(100.0 * AVG(subscribed), 2) AS conversion_rate_pct
            FROM campaign
            GROUP BY contact
            ORDER BY conversion_rate_pct DESC
            """,
            connection,
        )

    target = campaign["subscribed"]
    safe_features = campaign.drop(columns=["y", "subscribed", "duration"])
    train_features, test_features, train_target, test_target = train_test_split(
        safe_features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_rows: list[dict[str, object]] = []
    estimators = candidate_estimators()
    for model_name, estimator in estimators.items():
        pipeline = make_pipeline(train_features, estimator)
        scores = cross_validate(
            pipeline,
            train_features,
            train_target,
            cv=cross_validation,
            scoring={"ROC_AUC": "roc_auc", "PR_AUC": "average_precision"},
            n_jobs=1,
        )
        for fold in range(CV_FOLDS):
            cv_rows.append(
                {
                    "Model": model_name,
                    "Fold": fold + 1,
                    "ROC_AUC": scores["test_ROC_AUC"][fold],
                    "PR_AUC": scores["test_PR_AUC"][fold],
                }
            )
    cv_results = pd.DataFrame(cv_rows)
    cv_summary = (
        cv_results.groupby("Model", as_index=False)
        .agg(
            ROC_AUC_Mean=("ROC_AUC", "mean"),
            ROC_AUC_Std=("ROC_AUC", "std"),
            PR_AUC_Mean=("PR_AUC", "mean"),
            PR_AUC_Std=("PR_AUC", "std"),
        )
        .sort_values("PR_AUC_Mean", ascending=False)
        .reset_index(drop=True)
    )
    selected_base_model = str(cv_summary.iloc[0]["Model"])

    selected_pipeline = make_pipeline(train_features, estimators[selected_base_model])
    selected_pipeline.fit(train_features, train_target)
    raw_probabilities = selected_pipeline.predict_proba(test_features)[:, 1]

    calibrated_model = CalibratedClassifierCV(
        make_pipeline(train_features, estimators[selected_base_model]),
        method="sigmoid",
        cv=CV_FOLDS,
        n_jobs=1,
    )
    calibrated_model.fit(train_features, train_target)
    final_probabilities = calibrated_model.predict_proba(test_features)[:, 1]
    predicted_at_half = (final_probabilities >= 0.5).astype(int)
    test_actual = test_target.to_numpy()

    holdout_metrics = pd.DataFrame(
        [
            {
                "Model": "Calibrated Random Forest",
                "ROC_AUC": roc_auc_score(test_actual, final_probabilities),
                "PR_AUC": average_precision_score(test_actual, final_probabilities),
                "Brier_Score": brier_score_loss(test_actual, final_probabilities),
                "Log_Loss": log_loss(test_actual, final_probabilities),
                "Precision_at_0.5": precision_score(test_actual, predicted_at_half, zero_division=0),
                "Recall_at_0.5": recall_score(test_actual, predicted_at_half, zero_division=0),
                "F1_at_0.5": f1_score(test_actual, predicted_at_half, zero_division=0),
            }
        ]
    )

    budget_rows = [
        evaluate_ranking(test_actual, final_probabilities, share)
        for share in np.arange(0.10, 1.01, 0.10)
    ]
    budget_metrics = pd.DataFrame(budget_rows)
    top_20 = budget_metrics.loc[np.isclose(budget_metrics["contact_share"], TOP_SHARE)].iloc[0]

    scored_test = pd.DataFrame({"actual": test_actual, "score": final_probabilities})
    scored_test = scored_test.sort_values("score", ascending=False).reset_index(drop=True)
    scored_test["decile"] = pd.qcut(
        scored_test["score"].rank(method="first", ascending=False),
        q=10,
        labels=range(1, 11),
    )
    decile_summary = (
        scored_test.groupby("decile", observed=True)
        .agg(
            ContactRecords=("actual", "size"),
            Subscribers=("actual", "sum"),
            ConversionRate=("actual", "mean"),
            AverageScore=("score", "mean"),
        )
        .reset_index()
    )
    decile_summary["CumulativeSubscribers"] = decile_summary["Subscribers"].cumsum()
    decile_summary["CumulativeCapture"] = (
        decile_summary["CumulativeSubscribers"] / decile_summary["Subscribers"].sum()
    )
    decile_summary["ContactShare"] = decile_summary["decile"].astype(int) / 10
    decile_summary["RandomCapture"] = decile_summary["ContactShare"]
    decile_summary["Lift"] = (
        decile_summary["ConversionRate"] / scored_test["actual"].mean()
    )

    point_estimates = {
        "ROC_AUC": float(holdout_metrics.iloc[0]["ROC_AUC"]),
        "PR_AUC": float(holdout_metrics.iloc[0]["PR_AUC"]),
        "Top20_Lift": float(top_20["lift"]),
        "Top20_Capture": float(top_20["responder_capture"]),
    }
    bootstrap = bootstrap_intervals(test_actual, final_probabilities, point_estimates)

    fraction_raw, mean_raw = calibration_curve(
        test_actual, raw_probabilities, n_bins=10, strategy="quantile"
    )
    fraction_calibrated, mean_calibrated = calibration_curve(
        test_actual, final_probabilities, n_bins=10, strategy="quantile"
    )
    calibration_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "Model": "Raw Random Forest",
                    "MeanPredictedProbability": mean_raw,
                    "ObservedSubscriptionRate": fraction_raw,
                }
            ),
            pd.DataFrame(
                {
                    "Model": "Calibrated Random Forest",
                    "MeanPredictedProbability": mean_calibrated,
                    "ObservedSubscriptionRate": fraction_calibrated,
                }
            ),
        ],
        ignore_index=True,
    )
    calibration_metrics = pd.DataFrame(
        [
            {
                "Model": "Raw Random Forest",
                "Brier_Score": brier_score_loss(test_actual, raw_probabilities),
                "Log_Loss": log_loss(test_actual, raw_probabilities),
            },
            {
                "Model": "Calibrated Random Forest",
                "Brier_Score": brier_score_loss(test_actual, final_probabilities),
                "Log_Loss": log_loss(test_actual, final_probabilities),
            },
        ]
    )

    leaky_features = campaign.drop(columns=["y", "subscribed"])
    leaky_train = leaky_features.loc[train_features.index]
    leaky_test = leaky_features.loc[test_features.index]
    leaky_pipeline = make_pipeline(leaky_train, estimators[selected_base_model])
    leaky_pipeline.fit(leaky_train, train_target)
    leaky_probabilities = leaky_pipeline.predict_proba(leaky_test)[:, 1]
    leakage_audit = pd.DataFrame(
        [
            {
                "Feature_Set": "Pre-call features only",
                "Includes_Duration": False,
                "ROC_AUC": roc_auc_score(test_actual, raw_probabilities),
                "PR_AUC": average_precision_score(test_actual, raw_probabilities),
            },
            {
                "Feature_Set": "Pre-call features plus duration",
                "Includes_Duration": True,
                "ROC_AUC": roc_auc_score(test_actual, leaky_probabilities),
                "PR_AUC": average_precision_score(test_actual, leaky_probabilities),
            },
        ]
    )

    logistic_pipeline = make_pipeline(train_features, estimators["Logistic Regression"])
    logistic_pipeline.fit(train_features, train_target)
    feature_names = logistic_pipeline.named_steps["preprocess"].get_feature_names_out()
    coefficients = logistic_pipeline.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "AbsoluteCoefficient": np.abs(coefficients),
        }
    )
    coefficient_table["Feature"] = (
        coefficient_table["Feature"]
        .str.replace("numeric__", "", regex=False)
        .str.replace("categorical__", "", regex=False)
    )
    top_coefficients = coefficient_table.nlargest(16, "AbsoluteCoefficient").sort_values("Coefficient")

    output_tables = {
        "data_quality.csv": data_quality,
        "conversion_by_previous_outcome.csv": previous_outcome,
        "conversion_by_job.csv": job_conversion,
        "conversion_by_contact.csv": contact_conversion,
        "cross_validation_folds.csv": cv_results,
        "model_selection_summary.csv": cv_summary,
        "model_metrics.csv": holdout_metrics,
        "budget_metrics.csv": budget_metrics,
        "targeting_deciles.csv": decile_summary,
        "bootstrap_intervals.csv": bootstrap,
        "calibration_curve.csv": calibration_table,
        "calibration_metrics.csv": calibration_metrics,
        "leakage_audit.csv": leakage_audit,
        "logistic_coefficients.csv": coefficient_table,
    }
    for filename, table in output_tables.items():
        table.to_csv(output_dir / filename, index=False)

    save_charts(
        chart_dir=chart_dir,
        previous_outcome=previous_outcome,
        cv_summary=cv_summary,
        decile_summary=decile_summary,
        calibration_table=calibration_table,
        top_coefficients=top_coefficients,
        top_capture=float(top_20["responder_capture"]),
    )

    selected_cv = cv_summary.loc[cv_summary["Model"] == selected_base_model].iloc[0]
    interval_lookup = bootstrap.set_index("Metric")
    validation_checks = {
        "source_hash_matches": data_hash == EXPECTED_DATA_SHA256,
        "duration_excluded_from_selection": "duration" not in safe_features.columns,
        "holdout_isolation": set(train_features.index).isdisjoint(test_features.index),
        "train_test_rows_reconcile": len(train_features) + len(test_features) == len(campaign),
        "probabilities_in_range": bool(
            ((final_probabilities >= 0) & (final_probabilities <= 1)).all()
        ),
        "decile_contact_records_reconcile": int(decile_summary["ContactRecords"].sum()) == len(test_features),
        "decile_subscribers_reconcile": int(decile_summary["Subscribers"].sum()) == int(test_actual.sum()),
        "bootstrap_intervals_contain_estimates": bool(
            (
                (bootstrap["CI95_Lower"] <= bootstrap["Estimate"])
                & (bootstrap["Estimate"] <= bootstrap["CI95_Upper"])
            ).all()
        ),
        "calibration_improves_brier": bool(
            calibration_metrics.iloc[1]["Brier_Score"] < calibration_metrics.iloc[0]["Brier_Score"]
        ),
        "leakage_audit_inflates_performance": bool(
            leakage_audit.iloc[1]["PR_AUC"] > leakage_audit.iloc[0]["PR_AUC"]
        ),
    }
    if not all(validation_checks.values()):
        raise AssertionError(validation_checks)

    summary = {
        "data_sha256": data_hash,
        "raw_rows": int(len(campaign_raw)),
        "duplicate_rows_removed": duplicate_rows,
        "analysis_rows": int(len(campaign)),
        "positive_outcomes": int(campaign["subscribed"].sum()),
        "subscription_rate": float(campaign["subscribed"].mean()),
        "rows_with_unknown": rows_with_unknown,
        "selected_base_model": selected_base_model,
        "selection_rule": "Highest mean five-fold training-set PR-AUC",
        "cv_pr_auc_mean": float(selected_cv["PR_AUC_Mean"]),
        "cv_pr_auc_std": float(selected_cv["PR_AUC_Std"]),
        "cv_roc_auc_mean": float(selected_cv["ROC_AUC_Mean"]),
        "cv_roc_auc_std": float(selected_cv["ROC_AUC_Std"]),
        "final_model": "Calibrated Random Forest",
        "holdout_rows": int(len(test_features)),
        "roc_auc": point_estimates["ROC_AUC"],
        "roc_auc_ci95": [
            float(interval_lookup.loc["ROC_AUC", "CI95_Lower"]),
            float(interval_lookup.loc["ROC_AUC", "CI95_Upper"]),
        ],
        "pr_auc": point_estimates["PR_AUC"],
        "pr_auc_ci95": [
            float(interval_lookup.loc["PR_AUC", "CI95_Lower"]),
            float(interval_lookup.loc["PR_AUC", "CI95_Upper"]),
        ],
        "brier_score": float(holdout_metrics.iloc[0]["Brier_Score"]),
        "top_20_conversion_rate": float(top_20["conversion_rate"]),
        "top_20_lift": point_estimates["Top20_Lift"],
        "top_20_lift_ci95": [
            float(interval_lookup.loc["Top20_Lift", "CI95_Lower"]),
            float(interval_lookup.loc["Top20_Lift", "CI95_Upper"]),
        ],
        "top_20_capture": point_estimates["Top20_Capture"],
        "top_20_capture_ci95": [
            float(interval_lookup.loc["Top20_Capture", "CI95_Lower"]),
            float(interval_lookup.loc["Top20_Capture", "CI95_Upper"]),
        ],
        "safe_model_pr_auc": float(leakage_audit.iloc[0]["PR_AUC"]),
        "leaky_model_pr_auc": float(leakage_audit.iloc[1]["PR_AUC"]),
        "safe_model_roc_auc": float(leakage_audit.iloc[0]["ROC_AUC"]),
        "leaky_model_roc_auc": float(leakage_audit.iloc[1]["ROC_AUC"]),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "validation_checks": validation_checks,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "data_quality": data_quality,
        "cv_summary": cv_summary,
        "holdout_metrics": holdout_metrics,
        "bootstrap_intervals": bootstrap,
        "budget_metrics": budget_metrics,
        "leakage_audit": leakage_audit,
        "previous_outcome": previous_outcome,
    }


if __name__ == "__main__":
    results = run_analysis()
    print(json.dumps(results["summary"], indent=2))
