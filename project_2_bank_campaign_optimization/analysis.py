"""Publication-oriented, reproducible bank campaign targeting analysis.

Model and calibration selection use training data only. The fixed holdout is used for
prespecified final comparisons and is explicitly treated as a development holdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CV_FOLDS = 5
CALIBRATION_FOLDS = 3
BOOTSTRAP_REPEATS = 1_000
EXPECTED_DATA_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"

CLIENT_FEATURES = ["age", "job", "marital", "education", "default", "housing", "loan"]
HISTORY_FEATURES = ["pdays", "previous", "poutcome"]
PRIMARY_FEATURES = CLIENT_FEATURES + HISTORY_FEATURES
CONTACT_FEATURES = ["contact", "month", "day_of_week", "campaign_prior"]
MACRO_FEATURES = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
INFORMATION_SETS = {
    "Strict planning (primary)": PRIMARY_FEATURES,
    "Planning plus macro context": PRIMARY_FEATURES + MACRO_FEATURES,
    "Operational pre-contact": PRIMARY_FEATURES + CONTACT_FEATURES,
    "Operational plus macro context": PRIMARY_FEATURES + CONTACT_FEATURES + MACRO_FEATURES,
}

BLUE, BLUE_DARK, BLUE_LIGHT = "#2F6BFF", "#163B77", "#AFC9FF"
GOLD, RED, INK, GREY = "#D8A227", "#C34A36", "#20242B", "#6F7782"


def resolve_project_dir() -> Path:
    current = Path.cwd()
    if (current / "data" / "bank-additional-full.csv").exists():
        return current
    candidate = current / "project_2_bank_campaign_optimization"
    return candidate if candidate.exists() else Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_availability_audit() -> pd.DataFrame:
    """Decision-time audit based on the UCI data dictionary and Moro et al. (2014)."""
    source = "UCI bank-additional-names.txt; Moro, Cortez, and Rita (2014)"
    rows = [
        ("age", "Client age.", True, True, False, True, "Bank-held client attribute."),
        ("job", "Client occupation category.", True, True, False, True, "Bank-held client attribute; unknown remains a category."),
        ("marital", "Client marital status.", True, True, False, True, "Bank-held client attribute; divorced includes widowed."),
        ("education", "Client education category.", True, True, False, True, "Bank-held client attribute; unknown remains a category."),
        ("default", "Whether the client has credit in default.", True, True, False, True, "Bank-held credit attribute; unknown remains a category."),
        ("housing", "Whether the client has a housing loan.", True, True, False, True, "Bank-held credit attribute; unknown remains a category."),
        ("loan", "Whether the client has a personal loan.", True, True, False, True, "Bank-held credit attribute; unknown remains a category."),
        ("contact", "Communication channel for the current/last contact.", False, True, False, False, "Known for a scheduled contact but not necessarily at campaign-list selection."),
        ("month", "Month of the current/last contact.", False, True, False, False, "Current-contact timing; operational sensitivity only."),
        ("day_of_week", "Weekday of the current/last contact.", False, True, False, False, "Current-contact timing; operational sensitivity only."),
        ("duration", "Duration of the current/last contact in seconds.", False, False, True, False, "UCI explicitly says it is unknown before the call and invalid for realistic targeting."),
        ("campaign", "Contacts during this campaign, including the current/last contact.", False, False, False, False, "Raw value includes the focal contact and is temporally misaligned."),
        ("campaign_prior", "Derived earlier current-campaign contacts: campaign minus one.", False, True, False, False, "UCI definition supports removing the included focal contact; operational sensitivity only."),
        ("pdays", "Days since prior-campaign contact; 999 means none.", True, True, False, True, "Historical campaign record available before selection."),
        ("previous", "Contacts performed before this campaign.", True, True, False, True, "Historical campaign record available before selection."),
        ("poutcome", "Outcome of the previous marketing campaign.", True, True, False, True, "Historical outcome available before selection."),
        ("emp.var.rate", "Quarterly employment variation rate.", False, True, False, False, "Observable context but a campaign-period proxy; sensitivity only."),
        ("cons.price.idx", "Monthly consumer price index.", False, True, False, False, "Release timing, revisions, and period proxying warrant sensitivity-only use."),
        ("cons.conf.idx", "Monthly consumer confidence index.", False, True, False, False, "Release timing, revisions, and period proxying warrant sensitivity-only use."),
        ("euribor3m", "Daily three-month Euribor rate.", False, True, False, False, "Observable market context but a strong campaign-period proxy; sensitivity only."),
        ("nr.employed", "Quarterly number of employees.", False, True, False, False, "Release timing, revisions, and period proxying warrant sensitivity-only use."),
    ]
    columns = ["Feature", "Description", "Planning_Time_Available", "Pre_Contact_Available", "Post_Contact_Only", "Primary_Model_Allowed", "Rationale"]
    audit = pd.DataFrame(rows, columns=columns)
    audit["Authoritative_Source"] = source
    return audit


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "string"]).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        sparse_threshold=0,
    )


def candidate_estimators() -> dict[str, object]:
    return {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=12, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.06, max_iter=160, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, class_weight="balanced", random_state=RANDOM_STATE),
    }


def make_pipeline(features: pd.DataFrame, estimator: object) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(features)), ("model", clone(estimator))])


def make_model(features: pd.DataFrame, estimator: object, calibration: str, folds: int = CV_FOLDS) -> object:
    base = make_pipeline(features, estimator)
    return base if calibration == "none" else CalibratedClassifierCV(base, method=calibration, cv=folds, n_jobs=1)


def probability_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "ROC_AUC": float(roc_auc_score(actual, probabilities)),
        "PR_AUC": float(average_precision_score(actual, probabilities)),
        "Brier_Score": float(brier_score_loss(actual, probabilities)),
        "Log_Loss": float(log_loss(actual, probabilities)),
    }


def evaluate_ranking(actual: np.ndarray, scores: np.ndarray, share: float) -> dict[str, float]:
    ranked = actual[np.argsort(-scores, kind="mergesort")]
    contacts = int(np.ceil(len(ranked) * share))
    selected = ranked[:contacts]
    subscribers = int(selected.sum())
    prevalence = float(ranked.mean())
    actual_share = contacts / len(ranked)
    return {
        "contact_share": float(share),
        "selected_record_share": actual_share,
        "contact_records": contacts,
        "subscribers_captured": subscribers,
        "conversion_rate": float(selected.mean()),
        "lift": float(selected.mean() / prevalence),
        "responder_capture": float(subscribers / ranked.sum()),
        "contacts_per_observed_subscriber": float(contacts / subscribers),
        "random_expected_subscribers": float(contacts * prevalence),
        "random_conversion_rate": prevalence,
        "random_lift": 1.0,
        "random_responder_capture": actual_share,
        "random_contacts_per_observed_subscriber": float(1.0 / prevalence),
    }


def cross_validate_candidates(features: pd.DataFrame, target: pd.Series, estimators: dict[str, object], cv: StratifiedKFold) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = list(cv.split(features, target))
    scoring = {"ROC_AUC": "roc_auc", "PR_AUC": "average_precision", "Brier_Score": "neg_brier_score", "Log_Loss": "neg_log_loss"}
    rows = []
    for name, estimator in estimators.items():
        scores = cross_validate(make_pipeline(features, estimator), features, target, cv=splits, scoring=scoring, n_jobs=1)
        for fold, (fit_index, validation_index) in enumerate(splits, 1):
            rows.append({
                "Model": name, "Fold": fold, "Train_Rows": len(fit_index), "Validation_Rows": len(validation_index),
                "Train_Positive_Rate": float(target.iloc[fit_index].mean()), "Validation_Positive_Rate": float(target.iloc[validation_index].mean()),
                "Index_Overlap": len(set(fit_index).intersection(validation_index)), "ROC_AUC": float(scores["test_ROC_AUC"][fold - 1]),
                "PR_AUC": float(scores["test_PR_AUC"][fold - 1]), "Brier_Score": float(-scores["test_Brier_Score"][fold - 1]),
                "Log_Loss": float(-scores["test_Log_Loss"][fold - 1]),
            })
    folds = pd.DataFrame(rows)
    summary = folds.groupby("Model", as_index=False).agg(
        ROC_AUC_Mean=("ROC_AUC", "mean"), ROC_AUC_Std=("ROC_AUC", "std"),
        PR_AUC_Mean=("PR_AUC", "mean"), PR_AUC_Std=("PR_AUC", "std"),
        Brier_Score_Mean=("Brier_Score", "mean"), Log_Loss_Mean=("Log_Loss", "mean"),
    ).sort_values(["PR_AUC_Mean", "ROC_AUC_Mean"], ascending=False).reset_index(drop=True)
    summary["Selection_Rank"] = np.arange(1, len(summary) + 1)
    summary["Selected"] = summary["Selection_Rank"] == 1
    return folds, summary


def select_calibration(features: pd.DataFrame, target: pd.Series, estimator: object) -> tuple[str, pd.DataFrame]:
    outer = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=RANDOM_STATE + 11)
    labels = {"none": "None (raw)", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}
    rows = []
    for method, label in labels.items():
        probabilities = cross_val_predict(make_model(features, estimator, method, CALIBRATION_FOLDS), features, target, cv=outer, method="predict_proba", n_jobs=1)[:, 1]
        rows.append({"Evaluation_Split": "Training nested OOF", "Method": label, **probability_metrics(target.to_numpy(), probabilities), "Selected": False})
    table = pd.DataFrame(rows).sort_values(["Brier_Score", "Log_Loss"]).reset_index(drop=True)
    table.loc[0, "Selected"] = True
    reverse = {label: method for method, label in labels.items()}
    return reverse[str(table.iloc[0]["Method"])], table


def metric_values(actual: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    base = probability_metrics(actual, scores)
    return {
        **base,
        "Top10_Capture": evaluate_ranking(actual, scores, 0.10)["responder_capture"],
        "Top20_Capture": evaluate_ranking(actual, scores, 0.20)["responder_capture"],
        "Top30_Capture": evaluate_ranking(actual, scores, 0.30)["responder_capture"],
        "Top20_Lift": evaluate_ranking(actual, scores, 0.20)["lift"],
    }


def bootstrap_results(actual: np.ndarray, final_scores: np.ndarray, comparators: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    final_names = ["ROC_AUC", "PR_AUC", "Brier_Score", "Top10_Capture", "Top20_Capture", "Top30_Capture", "Top20_Lift"]
    pair_names = ["ROC_AUC", "PR_AUC", "Brier_Score", "Top20_Capture"]
    final_samples = {metric: [] for metric in final_names}
    pair_samples = {(model, metric): [] for model in comparators for metric in pair_names}
    for _ in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, len(actual), len(actual))
        sampled_actual = actual[indices]
        if sampled_actual.min() == sampled_actual.max():
            continue
        final = metric_values(sampled_actual, final_scores[indices])
        for metric in final_names:
            final_samples[metric].append(final[metric])
        for model, scores in comparators.items():
            comparison = metric_values(sampled_actual, scores[indices])
            for metric in pair_names:
                difference = comparison[metric] - final[metric] if metric == "Brier_Score" else final[metric] - comparison[metric]
                pair_samples[(model, metric)].append(difference)
    point = metric_values(actual, final_scores)
    intervals = pd.DataFrame([
        {"Metric": metric, "Estimate": point[metric], "CI95_Lower": float(np.quantile(final_samples[metric], 0.025)), "CI95_Upper": float(np.quantile(final_samples[metric], 0.975)), "Bootstrap_Samples": len(final_samples[metric])}
        for metric in final_names
    ])
    pair_rows = []
    for model, scores in comparators.items():
        comparison = metric_values(actual, scores)
        for metric in pair_names:
            estimate = comparison[metric] - point[metric] if metric == "Brier_Score" else point[metric] - comparison[metric]
            distribution = pair_samples[(model, metric)]
            pair_rows.append({
                "Final_Model": "Selected calibrated specification", "Comparator": model, "Metric": metric,
                "Difference_Definition": "Comparator minus final; positive favors final" if metric == "Brier_Score" else "Final minus comparator; positive favors final",
                "Estimate": estimate, "CI95_Lower": float(np.quantile(distribution, 0.025)), "CI95_Upper": float(np.quantile(distribution, 0.975)), "Bootstrap_Samples": len(distribution),
            })
    return intervals, pd.DataFrame(pair_rows)


def build_deciles(actual: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    scored = pd.DataFrame({"actual": actual, "score": scores}).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    scored["decile"] = pd.qcut(scored.index.to_series().rank(method="first"), 10, labels=range(1, 11))
    result = scored.groupby("decile", observed=True).agg(ContactRecords=("actual", "size"), Subscribers=("actual", "sum"), ConversionRate=("actual", "mean"), AverageScore=("score", "mean")).reset_index()
    result["CumulativeContacts"] = result["ContactRecords"].cumsum()
    result["CumulativeSubscribers"] = result["Subscribers"].cumsum()
    result["ContactShare"] = result["CumulativeContacts"] / len(scored)
    result["CumulativeCapture"] = result["CumulativeSubscribers"] / result["Subscribers"].sum()
    result["RandomCapture"] = result["ContactShare"]
    result["Lift"] = result["ConversionRate"] / scored["actual"].mean()
    return result


def capture_efficiency(actual: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    ranked = actual[np.argsort(-scores, kind="mergesort")]
    cumulative, total = np.cumsum(ranked), int(ranked.sum())
    rows = []
    for capture in (0.20, 0.50, 0.80):
        target = int(np.ceil(total * capture))
        model_contacts = int(np.argmax(cumulative >= target) + 1)
        random_contacts = float(capture * len(ranked))
        rows.append({"Target_Responder_Capture": capture, "Target_Subscribers": target, "Model_Contacts_Required": model_contacts, "Random_Expected_Contacts": random_contacts, "Expected_Contacts_Avoided_vs_Random": random_contacts - model_contacts})
    return pd.DataFrame(rows)


def evaluate_information_sets(campaign: pd.DataFrame, train_index: pd.Index, test_index: pd.Index, train_target: pd.Series, test_target: pd.Series, estimator: object, calibration: str, primary_cv: pd.Series, primary_probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, columns in INFORMATION_SETS.items():
        train_x, test_x = campaign.loc[train_index, columns], campaign.loc[test_index, columns]
        if name == "Strict planning (primary)":
            cv_pr, cv_roc, probabilities = float(primary_cv["PR_AUC_Mean"]), float(primary_cv["ROC_AUC_Mean"]), primary_probabilities
        else:
            scores = cross_validate(make_pipeline(train_x, estimator), train_x, train_target, cv=cv, scoring={"PR_AUC": "average_precision", "ROC_AUC": "roc_auc"}, n_jobs=1)
            cv_pr, cv_roc = float(scores["test_PR_AUC"].mean()), float(scores["test_ROC_AUC"].mean())
            model = make_model(train_x, estimator, calibration)
            model.fit(train_x, train_target)
            probabilities = model.predict_proba(test_x)[:, 1]
        metrics = probability_metrics(test_target.to_numpy(), probabilities)
        targeting = evaluate_ranking(test_target.to_numpy(), probabilities, 0.20)
        rows.append({"Information_Set": name, "Primary": name == "Strict planning (primary)", "Feature_Count": len(columns), "Features": "|".join(columns), "CV_PR_AUC": cv_pr, "CV_ROC_AUC": cv_roc, "Holdout_PR_AUC": metrics["PR_AUC"], "Holdout_ROC_AUC": metrics["ROC_AUC"], "Holdout_Brier_Score": metrics["Brier_Score"], "Top20_Capture": targeting["responder_capture"], "Top20_Lift": targeting["lift"]})
    return pd.DataFrame(rows)


def duplicate_sensitivity(raw: pd.DataFrame, primary_metrics: dict[str, float], primary_targeting: dict[str, float], estimator: object, calibration: str) -> pd.DataFrame:
    retained = raw.copy()
    retained["subscribed"] = (retained["y"] == "yes").astype(int)
    features, target = retained[PRIMARY_FEATURES], retained["subscribed"]
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target)
    model = make_model(train_x, estimator, calibration)
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    retained_metrics = probability_metrics(test_y.to_numpy(), probabilities)
    retained_targeting = evaluate_ranking(test_y.to_numpy(), probabilities, 0.20)
    return pd.DataFrame([
        {"Specification": "Exact duplicates removed (primary)", "Rows": len(raw.drop_duplicates()), **primary_metrics, "Top20_Capture": primary_targeting["responder_capture"], "Top20_Lift": primary_targeting["lift"]},
        {"Specification": "All source rows retained", "Rows": len(raw), **retained_metrics, "Top20_Capture": retained_targeting["responder_capture"], "Top20_Lift": retained_targeting["lift"]},
    ])


def save_charts(chart_dir: Path, cv: pd.DataFrame, deciles: pd.DataFrame, calibration: pd.DataFrame, information: pd.DataFrame, leakage: pd.DataFrame, importance: pd.DataFrame, model_label: str) -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#CDD3DA", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10, "axes.titleweight": "bold"})
    ordered = cv.sort_values("PR_AUC_Mean")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, label in [(axes[0], "PR_AUC", "PR-AUC"), (axes[1], "ROC_AUC", "ROC-AUC")]:
        ax.barh(ordered["Model"], ordered[f"{metric}_Mean"], color=BLUE)
        ax.errorbar(ordered[f"{metric}_Mean"], ordered["Model"], xerr=ordered[f"{metric}_Std"], fmt="none", ecolor=BLUE_DARK, capsize=4)
        ax.set(xlim=(0, 1), xlabel=f"Mean {label} (+/- 1 fold SD)")
        ax.grid(axis="x", color="#E8EBEF"); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Training-only Benchmark Performance", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.91, "Five stratified folds; strict planning information set", color=GREY, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(chart_dir / "model_validation.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Random targeting")
    ax.plot(np.r_[0, deciles["ContactShare"]], np.r_[0, deciles["CumulativeCapture"]], color=BLUE, marker="o", linewidth=2.4, label=model_label)
    row = deciles.iloc[1]; ax.scatter([row["ContactShare"]], [row["CumulativeCapture"]], s=70, color=GOLD, zorder=4)
    ax.annotate(f"Top 20% captures {row['CumulativeCapture']:.1%}", xy=(row["ContactShare"], row["CumulativeCapture"]), xytext=(0.34, max(0.15, row["CumulativeCapture"] - 0.12)), arrowprops={"arrowstyle": "->", "color": INK})
    ax.set(xlabel="Share of contact records prioritized", ylabel="Share of observed subscribers captured", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Cumulative Gains Under Contact Constraints", loc="left", pad=30)
    ax.text(0, 1.02, "Fixed holdout; observed ranking, not causal lift", transform=ax.transAxes, color=GREY, fontsize=9)
    ax.grid(color="#E8EBEF"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(chart_dir / "cumulative_gains.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], "--", color=GREY, label="Perfect calibration")
    for model, color, marker in [("Selected raw model", GOLD, "s"), ("Final selected model", BLUE, "o")]:
        data = calibration.loc[calibration["Model"] == model]
        ax.plot(data["Mean_Predicted_Probability"], data["Observed_Subscription_Rate"], color=color, marker=marker, label=model)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed subscription rate", xlim=(0, 0.8), ylim=(0, 0.8))
    ax.set_title("Holdout Probability Calibration", loc="left", pad=30)
    ax.text(0, 1.02, "Ten equal-frequency bins; strict planning information set", transform=ax.transAxes, color=GREY, fontsize=9)
    ax.grid(color="#E8EBEF"); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(chart_dir / "calibration_curve.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    comparison = information[["Information_Set", "Holdout_PR_AUC"]].copy()
    comparison.loc[len(comparison)] = ["Invalid: operational + duration", leakage.loc[leakage["Includes_Duration"], "PR_AUC"].iloc[0]]
    colors = [BLUE_DARK if value else BLUE_LIGHT for value in information["Primary"]] + [RED]
    fig, ax = plt.subplots(figsize=(10.5, 5.5)); ax.barh(comparison["Information_Set"], comparison["Holdout_PR_AUC"], color=colors)
    ax.set(xlabel="Holdout PR-AUC", xlim=(0, min(1, comparison["Holdout_PR_AUC"].max() + 0.12)))
    ax.set_title("Information Timing Changes Apparent Performance", loc="left", pad=30)
    ax.text(0, 1.02, "Red specification includes post-contact duration", transform=ax.transAxes, color=GREY, fontsize=9)
    ax.grid(axis="x", color="#E8EBEF"); ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(comparison["Holdout_PR_AUC"]): ax.text(value + 0.008, index, f"{value:.3f}", va="center")
    fig.tight_layout(); fig.savefig(chart_dir / "information_set_and_leakage.png", dpi=180, bbox_inches="tight"); plt.close(fig)

    top = importance.nlargest(15, "Importance_Mean").sort_values("Importance_Mean")
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(top["Feature"], top["Importance_Mean"], xerr=top["Importance_SD"], color=BLUE)
    ax.axvline(0, color=INK, linewidth=0.9); ax.set(xlabel="Mean PR-AUC decrease after permutation")
    ax.set_title("Predictive Associations in the Final Model", loc="left", pad=30)
    ax.text(0, 1.02, "Fixed-holdout permutation importance; not causal effects", transform=ax.transAxes, color=GREY, fontsize=9)
    ax.grid(axis="x", color="#E8EBEF"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(chart_dir / "permutation_importance.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def render_paper_results(summary: dict[str, object], cv: pd.DataFrame, budget: pd.DataFrame, intervals: pd.DataFrame, pairwise: pd.DataFrame, calibration: pd.DataFrame, leakage: pd.DataFrame, information: pd.DataFrame, duplicates: pd.DataFrame, importance: pd.DataFrame) -> str:
    ci = intervals.set_index("Metric"); top20 = budget.loc[np.isclose(budget["contact_share"], 0.20)].iloc[0]
    valid, leaky = leakage.iloc[0], leakage.iloc[1]
    training_cal = calibration.loc[(calibration["Evaluation_Split"] == "Training nested OOF") & calibration["Selected"]].iloc[0]
    holdout_cal = calibration.loc[(calibration["Evaluation_Split"] == "Fixed holdout") & calibration["Selected"]].iloc[0]
    lines = ["# Paper-Ready Results", "", "All values below are generated from saved machine-readable outputs. This is manuscript input, not a completed paper.", "", "## Sample and estimand", "", f"The UCI extract contains {summary['raw_rows']:,} contact records. The primary analysis removes {summary['duplicate_rows']:,} exact duplicate rows, leaving {summary['analysis_rows']:,} observations and {summary['positive_outcomes']:,} observed subscriptions ({summary['subscription_rate']:.1%}). The estimand is out-of-sample response propensity and ranking performance, not the causal effect of a marketing contact.", "", "## Decision-time information set", "", f"The conservative primary specification uses {len(summary['primary_features'])} planning-time predictors: {', '.join(f'`{x}`' for x in summary['primary_features'])}. Raw `campaign`, scheduled-contact fields, contemporaneous macroeconomic fields, and post-contact `duration` are excluded. `campaign_prior = campaign - 1` is used only in operational sensitivity analysis. This implements decision C: compare multiple information sets while retaining the strict planning set as primary.", "", "## Model development and validation", "", f"A stratified 80/20 split yields {summary['train_rows']:,} training and {summary['holdout_rows']:,} holdout observations. Five-fold training-only PR-AUC selected {summary['selected_base_model']}; nested training-only Brier score selected {summary['calibration_method_label']} calibration. The holdout was used in earlier repository analyses and is a development holdout, not a pristine prospective test.", ""]
    for row in cv.itertuples(): lines.append(f"- {row.Model}: PR-AUC {row.PR_AUC_Mean:.3f} (fold SD {row.PR_AUC_Std:.3f}); ROC-AUC {row.ROC_AUC_Mean:.3f} (fold SD {row.ROC_AUC_Std:.3f}).")
    lines += ["", "## Fixed-holdout results", "", f"The final {summary['final_model']} achieved PR-AUC {summary['pr_auc']:.3f} (95% bootstrap CI {ci.loc['PR_AUC', 'CI95_Lower']:.3f}-{ci.loc['PR_AUC', 'CI95_Upper']:.3f}), ROC-AUC {summary['roc_auc']:.3f} (95% CI {ci.loc['ROC_AUC', 'CI95_Lower']:.3f}-{ci.loc['ROC_AUC', 'CI95_Upper']:.3f}), Brier score {summary['brier_score']:.3f} (95% CI {ci.loc['Brier_Score', 'CI95_Lower']:.3f}-{ci.loc['Brier_Score', 'CI95_Upper']:.3f}), and log loss {summary['log_loss']:.3f}.", "", "### Paired model uncertainty", ""]
    for row in pairwise.loc[pairwise["Metric"].isin(["PR_AUC", "Top20_Capture"])].itertuples(): lines.append(f"- Final vs {row.Comparator}, {row.Metric}: delta {row.Estimate:+.3f} (95% CI {row.CI95_Lower:+.3f} to {row.CI95_Upper:+.3f}; positive favors final).")
    lines += ["", "## Probability calibration", "", f"The selected training-only specification had Brier score {training_cal['Brier_Score']:.3f} and log loss {training_cal['Log_Loss']:.3f}. On the holdout, Brier score was {holdout_cal['Brier_Score']:.3f} and log loss was {holdout_cal['Log_Loss']:.3f}.", "", "## Resource-constrained targeting", ""]
    for share in (0.10, 0.20, 0.30, 0.40):
        row = budget.loc[np.isclose(budget["contact_share"], share)].iloc[0]
        lines.append(f"- Top {share:.0%}: {int(row['contact_records']):,} contacts; {int(row['subscribers_captured']):,} observed subscribers; {row['conversion_rate']:.1%} conversion; {row['responder_capture']:.1%} capture; {row['lift']:.2f}x lift; {row['contacts_per_observed_subscriber']:.2f} contacts per observed subscriber.")
    lines += ["", f"At 20%, capture was {top20['responder_capture']:.1%} (95% CI {ci.loc['Top20_Capture', 'CI95_Lower']:.1%}-{ci.loc['Top20_Capture', 'CI95_Upper']:.1%}) and lift was {top20['lift']:.2f}x (95% CI {ci.loc['Top20_Lift', 'CI95_Lower']:.2f}-{ci.loc['Top20_Lift', 'CI95_Upper']:.2f}). These are observational ranking quantities, not incremental causal effects.", "", "## Information timing and leakage", ""]
    for row in information.itertuples(): lines.append(f"- {row.Information_Set}: holdout PR-AUC {row.Holdout_PR_AUC:.3f}, ROC-AUC {row.Holdout_ROC_AUC:.3f}, top-20% capture {row.Top20_Capture:.1%}.")
    lines += ["", f"Adding post-contact `duration` increased comparable holdout ROC-AUC from {valid['ROC_AUC']:.3f} to {leaky['ROC_AUC']:.3f}, PR-AUC from {valid['PR_AUC']:.3f} to {leaky['PR_AUC']:.3f}, and top-20% capture from {valid['Top20_Capture']:.1%} to {leaky['Top20_Capture']:.1%}. The leaky model is not deployable.", "", "## Duplicate sensitivity", "", f"Retaining all 12 exact duplicate rows changed holdout PR-AUC by {duplicates.iloc[1]['PR_AUC'] - duplicates.iloc[0]['PR_AUC']:+.3f}. Without customer IDs, accidental duplicates cannot be distinguished from distinct identical contacts.", "", "## Predictive associations", ""]
    for row in importance.nlargest(5, "Importance_Mean").itertuples(): lines.append(f"- `{row.Feature}`: mean PR-AUC decrease {row.Importance_Mean:.4f} (SD {row.Importance_SD:.4f}).")
    lines += ["", "These are predictive associations, not causal effects.", "", "## Material limitations", "", "- No stable customer identifier is available, so repeated customers cannot be grouped during splitting.", "- Month and weekday do not support a defensible row-level chronological split across May 2008-November 2010.", "- The historical sample is from one Portuguese bank; validity for modern campaigns, other countries, digital channels, or other populations is not established.", "- Observed subscription is not incremental response caused by calling; these are propensity models, not treatment-effect models.", "- Contact cost and customer lifetime value are unavailable, so monetary ROI is not estimated.", "- The repository's holdout was analyzed previously and is not a pristine prospective external validation.", "- Future research should use customer IDs, true timestamps, prospective temporal validation, and randomized policy evaluation with observed costs and value.", ""]
    return "\n".join(lines)


def render_upgrade_audit(summary: dict[str, object]) -> str:
    return f"""# Publication Upgrade Audit

## Original project

The prior pipeline removed exact duplicates, excluded `duration`, compared a dummy,
class-weighted logistic regression, and random forest by five-fold training PR-AUC,
sigmoid-calibrated the selected forest, and reported fixed-holdout results.

## Concerns identified

- The old `safe_features` mixed planning-time data with current-contact fields, raw
  `campaign`, and macro indicators. UCI defines `campaign` as including the focal contact.
- No boosting benchmark, paired model uncertainty, explicit calibration selection,
  information-set comparison, or duplicate-retention sensitivity was present.
- The holdout had already been reported and cannot be called pristine external validation.

## Changes made

1. Audited all 20 source predictors plus `campaign_prior` using official UCI definitions.
2. Made a strict planning set primary: {', '.join(summary['primary_features'])}.
3. Added planning-plus-macro, operational, and operational-plus-macro sensitivities.
4. Replaced raw `campaign` with nonnegative `campaign_prior = campaign - 1` only in
   operational sensitivity specifications.
5. Added histogram gradient boosting, nested training-only calibration selection,
   1,000 paired bootstrap comparisons, expanded targeting benchmarks, duplicate
   sensitivity, permutation importance, and environment recording.
6. Generated five publication-focused figures and made `analysis.py` the only source of
   analytical calculations used by the notebook and paper-ready results.

## Findings and remaining limitations

Verified values are generated in `PAPER_RESULTS.md`. No code can recover customer IDs,
complete timestamps, randomized treatment, external validation, contact costs, or
customer value. Exact duplicates therefore remain an explicit sensitivity analysis,
and the fixed holdout is labeled a previously analyzed development holdout.
"""


def run_analysis(project_dir: Path | None = None) -> dict[str, object]:
    project_dir = project_dir or resolve_project_dir()
    data_path, output_dir = project_dir / "data" / "bank-additional-full.csv", project_dir / "outputs"
    chart_dir = output_dir / "charts"; output_dir.mkdir(exist_ok=True); chart_dir.mkdir(exist_ok=True)
    data_hash = file_sha256(data_path)
    if data_hash != EXPECTED_DATA_SHA256: raise ValueError(f"Unexpected source data hash: {data_hash}")
    raw = pd.read_csv(data_path, sep=";")
    expected = {"age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"}
    if set(raw.columns) != expected: raise ValueError("Source columns do not match the UCI extract.")
    duplicate_rows = int(raw.duplicated().sum()); campaign = raw.drop_duplicates().copy()
    campaign["subscribed"] = (campaign["y"] == "yes").astype(int); campaign["campaign_prior"] = campaign["campaign"] - 1
    if (campaign["campaign_prior"] < 0).any(): raise ValueError("campaign_prior contains a negative value.")
    audit = feature_availability_audit(); audited = set(audit["Feature"]) - {"campaign_prior"}
    if audited != expected - {"y"}: raise AssertionError("Feature audit is incomplete.")
    objects = campaign.select_dtypes(include=["object", "string"])
    quality = pd.DataFrame([
        {"Check": "Source hash", "Value": data_hash, "Status": "Pass", "Analytical_Risk": "Wrong source version"},
        {"Check": "Raw rows", "Value": len(raw), "Status": "Pass", "Analytical_Risk": "Unexpected extract size"},
        {"Check": "Exact duplicate rows", "Value": duplicate_rows, "Status": "Sensitivity analyzed", "Analytical_Risk": "Unknown record identity"},
        {"Check": "Analysis rows", "Value": len(campaign), "Status": "Pass", "Analytical_Risk": "Row reconciliation"},
        {"Check": "Missing cells", "Value": int(campaign.isna().sum().sum()), "Status": "Pass", "Analytical_Risk": "Silent missingness"},
        {"Check": "Literal unknown cells", "Value": int((objects == "unknown").sum().sum()), "Status": "Documented category", "Analytical_Risk": "Unknown not missing at random"},
        {"Check": "Rows containing unknown", "Value": int((objects == "unknown").any(axis=1).sum()), "Status": "Documented category", "Analytical_Risk": "Coverage"},
        {"Check": "Positive outcomes", "Value": int(campaign["subscribed"].sum()), "Status": "Pass", "Analytical_Risk": "Class balance"},
        {"Check": "campaign_prior minimum", "Value": int(campaign["campaign_prior"].min()), "Status": "Pass", "Analytical_Risk": "Invalid transformation"},
        {"Check": "campaign_prior maximum", "Value": int(campaign["campaign_prior"].max()), "Status": "Documented", "Analytical_Risk": "Contact-frequency tail"},
    ])

    target, features = campaign["subscribed"], campaign[PRIMARY_FEATURES]
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE); estimators = candidate_estimators()
    folds, selection = cross_validate_candidates(train_x, train_y, estimators, cv)
    selected_name = str(selection.iloc[0]["Model"]); selected_estimator = estimators[selected_name]
    calibration_method, calibration_metrics = select_calibration(train_x, train_y, selected_estimator)
    label = {"none": "no", "sigmoid": "sigmoid", "isotonic": "isotonic"}[calibration_method]
    actual = test_y.to_numpy(); candidate_predictions, model_rows = {}, []
    for name, estimator in estimators.items():
        model = make_pipeline(train_x, estimator); model.fit(train_x, train_y); probabilities = model.predict_proba(test_x)[:, 1]
        candidate_predictions[name] = probabilities; metrics = probability_metrics(actual, probabilities); ranking = evaluate_ranking(actual, probabilities, 0.20)
        model_rows.append({"Model": name, "Role": "Locked benchmark", "Selected_Base_Model": name == selected_name, **metrics, "Top20_Capture": ranking["responder_capture"], "Top20_Lift": ranking["lift"]})
    final_model = make_model(train_x, selected_estimator, calibration_method); final_model.fit(train_x, train_y); final_probabilities = final_model.predict_proba(test_x)[:, 1]
    final_label = f"{selected_name} ({label} calibration)"; final_metrics = probability_metrics(actual, final_probabilities); final_ranking = evaluate_ranking(actual, final_probabilities, 0.20)
    model_rows.append({"Model": final_label, "Role": "Final selected model", "Selected_Base_Model": True, **final_metrics, "Top20_Capture": final_ranking["responder_capture"], "Top20_Lift": final_ranking["lift"]})
    model_metrics = pd.DataFrame(model_rows); raw_selected = candidate_predictions[selected_name]
    calibration_metrics = pd.concat([calibration_metrics, pd.DataFrame([
        {"Evaluation_Split": "Fixed holdout", "Method": "Selected raw model", **probability_metrics(actual, raw_selected), "Selected": False},
        {"Evaluation_Split": "Fixed holdout", "Method": f"Final ({label})", **final_metrics, "Selected": True},
    ])], ignore_index=True)
    curve_parts = []
    for name, probabilities in [("Selected raw model", raw_selected), ("Final selected model", final_probabilities)]:
        observed, predicted = calibration_curve(actual, probabilities, n_bins=10, strategy="quantile")
        curve_parts.append(pd.DataFrame({"Model": name, "Bin": np.arange(1, len(observed) + 1), "Mean_Predicted_Probability": predicted, "Observed_Subscription_Rate": observed}))
    calibration_table = pd.concat(curve_parts, ignore_index=True)

    budget = pd.DataFrame([evaluate_ranking(actual, final_probabilities, share) for share in np.arange(0.10, 1.01, 0.10)])
    deciles, efficiency = build_deciles(actual, final_probabilities), capture_efficiency(actual, final_probabilities)
    comparators = {name: scores for name, scores in candidate_predictions.items() if name != "Dummy Baseline"}
    if calibration_method == "none": comparators.pop(selected_name, None)
    else: comparators[f"{selected_name} raw"] = comparators.pop(selected_name)
    intervals, pairwise = bootstrap_results(actual, final_probabilities, comparators)
    primary_cv = selection.loc[selection["Model"] == selected_name].iloc[0]
    information = evaluate_information_sets(campaign, train_x.index, test_x.index, train_y, test_y, selected_estimator, calibration_method, primary_cv, final_probabilities)

    operational_columns = INFORMATION_SETS["Operational plus macro context"]
    operational_train, operational_test = campaign.loc[train_x.index, operational_columns], campaign.loc[test_x.index, operational_columns]
    operational_model = make_model(operational_train, selected_estimator, calibration_method); operational_model.fit(operational_train, train_y)
    operational_probabilities = operational_model.predict_proba(operational_test)[:, 1]
    leaky_columns = operational_columns + ["duration"]
    leaky_train, leaky_test = campaign.loc[train_x.index, leaky_columns], campaign.loc[test_x.index, leaky_columns]
    leaky_model = make_model(leaky_train, selected_estimator, calibration_method); leaky_model.fit(leaky_train, train_y)
    leaky_probabilities = leaky_model.predict_proba(leaky_test)[:, 1]
    leakage_rows = []
    for name, includes, probabilities in [("Operational plus macro context", False, operational_probabilities), ("Invalid: operational plus macro plus duration", True, leaky_probabilities)]:
        metrics, ranking = probability_metrics(actual, probabilities), evaluate_ranking(actual, probabilities, 0.20)
        leakage_rows.append({"Feature_Set": name, "Deployable": not includes, "Includes_Duration": includes, **metrics, "Top20_Capture": ranking["responder_capture"], "Top20_Lift": ranking["lift"]})
    leakage = pd.DataFrame(leakage_rows)
    duplicates = duplicate_sensitivity(raw, final_metrics, final_ranking, selected_estimator, calibration_method)

    logistic = make_pipeline(train_x, estimators["Logistic Regression"]); logistic.fit(train_x, train_y)
    names, coefficients = logistic.named_steps["preprocess"].get_feature_names_out(), logistic.named_steps["model"].coef_[0]
    coefficient_table = pd.DataFrame({"Feature": pd.Series(names).str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False), "Coefficient": coefficients, "Absolute_Coefficient": np.abs(coefficients)}).sort_values("Absolute_Coefficient", ascending=False)
    permutation = permutation_importance(final_model, test_x, test_y, scoring="average_precision", n_repeats=10, random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"Feature": test_x.columns, "Importance_Mean": permutation.importances_mean, "Importance_SD": permutation.importances_std, "Scoring": "PR-AUC decrease", "Interpretation": "Predictive association, not causal effect"}).sort_values("Importance_Mean", ascending=False)

    checks = {
        "source_hash_matches": data_hash == EXPECTED_DATA_SHA256,
        "all_source_predictors_audited": audited == expected - {"y"},
        "primary_feature_rules_match_audit": set(audit.loc[audit["Primary_Model_Allowed"], "Feature"]) == set(PRIMARY_FEATURES),
        "duration_excluded_from_primary": "duration" not in PRIMARY_FEATURES,
        "raw_campaign_excluded_from_models": all("campaign" not in values for values in INFORMATION_SETS.values()),
        "campaign_prior_nonnegative": bool((campaign["campaign_prior"] >= 0).all()),
        "holdout_isolation": set(train_x.index).isdisjoint(test_x.index),
        "train_holdout_rows_reconcile": len(train_x) + len(test_x) == len(campaign),
        "cv_fold_integrity": bool((folds["Index_Overlap"] == 0).all()),
        "probabilities_in_range": bool(((final_probabilities >= 0) & (final_probabilities <= 1)).all()),
        "budget_contacts_monotonic": bool(budget["contact_records"].is_monotonic_increasing),
        "budget_capture_monotonic": bool(budget["responder_capture"].is_monotonic_increasing),
        "budget_full_sample_reconciles": int(budget.iloc[-1]["contact_records"]) == len(test_x),
        "budget_full_subscribers_reconcile": int(budget.iloc[-1]["subscribers_captured"]) == int(actual.sum()),
        "random_baseline_reconciles": bool(np.allclose(budget["random_responder_capture"], budget["selected_record_share"])),
        "bootstrap_intervals_contain_estimates": bool(((intervals["CI95_Lower"] <= intervals["Estimate"]) & (intervals["Estimate"] <= intervals["CI95_Upper"])).all()),
        "bootstrap_count": bool((intervals["Bootstrap_Samples"] == BOOTSTRAP_REPEATS).all()),
        "leaky_model_not_final": bool(leakage.loc[leakage["Includes_Duration"], "Deployable"].eq(False).all()),
        "duration_inflates_pr_auc": bool(leakage.iloc[1]["PR_AUC"] > leakage.iloc[0]["PR_AUC"]),
    }
    if not all(checks.values()): raise AssertionError(checks)
    primary_cv = selection.loc[selection["Model"] == selected_name].iloc[0]; interval_lookup = intervals.set_index("Metric")
    summary = {
        "data_sha256": data_hash, "raw_rows": int(len(raw)), "duplicate_rows": duplicate_rows, "analysis_rows": int(len(campaign)),
        "positive_outcomes": int(campaign["subscribed"].sum()), "subscription_rate": float(campaign["subscribed"].mean()),
        "rows_with_unknown": int((objects == "unknown").any(axis=1).sum()), "information_set_decision": "C: compare multiple sets; strict planning is primary",
        "primary_information_set": "Strict planning (primary)", "primary_features": PRIMARY_FEATURES, "train_rows": int(len(train_x)), "holdout_rows": int(len(test_x)),
        "holdout_status": "Previously analyzed development holdout; not pristine prospective validation", "candidate_models": list(estimators),
        "selected_base_model": selected_name, "selection_rule": "Highest mean five-fold training-set PR-AUC; ROC-AUC breaks an exact tie",
        "cv_pr_auc_mean": float(primary_cv["PR_AUC_Mean"]), "cv_pr_auc_std": float(primary_cv["PR_AUC_Std"]), "cv_roc_auc_mean": float(primary_cv["ROC_AUC_Mean"]), "cv_roc_auc_std": float(primary_cv["ROC_AUC_Std"]),
        "calibration_selection_rule": "Lowest nested training-only OOF Brier score", "calibration_method": calibration_method, "calibration_method_label": label, "final_model": final_label,
        "roc_auc": final_metrics["ROC_AUC"], "roc_auc_ci95": [float(interval_lookup.loc["ROC_AUC", "CI95_Lower"]), float(interval_lookup.loc["ROC_AUC", "CI95_Upper"])],
        "pr_auc": final_metrics["PR_AUC"], "pr_auc_ci95": [float(interval_lookup.loc["PR_AUC", "CI95_Lower"]), float(interval_lookup.loc["PR_AUC", "CI95_Upper"])],
        "brier_score": final_metrics["Brier_Score"], "brier_score_ci95": [float(interval_lookup.loc["Brier_Score", "CI95_Lower"]), float(interval_lookup.loc["Brier_Score", "CI95_Upper"])],
        "log_loss": final_metrics["Log_Loss"], "top_20_conversion_rate": final_ranking["conversion_rate"], "top_20_lift": final_ranking["lift"],
        "top_20_lift_ci95": [float(interval_lookup.loc["Top20_Lift", "CI95_Lower"]), float(interval_lookup.loc["Top20_Lift", "CI95_Upper"])], "top_20_capture": final_ranking["responder_capture"],
        "top_20_capture_ci95": [float(interval_lookup.loc["Top20_Capture", "CI95_Lower"]), float(interval_lookup.loc["Top20_Capture", "CI95_Upper"])],
        "leaky_model_pr_auc": float(leakage.iloc[1]["PR_AUC"]), "leaky_model_roc_auc": float(leakage.iloc[1]["ROC_AUC"]), "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "environment": {"python": platform.python_version(), "pandas": version("pandas"), "numpy": version("numpy"), "scikit-learn": version("scikit-learn"), "matplotlib": version("matplotlib")}, "validation_checks": checks,
    }
    tables = {"data_quality.csv": quality, "feature_availability_audit.csv": audit, "cross_validation_folds.csv": folds, "model_selection_summary.csv": selection, "model_metrics.csv": model_metrics, "model_pairwise_bootstrap.csv": pairwise, "budget_metrics.csv": budget, "targeting_deciles.csv": deciles, "capture_efficiency.csv": efficiency, "bootstrap_intervals.csv": intervals, "calibration_metrics.csv": calibration_metrics, "calibration_curve.csv": calibration_table, "information_set_comparison.csv": information, "duplicate_sensitivity.csv": duplicates, "leakage_audit.csv": leakage, "logistic_coefficients.csv": coefficient_table, "permutation_importance.csv": importance}
    for filename, table in tables.items(): table.to_csv(output_dir / filename, index=False)
    save_charts(chart_dir, selection, deciles, calibration_table, information, leakage, importance, final_label)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (project_dir / "PAPER_RESULTS.md").write_text(render_paper_results(summary, selection, budget, intervals, pairwise, calibration_metrics, leakage, information, duplicates, importance), encoding="utf-8")
    (project_dir / "PUBLICATION_UPGRADE_AUDIT.md").write_text(render_upgrade_audit(summary), encoding="utf-8")
    return {"summary": summary, "data_quality": quality, "feature_availability_audit": audit, "cv_summary": selection, "holdout_metrics": model_metrics, "bootstrap_intervals": intervals, "model_pairwise_bootstrap": pairwise, "budget_metrics": budget, "information_set_comparison": information, "duplicate_sensitivity": duplicates, "leakage_audit": leakage, "permutation_importance": importance}


if __name__ == "__main__":
    print(json.dumps(run_analysis()["summary"], indent=2))
