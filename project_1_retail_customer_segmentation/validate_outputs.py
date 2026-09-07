"""Validate the committed outputs for the retail segmentation project."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main() -> None:
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
    segments = pd.read_csv(OUTPUT_DIR / "segment_summary.csv")
    diagnostics = pd.read_csv(OUTPUT_DIR / "cluster_diagnostics.csv")
    stability = pd.read_csv(OUTPUT_DIR / "cluster_stability.csv")

    assert summary["raw_rows"] == 541_909
    assert summary["customers"] == 4_338
    assert summary["orders"] == 18_532
    assert np.isclose(summary["revenue_gbp"], 8_911_407.90, atol=0.01)
    assert segments["Customers"].sum() == summary["customers"]
    assert np.isclose(segments["CustomerShare"].sum(), 1.0)
    assert np.isclose(segments["RevenueShare"].sum(), 1.0)
    assert diagnostics["k"].tolist() == list(range(2, 9))
    assert np.isclose(
        diagnostics.loc[diagnostics["k"] == 4, "silhouette_score"].iloc[0],
        summary["silhouette_score"],
    )
    assert len(stability) == 25
    assert stability["adjusted_rand_index"].between(-1, 1).all()
    assert np.isclose(stability["adjusted_rand_index"].mean(), summary["mean_stability_ari"])
    assert all(summary["validation_checks"].values())

    print("All retail output checks passed.")


if __name__ == "__main__":
    main()
