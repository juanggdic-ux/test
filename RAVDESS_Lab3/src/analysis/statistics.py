"""Statistical summary and paired tests for windowed acoustic measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import shapiro, wilcoxon
from statsmodels.stats.multitest import multipletests

STATISTICS = ("Mean", "Std", "Median", "Min", "Max", "Range")


def summarize_reference(features: dict[str, np.ndarray], actors: list[str]) -> pd.DataFrame:
    """Original notebook-style stacked DataFrame + unstack/swaplevel summary."""
    intermediate = pd.concat(
        [pd.DataFrame(values, index=actors) for values in features.values()],
        keys=features.keys(), names=["Characteristic", "Actor"]
    )
    stats = pd.DataFrame({
        "Mean": intermediate.mean(axis=1),
        "Std": intermediate.std(axis=1),
        "Median": intermediate.median(axis=1),
        "Min": intermediate.min(axis=1),
        "Max": intermediate.max(axis=1),
        "Range": intermediate.max(axis=1) - intermediate.min(axis=1),
    })
    stats = stats.unstack("Characteristic").swaplevel(0, 1, axis=1)
    stats = stats.reindex(columns=list(features), level=0)
    stats.columns.names = ("Characteristic", "Statistic")
    stats.index.name = "Actor"
    return stats


def summarize_optimized(features: dict[str, np.ndarray], actors: list[str]) -> pd.DataFrame:
    """Construct actor-row/feature-statistic-column MultiIndex without one giant frame table.

    Pandas ignores undefined (unvoiced) pYIN F0 values by default.
    """
    output = {}
    for characteristic, values in features.items():
        small = pd.DataFrame(values, index=actors)
        minimum = small.min(axis=1)
        maximum = small.max(axis=1)
        output[(characteristic, "Mean")] = small.mean(axis=1)
        output[(characteristic, "Std")] = small.std(axis=1)
        output[(characteristic, "Median")] = small.median(axis=1)
        output[(characteristic, "Min")] = minimum
        output[(characteristic, "Max")] = maximum
        output[(characteristic, "Range")] = maximum - minimum
    summary = pd.DataFrame(output, index=actors).astype("float64")
    summary.columns.names = ("Characteristic", "Statistic")
    summary.index.name = "Actor"
    return summary


def paired_tests(neutral: pd.DataFrame, sad: pd.DataFrame) -> pd.DataFrame:
    """Paired Shapiro–Wilk and Wilcoxon tests with Holm multiplicity correction.

    Same actor must occur in both conditions. Test statistics are descriptive
    and not used for selecting classifier features before cross-validation.
    """
    if not neutral.index.equals(sad.index) or not neutral.columns.equals(sad.columns):
        raise ValueError("Paired tables require identical actor order and feature columns")
    differences = sad - neutral
    rows = []
    for feature, statistic in differences.columns:
        delta = differences[(feature, statistic)].dropna()
        if len(delta) >= 3 and delta.nunique() > 1:
            shapiro_p = float(shapiro(delta).pvalue)
        else:
            shapiro_p = np.nan
        if delta.empty:
            wilcoxon_p = np.nan
        elif (delta == 0).all():
            wilcoxon_p = 1.0
        else:
            wilcoxon_p = float(wilcoxon(delta, alternative="two-sided").pvalue)
        rows.append((feature, statistic, len(delta), shapiro_p, wilcoxon_p))
    result = pd.DataFrame(rows, columns=["Feature", "Statistic", "N pairs", "Shapiro p", "Wilcoxon p"])
    result = result.set_index(["Feature", "Statistic"])
    valid = result["Wilcoxon p"].notna()
    result["Holm-adjusted p"] = np.nan
    if valid.any():
        result.loc[valid, "Holm-adjusted p"] = multipletests(result.loc[valid, "Wilcoxon p"], method="holm")[1]
    return result
