"""
Correlation Engine
==================
Cross-metric correlation analysis using Pearson and Spearman methods
with per-workload breakdowns.
"""

import pandas as pd
import numpy as np
from scipy import stats


CORRELATION_PAIRS = [
    {"x": "gpu_usage", "y": "power_consumption",
     "label": "GPU Usage × Power Consumption",
     "expected": "Strong positive — GPU workloads draw more power"},
    {"x": "temperature", "y": "gpu_usage",
     "label": "Temperature × GPU Usage",
     "expected": "Moderate positive — heat builds with compute load"},
    {"x": "memory_usage", "y": "gpu_usage",
     "label": "Memory Usage × GPU Usage",
     "expected": "Moderate positive — training uses both GPU and memory"},
    {"x": "cpu_usage", "y": "gpu_usage",
     "label": "CPU Usage × GPU Usage",
     "expected": "Weak to moderate — depends on pipeline balance"},
    {"x": "temperature", "y": "power_efficiency",
     "label": "Temperature × Power Efficiency",
     "expected": "Negative — efficiency drops as temperature rises"},
    {"x": "gpu_usage", "y": "thermal_headroom",
     "label": "GPU Usage × Thermal Headroom",
     "expected": "Strong negative — higher load reduces safety margin"},
]


def compute_correlation_pair(df, x_col, y_col):
    """Compute Pearson and Spearman correlation for a column pair."""
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 10:
        return {"pearson_r": 0, "pearson_p": 1, "spearman_r": 0, "spearman_p": 1}
    pr, pp = stats.pearsonr(valid[x_col], valid[y_col])
    sr, sp = stats.spearmanr(valid[x_col], valid[y_col])
    return {"pearson_r": round(pr, 3), "pearson_p": round(pp, 5),
            "spearman_r": round(sr, 3), "spearman_p": round(sp, 5)}


def interpret_strength(r):
    """Interpret correlation coefficient strength."""
    a = abs(r)
    if a >= 0.8: return "Strong"
    elif a >= 0.5: return "Moderate"
    elif a >= 0.3: return "Weak"
    return "Very weak / None"


def run_correlation_analysis(df):
    """Run full correlation analysis on all defined pairs with per-workload breakdowns."""
    results = []
    for pair in CORRELATION_PAIRS:
        x_col, y_col = pair["x"], pair["y"]
        if x_col not in df.columns or y_col not in df.columns:
            continue
        overall = compute_correlation_pair(df, x_col, y_col)
        by_workload = {}
        if "workload_type" in df.columns:
            for wl in df["workload_type"].unique():
                sub = df[df["workload_type"] == wl]
                if len(sub) > 20:
                    wc = compute_correlation_pair(sub, x_col, y_col)
                    by_workload[wl] = {"pearson_r": wc["pearson_r"], "spearman_r": wc["spearman_r"]}
        strength = interpret_strength(overall["pearson_r"])
        direction = "positive" if overall["pearson_r"] > 0 else "negative"
        interpretation = (
            f"{strength} {direction} correlation "
            f"(Pearson r={overall['pearson_r']}, Spearman ρ={overall['spearman_r']}). "
            f"{pair['expected']}."
        )
        results.append({
            "pair": pair["label"], "x_col": x_col, "y_col": y_col,
            **overall, "strength": strength, "direction": direction,
            "interpretation": interpretation, "by_workload": by_workload,
        })
    return results


def get_correlation_matrix(df):
    """Compute full correlation matrix for numeric metric columns."""
    cols = [c for c in ["gpu_usage", "cpu_usage", "memory_usage",
            "temperature", "power_consumption", "power_efficiency",
            "thermal_headroom"] if c in df.columns]
    return df[cols].corr().round(3)
