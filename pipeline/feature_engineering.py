"""
Feature Engineering
===================
Derives advanced features from cleaned GPU/CPU metrics for downstream
analysis: rolling averages, utilization ratios, thermal headroom, etc.
"""

import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Add derived features to the cleaned metrics DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned metrics with columns: timestamp, gpu_usage, cpu_usage,
        memory_usage, temperature, power_consumption, workload_type.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered feature columns.
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Utilization ratios ───────────────────────────────────────────────
    df["gpu_cpu_ratio"] = (
        df["gpu_usage"] / df["cpu_usage"].clip(lower=1)
    ).round(3)

    # ── Rolling averages (smoothed trends) ───────────────────────────────
    df["rolling_avg_gpu_5"] = (
        df["gpu_usage"].rolling(window=5, min_periods=1).mean().round(2)
    )
    df["rolling_avg_gpu_15"] = (
        df["gpu_usage"].rolling(window=15, min_periods=1).mean().round(2)
    )
    df["rolling_avg_temp_10"] = (
        df["temperature"].rolling(window=10, min_periods=1).mean().round(2)
    )

    # ── Rolling volatility ───────────────────────────────────────────────
    df["rolling_std_gpu_5"] = (
        df["gpu_usage"].rolling(window=5, min_periods=1).std().fillna(0).round(2)
    )

    # ── Power efficiency (GPU usage per watt) ────────────────────────────
    df["power_efficiency"] = (
        df["gpu_usage"] / df["power_consumption"].clip(lower=1)
    ).round(4)

    # ── Thermal headroom (distance to critical 90°C) ─────────────────────
    df["thermal_headroom"] = (90.0 - df["temperature"]).round(1)

    # ── Memory pressure (rate of change) ─────────────────────────────────
    df["memory_pressure"] = df["memory_usage"].diff().fillna(0).round(2)

    # ── Time-based features ──────────────────────────────────────────────
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_peak_hour"] = df["hour_of_day"].apply(
        lambda h: 1 if 9 <= h <= 18 else 0
    )

    return df


if __name__ == "__main__":
    from data_generator import generate_system_metrics
    from data_cleaner import run_etl_pipeline

    df_raw = generate_system_metrics(days=7)
    df_clean, _, _ = run_etl_pipeline(df_raw)
    df_feat = engineer_features(df_clean)

    print(f"✅ Feature engineering complete! {len(df_feat.columns)} columns:")
    print(df_feat.columns.tolist())
    print(f"\nSample:\n{df_feat.head()}")
