"""
Anomaly Detection Engine
========================
Multi-method anomaly detection using Z-Score, IQR, and Rolling Deviation
with ensemble confidence scoring.
"""

import pandas as pd
import numpy as np


METRIC_COLUMNS = [
    "gpu_usage", "cpu_usage", "memory_usage",
    "temperature", "power_consumption",
]


def detect_zscore_anomalies(series, threshold=3.0):
    """
    Detect anomalies using the Z-Score method.

    |z| = |(x - μ) / σ|
    If |z| > threshold → anomaly.

    Parameters
    ----------
    series : pd.Series
        Numeric values to analyze.
    threshold : float
        Z-score threshold (default: 3.0).

    Returns
    -------
    tuple
        (z_scores: pd.Series, flags: pd.Series of bool)
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index), pd.Series(False, index=series.index)

    z_scores = ((series - mean) / std).abs()
    flags = z_scores > threshold
    return z_scores.round(3), flags


def detect_iqr_anomalies(series, multiplier=1.5):
    """
    Detect anomalies using the IQR (Interquartile Range) method.

    Anomaly if: x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR

    Parameters
    ----------
    series : pd.Series
        Numeric values to analyze.
    multiplier : float
        IQR multiplier (default: 1.5).

    Returns
    -------
    tuple
        (lower_fence, upper_fence, flags: pd.Series of bool)
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    flags = (series < lower) | (series > upper)
    return lower, upper, flags


def detect_rolling_anomalies(series, window=15, num_std=2.0):
    """
    Detect anomalies using Rolling Mean ± Rolling Std deviation.

    Anomaly if: x > rolling_mean ± num_std × rolling_std

    Parameters
    ----------
    series : pd.Series
        Numeric values to analyze.
    window : int
        Rolling window size in minutes.
    num_std : float
        Number of standard deviations for threshold.

    Returns
    -------
    pd.Series
        Boolean flags indicating anomalies.
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0)

    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std

    flags = (series > upper) | (series < lower)
    return flags


def run_anomaly_detection(df):
    """
    Run all three anomaly detection methods on each metric column
    and compute ensemble confidence scores.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered metrics DataFrame.

    Returns
    -------
    pd.DataFrame
        Anomaly report with columns: timestamp, metric, value,
        z_score, zscore_flag, iqr_flag, rolling_flag, methods_flagged,
        confidence, is_anomaly
    """
    anomaly_records = []

    for col in METRIC_COLUMNS:
        if col not in df.columns:
            continue

        series = df[col]

        # Method 1: Z-Score
        z_scores, z_flags = detect_zscore_anomalies(series)

        # Method 2: IQR
        _, _, iqr_flags = detect_iqr_anomalies(series)

        # Method 3: Rolling Deviation
        rolling_flags = detect_rolling_anomalies(series)

        # Ensemble: count how many methods flagged each point
        for i in range(len(df)):
            methods = int(z_flags.iloc[i]) + int(iqr_flags.iloc[i]) + int(rolling_flags.iloc[i])
            if methods >= 1:
                if methods >= 3:
                    confidence = "high"
                elif methods >= 2:
                    confidence = "medium"
                else:
                    confidence = "low"

                anomaly_records.append({
                    "timestamp": df["timestamp"].iloc[i],
                    "metric": col,
                    "value": round(series.iloc[i], 2),
                    "z_score": z_scores.iloc[i],
                    "zscore_flag": bool(z_flags.iloc[i]),
                    "iqr_flag": bool(iqr_flags.iloc[i]),
                    "rolling_flag": bool(rolling_flags.iloc[i]),
                    "methods_flagged": methods,
                    "confidence": confidence,
                    "is_anomaly": methods >= 2,
                })

    anomaly_df = pd.DataFrame(anomaly_records)
    if len(anomaly_df) > 0:
        anomaly_df = anomaly_df.sort_values("timestamp").reset_index(drop=True)

    return anomaly_df


def get_anomaly_summary(anomaly_df):
    """
    Generate summary statistics from detected anomalies.

    Returns
    -------
    dict
        Summary with counts by metric, confidence level, and hourly distribution.
    """
    if anomaly_df.empty:
        return {
            "total": 0,
            "confirmed": 0,
            "by_metric": {},
            "by_confidence": {},
            "hourly_distribution": {},
        }

    confirmed = anomaly_df[anomaly_df["is_anomaly"]]

    by_metric = confirmed["metric"].value_counts().to_dict()
    by_confidence = anomaly_df["confidence"].value_counts().to_dict()

    # Hourly distribution of confirmed anomalies
    if len(confirmed) > 0:
        confirmed_copy = confirmed.copy()
        confirmed_copy["_hour"] = pd.to_datetime(confirmed_copy["timestamp"]).dt.hour
        hourly = confirmed_copy["_hour"].value_counts().sort_index().to_dict()
    else:
        hourly = {}

    return {
        "total": len(anomaly_df),
        "confirmed": len(confirmed),
        "by_metric": by_metric,
        "by_confidence": by_confidence,
        "hourly_distribution": hourly,
    }


def save_anomalies_to_sqlite(anomaly_df, db_path):
    """Save anomaly events to SQLite."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    # Only save confirmed anomalies to keep the table lean
    confirmed = anomaly_df[anomaly_df["is_anomaly"]] if not anomaly_df.empty else anomaly_df
    confirmed.to_sql("anomalies", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Anomalies saved to SQLite ({len(confirmed)} confirmed events)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from pipeline.data_generator import generate_system_metrics
    from pipeline.data_cleaner import run_etl_pipeline
    from pipeline.feature_engineering import engineer_features

    df = generate_system_metrics(days=7)
    df_clean, _, _ = run_etl_pipeline(df)
    df_feat = engineer_features(df_clean)
    anomalies = run_anomaly_detection(df_feat)
    summary = get_anomaly_summary(anomalies)

    print(f"\n🔬 Total flagged: {summary['total']}")
    print(f"   Confirmed anomalies: {summary['confirmed']}")
    print(f"   By metric: {summary['by_metric']}")
    print(f"   By confidence: {summary['by_confidence']}")
