"""
Bottleneck Detector
===================
Rule-based detection engine that identifies GPU, CPU, memory, thermal,
and power bottlenecks with severity levels and actionable recommendations.
"""

import pandas as pd
import numpy as np


# ── Detection rules ───────────────────────────────────────────────────────────
RULES = [
    {
        "id": "gpu_bottleneck",
        "condition": lambda row: row["gpu_usage"] > 90,
        "issue": "GPU Bottleneck",
        "severity": "Critical",
        "severity_icon": "🔴",
        "reason_template": "GPU usage at {gpu_usage}%",
        "recommendation": (
            "Reduce batch size or enable mixed precision (FP16) training. "
            "Consider gradient accumulation to reduce per-step memory."
        ),
    },
    {
        "id": "cpu_bottleneck",
        "condition": lambda row: row["cpu_usage"] > 85,
        "issue": "CPU Bottleneck",
        "severity": "Critical",
        "severity_icon": "🔴",
        "reason_template": "CPU usage at {cpu_usage}%",
        "recommendation": (
            "Optimize data loading pipeline — increase DataLoader workers, "
            "enable pin_memory, or use pre-processed datasets."
        ),
    },
    {
        "id": "memory_pressure",
        "condition": lambda row: 85 < row["memory_usage"] <= 95,
        "issue": "Memory Pressure",
        "severity": "Warning",
        "severity_icon": "🟡",
        "reason_template": "Memory usage at {memory_usage}%",
        "recommendation": (
            "Monitor for memory leaks. Consider reducing batch size or "
            "implementing periodic cache clearing."
        ),
    },
    {
        "id": "memory_critical",
        "condition": lambda row: row["memory_usage"] > 95,
        "issue": "Memory Critical",
        "severity": "Critical",
        "severity_icon": "🔴",
        "reason_template": "Memory usage at {memory_usage}% — CRITICAL",
        "recommendation": (
            "Immediate action required: restart process or reduce workload. "
            "Set memory watchdog with auto-restart at 90% threshold."
        ),
    },
    {
        "id": "overheating",
        "condition": lambda row: 85 < row["temperature"] <= 90,
        "issue": "Overheating",
        "severity": "Warning",
        "severity_icon": "🟡",
        "reason_template": "Temperature at {temperature}°C",
        "recommendation": (
            "Improve rack cooling and check airflow. Ensure intake vents "
            "are unobstructed."
        ),
    },
    {
        "id": "thermal_throttle",
        "condition": lambda row: row["temperature"] > 90,
        "issue": "Thermal Throttle",
        "severity": "Critical",
        "severity_icon": "🔴",
        "reason_template": "Temperature at {temperature}°C — THROTTLING RISK",
        "recommendation": (
            "Reduce workload immediately. Consider liquid cooling for "
            "sustained training. Set thermal throttle limit at 85°C."
        ),
    },
    {
        "id": "cpu_underutilization",
        "condition": lambda row: row["gpu_usage"] > 90 and row["cpu_usage"] < 30,
        "issue": "CPU Underutilization",
        "severity": "Medium",
        "severity_icon": "🟠",
        "reason_template": "GPU at {gpu_usage}% but CPU only at {cpu_usage}%",
        "recommendation": (
            "Data pipeline is likely the bottleneck. Increase DataLoader "
            "workers from 4→8, enable prefetching, add data augmentation on GPU."
        ),
    },
    {
        "id": "power_limit",
        "condition": lambda row: row["power_consumption"] > 320,
        "issue": "Power Limit",
        "severity": "Warning",
        "severity_icon": "🟡",
        "reason_template": "Power consumption at {power_consumption}W",
        "recommendation": (
            "Check PSU capacity. Enable GPU power capping via nvidia-smi "
            "(typical performance impact <5%, reduces thermal stress ~12%)."
        ),
    },
]


def detect_bottlenecks(df):
    """
    Scan all rows for bottleneck conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered metrics DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame of detected bottleneck events with columns:
        timestamp, issue, reason, severity, severity_icon, recommendation
    """
    events = []

    for _, row in df.iterrows():
        for rule in RULES:
            try:
                if rule["condition"](row):
                    reason = rule["reason_template"].format(**row.to_dict())
                    events.append({
                        "timestamp": row["timestamp"],
                        "issue": rule["issue"],
                        "reason": reason,
                        "severity": rule["severity"],
                        "severity_icon": rule["severity_icon"],
                        "recommendation": rule["recommendation"],
                    })
            except (KeyError, TypeError):
                continue

    bottleneck_df = pd.DataFrame(events)
    if len(bottleneck_df) > 0:
        bottleneck_df = bottleneck_df.sort_values("timestamp").reset_index(drop=True)

    return bottleneck_df


def get_bottleneck_summary(bottleneck_df):
    """
    Generate summary statistics from detected bottlenecks.

    Parameters
    ----------
    bottleneck_df : pd.DataFrame
        Output from detect_bottlenecks().

    Returns
    -------
    dict
        Summary with counts by type, severity, and most affected windows.
    """
    if bottleneck_df.empty:
        return {
            "total": 0,
            "by_type": {},
            "by_severity": {},
            "most_affected_hours": [],
            "by_workload": {},
        }

    by_type = bottleneck_df["issue"].value_counts().to_dict()
    by_severity = bottleneck_df["severity"].value_counts().to_dict()

    # Most affected hours
    if "timestamp" in bottleneck_df.columns:
        bottleneck_df["_hour"] = pd.to_datetime(bottleneck_df["timestamp"]).dt.hour
        hour_counts = bottleneck_df["_hour"].value_counts().head(5).to_dict()
        bottleneck_df.drop(columns=["_hour"], inplace=True)
    else:
        hour_counts = {}

    return {
        "total": len(bottleneck_df),
        "by_type": by_type,
        "by_severity": by_severity,
        "most_affected_hours": hour_counts,
    }


def save_bottlenecks_to_sqlite(bottleneck_df, db_path):
    """Save bottleneck events to SQLite."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    bottleneck_df.to_sql("bottlenecks", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ Bottlenecks saved to SQLite ({len(bottleneck_df)} events)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from pipeline.data_generator import generate_system_metrics
    from pipeline.data_cleaner import run_etl_pipeline
    from pipeline.feature_engineering import engineer_features

    df = generate_system_metrics(days=7)
    df_clean, _, _ = run_etl_pipeline(df)
    df_feat = engineer_features(df_clean)
    bottlenecks = detect_bottlenecks(df_feat)
    summary = get_bottleneck_summary(bottlenecks)

    print(f"\n🚨 Total bottlenecks: {summary['total']}")
    print(f"By type: {summary['by_type']}")
    print(f"By severity: {summary['by_severity']}")
