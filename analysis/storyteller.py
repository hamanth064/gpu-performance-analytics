"""
Storytelling Layer
==================
Auto-generates human-readable Key Findings narratives and production
mitigation strategies from all analysis results.
"""

import pandas as pd
import numpy as np


def generate_key_findings(df, bottleneck_df, anomaly_df, correlation_results):
    """
    Generate structured key findings from analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered metrics.
    bottleneck_df : pd.DataFrame
        Detected bottlenecks from bottleneck_detector.
    anomaly_df : pd.DataFrame
        Detected anomalies from anomaly_detection.
    correlation_results : list[dict]
        Results from correlation_engine.

    Returns
    -------
    list[dict]
        Ordered list of findings with severity, title, detail, and category.
    """
    findings = []

    # ── Bottleneck findings ──────────────────────────────────────────────
    if not bottleneck_df.empty:
        # GPU bottleneck finding
        gpu_bn = bottleneck_df[bottleneck_df["issue"] == "GPU Bottleneck"]
        if len(gpu_bn) > 0:
            # Find which workload is most affected
            bn_with_ts = gpu_bn.copy()
            bn_with_ts["_hour"] = pd.to_datetime(bn_with_ts["timestamp"]).dt.hour
            peak_hours = bn_with_ts["_hour"].value_counts().head(3)
            peak_range = f"{peak_hours.index.min()}:00–{peak_hours.index.max() + 1}:00"

            findings.append({
                "severity": "critical",
                "icon": "🔴",
                "title": "Frequent GPU bottlenecks during training workloads",
                "detail": (
                    f"GPU usage exceeded 90% in {len(gpu_bn)} events. "
                    f"Most concentrated between {peak_range} across monitored days. "
                    f"This indicates the system is GPU-bound during peak training hours."
                ),
                "category": "bottleneck",
            })

        # Memory findings
        mem_bn = bottleneck_df[bottleneck_df["issue"].isin(["Memory Pressure", "Memory Critical"])]
        if len(mem_bn) > 0:
            critical_count = len(bottleneck_df[bottleneck_df["issue"] == "Memory Critical"])
            findings.append({
                "severity": "warning" if critical_count == 0 else "critical",
                "icon": "🟡" if critical_count == 0 else "🔴",
                "title": "Memory pressure detected — potential leak patterns",
                "detail": (
                    f"Memory usage exceeded safe thresholds in {len(mem_bn)} events "
                    f"({critical_count} critical). Gradual memory climb patterns suggest "
                    f"possible memory leaks during long-running inference workloads."
                ),
                "category": "bottleneck",
            })

        # Thermal findings
        thermal_bn = bottleneck_df[bottleneck_df["issue"].isin(["Overheating", "Thermal Throttle"])]
        if len(thermal_bn) > 0:
            avg_temp = df["temperature"].mean()
            max_temp = df["temperature"].max()
            throttle_count = len(bottleneck_df[bottleneck_df["issue"] == "Thermal Throttle"])
            pct_safe = round((1 - len(thermal_bn) / len(df)) * 100, 1)

            if throttle_count > 0:
                findings.append({
                    "severity": "critical",
                    "icon": "🔴",
                    "title": f"Thermal throttling risk — {throttle_count} critical events",
                    "detail": (
                        f"Temperature exceeded 90°C in {throttle_count} events (max: {max_temp}°C). "
                        f"Thermal management adequate for {pct_safe}% of time, but sustained "
                        f"training sessions push temperatures beyond safe limits."
                    ),
                    "category": "thermal",
                })
            else:
                findings.append({
                    "severity": "positive",
                    "icon": "🟢",
                    "title": f"System thermal management adequate for {pct_safe}% of time",
                    "detail": (
                        f"Temperature exceeded 85°C in {len(thermal_bn)} events but stayed "
                        f"below throttle threshold. Average temperature: {avg_temp:.1f}°C. "
                        f"Average thermal headroom: {(90 - avg_temp):.1f}°C during normal operation."
                    ),
                    "category": "thermal",
                })

        # Power findings
        power_bn = bottleneck_df[bottleneck_df["issue"] == "Power Limit"]
        if len(power_bn) > 0:
            avg_power = df["power_consumption"].mean()
            findings.append({
                "severity": "warning",
                "icon": "🟡",
                "title": f"Power consumption spikes detected ({len(power_bn)} events)",
                "detail": (
                    f"Power draw exceeded 320W in {len(power_bn)} events. "
                    f"Average consumption: {avg_power:.0f}W. High power events correlate "
                    f"with GPU-intensive training workloads."
                ),
                "category": "power",
            })

    # ── Anomaly findings ─────────────────────────────────────────────────
    if not anomaly_df.empty:
        confirmed = anomaly_df[anomaly_df["is_anomaly"]]
        high_conf = anomaly_df[anomaly_df["confidence"] == "high"]

        if len(confirmed) > 0:
            most_anomalous = confirmed["metric"].value_counts().index[0]
            findings.append({
                "severity": "warning",
                "icon": "🟡",
                "title": f"Anomalous behavior detected — {most_anomalous} most affected",
                "detail": (
                    f"{len(confirmed)} confirmed anomalies detected across all metrics "
                    f"({len(high_conf)} high-confidence). {most_anomalous} shows the most "
                    f"anomalous patterns, warranting closer investigation."
                ),
                "category": "anomaly",
            })

    # ── Correlation findings ─────────────────────────────────────────────
    if correlation_results:
        for cr in correlation_results:
            if cr["strength"] == "Strong":
                findings.append({
                    "severity": "insight",
                    "icon": "🔵",
                    "title": f"{cr['strength']} {cr['direction']} correlation: {cr['pair']}",
                    "detail": cr["interpretation"],
                    "category": "correlation",
                })

    # Sort: critical first, then warning, then insight, then positive
    severity_order = {"critical": 0, "warning": 1, "insight": 2, "positive": 3}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 99))

    return findings


def generate_mitigation_strategies(df, bottleneck_df, anomaly_df):
    """
    Generate "What would you do in real life?" production mitigation strategies.

    Returns
    -------
    list[dict]
        Each with title, strategy, impact, and implementation details.
    """
    strategies = []

    if not bottleneck_df.empty:
        # Strategy 1: Dynamic Workload Scheduling
        gpu_bn = bottleneck_df[bottleneck_df["issue"] == "GPU Bottleneck"]
        if len(gpu_bn) > 0:
            gpu_bn_hours = pd.to_datetime(gpu_bn["timestamp"]).dt.hour
            peak_hour = gpu_bn_hours.mode().iloc[0] if len(gpu_bn_hours) > 0 else 14
            off_peak_avg = df[df["hour_of_day"].isin([22, 23, 0, 1, 2, 3, 4, 5])]["gpu_usage"].mean()

            strategies.append({
                "title": "Dynamic Workload Scheduling",
                "icon": "📅",
                "strategy": (
                    f"Defer non-critical training jobs to off-peak hours (10 PM–6 AM) "
                    f"where GPU availability averages {off_peak_avg:.0f}% vs peak usage. "
                    f"Most bottlenecks cluster around {peak_hour}:00."
                ),
                "impact": "Estimated 30–40% reduction in GPU bottleneck events",
                "implementation": [
                    "Set up job scheduler (SLURM / Kubernetes CronJobs)",
                    "Prioritize inference during peak, training during off-peak",
                    "Add GPU reservation system for critical jobs",
                ],
            })

        # Strategy 2: Cooling Optimization
        thermal = bottleneck_df[bottleneck_df["issue"].isin(["Overheating", "Thermal Throttle"])]
        if len(thermal) > 0:
            strategies.append({
                "title": "Cooling System Optimization",
                "icon": "❄️",
                "strategy": (
                    f"{len(thermal)} overheating events detected with thermal lag after GPU spikes. "
                    f"Sustained training pushes temperatures beyond safe operating range."
                ),
                "impact": "Prevents thermal throttling, maintains consistent performance",
                "implementation": [
                    "Improve rack airflow — ensure 3U spacing between GPU nodes",
                    "Consider liquid cooling for sustained training workloads",
                    "Set thermal throttle at 85°C to prevent degradation",
                    "Add temperature-based workload migration between nodes",
                ],
            })

        # Strategy 3: Memory Management
        mem_bn = bottleneck_df[bottleneck_df["issue"].isin(["Memory Pressure", "Memory Critical"])]
        if len(mem_bn) > 0:
            strategies.append({
                "title": "Memory Leak Mitigation",
                "icon": "🧠",
                "strategy": (
                    f"Memory leak patterns during inference suggest unbounded tensor caching. "
                    f"{len(mem_bn)} memory pressure events detected."
                ),
                "impact": "Prevents OOM crashes, improves system stability",
                "implementation": [
                    "Implement periodic cache clearing every 30 minutes",
                    "Set per-process memory caps (ulimit / cgroups)",
                    "Add memory watchdog with auto-restart at 90% threshold",
                    "Profile memory with torch.cuda.memory_stats()",
                ],
            })

        # Strategy 4: Power Capping
        power_bn = bottleneck_df[bottleneck_df["issue"] == "Power Limit"]
        if len(power_bn) > 0:
            strategies.append({
                "title": "Power Capping Strategy",
                "icon": "⚡",
                "strategy": (
                    f"Training workloads push power to 320W+ ({len(power_bn)} events). "
                    f"Power capping reduces thermal stress with minimal performance impact."
                ),
                "impact": "Performance impact <5%, reduces power costs ~12%",
                "implementation": [
                    "Enable GPU power capping: nvidia-smi -pl 300",
                    "Monitor perf/watt ratio after capping",
                    "Consider time-of-use electricity pricing for scheduling",
                ],
            })

    # Strategy 5: Pipeline Efficiency (always relevant)
    if "cpu_usage" in df.columns and "gpu_usage" in df.columns:
        training_data = df[df["workload_type"] == "training"]
        if len(training_data) > 0:
            avg_cpu_train = training_data["cpu_usage"].mean()
            avg_gpu_train = training_data["gpu_usage"].mean()
            if avg_cpu_train < avg_gpu_train * 0.6:
                strategies.append({
                    "title": "Data Pipeline Efficiency",
                    "icon": "🔧",
                    "strategy": (
                        f"CPU underutilization during training (avg {avg_cpu_train:.0f}%) "
                        f"vs GPU (avg {avg_gpu_train:.0f}%) suggests data loading bottleneck."
                    ),
                    "impact": "Can improve GPU utilization by 15–25%",
                    "implementation": [
                        "Increase DataLoader workers from 4→8",
                        "Enable pin_memory=True for faster CPU→GPU transfer",
                        "Implement data prefetching pipeline",
                        "Pre-process datasets to binary format (TFRecord / WebDataset)",
                    ],
                })

    return strategies
