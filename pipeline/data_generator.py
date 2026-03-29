"""
GPU/System Performance Data Generator
=====================================
Generates realistic synthetic GPU/CPU system metrics with configurable
anomaly injection for performance monitoring simulation.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


def generate_system_metrics(days=7, anomaly_rate=0.10, seed=42):
    """
    Generate synthetic GPU/CPU system metrics data.

    Parameters
    ----------
    days : int
        Number of days to simulate (7 → ~10K rows, 30 → ~43K rows).
    anomaly_rate : float
        Fraction of data points that will contain anomalies (0.05–0.25).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, gpu_usage, cpu_usage, memory_usage,
        temperature, power_consumption, workload_type
    """
    np.random.seed(seed)

    total_minutes = days * 24 * 60
    timestamps = [
        datetime(2026, 3, 1, 0, 0) + timedelta(minutes=i)
        for i in range(total_minutes)
    ]

    # ── Workload schedule ─────────────────────────────────────────────────
    # Realistic workload pattern: training during work hours, inference at
    # night, idle in gaps.
    workload_types = []
    for ts in timestamps:
        hour = ts.hour
        if 9 <= hour < 12 or 14 <= hour < 18:
            # Peak work hours → mostly training
            workload_types.append(
                np.random.choice(["training", "inference", "idle"],
                                 p=[0.65, 0.25, 0.10])
            )
        elif 18 <= hour < 22:
            # Evening → inference-heavy
            workload_types.append(
                np.random.choice(["training", "inference", "idle"],
                                 p=[0.15, 0.65, 0.20])
            )
        elif 22 <= hour or hour < 6:
            # Night → batch inference + idle
            workload_types.append(
                np.random.choice(["training", "inference", "idle"],
                                 p=[0.10, 0.40, 0.50])
            )
        else:
            # Early morning 6–9 → ramp-up
            workload_types.append(
                np.random.choice(["training", "inference", "idle"],
                                 p=[0.30, 0.30, 0.40])
            )

    workload_arr = np.array(workload_types)

    # ── Base metrics (workload-dependent) ────────────────────────────────
    gpu_usage = np.zeros(total_minutes)
    cpu_usage = np.zeros(total_minutes)
    memory_usage = np.zeros(total_minutes)

    for i, wl in enumerate(workload_arr):
        if wl == "training":
            gpu_usage[i] = np.random.normal(72, 8)
            cpu_usage[i] = np.random.normal(55, 10)
            memory_usage[i] = np.random.normal(65, 8)
        elif wl == "inference":
            gpu_usage[i] = np.random.normal(55, 10)
            cpu_usage[i] = np.random.normal(40, 8)
            memory_usage[i] = np.random.normal(55, 7)
        else:  # idle
            gpu_usage[i] = np.random.normal(15, 5)
            cpu_usage[i] = np.random.normal(12, 4)
            memory_usage[i] = np.random.normal(30, 5)

    # ── Temperature (correlated with GPU, with thermal lag) ──────────────
    temperature = np.zeros(total_minutes)
    temperature[0] = 55.0
    for i in range(1, total_minutes):
        # Temperature follows GPU usage with inertia (thermal lag)
        target_temp = 45 + gpu_usage[i] * 0.35 + np.random.normal(0, 1.5)
        # Exponential smoothing (thermal inertia ~3 min lag)
        alpha = 0.3
        temperature[i] = alpha * target_temp + (1 - alpha) * temperature[i - 1]

    # ── Power consumption (correlated with GPU usage) ────────────────────
    power_consumption = (
        120  # base power draw
        + gpu_usage * 2.2  # GPU contribution
        + cpu_usage * 0.5  # CPU contribution
        + np.random.normal(0, 8, total_minutes)  # noise
    )

    # ── Inject anomalies ─────────────────────────────────────────────────
    num_anomalies = int(total_minutes * anomaly_rate)

    # Strategy 1: GPU spikes (40% of anomalies)
    n_gpu_spikes = int(num_anomalies * 0.35)
    gpu_spike_starts = np.random.choice(
        range(100, total_minutes - 20), n_gpu_spikes // 5, replace=False
    )
    for start in gpu_spike_starts:
        duration = np.random.randint(5, 16)  # 5–15 minute bursts
        end = min(start + duration, total_minutes)
        gpu_usage[start:end] = np.random.uniform(93, 100, end - start)
        # CPU may or may not spike (CPU underutilization pattern)
        if np.random.random() < 0.3:
            cpu_usage[start:end] = np.random.uniform(15, 30, end - start)

    # Strategy 2: Memory leaks (25% of anomalies)
    n_mem_leaks = max(1, int(num_anomalies * 0.15))
    mem_leak_starts = np.random.choice(
        range(200, total_minutes - 120), n_mem_leaks, replace=False
    )
    for start in mem_leak_starts:
        leak_duration = np.random.randint(30, 90)  # 30–90 min gradual climb
        end = min(start + leak_duration, total_minutes)
        leak_curve = np.linspace(
            memory_usage[start], np.random.uniform(88, 98), end - start
        )
        memory_usage[start:end] = leak_curve
        # Add a "reset" after the leak (simulating restart)
        reset_end = min(end + 5, total_minutes)
        memory_usage[end:reset_end] = np.random.uniform(30, 40, reset_end - end)

    # Strategy 3: Overheating events (20% of anomalies)
    n_overheat = max(1, int(num_anomalies * 0.10))
    overheat_starts = np.random.choice(
        range(300, total_minutes - 30), n_overheat, replace=False
    )
    for start in overheat_starts:
        duration = np.random.randint(8, 25)
        end = min(start + duration, total_minutes)
        temperature[start:end] = np.random.uniform(86, 96, end - start)
        # Overheating causes power surge
        power_consumption[start:end] += np.random.uniform(30, 80, end - start)

    # Strategy 4: Multi-metric failure (rare ~5% of anomalies)
    n_multi = max(1, int(num_anomalies * 0.03))
    multi_starts = np.random.choice(
        range(500, total_minutes - 20), n_multi, replace=False
    )
    for start in multi_starts:
        duration = np.random.randint(5, 12)
        end = min(start + duration, total_minutes)
        gpu_usage[start:end] = np.random.uniform(95, 100, end - start)
        memory_usage[start:end] = np.random.uniform(90, 99, end - start)
        temperature[start:end] = np.random.uniform(88, 96, end - start)
        power_consumption[start:end] = np.random.uniform(350, 420, end - start)
        cpu_usage[start:end] = np.random.uniform(85, 98, end - start)

    # ── Clip to valid ranges ─────────────────────────────────────────────
    gpu_usage = np.clip(gpu_usage, 0, 100).round(1)
    cpu_usage = np.clip(cpu_usage, 0, 100).round(1)
    memory_usage = np.clip(memory_usage, 0, 100).round(1)
    temperature = np.clip(temperature, 20, 105).round(1)
    power_consumption = np.clip(power_consumption, 50, 500).round(1)

    # ── Build DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "timestamp": timestamps,
        "gpu_usage": gpu_usage,
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "temperature": temperature,
        "power_consumption": power_consumption,
        "workload_type": workload_arr,
    })

    return df


def save_raw_data(df, days, output_dir=None):
    """Save generated data to data/raw/ as CSV."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"system_metrics_{days}d.csv")
    df.to_csv(path, index=False)
    print(f"✅ Generated {len(df):,} rows → {path}")
    return path


if __name__ == "__main__":
    df = generate_system_metrics(days=7, anomaly_rate=0.10)
    save_raw_data(df, 7)
    print(df.describe())
    print(f"\nWorkload distribution:\n{df['workload_type'].value_counts()}")
