-- GPU/System Performance Analytics — Pre-Built Analytical Queries
-- ================================================================

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 1. Average GPU/CPU/Memory usage per workload type               │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Workload Analysis
SELECT
    workload_type,
    ROUND(AVG(gpu_usage), 1)   AS avg_gpu,
    ROUND(AVG(cpu_usage), 1)   AS avg_cpu,
    ROUND(AVG(memory_usage), 1) AS avg_memory,
    ROUND(AVG(temperature), 1)  AS avg_temp,
    ROUND(AVG(power_consumption), 0) AS avg_power,
    COUNT(*) AS sample_count
FROM system_metrics
GROUP BY workload_type
ORDER BY avg_gpu DESC;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 2. Top 5 high-temperature periods (15-min windows)              │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Thermal
SELECT
    SUBSTR(timestamp, 1, 15) || '0' AS time_window,
    ROUND(MAX(temperature), 1)      AS max_temp,
    ROUND(AVG(temperature), 1)      AS avg_temp,
    ROUND(AVG(gpu_usage), 1)        AS avg_gpu_during,
    COUNT(*) AS readings
FROM system_metrics
GROUP BY time_window
ORDER BY max_temp DESC
LIMIT 5;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 3. Hourly memory usage trend                                    │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Memory
SELECT
    hour_of_day,
    ROUND(AVG(memory_usage), 1) AS avg_memory,
    ROUND(MAX(memory_usage), 1) AS max_memory,
    ROUND(MIN(memory_usage), 1) AS min_memory,
    ROUND(AVG(memory_pressure), 3) AS avg_pressure
FROM system_metrics
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 4. Bottleneck frequency by type and severity                    │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Bottleneck
SELECT
    issue,
    severity,
    COUNT(*) AS event_count,
    MIN(timestamp) AS first_occurrence,
    MAX(timestamp) AS last_occurrence
FROM bottlenecks
GROUP BY issue, severity
ORDER BY event_count DESC;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 5. Peak vs off-peak performance comparison                      │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Time Analysis
SELECT
    CASE WHEN is_peak_hour = 1 THEN 'Peak (9AM-6PM)' ELSE 'Off-Peak' END AS period,
    ROUND(AVG(gpu_usage), 1) AS avg_gpu,
    ROUND(AVG(cpu_usage), 1) AS avg_cpu,
    ROUND(AVG(temperature), 1) AS avg_temp,
    ROUND(AVG(power_consumption), 0) AS avg_power,
    ROUND(AVG(power_efficiency), 4) AS avg_efficiency,
    COUNT(*) AS total_readings
FROM system_metrics
GROUP BY is_peak_hour;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 6. Power consumption statistics by workload type                │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Power
SELECT
    workload_type,
    ROUND(AVG(power_consumption), 0) AS avg_power,
    ROUND(MAX(power_consumption), 0) AS max_power,
    ROUND(MIN(power_consumption), 0) AS min_power,
    ROUND(AVG(power_efficiency), 4)  AS avg_efficiency,
    COUNT(*) AS samples
FROM system_metrics
GROUP BY workload_type
ORDER BY avg_power DESC;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 7. Anomaly count per metric per hour                            │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Anomaly
SELECT
    metric,
    CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) AS hour,
    COUNT(*) AS anomaly_count,
    ROUND(AVG(z_score), 2) AS avg_zscore
FROM anomalies
WHERE is_anomaly = 1
GROUP BY metric, hour
ORDER BY anomaly_count DESC;

-- ┌──────────────────────────────────────────────────────────────────┐
-- │ 8. Correlated events — bottlenecks + anomalies in same window   │
-- └──────────────────────────────────────────────────────────────────┘
-- Category: Cross-Analysis
SELECT
    b.timestamp,
    b.issue AS bottleneck_type,
    b.severity,
    a.metric AS anomaly_metric,
    a.z_score,
    a.confidence
FROM bottlenecks b
JOIN anomalies a
  ON SUBSTR(b.timestamp, 1, 16) = SUBSTR(a.timestamp, 1, 16)
WHERE a.is_anomaly = 1
ORDER BY b.timestamp
LIMIT 50;
