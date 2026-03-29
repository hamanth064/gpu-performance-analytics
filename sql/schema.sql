-- GPU/System Performance Analytics — Database Schema
-- ===================================================

CREATE TABLE IF NOT EXISTS system_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT NOT NULL,
    gpu_usage               REAL,
    cpu_usage               REAL,
    memory_usage            REAL,
    temperature             REAL,
    power_consumption       REAL,
    workload_type           TEXT,
    gpu_usage_normalized    REAL,
    cpu_usage_normalized    REAL,
    memory_usage_normalized REAL,
    temperature_normalized  REAL,
    power_consumption_normalized REAL,
    gpu_cpu_ratio           REAL,
    rolling_avg_gpu_5       REAL,
    rolling_avg_gpu_15      REAL,
    rolling_avg_temp_10     REAL,
    rolling_std_gpu_5       REAL,
    power_efficiency        REAL,
    thermal_headroom        REAL,
    memory_pressure         REAL,
    hour_of_day             INTEGER,
    day_of_week             TEXT,
    is_peak_hour            INTEGER
);

CREATE TABLE IF NOT EXISTS bottlenecks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    issue           TEXT NOT NULL,
    reason          TEXT,
    severity        TEXT,
    severity_icon   TEXT,
    recommendation  TEXT
);

CREATE TABLE IF NOT EXISTS anomalies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL,
    z_score         REAL,
    zscore_flag     INTEGER,
    iqr_flag        INTEGER,
    rolling_flag    INTEGER,
    methods_flagged INTEGER,
    confidence      TEXT,
    is_anomaly      INTEGER
);
