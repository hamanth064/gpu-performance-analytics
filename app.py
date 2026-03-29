"""
GPU/System Performance Analytics Platform
==========================================
Premium 6-tab Streamlit dashboard with NVIDIA-inspired dark theme.
Tabs: System Overview | Performance Trends | Bottleneck Detection |
      Anomaly Analysis | Insights & Story | SQL Explorer & Reports
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from pipeline.data_generator import generate_system_metrics, save_raw_data
from pipeline.data_cleaner import run_etl_pipeline
from pipeline.feature_engineering import engineer_features
from analysis.bottleneck_detector import detect_bottlenecks, get_bottleneck_summary, save_bottlenecks_to_sqlite
from analysis.anomaly_detection import run_anomaly_detection, get_anomaly_summary, save_anomalies_to_sqlite
from analysis.correlation_engine import run_correlation_analysis, get_correlation_matrix
from analysis.storyteller import generate_key_findings, generate_mitigation_strategies
from visualization.dashboard import (
    chart_gpu_cpu_over_time, chart_temperature_timeline, chart_memory_usage,
    chart_power_by_workload, chart_bottleneck_timeline, chart_anomaly_heatmap,
    chart_workload_distribution, chart_health_gauge, chart_correlation_scatter,
    chart_correlation_heatmap, chart_bottleneck_frequency,
)
from reports.report_generator import generate_performance_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPU Performance Analytics",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
:root {
  --bg:#0E1117; --surface:#1a1d27; --surface2:#22263a;
  --gpu:#22c55e; --cpu:#22d3ee; --mem:#a78bfa;
  --temp:#f59e0b; --power:#3b82f6;
  --critical:#ef4444; --warn:#f59e0b; --medium:#f97316;
  --text:#e2e8f0; --muted:#64748b;
  --border:rgba(255,255,255,0.07); --radius:14px;
}
html,body,[class*="css"]{
  font-family:'DM Sans',sans-serif!important;
  background-color:var(--bg)!important;
  color:var(--text)!important;
}
section[data-testid="stSidebar"]{
  background:var(--surface)!important;
  border-right:1px solid var(--border)!important;
}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{
  gap:4px;background:var(--surface);
  border-radius:12px;padding:6px;
  border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px!important;padding:8px 16px!important;
  font-weight:600!important;font-size:13px!important;
  color:var(--muted)!important;background:transparent!important;border:none!important;
}
.stTabs [aria-selected="true"]{background:var(--gpu)!important;color:#000!important;}
.stButton>button{
  width:100%;border-radius:10px!important;height:3em;
  background:linear-gradient(135deg,#16a34a,#22c55e)!important;
  color:white!important;font-weight:700!important;font-size:14px!important;
  border:none!important;transition:opacity 0.2s,transform 0.15s;
}
.stButton>button:hover{opacity:.9;transform:translateY(-1px);}
.stDownloadButton>button{
  width:100%;border-radius:10px!important;height:3em;
  background:linear-gradient(135deg,#1d4ed8,#3b82f6)!important;
  color:white!important;font-weight:700!important;border:none!important;
}
[data-testid="stMetric"]{
  background:var(--surface)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;padding:20px!important;
}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:11px!important;text-transform:uppercase;letter-spacing:1px;}
[data-testid="stMetricValue"]{color:var(--text)!important;font-weight:700!important;font-size:1.8rem!important;}
.stDataFrame{border-radius:var(--radius)!important;overflow:hidden;border:1px solid var(--border)!important;}
.stSuccess{background:rgba(34,197,94,.12)!important;border:1px solid rgba(34,197,94,.3)!important;border-radius:10px!important;}
.stError{background:rgba(239,68,68,.10)!important;border:1px solid rgba(239,68,68,.3)!important;border-radius:10px!important;}
.stWarning{background:rgba(245,158,11,.10)!important;border:1px solid rgba(245,158,11,.3)!important;border-radius:10px!important;}
hr{border-color:var(--border)!important;}
.stTextArea textarea,.stTextInput input{
  background:var(--surface2)!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--text)!important;
}
.stSelectbox>div>div{
  background:var(--surface2)!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--text)!important;
}
/* Hero */
.hero-title{
  font-size:2.8rem;font-weight:800;
  background:linear-gradient(135deg,#fff 10%,#22c55e 60%,#22d3ee 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  line-height:1.15;margin-bottom:6px;
}
.hero-sub{color:var(--muted);font-size:1rem;margin-bottom:24px;}
/* Section labels */
.slabel{
  font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:2px;color:var(--cpu);margin-bottom:10px;
}
/* Pills */
.pill{display:inline-block;padding:3px 14px;border-radius:20px;font-size:12px;font-weight:600;margin-right:4px;margin-bottom:4px;}
.pill-green{background:rgba(34,197,94,.15);color:#4ade80;}
.pill-cyan{background:rgba(34,211,238,.12);color:#67e8f9;}
.pill-purple{background:rgba(167,139,250,.12);color:#c4b5fd;}
.pill-amber{background:rgba(245,158,11,.12);color:#fcd34d;}
.pill-red{background:rgba(239,68,68,.12);color:#fca5a5;}
/* Finding cards */
.finding{border-left:4px solid;border-radius:0 10px 10px 0;padding:16px 20px;margin:10px 0;background:var(--surface);}
.finding.critical{border-color:var(--critical);background:rgba(239,68,68,.04);}
.finding.warning{border-color:var(--warn);background:rgba(245,158,11,.04);}
.finding.insight{border-color:var(--cpu);background:rgba(34,211,238,.04);}
.finding.positive{border-color:var(--gpu);background:rgba(34,197,94,.04);}
.finding-title{font-weight:700;font-size:15px;margin-bottom:6px;}
/* Strategy cards */
.strategy{
  background:linear-gradient(135deg,rgba(34,197,94,.06),rgba(34,211,238,.04));
  border:1px solid rgba(34,197,94,.2);border-radius:12px;
  padding:20px 24px;margin:12px 0;
}
.strategy-title{font-weight:700;font-size:16px;margin-bottom:8px;}
.strategy-impact{color:var(--gpu);font-size:13px;font-weight:600;margin:10px 0;}
.impl-li{color:#94a3b8;font-size:14px;padding:3px 0;}
.impl-li::before{content:"→ ";color:var(--cpu);}
/* Alert banner */
.alert-banner{
  border-radius:12px;padding:16px 20px;margin:12px 0;
  display:flex;align-items:center;gap:12px;font-weight:600;
}
.alert-critical{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);color:#fca5a5;}
.alert-warning{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);color:#fcd34d;}
.alert-good{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.3);color:#4ade80;}
/* Corr card */
.corr-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:16px 20px;margin:8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init():
    defaults = {
        "df": None, "bottleneck_df": None, "anomaly_df": None,
        "bottleneck_summary": {}, "anomaly_summary": {},
        "correlation_results": [], "findings": [], "strategies": [],
        "pipeline_done": False, "db_path": None, "report_path": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_full_pipeline(days, anomaly_rate):
    """Run the complete data pipeline and all analysis modules."""
    db_path = os.path.join(APP_DIR, "data", "performance.db")

    with st.spinner("⚡ Generating system metrics..."):
        df_raw = generate_system_metrics(days=days, anomaly_rate=anomaly_rate)
        save_raw_data(df_raw, days, os.path.join(APP_DIR, "data", "raw"))

    with st.spinner("🔄 Running ETL pipeline..."):
        df_clean, _, _ = run_etl_pipeline(df_raw)

    with st.spinner("🔧 Engineering features..."):
        df = engineer_features(df_clean)

    with st.spinner("🚨 Detecting bottlenecks..."):
        bottleneck_df = detect_bottlenecks(df)
        bottleneck_summary = get_bottleneck_summary(bottleneck_df)
        if not bottleneck_df.empty:
            save_bottlenecks_to_sqlite(bottleneck_df, db_path)

    with st.spinner("🔬 Running anomaly detection..."):
        anomaly_df = run_anomaly_detection(df)
        anomaly_summary = get_anomaly_summary(anomaly_df)
        if not anomaly_df.empty:
            save_anomalies_to_sqlite(anomaly_df, db_path)

    with st.spinner("🔗 Analysing correlations..."):
        correlation_results = run_correlation_analysis(df)

    with st.spinner("📖 Generating key findings..."):
        findings = generate_key_findings(df, bottleneck_df, anomaly_df, correlation_results)
        strategies = generate_mitigation_strategies(df, bottleneck_df, anomaly_df)

    with st.spinner("📄 Generating HTML report..."):
        report_path = generate_performance_report(
            df, bottleneck_df, bottleneck_summary,
            anomaly_df, anomaly_summary,
            correlation_results, findings, strategies,
            os.path.join(APP_DIR, "reports", "performance_report.html"),
        )

    return df, bottleneck_df, bottleneck_summary, anomaly_df, anomaly_summary, \
           correlation_results, findings, strategies, db_path, report_path


def apply_filters(df, workloads, date_range=None):
    """Apply sidebar filters to the DataFrame."""
    filtered = df.copy()
    if workloads:
        filtered = filtered[filtered["workload_type"].isin(workloads)]
    return filtered


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Performance Analytics")
    st.markdown("<p style='color:#64748b;font-size:13px;margin-top:-8px;'>GPU/System Monitoring Platform</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='slabel'>🔧 Data Generation</div>", unsafe_allow_html=True)
    days = st.selectbox("Simulation Period", [7, 14, 30], index=0,
                        format_func=lambda d: f"{d} days (~{d*1440:,} rows)")
    anomaly_rate_pct = st.slider("Anomaly Rate", 5, 25, 10, 1,
                                  format="%d%%", help="% of data with injected anomalies")
    anomaly_rate = anomaly_rate_pct / 100.0
    run_clicked = st.button("🚀 Generate & Run Pipeline")
    st.divider()

    if st.session_state.pipeline_done and st.session_state.df is not None:
        st.markdown("<div class='slabel'>📊 Filters</div>", unsafe_allow_html=True)
        all_workloads = st.session_state.df["workload_type"].unique().tolist()
        selected_workloads = st.multiselect("Workload Types", all_workloads,
                                            default=all_workloads)
        st.divider()
        st.markdown("<div class='slabel'>📥 Export</div>", unsafe_allow_html=True)
        df_filtered = apply_filters(st.session_state.df, selected_workloads)
        st.download_button(
            "📊 Download Cleaned CSV",
            data=df_filtered.to_csv(index=False),
            file_name="cleaned_metrics.csv", mime="text/csv",
            use_container_width=True,
        )
        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, "rb") as f:
                st.download_button(
                    "📄 Download HTML Report",
                    data=f, file_name="performance_report.html",
                    mime="text/html", use_container_width=True,
                )
        st.divider()
    else:
        selected_workloads = []

    st.markdown("""<div style='font-size:11px;color:#475569;line-height:2;'>
    <b>Pipeline Steps</b><br>
    1. Generate synthetic metrics<br>
    2. ETL: clean + normalize<br>
    3. Feature engineering<br>
    4. Bottleneck detection<br>
    5. Anomaly detection (3-method)<br>
    6. Correlation analysis<br>
    7. Storytelling + insights<br>
    8. HTML report generation
    </div>""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='hero-title'>GPU/System Performance Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub'>Simulate → Detect → Analyse → Optimise. NVIDIA-inspired performance monitoring platform.</div>", unsafe_allow_html=True)
st.markdown("""
<span class='pill pill-green'>GPU Monitoring</span>
<span class='pill pill-cyan'>Bottleneck Detection</span>
<span class='pill pill-purple'>Anomaly Detection</span>
<span class='pill pill-amber'>Correlation Engine</span>
<span class='pill pill-green'>SQL Layer</span>
<span class='pill pill-cyan'>Storytelling</span>
<span class='pill pill-red'>Real-World Mitigations</span>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
if run_clicked:
    result = run_full_pipeline(days, anomaly_rate)
    (st.session_state.df, st.session_state.bottleneck_df,
     st.session_state.bottleneck_summary, st.session_state.anomaly_df,
     st.session_state.anomaly_summary, st.session_state.correlation_results,
     st.session_state.findings, st.session_state.strategies,
     st.session_state.db_path, st.session_state.report_path) = result
    st.session_state.pipeline_done = True
    st.success(f"✅ Pipeline complete! {len(st.session_state.df):,} rows processed.")
    st.rerun()

# ── No data state ─────────────────────────────────────────────────────────────
if not st.session_state.pipeline_done:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""<div style='text-align:center;background:#1a1d27;border:1px dashed rgba(34,197,94,0.4);
             border-radius:20px;padding:70px 40px;margin-top:20px;'>
          <div style='font-size:3.5rem;margin-bottom:16px;'>⚡</div>
          <div style='font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:8px;'>Ready to Analyse</div>
          <div style='color:#64748b;font-size:14px;line-height:1.8;'>
            Select your simulation period and anomaly rate<br>
            in the sidebar, then click<br>
            <b style='color:#22c55e'>🚀 Generate &amp; Run Pipeline</b>
          </div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ── Get data ──────────────────────────────────────────────────────────────────
df_all = st.session_state.df
df = apply_filters(df_all, selected_workloads) if selected_workloads else df_all
bottleneck_df = st.session_state.bottleneck_df
anomaly_df = st.session_state.anomaly_df
bn_summary = st.session_state.bottleneck_summary
an_summary = st.session_state.anomaly_summary
corr_results = st.session_state.correlation_results
findings = st.session_state.findings
strategies = st.session_state.strategies


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🖥️  System Overview",
    "📊  Performance Trends",
    "🚨  Bottleneck Detection",
    "🔬  Anomaly Analysis",
    "📖  Insights & Story",
    "📋  SQL Explorer & Reports",
])


# ── TAB 1: System Overview ────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='slabel'>System Health — Key Performance Indicators</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Avg GPU Usage", f"{df['gpu_usage'].mean():.1f}%",
              delta=f"Max {df['gpu_usage'].max():.0f}%")
    c2.metric("Avg CPU Usage", f"{df['cpu_usage'].mean():.1f}%",
              delta=f"Max {df['cpu_usage'].max():.0f}%")
    c3.metric("Avg Memory", f"{df['memory_usage'].mean():.1f}%",
              delta=f"Max {df['memory_usage'].max():.0f}%")
    c4.metric("Avg Temp", f"{df['temperature'].mean():.1f}°C",
              delta=f"Max {df['temperature'].max():.0f}°C")
    c5.metric("Avg Power", f"{df['power_consumption'].mean():.0f}W",
              delta=f"Max {df['power_consumption'].max():.0f}W")
    c6.metric("Bottleneck Events", f"{bn_summary.get('total', 0):,}",
              delta=f"{an_summary.get('confirmed', 0)} anomalies")

    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(chart_health_gauge(df["gpu_usage"].mean(), "GPU Usage",
                        thresholds={"good": 70, "warn": 85}), use_container_width=True)
    with g2:
        st.plotly_chart(chart_health_gauge(df["temperature"].mean(), "Temperature",
                        max_val=105, thresholds={"good": 70, "warn": 85}), use_container_width=True)
    with g3:
        st.plotly_chart(chart_health_gauge(df["memory_usage"].mean(), "Memory Usage",
                        thresholds={"good": 70, "warn": 85}), use_container_width=True)
    with g4:
        st.plotly_chart(chart_health_gauge(
            min(100, df["power_consumption"].mean() / 5),
            "Power Draw", thresholds={"good": 60, "warn": 80}
        ), use_container_width=True)

    st.divider()
    lo, ro = st.columns([2, 1], gap="large")
    with lo:
        st.markdown("<div class='slabel'>System Metrics Summary</div>", unsafe_allow_html=True)
        summary_data = {
            "Metric": ["GPU Usage (%)", "CPU Usage (%)", "Memory Usage (%)", "Temperature (°C)", "Power (W)"],
            "Mean": [df["gpu_usage"].mean(), df["cpu_usage"].mean(), df["memory_usage"].mean(),
                     df["temperature"].mean(), df["power_consumption"].mean()],
            "Max": [df["gpu_usage"].max(), df["cpu_usage"].max(), df["memory_usage"].max(),
                    df["temperature"].max(), df["power_consumption"].max()],
            "Min": [df["gpu_usage"].min(), df["cpu_usage"].min(), df["memory_usage"].min(),
                    df["temperature"].min(), df["power_consumption"].min()],
            "Std": [df["gpu_usage"].std(), df["cpu_usage"].std(), df["memory_usage"].std(),
                    df["temperature"].std(), df["power_consumption"].std()],
        }
        summary_df = pd.DataFrame(summary_data)
        for col in ["Mean", "Max", "Min", "Std"]:
            summary_df[col] = summary_df[col].round(1)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    with ro:
        st.markdown("<div class='slabel'>Workload Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_workload_distribution(df), use_container_width=True)


# ── TAB 2: Performance Trends ────────────────────────────────────────────────
with tab2:
    # Downsample for display (show max 2000 points for performance)
    step = max(1, len(df) // 2000)
    df_plot = df.iloc[::step].copy()

    st.markdown("<div class='slabel'>GPU & CPU Utilisation Over Time</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_gpu_cpu_over_time(df_plot), use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("<div class='slabel'>Temperature Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_temperature_timeline(df_plot), use_container_width=True)
    with col_b:
        st.markdown("<div class='slabel'>Memory Usage Over Time</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_memory_usage(df_plot), use_container_width=True)

    st.divider()
    st.markdown("<div class='slabel'>Power Consumption by Workload</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_power_by_workload(df), use_container_width=True)

    st.divider()
    st.markdown("<div class='slabel'>Hourly Averages</div>", unsafe_allow_html=True)
    if "hour_of_day" in df.columns:
        hourly = df.groupby("hour_of_day")[
            ["gpu_usage", "cpu_usage", "memory_usage", "temperature", "power_consumption"]
        ].mean().round(1).reset_index()
        hourly.columns = ["Hour", "GPU %", "CPU %", "Memory %", "Temp °C", "Power W"]
        st.dataframe(hourly, use_container_width=True, hide_index=True)


# ── TAB 3: Bottleneck Detection ───────────────────────────────────────────────
with tab3:
    critical_count = bn_summary.get("by_severity", {}).get("Critical", 0)
    warning_count = bn_summary.get("by_severity", {}).get("Warning", 0)
    medium_count = bn_summary.get("by_severity", {}).get("Medium", 0)
    total_bn = bn_summary.get("total", 0)

    if total_bn == 0:
        st.markdown("<div class='alert-banner alert-good'>✅ No bottlenecks detected — system is performing optimally!</div>", unsafe_allow_html=True)
    else:
        if critical_count > 0:
            st.markdown(f"<div class='alert-banner alert-critical'>🔴 {critical_count} Critical events require immediate attention</div>", unsafe_allow_html=True)
        if warning_count > 0:
            st.markdown(f"<div class='alert-banner alert-warning'>🟡 {warning_count} Warning events detected</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    b1.metric("🔴 Critical", critical_count)
    b2.metric("🟡 Warning", warning_count)
    b3.metric("🟠 Medium", medium_count)

    st.divider()
    bc1, bc2 = st.columns([3, 2], gap="large")
    with bc1:
        st.markdown("<div class='slabel'>Bottleneck Events Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_bottleneck_timeline(bottleneck_df), use_container_width=True)
    with bc2:
        st.markdown("<div class='slabel'>Frequency by Type</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_bottleneck_frequency(bottleneck_df), use_container_width=True)

    if not bottleneck_df.empty:
        st.divider()
        st.markdown("<div class='slabel'>Bottleneck Events Table</div>", unsafe_allow_html=True)
        display_bn = bottleneck_df[["timestamp", "issue", "severity", "reason", "recommendation"]].copy()
        display_bn["timestamp"] = pd.to_datetime(display_bn["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            display_bn.head(200),
            use_container_width=True, hide_index=True,
            column_config={
                "recommendation": st.column_config.TextColumn("Recommendation", width="large"),
            }
        )


# ── TAB 4: Anomaly Analysis ───────────────────────────────────────────────────
with tab4:
    high_conf = len(anomaly_df[anomaly_df["confidence"] == "high"]) if not anomaly_df.empty else 0
    med_conf = len(anomaly_df[anomaly_df["confidence"] == "medium"]) if not anomaly_df.empty else 0
    low_conf = len(anomaly_df[anomaly_df["confidence"] == "low"]) if not anomaly_df.empty else 0
    confirmed = an_summary.get("confirmed", 0)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Flagged", an_summary.get("total", 0))
    a2.metric("🔴 High Confidence", high_conf)
    a3.metric("🟡 Medium Confidence", med_conf)
    a4.metric("⚪ Low Confidence", low_conf)

    st.divider()
    st.markdown("<div class='slabel'>Anomaly Distribution Heatmap (Hour × Metric)</div>", unsafe_allow_html=True)
    st.plotly_chart(chart_anomaly_heatmap(anomaly_df), use_container_width=True)

    if not anomaly_df.empty and confirmed > 0:
        st.divider()
        st.markdown("<div class='slabel'>Confirmed Anomalies Table</div>", unsafe_allow_html=True)
        conf_df = anomaly_df[anomaly_df["is_anomaly"]].copy()
        conf_df["timestamp"] = pd.to_datetime(conf_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            conf_df[["timestamp", "metric", "value", "z_score", "confidence", "methods_flagged"]].head(200),
            use_container_width=True, hide_index=True,
        )

        st.divider()
        st.markdown("<div class='slabel'>Anomaly Breakdown by Metric</div>", unsafe_allow_html=True)
        by_metric = an_summary.get("by_metric", {})
        if by_metric:
            mc1, mc2 = st.columns(2)
            for i, (metric, count) in enumerate(by_metric.items()):
                col = mc1 if i % 2 == 0 else mc2
                col.metric(metric.replace("_", " ").title(), f"{count} anomalies")


# ── TAB 5: Insights & Story ───────────────────────────────────────────────────
with tab5:
    # Key Findings
    st.markdown("<div class='slabel'>📖 Key Findings</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:14px;margin-bottom:16px;'>Auto-generated narratives from system analysis — the story your data tells.</p>", unsafe_allow_html=True)

    if findings:
        for f in findings:
            st.markdown(f"""<div class='finding {f["severity"]}'>
                <div class='finding-title'>{f['icon']} {f['title']}</div>
                <div style='color:#94a3b8;font-size:14px;'>{f['detail']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Run the pipeline to generate key findings.")

    st.divider()

    # Correlation Analysis
    st.markdown("<div class='slabel'>🔗 Correlation Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:14px;margin-bottom:16px;'>Cross-metric relationships using Pearson and Spearman methods.</p>", unsafe_allow_html=True)

    if corr_results:
        # Correlation heatmap
        corr_matrix = get_correlation_matrix(df)
        st.plotly_chart(chart_correlation_heatmap(corr_matrix), use_container_width=True)

        st.divider()
        for cr in corr_results:
            if cr["strength"] in ["Strong", "Moderate"]:
                wl_tags = "".join(
                    f"<span class='pill pill-cyan'>{wl}: r={vals['pearson_r']}</span>"
                    for wl, vals in cr.get("by_workload", {}).items()
                )
                st.markdown(f"""<div class='corr-card'>
                    <div style='font-weight:700;font-size:15px;margin-bottom:6px;'>
                        🔗 {cr['pair']}
                    </div>
                    <div style='color:#94a3b8;font-size:14px;margin-bottom:10px;'>
                        {cr['interpretation']}
                    </div>
                    <div>
                        <span class='pill pill-green'>Pearson: {cr['pearson_r']}</span>
                        <span class='pill pill-purple'>Spearman: {cr['spearman_r']}</span>
                        <span class='pill pill-amber'>{cr['strength']}</span>
                        {wl_tags}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Scatter plots for top pairs
        st.divider()
        st.markdown("<div class='slabel'>Correlation Scatter Plots</div>", unsafe_allow_html=True)
        top_pairs = [cr for cr in corr_results if cr["strength"] == "Strong"][:4]
        if top_pairs:
            for i in range(0, len(top_pairs), 2):
                sc1, sc2 = st.columns(2, gap="large")
                with sc1:
                    cr = top_pairs[i]
                    if cr["x_col"] in df.columns and cr["y_col"] in df.columns:
                        st.plotly_chart(chart_correlation_scatter(
                            df, cr["x_col"], cr["y_col"], cr["pair"], cr["pearson_r"]
                        ), use_container_width=True)
                if i + 1 < len(top_pairs):
                    with sc2:
                        cr = top_pairs[i + 1]
                        if cr["x_col"] in df.columns and cr["y_col"] in df.columns:
                            st.plotly_chart(chart_correlation_scatter(
                                df, cr["x_col"], cr["y_col"], cr["pair"], cr["pearson_r"]
                            ), use_container_width=True)

    st.divider()

    # What would you do in real life?
    st.markdown("<div class='slabel'>🏢 What Would You Do In Real Life?</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:14px;margin-bottom:16px;'>Production mitigation strategies derived from detected issues.</p>", unsafe_allow_html=True)

    if strategies:
        for s in strategies:
            impl_items = "".join(f"<div class='impl-li'>{step}</div>" for step in s["implementation"])
            st.markdown(f"""<div class='strategy'>
                <div class='strategy-title'>{s['icon']} {s['title']}</div>
                <div style='color:#94a3b8;font-size:14px;margin-bottom:10px;'>{s['strategy']}</div>
                <div class='strategy-impact'>📈 Impact: {s['impact']}</div>
                <div style='margin-top:8px;'>{impl_items}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No specific strategies generated — system appears healthy!")


# ── TAB 6: SQL Explorer & Reports ────────────────────────────────────────────
with tab6:
    db_path = st.session_state.db_path

    # Pre-built query library
    st.markdown("<div class='slabel'>📚 Pre-Built Query Library</div>", unsafe_allow_html=True)
    QUERIES = [
        {"title": "Avg Usage per Workload", "category": "Workload",
         "description": "Average GPU/CPU/Memory/Temperature/Power per workload type",
         "sql": "SELECT workload_type, ROUND(AVG(gpu_usage),1) AS avg_gpu, ROUND(AVG(cpu_usage),1) AS avg_cpu, ROUND(AVG(memory_usage),1) AS avg_memory, ROUND(AVG(temperature),1) AS avg_temp, ROUND(AVG(power_consumption),0) AS avg_power, COUNT(*) AS samples FROM system_metrics GROUP BY workload_type ORDER BY avg_gpu DESC;"},
        {"title": "Top 5 Hottest Periods", "category": "Thermal",
         "description": "Highest temperature 15-minute windows",
         "sql": "SELECT SUBSTR(timestamp, 1, 15) || '0' AS window, ROUND(MAX(temperature),1) AS max_temp, ROUND(AVG(temperature),1) AS avg_temp, ROUND(AVG(gpu_usage),1) AS avg_gpu FROM system_metrics GROUP BY window ORDER BY max_temp DESC LIMIT 5;"},
        {"title": "Hourly Memory Trend", "category": "Memory",
         "description": "Average and max memory usage per hour of day",
         "sql": "SELECT hour_of_day, ROUND(AVG(memory_usage),1) AS avg_memory, ROUND(MAX(memory_usage),1) AS max_memory, ROUND(AVG(memory_pressure),3) AS avg_pressure FROM system_metrics GROUP BY hour_of_day ORDER BY hour_of_day;"},
        {"title": "Bottleneck Frequency", "category": "Bottleneck",
         "description": "Count of each bottleneck type by severity",
         "sql": "SELECT issue, severity, COUNT(*) AS event_count, MIN(timestamp) AS first_seen, MAX(timestamp) AS last_seen FROM bottlenecks GROUP BY issue, severity ORDER BY event_count DESC;"},
        {"title": "Peak vs Off-Peak", "category": "Time Analysis",
         "description": "Performance comparison between peak hours (9AM-6PM) and off-peak",
         "sql": "SELECT CASE WHEN is_peak_hour=1 THEN 'Peak (9AM-6PM)' ELSE 'Off-Peak' END AS period, ROUND(AVG(gpu_usage),1) AS avg_gpu, ROUND(AVG(cpu_usage),1) AS avg_cpu, ROUND(AVG(temperature),1) AS avg_temp, ROUND(AVG(power_consumption),0) AS avg_power, COUNT(*) AS readings FROM system_metrics GROUP BY is_peak_hour;"},
        {"title": "Power by Workload", "category": "Power",
         "description": "Power consumption statistics per workload type",
         "sql": "SELECT workload_type, ROUND(AVG(power_consumption),0) AS avg_power, ROUND(MAX(power_consumption),0) AS max_power, ROUND(AVG(power_efficiency),4) AS avg_efficiency FROM system_metrics GROUP BY workload_type ORDER BY avg_power DESC;"},
        {"title": "Anomaly Count by Hour", "category": "Anomaly",
         "description": "Confirmed anomalies per metric per hour",
         "sql": "SELECT metric, CAST(SUBSTR(timestamp,12,2) AS INTEGER) AS hour, COUNT(*) AS anomaly_count, ROUND(AVG(z_score),2) AS avg_zscore FROM anomalies WHERE is_anomaly=1 GROUP BY metric, hour ORDER BY anomaly_count DESC;"},
        {"title": "Correlated Events", "category": "Cross-Analysis",
         "description": "Bottlenecks and anomalies occurring in the same minute",
         "sql": "SELECT b.timestamp, b.issue AS bottleneck, b.severity, a.metric AS anomaly_metric, ROUND(a.z_score,2) AS z_score, a.confidence FROM bottlenecks b JOIN anomalies a ON SUBSTR(b.timestamp,1,16)=SUBSTR(a.timestamp,1,16) WHERE a.is_anomaly=1 ORDER BY b.timestamp LIMIT 50;"},
    ]

    cats = ["All"] + list(dict.fromkeys(q["category"] for q in QUERIES))
    selected_cat = st.selectbox("Filter by category", cats)
    filtered_queries = QUERIES if selected_cat == "All" else [q for q in QUERIES if q["category"] == selected_cat]

    for i, q in enumerate(filtered_queries):
        with st.expander(f"**{q['title']}** — {q['description']}"):
            st.code(q["sql"], language="sql")
            if st.button(f"▶ Run", key=f"qbtn_{i}"):
                if db_path and os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path)
                        res = pd.read_sql_query(q["sql"], conn)
                        conn.close()
                        st.success(f"✅ {len(res)} rows returned.")
                        st.dataframe(res, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query error: {e}")
                else:
                    st.warning("Run the pipeline first to create the database.")

    st.divider()

    # Custom SQL editor
    st.markdown("<div class='slabel'>⌨️ Custom SQL Editor</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8;font-size:13px;'>Tables: <code style='color:#22d3ee'>system_metrics</code> · <code style='color:#f59e0b'>bottlenecks</code> · <code style='color:#a78bfa'>anomalies</code></p>", unsafe_allow_html=True)
    custom_sql = st.text_area("SQL", value="SELECT * FROM system_metrics LIMIT 10;",
                              height=130, label_visibility="collapsed")
    if st.button("▶ Execute SQL"):
        if db_path and os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                res = pd.read_sql_query(custom_sql, conn)
                conn.close()
                st.success(f"✅ {len(res)} rows returned.")
                st.dataframe(res, use_container_width=True)
                st.download_button("📥 Download CSV", data=res.to_csv(index=False),
                                   file_name="query_result.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Query error: {e}")
        else:
            st.warning("Run the pipeline first.")

    st.divider()

    # Report download
    st.markdown("<div class='slabel'>📄 Performance Report</div>", unsafe_allow_html=True)
    report_path = st.session_state.report_path
    if report_path and os.path.exists(report_path):
        with open(report_path, "rb") as f:
            st.download_button(
                "📄 Download Full HTML Report",
                data=f, file_name="performance_report.html",
                mime="text/html", use_container_width=True,
            )
        st.caption(f"Report includes: Executive Summary · Key Findings · Bottlenecks · Anomalies · Correlations · Mitigation Strategies")
    else:
        st.info("Run the pipeline to generate the HTML report.")
