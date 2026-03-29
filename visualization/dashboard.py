"""
Dashboard Visualization Module
==============================
Plotly chart builders for the GPU/System Performance Analytics dashboard.
10 chart types with consistent NVIDIA-inspired dark theming.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ── Theme ─────────────────────────────────────────────────────────────────────
THEME = {
    "bg": "#0E1117",
    "surface": "#1a1d27",
    "grid": "rgba(255,255,255,0.05)",
    "gpu_green": "#22c55e",
    "cpu_cyan": "#22d3ee",
    "memory_purple": "#a78bfa",
    "temp_amber": "#f59e0b",
    "power_blue": "#3b82f6",
    "critical_red": "#ef4444",
    "warning_amber": "#f59e0b",
    "text": "#e2e8f0",
    "muted": "#64748b",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=THEME["bg"],
    plot_bgcolor=THEME["surface"],
    font=dict(family="DM Sans, sans-serif", color=THEME["text"], size=12),
    margin=dict(l=50, r=30, t=50, b=40),
    xaxis=dict(gridcolor=THEME["grid"], showgrid=True),
    yaxis=dict(gridcolor=THEME["grid"], showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hovermode="x unified",
)


def _apply_theme(fig):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ── Chart 1: GPU/CPU Usage Over Time ─────────────────────────────────────────
def chart_gpu_cpu_over_time(df):
    """Dual-line area chart showing GPU and CPU usage trends."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["gpu_usage"],
        name="GPU Usage", fill="tozeroy",
        line=dict(color=THEME["gpu_green"], width=1),
        fillcolor="rgba(34,197,94,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["cpu_usage"],
        name="CPU Usage", fill="tozeroy",
        line=dict(color=THEME["cpu_cyan"], width=1),
        fillcolor="rgba(34,211,238,0.10)",
    ))
    fig.update_layout(
        title="GPU & CPU Usage Over Time",
        yaxis_title="Usage (%)", xaxis_title="",
        yaxis=dict(range=[0, 105]),
    )
    # Add danger zone
    fig.add_hline(y=90, line_dash="dash", line_color=THEME["critical_red"],
                  annotation_text="Critical (90%)", annotation_font_color=THEME["critical_red"])
    return _apply_theme(fig)


# ── Chart 2: Temperature Timeline ────────────────────────────────────────────
def chart_temperature_timeline(df):
    """Temperature line chart with danger zone overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temperature"],
        name="Temperature", fill="tozeroy",
        line=dict(color=THEME["temp_amber"], width=1.5),
        fillcolor="rgba(245,158,11,0.12)",
    ))
    # Danger zone
    fig.add_hrect(y0=85, y1=100, fillcolor="rgba(239,68,68,0.08)",
                  line_width=0, annotation_text="⚠️ Danger Zone",
                  annotation_position="top left",
                  annotation_font_color=THEME["critical_red"])
    fig.add_hline(y=90, line_dash="dash", line_color=THEME["critical_red"],
                  annotation_text="Throttle (90°C)")
    fig.update_layout(title="Temperature Timeline", yaxis_title="°C")
    return _apply_theme(fig)


# ── Chart 3: Memory Usage ────────────────────────────────────────────────────
def chart_memory_usage(df):
    """Memory usage area chart with leak highlighting."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["memory_usage"],
        name="Memory Usage", fill="tozeroy",
        line=dict(color=THEME["memory_purple"], width=1.5),
        fillcolor="rgba(167,139,250,0.15)",
    ))
    fig.add_hline(y=85, line_dash="dash", line_color=THEME["warning_amber"],
                  annotation_text="Warning (85%)")
    fig.add_hline(y=95, line_dash="dash", line_color=THEME["critical_red"],
                  annotation_text="Critical (95%)")
    fig.update_layout(title="Memory Usage Over Time",
                      yaxis_title="Usage (%)", yaxis=dict(range=[0, 105]))
    return _apply_theme(fig)


# ── Chart 4: Power Consumption ───────────────────────────────────────────────
def chart_power_by_workload(df):
    """Grouped bar chart of power consumption by workload type."""
    agg = df.groupby("workload_type").agg(
        avg_power=("power_consumption", "mean"),
        max_power=("power_consumption", "max"),
    ).reset_index()

    colors = {"training": THEME["critical_red"], "inference": THEME["cpu_cyan"],
              "idle": THEME["muted"]}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["workload_type"], y=agg["avg_power"],
        name="Avg Power", marker_color=[colors.get(w, THEME["muted"]) for w in agg["workload_type"]],
        text=agg["avg_power"].round(0), textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=agg["workload_type"], y=agg["max_power"],
        name="Max Power", marker_color=[f"rgba(255,255,255,0.2)"] * len(agg),
        text=agg["max_power"].round(0), textposition="auto",
    ))
    fig.update_layout(title="Power Consumption by Workload",
                      yaxis_title="Watts", barmode="group")
    return _apply_theme(fig)


# ── Chart 5: Bottleneck Timeline ─────────────────────────────────────────────
def chart_bottleneck_timeline(bottleneck_df):
    """Scatter plot of bottleneck events with severity coloring."""
    if bottleneck_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No bottlenecks detected", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=THEME["muted"]))
        return _apply_theme(fig)

    color_map = {"Critical": THEME["critical_red"], "Warning": THEME["warning_amber"],
                 "Medium": "#f97316"}
    bn = bottleneck_df.copy()
    bn["color"] = bn["severity"].map(color_map).fillna(THEME["muted"])

    fig = go.Figure()
    for sev in ["Critical", "Warning", "Medium"]:
        sub = bn[bn["severity"] == sev]
        if len(sub) > 0:
            fig.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["issue"],
                mode="markers", name=sev,
                marker=dict(color=color_map.get(sev, THEME["muted"]), size=8, opacity=0.7),
                text=sub["reason"], hovertemplate="%{text}<extra></extra>",
            ))
    fig.update_layout(title="Bottleneck Events Timeline", yaxis_title="Issue Type")
    return _apply_theme(fig)


# ── Chart 6: Anomaly Heatmap ─────────────────────────────────────────────────
def chart_anomaly_heatmap(anomaly_df):
    """Heatmap of anomaly counts (hour × metric)."""
    if anomaly_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No anomalies detected", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=THEME["muted"]))
        return _apply_theme(fig)

    confirmed = anomaly_df[anomaly_df["is_anomaly"]].copy()
    if confirmed.empty:
        fig = go.Figure()
        fig.add_annotation(text="No confirmed anomalies", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=THEME["muted"]))
        return _apply_theme(fig)

    confirmed["hour"] = pd.to_datetime(confirmed["timestamp"]).dt.hour
    pivot = confirmed.groupby(["metric", "hour"]).size().unstack(fill_value=0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, THEME["surface"]], [0.5, THEME["warning_amber"]], [1, THEME["critical_red"]]],
        text=pivot.values, texttemplate="%{text}", hovertemplate="Hour %{x}, %{y}: %{z} anomalies<extra></extra>",
    ))
    fig.update_layout(title="Anomaly Distribution (Hour × Metric)",
                      xaxis_title="Hour of Day", yaxis_title="Metric")
    return _apply_theme(fig)


# ── Chart 7: Workload Distribution ───────────────────────────────────────────
def chart_workload_distribution(df):
    """Donut chart of workload type distribution."""
    counts = df["workload_type"].value_counts()
    colors = [THEME["critical_red"], THEME["cpu_cyan"], THEME["muted"]]

    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55, marker=dict(colors=colors[:len(counts)]),
        textinfo="label+percent", textfont=dict(size=13),
    ))
    fig.update_layout(title="Workload Distribution", showlegend=False)
    return _apply_theme(fig)


# ── Chart 8: System Health Gauges ─────────────────────────────────────────────
def chart_health_gauge(value, title, max_val=100, thresholds=None):
    """Single gauge indicator for system health metrics."""
    if thresholds is None:
        thresholds = {"good": 60, "warn": 80}

    if value <= thresholds["good"]:
        bar_color = THEME["gpu_green"]
    elif value <= thresholds["warn"]:
        bar_color = THEME["warning_amber"]
    else:
        bar_color = THEME["critical_red"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title=dict(text=title, font=dict(size=14, color=THEME["text"])),
        number=dict(suffix="%", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[0, max_val], tickcolor=THEME["muted"]),
            bar=dict(color=bar_color),
            bgcolor=THEME["surface"],
            steps=[
                dict(range=[0, thresholds["good"]], color="rgba(34,197,94,0.1)"),
                dict(range=[thresholds["good"], thresholds["warn"]], color="rgba(245,158,11,0.1)"),
                dict(range=[thresholds["warn"], max_val], color="rgba(239,68,68,0.1)"),
            ],
        ),
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor=THEME["bg"], font=dict(color=THEME["text"]))
    return fig


# ── Chart 9: Correlation Scatter ──────────────────────────────────────────────
def chart_correlation_scatter(df, x_col, y_col, title, r_value):
    """Scatter plot with trend line and R² overlay."""
    sample = df.sample(min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample[x_col], y=sample[y_col],
        mode="markers", marker=dict(color=THEME["cpu_cyan"], size=3, opacity=0.3),
        name="Data Points",
    ))

    # Trend line
    if len(sample) > 10:
        z = np.polyfit(sample[x_col].values, sample[y_col].values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample[x_col].min(), sample[x_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p(x_line), mode="lines",
            line=dict(color=THEME["warning_amber"], width=2, dash="dash"),
            name=f"Trend (r={r_value})",
        ))

    fig.update_layout(title=f"{title} (r={r_value})",
                      xaxis_title=x_col.replace("_", " ").title(),
                      yaxis_title=y_col.replace("_", " ").title())
    return _apply_theme(fig)


# ── Chart 10: Correlation Heatmap ─────────────────────────────────────────────
def chart_correlation_heatmap(corr_matrix):
    """Annotated heatmap of metric correlations."""
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=[c.replace("_", " ").title() for c in corr_matrix.columns],
        y=[c.replace("_", " ").title() for c in corr_matrix.index],
        colorscale=[[0, "#3b82f6"], [0.5, THEME["surface"]], [1, THEME["critical_red"]]],
        zmid=0, text=corr_matrix.values.round(2),
        texttemplate="%{text}", hovertemplate="%{x} × %{y}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Metric Correlation Matrix", height=450)
    return _apply_theme(fig)


# ── Chart: Bottleneck Frequency Bar ──────────────────────────────────────────
def chart_bottleneck_frequency(bottleneck_df):
    """Bar chart of bottleneck frequency by type."""
    if bottleneck_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No bottlenecks detected", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=THEME["muted"]))
        return _apply_theme(fig)

    counts = bottleneck_df["issue"].value_counts()
    severity_map = {}
    for _, row in bottleneck_df.drop_duplicates("issue").iterrows():
        severity_map[row["issue"]] = row["severity"]

    color_map = {"Critical": THEME["critical_red"], "Warning": THEME["warning_amber"], "Medium": "#f97316"}
    colors = [color_map.get(severity_map.get(i, ""), THEME["muted"]) for i in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=colors, text=counts.values, textposition="auto",
    ))
    fig.update_layout(title="Bottleneck Frequency by Type", yaxis_title="Count")
    return _apply_theme(fig)
