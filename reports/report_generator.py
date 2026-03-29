"""
Performance Report Generator
=============================
Auto-generates a premium HTML report using Jinja2 templating with
embedded analysis results, key findings, and mitigation strategies.
"""

import os
from datetime import datetime
from jinja2 import Template


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU/System Performance Report</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'DM Sans',sans-serif;background:#0E1117;color:#e2e8f0;padding:40px;line-height:1.7}
.container{max-width:1100px;margin:0 auto}
h1{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#fff 20%,#22d3ee 80%);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
h2{font-size:1.4rem;font-weight:700;color:#22d3ee;margin:32px 0 16px;
   padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.07)}
h3{font-size:1.1rem;font-weight:600;color:#e2e8f0;margin:20px 0 10px}
.subtitle{color:#64748b;font-size:1rem;margin-bottom:32px}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;margin-right:6px}
.badge-green{background:rgba(34,197,94,.15);color:#4ade80}
.badge-cyan{background:rgba(34,211,238,.12);color:#67e8f9}
.badge-amber{background:rgba(245,158,11,.12);color:#fcd34d}
.badge-red{background:rgba(239,68,68,.12);color:#f87171}
.card{background:#1a1d27;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:24px;margin:12px 0}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin:20px 0}
.stat{background:#1a1d27;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:20px;text-align:center}
.stat-value{font-size:2rem;font-weight:800;color:#fff}
.stat-label{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin-top:4px}
.finding{border-left:3px solid;border-radius:8px;padding:16px 20px;margin:10px 0;background:#1a1d27}
.finding.critical{border-color:#ef4444}
.finding.warning{border-color:#f59e0b}
.finding.insight{border-color:#22d3ee}
.finding.positive{border-color:#22c55e}
.finding-title{font-weight:700;font-size:15px;margin-bottom:6px}
.strategy{background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(34,211,238,.05));
          border:1px solid rgba(99,102,241,.2);border-radius:12px;padding:20px;margin:12px 0}
.strategy-title{font-weight:700;font-size:16px;margin-bottom:8px}
.strategy-impact{color:#22c55e;font-size:13px;font-weight:600;margin:8px 0}
.impl-list{list-style:none;padding:0}
.impl-list li{padding:4px 0;color:#94a3b8;font-size:14px}
.impl-list li::before{content:"→ ";color:#22d3ee}
table{width:100%;border-collapse:collapse;margin:16px 0}
th{background:#22263a;color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:1px;
   padding:12px 16px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.07)}
td{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.04);font-size:14px}
tr:hover{background:rgba(255,255,255,0.02)}
.footer{text-align:center;color:#475569;font-size:12px;margin-top:48px;padding-top:24px;
        border-top:1px solid rgba(255,255,255,0.05)}
.score-circle{display:inline-flex;align-items:center;justify-content:center;
              width:80px;height:80px;border-radius:50%;font-size:1.8rem;font-weight:800}
.score-good{background:rgba(34,197,94,.15);color:#4ade80;border:2px solid rgba(34,197,94,.3)}
.score-medium{background:rgba(245,158,11,.15);color:#fcd34d;border:2px solid rgba(245,158,11,.3)}
.score-bad{background:rgba(239,68,68,.12);color:#f87171;border:2px solid rgba(239,68,68,.3)}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<h1>⚡ GPU/System Performance Report</h1>
<p class="subtitle">Generated on {{ generated_at }} · {{ data_range }} · {{ total_rows }} data points</p>
<span class="badge badge-cyan">Performance Analytics</span>
<span class="badge badge-green">Bottleneck Detection</span>
<span class="badge badge-amber">Anomaly Analysis</span>

<!-- Executive Summary -->
<h2>📋 Executive Summary</h2>
<div class="stat-grid">
    <div class="stat">
        <div class="stat-value">{{ health_score }}</div>
        <div class="stat-label">Health Score</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ avg_gpu }}%</div>
        <div class="stat-label">Avg GPU Usage</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ avg_temp }}°C</div>
        <div class="stat-label">Avg Temperature</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ total_bottlenecks }}</div>
        <div class="stat-label">Bottleneck Events</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ total_anomalies }}</div>
        <div class="stat-label">Anomalies Detected</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ avg_power }}W</div>
        <div class="stat-label">Avg Power Draw</div>
    </div>
</div>

<!-- Key Findings -->
<h2>📖 Key Findings</h2>
{% for f in findings %}
<div class="finding {{ f.severity }}">
    <div class="finding-title">{{ f.icon }} {{ f.title }}</div>
    <div>{{ f.detail }}</div>
</div>
{% endfor %}

<!-- Bottleneck Summary -->
<h2>🚨 Bottleneck Analysis</h2>
{% if bottleneck_summary.total > 0 %}
<table>
    <thead><tr><th>Issue Type</th><th>Count</th><th>Severity</th></tr></thead>
    <tbody>
    {% for issue, count in bottleneck_summary.by_type.items() %}
    <tr><td>{{ issue }}</td><td>{{ count }}</td>
        <td>{% if 'Critical' in issue or count > 50 %}<span class="badge badge-red">Critical</span>
            {% else %}<span class="badge badge-amber">Warning</span>{% endif %}</td></tr>
    {% endfor %}
    </tbody>
</table>
{% else %}
<div class="card"><p style="color:#64748b">No bottlenecks detected — system running smoothly.</p></div>
{% endif %}

<!-- Anomaly Summary -->
<h2>🔬 Anomaly Summary</h2>
{% if anomaly_summary.confirmed > 0 %}
<div class="stat-grid">
    <div class="stat">
        <div class="stat-value">{{ anomaly_summary.total }}</div>
        <div class="stat-label">Total Flagged</div>
    </div>
    <div class="stat">
        <div class="stat-value">{{ anomaly_summary.confirmed }}</div>
        <div class="stat-label">Confirmed (≥2 methods)</div>
    </div>
</div>
<table>
    <thead><tr><th>Metric</th><th>Anomaly Count</th></tr></thead>
    <tbody>
    {% for metric, count in anomaly_summary.by_metric.items() %}
    <tr><td>{{ metric }}</td><td>{{ count }}</td></tr>
    {% endfor %}
    </tbody>
</table>
{% else %}
<div class="card"><p style="color:#64748b">No confirmed anomalies.</p></div>
{% endif %}

<!-- Correlation Insights -->
<h2>🔗 Correlation Insights</h2>
{% for cr in correlations %}
<div class="card">
    <h3>{{ cr.pair }}</h3>
    <p>{{ cr.interpretation }}</p>
    {% if cr.by_workload %}
    <p style="margin-top:8px;font-size:13px;color:#94a3b8">
        Per-workload: {% for wl, vals in cr.by_workload.items() %}
        <span class="badge badge-cyan">{{ wl }}: r={{ vals.pearson_r }}</span>
        {% endfor %}
    </p>
    {% endif %}
</div>
{% endfor %}

<!-- Production Recommendations -->
<h2>🏢 Production Recommendations</h2>
<p style="color:#94a3b8;margin-bottom:16px;font-size:14px">
    "What would you do in real life?" — actionable strategies for production systems.
</p>
{% for s in strategies %}
<div class="strategy">
    <div class="strategy-title">{{ s.icon }} {{ s.title }}</div>
    <p>{{ s.strategy }}</p>
    <div class="strategy-impact">📈 Impact: {{ s.impact }}</div>
    <ul class="impl-list">
    {% for step in s.implementation %}
        <li>{{ step }}</li>
    {% endfor %}
    </ul>
</div>
{% endfor %}

<!-- Footer -->
<div class="footer">
    GPU/System Performance Analytics Platform · Built with Python, Streamlit, & Plotly<br>
    Report generated {{ generated_at }}
</div>

</div>
</body>
</html>"""


def generate_performance_report(
    df, bottleneck_df, bottleneck_summary, anomaly_df, anomaly_summary,
    correlation_results, findings, strategies, output_path=None
):
    """
    Generate a premium HTML performance report.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered metrics.
    bottleneck_df, anomaly_df : pd.DataFrame
        Detection results.
    bottleneck_summary, anomaly_summary : dict
        Summary statistics.
    correlation_results : list[dict]
        From correlation_engine.
    findings : list[dict]
        From storyteller.
    strategies : list[dict]
        From storyteller.
    output_path : str, optional
        Where to save the report.

    Returns
    -------
    str
        Path to the generated HTML report.
    """
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "reports", "performance_report.html"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Compute health score
    issues = bottleneck_summary.get("total", 0) + anomaly_summary.get("confirmed", 0)
    health_score = max(0, min(100, 100 - issues // 5))

    # Data range
    ts = df["timestamp"]
    data_range = f"{str(ts.min())[:10]} to {str(ts.max())[:10]}"

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_range=data_range,
        total_rows=f"{len(df):,}",
        health_score=health_score,
        avg_gpu=round(df["gpu_usage"].mean(), 1),
        avg_temp=round(df["temperature"].mean(), 1),
        avg_power=round(df["power_consumption"].mean(), 0),
        total_bottlenecks=bottleneck_summary.get("total", 0),
        total_anomalies=anomaly_summary.get("confirmed", 0),
        findings=findings,
        bottleneck_summary=bottleneck_summary,
        anomaly_summary=anomaly_summary,
        correlations=correlation_results,
        strategies=strategies,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report generated → {output_path}")
    return output_path
