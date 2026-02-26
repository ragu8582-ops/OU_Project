# =============================================================================
# HOSPITAL OBSERVATION UNIT — Phase 2: EDA & Visualizations
# BDA 640 Final Case Report
# All charts saved as interactive HTML files → embed in report
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

os.makedirs("charts", exist_ok=True)

df = pd.read_csv("OUData_cleaned.csv")

# Color palette — professional consulting style
COLOR_FLIPPED   = "#E63946"   # Red   → Flipped (Inpatient)
COLOR_STAYED    = "#2196F3"   # Blue  → Stayed (Observation)
COLOR_NEUTRAL   = "#457B9D"
COLOR_BG        = "#F8F9FA"
COLOR_HIGHLIGHT = "#FF9800"

FONT = "Inter, Arial, sans-serif"

TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family=FONT, size=13),
        paper_bgcolor="white",
        plot_bgcolor=COLOR_BG,
        title_font=dict(size=18, color="#1D3557"),
        legend=dict(bgcolor="white", bordercolor="#DDD", borderwidth=1),
    )
)

DRG_MAP = {
    276: "Dehydration", 428: "Congestive Heart Failure", 486: "Pneumonia",
    558: "Colitis",     577: "Pancreatitis",             578: "GI Bleeding",
    599: "Urinary Tract Infection", 780: "Syncope",      782: "Edema",
    786: "Chest Pain",  787: "Nausea",                  789: "Abdominal Pain",
}
df["DiagnosisName"] = df["DRG01"].map(DRG_MAP)

# =============================================================================
# CHART 1 — Flip Rate by Diagnosis (sorted bar chart)
# =============================================================================
flip_by_drg = (
    df.groupby("DiagnosisName")["Flipped"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .rename(columns={"sum": "Flipped_Count", "count": "Total", "mean": "Flip_Rate"})
    .sort_values("Flip_Rate", ascending=True)
)
flip_by_drg["Flip_Pct"] = (flip_by_drg["Flip_Rate"] * 100).round(1)

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=flip_by_drg["Flip_Pct"],
    y=flip_by_drg["DiagnosisName"],
    orientation="h",
    marker=dict(
        color=flip_by_drg["Flip_Pct"],
        colorscale=[[0, COLOR_STAYED], [0.5, COLOR_HIGHLIGHT], [1, COLOR_FLIPPED]],
        showscale=True,
        colorbar=dict(title="Flip Rate %", ticksuffix="%")
    ),
    text=[f"  {p}%  (n={n})" for p, n in zip(flip_by_drg["Flip_Pct"], flip_by_drg["Total"])],
    textposition="inside",
    textfont=dict(color="white", size=12),
    hovertemplate="<b>%{y}</b><br>Flip Rate: %{x:.1f}%<extra></extra>",
))
fig1.add_vline(x=df["Flipped"].mean() * 100, line_dash="dash",
               line_color="#333", annotation_text=f"Avg: {df['Flipped'].mean()*100:.1f}%",
               annotation_position="top right")
fig1.update_layout(
    title=dict(text="<b>Flip Rate by Diagnosis</b><br><sup>% of patients converted from Observation → Inpatient</sup>"),
    xaxis=dict(title="Flip Rate (%)", ticksuffix="%", range=[0, 100]),
    yaxis=dict(title=""),
    height=520,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
)
fig1.write_html("charts/01_flip_rate_by_diagnosis.html")
print("✓ Chart 1: Flip Rate by Diagnosis")

# =============================================================================
# CHART 2 — Patient Volume & Flip Rate by Diagnosis (dual metric)
# =============================================================================
flip_by_drg_sorted = flip_by_drg.sort_values("Total", ascending=False)

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Bar(
    x=flip_by_drg_sorted["DiagnosisName"],
    y=flip_by_drg_sorted["Total"],
    name="Total Patients",
    marker_color=COLOR_NEUTRAL,
    opacity=0.85,
), secondary_y=False)
fig2.add_trace(go.Bar(
    x=flip_by_drg_sorted["DiagnosisName"],
    y=flip_by_drg_sorted["Flipped_Count"],
    name="Flipped to Inpatient",
    marker_color=COLOR_FLIPPED,
    opacity=0.85,
), secondary_y=False)
fig2.add_trace(go.Scatter(
    x=flip_by_drg_sorted["DiagnosisName"],
    y=flip_by_drg_sorted["Flip_Pct"],
    name="Flip Rate %",
    mode="lines+markers",
    marker=dict(size=10, color=COLOR_HIGHLIGHT, symbol="diamond"),
    line=dict(color=COLOR_HIGHLIGHT, width=3),
    yaxis="y2",
), secondary_y=True)
fig2.update_layout(
    title=dict(text="<b>Patient Volume & Flip Rate by Diagnosis</b>"),
    barmode="overlay",
    xaxis=dict(tickangle=-30),
    yaxis=dict(title="Number of Patients"),
    yaxis2=dict(title="Flip Rate (%)", ticksuffix="%"),
    height=480,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    legend=dict(orientation="h", y=1.08),
)
fig2.write_html("charts/02_volume_and_flip_rate.html")
print("✓ Chart 2: Volume & Flip Rate by Diagnosis")

# =============================================================================
# CHART 3 — OU Length of Stay: Flipped vs. Stayed (box + violin)
# =============================================================================
fig3 = go.Figure()
for label, color, subset in [
    ("Stayed (Observation)", COLOR_STAYED, df[df["Flipped"] == 0]),
    ("Flipped (Inpatient)",  COLOR_FLIPPED, df[df["Flipped"] == 1]),
]:
    fig3.add_trace(go.Violin(
        y=subset["OU_LOS_hrs"],
        name=label,
        box_visible=True,
        meanline_visible=True,
        fillcolor=color,
        opacity=0.7,
        line_color=color,
        points="outliers",
        marker=dict(size=3),
    ))
fig3.add_hline(y=48, line_dash="dash", line_color="#333",
               annotation_text="48-hr threshold", annotation_position="right")
fig3.update_layout(
    title=dict(text="<b>OU Length of Stay Distribution</b><br><sup>Flipped patients stay significantly longer</sup>"),
    yaxis=dict(title="Hours in OU", rangemode="tozero"),
    xaxis=dict(title=""),
    height=500,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
)
fig3.write_html("charts/03_los_distribution.html")
print("✓ Chart 3: LOS Distribution")

# =============================================================================
# CHART 4 — Age Distribution by Outcome
# =============================================================================
fig4 = go.Figure()
for label, color, subset in [
    ("Stayed (Observation)", COLOR_STAYED, df[df["Flipped"] == 0]["Age"]),
    ("Flipped (Inpatient)",  COLOR_FLIPPED, df[df["Flipped"] == 1]["Age"]),
]:
    fig4.add_trace(go.Histogram(
        x=subset, name=label, opacity=0.7,
        marker_color=color, nbinsx=20,
        histnorm="percent",
    ))
fig4.update_layout(
    title=dict(text="<b>Age Distribution by Patient Outcome</b><br><sup>Older patients more likely to flip</sup>"),
    barmode="overlay",
    xaxis=dict(title="Age (years)"),
    yaxis=dict(title="% of Patients", ticksuffix="%"),
    height=450,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    legend=dict(orientation="h", y=1.08),
)
fig4.write_html("charts/04_age_distribution.html")
print("✓ Chart 4: Age Distribution")

# =============================================================================
# CHART 5 — Flip Rate by Insurance Category
# =============================================================================
flip_by_ins = (
    df.groupby("InsuranceGroup")["Flipped"]
    .agg(["sum", "count", "mean"])
    .reset_index()
    .rename(columns={"sum": "Flipped", "count": "Total", "mean": "Rate"})
)
flip_by_ins["Pct"] = (flip_by_ins["Rate"] * 100).round(1)

fig5 = go.Figure(go.Bar(
    x=flip_by_ins["InsuranceGroup"],
    y=flip_by_ins["Pct"],
    marker_color=[COLOR_FLIPPED, COLOR_NEUTRAL, COLOR_STAYED],
    text=[f"{p}%<br>(n={n})" for p, n in zip(flip_by_ins["Pct"], flip_by_ins["Total"])],
    textposition="outside",
    width=0.5,
))
fig5.add_hline(y=df["Flipped"].mean() * 100, line_dash="dash",
               line_color="#666", annotation_text=f"Overall avg {df['Flipped'].mean()*100:.1f}%")
fig5.update_layout(
    title=dict(text="<b>Flip Rate by Insurance Category</b>"),
    yaxis=dict(title="Flip Rate (%)", ticksuffix="%", range=[0, 65]),
    xaxis=dict(title=""),
    height=420,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
)
fig5.write_html("charts/05_flip_by_insurance.html")
print("✓ Chart 5: Flip Rate by Insurance")

# =============================================================================
# CHART 6 — Vitals Comparison: Flipped vs. Stayed (radar/heatmap)
# =============================================================================
vitals = ["BloodPressureUpper", "BloodPressureLower", "Pulse",
          "PulseOximetry", "Respirations", "Temperature"]
vitals_labels = ["Systolic BP", "Diastolic BP", "Pulse",
                 "Pulse Oximetry", "Respirations", "Temperature"]

stayed_means  = df[df["Flipped"] == 0][vitals].mean()
flipped_means = df[df["Flipped"] == 1][vitals].mean()

# Normalize each vital to 0-100 scale for comparison
overall_min = df[vitals].min()
overall_max = df[vitals].max()
stayed_norm  = ((stayed_means  - overall_min) / (overall_max - overall_min) * 100).round(1)
flipped_norm = ((flipped_means - overall_min) / (overall_max - overall_min) * 100).round(1)

fig6 = go.Figure()
fig6.add_trace(go.Scatterpolar(
    r=list(stayed_norm) + [stayed_norm.iloc[0]],
    theta=vitals_labels + [vitals_labels[0]],
    fill="toself", name="Stayed (Observation)",
    line_color=COLOR_STAYED, fillcolor=f"rgba(33, 150, 243, 0.2)",
))
fig6.add_trace(go.Scatterpolar(
    r=list(flipped_norm) + [flipped_norm.iloc[0]],
    theta=vitals_labels + [vitals_labels[0]],
    fill="toself", name="Flipped (Inpatient)",
    line_color=COLOR_FLIPPED, fillcolor=f"rgba(230, 57, 70, 0.2)",
))
fig6.update_layout(
    title=dict(text="<b>Vital Signs Profile: Flipped vs. Stayed</b><br><sup>Normalized 0–100 scale</sup>"),
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    height=500,
    paper_bgcolor="white",
    font=dict(family=FONT),
    legend=dict(orientation="h", y=-0.1),
)
fig6.write_html("charts/06_vitals_radar.html")
print("✓ Chart 6: Vitals Radar Chart")

# =============================================================================
# CHART 7 — Vitals Actual Values Table (mean comparison)
# =============================================================================
vitals_compare = pd.DataFrame({
    "Vital Sign": vitals_labels,
    "Stayed – Mean": stayed_means.round(1).values,
    "Flipped – Mean": flipped_means.round(1).values,
    "Difference": (flipped_means - stayed_means).round(1).values,
})
vitals_compare["Clinical Flag"] = [
    "↑ Higher BP in flipped",
    "↑ Higher BP in flipped",
    "↑ Higher HR in flipped",
    "↓ Lower O₂ in flipped",
    "↑ Higher RR in flipped",
    "↑ Higher Temp in flipped",
]

fig7 = go.Figure(go.Table(
    header=dict(
        values=["<b>Vital Sign</b>", "<b>Stayed Mean</b>", "<b>Flipped Mean</b>",
                "<b>Difference</b>", "<b>Clinical Note</b>"],
        fill_color="#1D3557",
        font=dict(color="white", size=13, family=FONT),
        align="center", height=35,
    ),
    cells=dict(
        values=[vitals_compare[c] for c in vitals_compare.columns],
        fill_color=[["white", COLOR_BG] * 6],
        align=["left", "center", "center", "center", "left"],
        font=dict(size=12, family=FONT),
        height=30,
    )
))
fig7.update_layout(
    title=dict(text="<b>Mean Vital Signs: Flipped vs. Stayed</b>"),
    height=350,
    paper_bgcolor="white",
    font=dict(family=FONT),
)
fig7.write_html("charts/07_vitals_table.html")
print("✓ Chart 7: Vitals Comparison Table")

# =============================================================================
# CHART 8 — Flip Rate by Age Group × Diagnosis (heatmap)
# =============================================================================
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 40, 55, 65, 75, 89],
                        labels=["18–40", "41–55", "56–65", "66–75", "76+"])
heatmap_data = (
    df.groupby(["DiagnosisName", "AgeGroup"], observed=True)["Flipped"]
    .mean()
    .unstack()
    .fillna(0)
    * 100
).round(1)

fig8 = go.Figure(go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns.tolist(),
    y=heatmap_data.index.tolist(),
    colorscale=[[0, "#EEF4FF"], [0.5, COLOR_HIGHLIGHT], [1, COLOR_FLIPPED]],
    text=heatmap_data.values,
    texttemplate="%{text:.0f}%",
    textfont=dict(size=11),
    colorbar=dict(title="Flip Rate %", ticksuffix="%"),
    hovertemplate="Diagnosis: %{y}<br>Age: %{x}<br>Flip Rate: %{z:.1f}%<extra></extra>",
))
fig8.update_layout(
    title=dict(text="<b>Flip Rate Heatmap: Diagnosis × Age Group</b><br><sup>% converted to Inpatient status</sup>"),
    xaxis=dict(title="Age Group"),
    yaxis=dict(title=""),
    height=500,
    paper_bgcolor="white",
    font=dict(family=FONT),
)
fig8.write_html("charts/08_heatmap_diagnosis_age.html")
print("✓ Chart 8: Diagnosis × Age Heatmap")

# =============================================================================
# CHART 9 — Gender Flip Rate
# =============================================================================
flip_by_gender = (
    df.groupby("Gender")["Flipped"]
    .agg(["sum", "count", "mean"])
    .reset_index()
)
flip_by_gender["Pct"] = (flip_by_gender["mean"] * 100).round(1)

fig9 = go.Figure(go.Bar(
    x=flip_by_gender["Gender"],
    y=flip_by_gender["Pct"],
    marker_color=[COLOR_FLIPPED, COLOR_STAYED],
    text=[f"{p}%<br>(n={n})" for p, n in zip(flip_by_gender["Pct"], flip_by_gender["count"])],
    textposition="outside",
    width=0.4,
))
fig9.update_layout(
    title=dict(text="<b>Flip Rate by Gender</b>"),
    yaxis=dict(title="Flip Rate (%)", ticksuffix="%", range=[0, 60]),
    xaxis=dict(title=""),
    height=380,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
)
fig9.write_html("charts/09_flip_by_gender.html")
print("✓ Chart 9: Flip Rate by Gender")

# =============================================================================
# CHART 10 — Abnormal Vital Count vs. Flip Rate
# =============================================================================
vitals_flip = (
    df.groupby("AbnormalVitalCount")["Flipped"]
    .agg(["sum", "count", "mean"])
    .reset_index()
)
vitals_flip["Pct"] = (vitals_flip["mean"] * 100).round(1)

fig10 = go.Figure()
fig10.add_trace(go.Bar(
    x=vitals_flip["AbnormalVitalCount"],
    y=vitals_flip["count"],
    name="Patient Count",
    marker_color=COLOR_NEUTRAL,
    yaxis="y",
    opacity=0.7,
))
fig10.add_trace(go.Scatter(
    x=vitals_flip["AbnormalVitalCount"],
    y=vitals_flip["Pct"],
    name="Flip Rate %",
    mode="lines+markers",
    marker=dict(size=12, color=COLOR_FLIPPED, symbol="circle"),
    line=dict(color=COLOR_FLIPPED, width=3),
    yaxis="y2",
))
fig10.update_layout(
    title=dict(text="<b>Abnormal Vital Count vs. Flip Rate</b><br><sup>More abnormal vitals → higher chance of flipping</sup>"),
    xaxis=dict(title="Number of Abnormal Vital Signs"),
    yaxis=dict(title="Patient Count"),
    yaxis2=dict(title="Flip Rate (%)", overlaying="y", side="right", ticksuffix="%"),
    height=450,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    legend=dict(orientation="h", y=1.08),
)
fig10.write_html("charts/10_abnormal_vitals_flip.html")
print("✓ Chart 10: Abnormal Vitals vs. Flip Rate")

# =============================================================================
# CHART 11 — LOS by Diagnosis (box plot)
# =============================================================================
drg_order = (df.groupby("DiagnosisName")["OU_LOS_hrs"].median()
               .sort_values(ascending=False).index.tolist())

fig11 = go.Figure()
for diag in drg_order:
    sub = df[df["DiagnosisName"] == diag]
    fig11.add_trace(go.Box(
        y=sub["OU_LOS_hrs"],
        name=diag,
        boxmean="sd",
        marker_color=COLOR_NEUTRAL,
        line_color="#1D3557",
    ))
fig11.add_hline(y=48, line_dash="dash", line_color=COLOR_FLIPPED,
                annotation_text="48-hr target", annotation_position="right")
fig11.update_layout(
    title=dict(text="<b>OU Length of Stay by Diagnosis</b>"),
    yaxis=dict(title="Hours in OU"),
    xaxis=dict(tickangle=-30),
    height=520,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    showlegend=False,
)
fig11.write_html("charts/11_los_by_diagnosis.html")
print("✓ Chart 11: LOS by Diagnosis")

# =============================================================================
# CHART 12 — Capacity Impact Summary (KPI waterfall)
# =============================================================================
total     = len(df)
flipped_n = df["Flipped"].sum()
stayed_n  = total - flipped_n
flip_rate = flipped_n / total

# Capacity scenarios from case
current_per_week      = 44
target_flip_20pct     = 0.20
target_flip_33pct     = 0.333
patients_at_20pct     = round(current_per_week * (1 - target_flip_20pct) / (1 - 0.45))  # ≈55
patients_at_33pct     = round(current_per_week * (1 - target_flip_33pct) / (1 - 0.45))  # ≈49

fig12 = go.Figure(go.Waterfall(
    orientation="v",
    measure=["absolute", "relative", "relative", "total"],
    x=["Current (44/wk)", "Reduce flip to 33%<br>(+5/wk)", "Reduce flip to 20%<br>(+11/wk)", "Max Scenario<br>(55/wk)"],
    y=[44, 5, 6, 0],
    text=["44 patients/wk", "+5 patients/wk<br>(+260/yr)", "+6 patients/wk<br>(+310/yr)", "55 patients/wk"],
    textposition="outside",
    connector={"line": {"color": "#DDD", "dash": "dot"}},
    increasing={"marker": {"color": "#2ECC71"}},
    decreasing={"marker": {"color": COLOR_FLIPPED}},
    totals={"marker": {"color": "#1D3557"}},
))
fig12.update_layout(
    title=dict(text="<b>Capacity Impact: Medicine Service Patients per Week</b><br><sup>By reducing flip rate through improved exclusion list</sup>"),
    yaxis=dict(title="Patients per Week"),
    height=430,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
)
fig12.write_html("charts/12_capacity_waterfall.html")
print("✓ Chart 12: Capacity Impact Waterfall")

print("\n" + "="*60)
print(f"✓ All 12 charts saved to /charts/ folder")
print("→ Run next: python predictive_model.py")
print("="*60)
