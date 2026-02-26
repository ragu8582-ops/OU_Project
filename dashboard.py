# =============================================================================
# HOSPITAL OBSERVATION UNIT — Premium Analytics Dashboard v2.0
# BDA 640 Final Case Report
# Run: python3 dashboard.py → open http://127.0.0.1:8050
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# =============================================================================
# DATA
# =============================================================================
# ── Streamlit Cloud compatible path resolution ──
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "OUData_cleaned.csv")
df = pd.read_csv(DATA_FILE)

DRG_MAP = {
    276:"Dehydration", 428:"Congestive Heart Failure", 486:"Pneumonia",
    558:"Colitis",     577:"Pancreatitis",             578:"GI Bleeding",
    599:"Urinary Tract Infection", 780:"Syncope",       782:"Edema",
    786:"Chest Pain",  787:"Nausea",                   789:"Abdominal Pain",
}
df["DiagnosisName"] = df["DRG01"].map(DRG_MAP)
df["AgeGroup"]      = pd.cut(df["Age"], bins=[0,40,55,65,75,89],
                             labels=["18-40","41-55","56-65","66-75","76+"])

try:
    drg_risk         = pd.read_csv(os.path.join(BASE_DIR, "model_outputs", "drg_risk_scores.csv"))
    model_comparison = pd.read_csv(os.path.join(BASE_DIR, "model_outputs", "model_comparison.csv"))
    MODEL_RAN        = True
except FileNotFoundError:
    MODEL_RAN = False
    drg_risk  = (
        df.groupby(["DRG01","DiagnosisName"])["Flipped"]
        .agg(Avg_Prob="mean", Count="count").reset_index()
    )
    drg_risk["Actual_Rate"]  = drg_risk["Avg_Prob"]
    drg_risk["Avg_Prob_Pct"] = (drg_risk["Avg_Prob"] * 100).round(1)

# =============================================================================
# PALETTE
# =============================================================================
BG_DARK  = "#0A0E1A"
BG_CARD  = "#111827"
BG_CARD2 = "#1A2233"
ACCENT1  = "#6366F1"
ACCENT2  = "#EC4899"
ACCENT3  = "#10B981"
ACCENT4  = "#F59E0B"
ACCENT5  = "#3B82F6"
TEXT_PRI = "#F1F5F9"
TEXT_SEC = "#94A3B8"
BORDER   = "#1E293B"
FONT     = "Inter, Segoe UI, system-ui, sans-serif"

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family=FONT, color=TEXT_PRI, size=12),
    margin=dict(l=16, r=16, t=16, b=16),
    legend=dict(bgcolor="rgba(255,255,255,0.05)", bordercolor=BORDER,
                borderwidth=1, font=dict(color=TEXT_SEC, size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color=TEXT_SEC, linecolor=BORDER),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color=TEXT_SEC, linecolor=BORDER),
)

# =============================================================================
# KPIs
# =============================================================================
total_patients  = len(df)
flip_n          = int(df["Flipped"].sum())
flip_rate       = df["Flipped"].mean() * 100
avg_los_all     = df["OU_LOS_hrs"].mean()
avg_los_flipped = df[df["Flipped"]==1]["OU_LOS_hrs"].mean()
avg_los_stayed  = df[df["Flipped"]==0]["OU_LOS_hrs"].mean()
avg_age         = df["Age"].mean()
current_per_wk  = 44

flip_by_drg = (
    df.groupby("DiagnosisName")["Flipped"]
    .agg(Flipped_Count="sum", Total="count", Flip_Rate="mean")
    .reset_index().sort_values("Flip_Rate", ascending=True)
)
flip_by_drg["Pct"] = (flip_by_drg["Flip_Rate"] * 100).round(1)

# =============================================================================
# CSS
# =============================================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0A0E1A; font-family: 'Inter', system-ui, sans-serif; color: #F1F5F9; overflow-x: hidden; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #111827; }
::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }

.sidebar {
    position: fixed; left: 0; top: 0; width: 220px; height: 100vh;
    background: linear-gradient(180deg,#111827 0%,#0d1526 100%);
    border-right: 1px solid #1E293B; z-index: 100;
    display: flex; flex-direction: column;
}
.sidebar-logo { padding: 22px 18px 18px; border-bottom: 1px solid #1E293B; }
.sidebar-logo h3 { font-size: 0.95rem; font-weight: 700; color: #F1F5F9; line-height: 1.3; }
.sidebar-logo p  { font-size: 0.7rem;  color: #64748B; margin-top: 3px; }
.sidebar-nav  { padding: 14px 10px; flex: 1; }
.nav-section-label { font-size: 0.62rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 1px; padding: 8px 8px 4px; }
.nav-btn {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px; border-radius: 10px; cursor: pointer;
    transition: all 0.2s ease; margin-bottom: 2px;
    border: 1px solid transparent; background: transparent;
    color: #94A3B8; font-size: 0.83rem; font-weight: 500;
    width: 100%; text-align: left; font-family: 'Inter', system-ui, sans-serif;
}
.nav-btn:hover { background: rgba(99,102,241,0.1); color: #C7D2FE; }
.nav-btn.active {
    background: linear-gradient(135deg,rgba(99,102,241,0.2) 0%,rgba(236,72,153,0.1) 100%);
    color: #A5B4FC; border-color: rgba(99,102,241,0.25);
}
.nav-icon { font-size: 1rem; width: 18px; text-align: center; }
.sidebar-footer { padding: 14px 18px; border-top: 1px solid #1E293B; }
.sidebar-footer p { font-size: 0.68rem; color: #475569; }

.main-content { margin-left: 220px; min-height: 100vh; background: #0A0E1A; }

.topbar {
    background: rgba(17,24,39,0.96); backdrop-filter: blur(12px);
    border-bottom: 1px solid #1E293B; padding: 13px 26px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 50;
}
.topbar-title { font-size: 1.05rem; font-weight: 700; color: #F1F5F9; }
.topbar-badge {
    background: linear-gradient(135deg,#6366F1,#EC4899);
    color: white; font-size: 0.7rem; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
.live-dot {
    width: 7px; height: 7px; background: #10B981; border-radius: 50%;
    display: inline-block; margin-right: 6px;
    animation: pulseDot 2s infinite;
}
@keyframes pulseDot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(1.4); }
}
.page-content { padding: 22px 26px; }

.glass-card {
    background: rgba(17,24,39,0.85); border: 1px solid #1E293B;
    border-radius: 16px; padding: 18px; backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease; margin-bottom: 18px;
}
.glass-card:hover { transform: translateY(-2px); box-shadow: 0 10px 36px rgba(0,0,0,0.45); }

.kpi-card {
    background: rgba(17,24,39,0.9); border: 1px solid #1E293B;
    border-radius: 16px; padding: 16px 18px; position: relative;
    overflow: hidden; transition: all 0.3s ease; height: 100%;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 14px 44px rgba(0,0,0,0.5); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; border-radius: 16px 16px 0 0;
}
.kpi-card.red::before    { background: linear-gradient(90deg,#EC4899,#F43F5E); }
.kpi-card.blue::before   { background: linear-gradient(90deg,#3B82F6,#6366F1); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10B981,#06B6D4); }
.kpi-card.amber::before  { background: linear-gradient(90deg,#F59E0B,#EF4444); }
.kpi-card.indigo::before { background: linear-gradient(90deg,#6366F1,#8B5CF6); }
.kpi-card.teal::before   { background: linear-gradient(90deg,#06B6D4,#3B82F6); }
.kpi-icon  { font-size: 1.5rem; margin-bottom: 8px; display: block; }
.kpi-value { font-size: 1.85rem; font-weight: 800; line-height: 1; margin-bottom: 4px; letter-spacing: -0.5px; }
.kpi-value.red    { color: #F87171; }
.kpi-value.blue   { color: #60A5FA; }
.kpi-value.green  { color: #34D399; }
.kpi-value.amber  { color: #FCD34D; }
.kpi-value.indigo { color: #A5B4FC; }
.kpi-value.teal   { color: #67E8F9; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.6px; }
.kpi-sub   { font-size: 0.7rem; color: #475569; margin-top: 5px; }

.section-header { margin-bottom: 18px; }
.section-header h4 { font-size: 1.05rem; font-weight: 700; color: #F1F5F9; margin-bottom: 3px; }
.section-header p  { font-size: 0.8rem; color: #64748B; }

.chart-title { font-size: 0.85rem; font-weight: 600; color: #CBD5E1; margin-bottom: 10px; display: flex; align-items: center; gap: 6px; }
.filter-label { font-size: 0.72rem; font-weight: 600; color: #64748B; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 6px; display: block; }
.filter-panel { background: rgba(17,24,39,0.9); border: 1px solid #1E293B; border-radius: 16px; padding: 18px; margin-bottom: 18px; }
.divider { border: none; border-top: 1px solid #1E293B; margin: 14px 0; }

.excl-item { display: flex; align-items: center; gap: 10px; padding: 9px 13px; border-radius: 10px; margin-bottom: 5px; font-size: 0.83rem; color: #CBD5E1; }
.excl-item.current { background: rgba(100,116,139,0.08); border: 1px solid rgba(100,116,139,0.15); }
.excl-item.add     { background: rgba(16,185,129,0.07); border: 1px solid rgba(16,185,129,0.18); color: #6EE7B7; }

.risk-high   { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.25);  padding: 2px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.risk-medium { background: rgba(245,158,11,0.12); color: #FCD34D; border: 1px solid rgba(245,158,11,0.25); padding: 2px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }
.risk-low    { background: rgba(16,185,129,0.12); color: #34D399; border: 1px solid rgba(16,185,129,0.25); padding: 2px 9px; border-radius: 20px; font-size: 0.73rem; font-weight: 600; }

.impact-num { font-size: 2rem; font-weight: 800; letter-spacing: -1px; line-height: 1; }

.vitals-row-item {
    padding: 10px 14px; background: rgba(255,255,255,0.03);
    border-radius: 10px; border: 1px solid #1E293B; margin-bottom: 7px;
}
.progress-bg  { background: rgba(255,255,255,0.06); border-radius: 4px; height: 5px; overflow: hidden; margin-top: 5px; }
.progress-fill { height: 100%; border-radius: 4px; }

.info-banner {
    background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 12px 16px; margin-bottom: 18px;
    display: flex; align-items: center; gap: 10px; color: #94A3B8; font-size: 0.82rem;
}
"""

# =============================================================================
# CHART BUILDERS
# =============================================================================
def CL(**kw):
    """Merge base chart layout with overrides."""
    out = {**CHART_BASE}
    out.update(kw)
    return out

def make_flip_drg():
    data = flip_by_drg.copy()
    colors = [ACCENT2 if p>60 else (ACCENT4 if p>45 else ACCENT5) for p in data["Pct"]]
    fig = go.Figure(go.Bar(
        x=data["Pct"], y=data["DiagnosisName"], orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.04)",width=1)),
        text=[f"{p}%  (n={n})" for p,n in zip(data["Pct"],data["Total"])],
        textposition="inside", textfont=dict(color="rgba(255,255,255,0.9)",size=11),
        hovertemplate="<b>%{y}</b><br>Flip Rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=flip_rate, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                  annotation_text=f"Avg {flip_rate:.1f}%",
                  annotation_font=dict(color=TEXT_SEC,size=10))
    fig.update_layout(**CL(
        xaxis=dict(title="Flip Rate (%)",ticksuffix="%",range=[0,100],
                   gridcolor="rgba(255,255,255,0.04)",color=TEXT_SEC),
        yaxis=dict(title="",color=TEXT_SEC),height=400,
    ))
    return fig

def make_los():
    fig = go.Figure()
    for lbl,color,fill,sub in [
        ("Stayed",  ACCENT5,"rgba(59,130,246,0.15)",  df[df["Flipped"]==0]),
        ("Flipped", ACCENT2,"rgba(236,72,153,0.15)",  df[df["Flipped"]==1]),
    ]:
        fig.add_trace(go.Violin(y=sub["OU_LOS_hrs"],name=lbl,box_visible=True,
            meanline_visible=True,fillcolor=fill,line_color=color,opacity=0.9,
            points="outliers",marker=dict(size=3,color=color,opacity=0.4)))
    fig.add_hline(y=48,line_dash="dot",line_color="rgba(255,255,255,0.2)",
                  annotation_text="48-hr target",
                  annotation_font=dict(color=TEXT_SEC,size=10),
                  annotation_position="top right")
    fig.update_layout(**CL(
        yaxis=dict(title="Hours in OU",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=340,
        legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_SEC,size=11)),
    ))
    return fig

def make_heatmap():
    hm = (df.groupby(["DiagnosisName","AgeGroup"],observed=True)["Flipped"]
          .mean().unstack().fillna(0)*100).round(1)
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
        colorscale=[[0,"rgba(99,102,241,0.08)"],[0.4,"rgba(99,102,241,0.5)"],
                    [0.7,ACCENT4],[1.0,ACCENT2]],
        text=hm.values, texttemplate="%{text:.0f}%",
        textfont=dict(size=11,color="white"),
        colorbar=dict(ticksuffix="%",
                      tickfont=dict(color=TEXT_SEC),
                      title=dict(text="Flip %",font=dict(color=TEXT_SEC))),
        hovertemplate="<b>%{y}</b><br>Age: %{x}<br>Flip Rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**CL(
        xaxis=dict(title="Age Group",color=TEXT_SEC),
        yaxis=dict(title="",color=TEXT_SEC),height=430,
    ))
    return fig

def make_radar():
    vitals = ["BloodPressureUpper","BloodPressureLower","Pulse","PulseOximetry","Respirations","Temperature"]
    labels = ["Systolic BP","Diastolic BP","Heart Rate","O2 Sat","Respirations","Temperature"]
    s = df[df["Flipped"]==0][vitals].mean()
    f = df[df["Flipped"]==1][vitals].mean()
    mn,mx = df[vitals].min(),df[vitals].max()
    sn = ((s-mn)/(mx-mn)*100).round(1)
    fn = ((f-mn)/(mx-mn)*100).round(1)
    fig = go.Figure()
    for name,vals,lc,fc in [
        ("Stayed",  sn, ACCENT5,"rgba(59,130,246,0.12)"),
        ("Flipped", fn, ACCENT2,"rgba(236,72,153,0.12)"),
    ]:
        fig.add_trace(go.Scatterpolar(
            r=list(vals)+[vals.iloc[0]], theta=labels+[labels[0]],
            fill="toself", name=name, line=dict(color=lc,width=2.5), fillcolor=fc,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True,range=[0,100],
                gridcolor="rgba(255,255,255,0.07)",
                tickfont=dict(color=TEXT_SEC,size=9),color=TEXT_SEC),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.07)",
                tickfont=dict(color=TEXT_SEC,size=10),color=TEXT_SEC),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT,color=TEXT_PRI),
        margin=dict(l=46,r=46,t=26,b=46), height=380,
        legend=dict(orientation="h",y=-0.1,bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_SEC,size=11)),
    )
    return fig

def make_flags():
    cols   = ["Flag_Tachycardia","Flag_Hypo_O2","Flag_Fever","Flag_Tachypnea","Flag_Hypertension","Flag_Hypotension"]
    labels = ["Tachycardia","Low O2","Fever","Tachypnea","Hypertension","Hypotension"]
    fp = (df[df["Flipped"]==1][cols].mean()*100).round(1)
    sp = (df[df["Flipped"]==0][cols].mean()*100).round(1)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stayed", x=labels, y=sp,
        marker=dict(color=ACCENT5,line=dict(color="rgba(255,255,255,0.04)",width=1)),opacity=0.85))
    fig.add_trace(go.Bar(name="Flipped",x=labels, y=fp,
        marker=dict(color=ACCENT2,line=dict(color="rgba(255,255,255,0.04)",width=1)),opacity=0.85))
    fig.update_layout(**CL(
        barmode="group",
        yaxis=dict(title="% with Flag",ticksuffix="%",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=310,
        legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
    ))
    return fig

def make_model_bar():
    def blank(msg):
        fig=go.Figure()
        fig.add_annotation(text=msg,xref="paper",yref="paper",x=0.5,y=0.5,
            showarrow=False,font=dict(size=13,color=TEXT_SEC,family=FONT),align="center")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",height=320,
            xaxis=dict(visible=False),yaxis=dict(visible=False))
        return fig
    if not MODEL_RAN:
        return blank("Run python3 predictive_model.py<br>then restart the dashboard")
    try:
        mc   = model_comparison.copy()
        aucs = mc["ROC-AUC"].astype(str).str.extract(r"([\d\.]+)")[0].astype(float)
        colors=[ACCENT5,ACCENT2,ACCENT3]
        fig=go.Figure()
        for m,a,c in zip(mc["Model"],aucs,colors):
            fig.add_trace(go.Bar(x=[m],y=[a],name=m,width=0.5,
                marker=dict(color=c,line=dict(color="rgba(255,255,255,0.06)",width=1)),
                text=[f"AUC = {a:.3f}"],textposition="outside",
                textfont=dict(color=TEXT_PRI,size=12)))
        fig.add_hline(y=0.5,line_dash="dot",line_color="rgba(255,255,255,0.15)",
                      annotation_text="Random (0.50)",
                      annotation_font=dict(color=TEXT_SEC,size=10))
        fig.update_layout(**CL(
            barmode="group",showlegend=False,
            yaxis=dict(title="ROC-AUC",range=[0.4,1.0],color=TEXT_SEC,
                       gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(color=TEXT_SEC),height=320,
        ))
        return fig
    except Exception as e:
        return blank(f"Error: {e}")

def make_drg_risk():
    try:
        data=drg_risk.copy()
        if "Avg_Prob_Pct" not in data.columns:
            data["Avg_Prob_Pct"]=(data["Avg_Prob"]*100).round(1)
        if "DiagnosisName" not in data.columns and "DRG01" in data.columns:
            data["DiagnosisName"]=data["DRG01"].map(DRG_MAP).fillna(data["DRG01"].astype(str))
        data=data.sort_values("Avg_Prob_Pct",ascending=True)
        colors=[ACCENT2 if p>55 else (ACCENT4 if p>35 else ACCENT5) for p in data["Avg_Prob_Pct"]]
        fig=go.Figure(go.Bar(
            x=data["Avg_Prob_Pct"],y=data["DiagnosisName"],orientation="h",
            marker=dict(color=colors,line=dict(color="rgba(255,255,255,0.04)",width=1)),
            text=[f"  {p:.0f}%  " for p in data["Avg_Prob_Pct"]],
            textposition="inside",textfont=dict(color="rgba(255,255,255,0.9)",size=11),
            hovertemplate="<b>%{y}</b><br>Flip Prob: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=55,line_dash="dash",line_color="rgba(255,255,255,0.18)",
                      annotation_text="Exclusion threshold",
                      annotation_font=dict(color=TEXT_SEC,size=10),
                      annotation_position="top right")
        fig.update_layout(**CL(
            xaxis=dict(title="Predicted Flip Probability (%)",ticksuffix="%",range=[0,100],
                       color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="",color=TEXT_SEC),height=420,
        ))
        return fig
    except Exception as e:
        fig=go.Figure()
        fig.add_annotation(text=f"Error: {e}",xref="paper",yref="paper",x=0.5,y=0.5,
            showarrow=False,font=dict(size=12,color=ACCENT2,family=FONT))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=340,
            xaxis=dict(visible=False),yaxis=dict(visible=False))
        return fig

def make_waterfall(target_pct):
    extra=max(0,round(current_per_wk*(1-target_pct/100)/(1-0.46))-current_per_wk)
    fig=go.Figure(go.Waterfall(
        orientation="v",measure=["absolute","relative","total"],
        x=["Current\nBaseline",f"Reduce flip\nto {target_pct}%","Projected\nCapacity"],
        y=[current_per_wk,extra,0],
        text=[f"44/wk",f"+{extra}/wk",f"{current_per_wk+extra}/wk"],
        textposition="outside",textfont=dict(color=TEXT_PRI,size=12),
        connector=dict(line=dict(color="rgba(255,255,255,0.08)",dash="dot")),
        increasing=dict(marker=dict(color=ACCENT3,line=dict(color=ACCENT3,width=1))),
        totals=dict(marker=dict(color=ACCENT1,line=dict(color=ACCENT1,width=1))),
    ))
    fig.update_layout(**CL(
        yaxis=dict(title="Patients / Week",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=320,
    ))
    return fig

def make_age_hist_fig(dff):
    fig=go.Figure()
    for lbl,color,sub in [("Stayed",ACCENT5,dff[dff["Flipped"]==0]),
                           ("Flipped",ACCENT2,dff[dff["Flipped"]==1])]:
        fig.add_trace(go.Histogram(x=sub["Age"],name=lbl,opacity=0.8,
            marker=dict(color=color),nbinsx=16,histnorm="percent"))
    fig.update_layout(**CL(
        barmode="overlay",
        xaxis=dict(title="Age (years)",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="%",ticksuffix="%",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        height=270,
        legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
    ))
    return fig

def make_ins_bar_fig(dff):
    ins=dff.groupby("InsuranceGroup")["Flipped"].agg(["mean","count"]).reset_index()
    ins["Pct"]=(ins["mean"]*100).round(1)
    fig=go.Figure(go.Bar(
        x=ins["InsuranceGroup"],y=ins["Pct"],width=0.5,
        marker=dict(color=[ACCENT2,ACCENT5,ACCENT3][:len(ins)],
                    line=dict(color="rgba(255,255,255,0.04)",width=1)),
        text=[f"{p}%" for p in ins["Pct"]],textposition="outside",
        textfont=dict(color=TEXT_PRI,size=12),
    ))
    fig.update_layout(**CL(
        yaxis=dict(title="Flip Rate (%)",ticksuffix="%",range=[0,70],
                   color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=270,
    ))
    return fig

def make_drg_box_fig(dff,title):
    fig=go.Figure()
    for lbl,color,sub in [("Stayed",ACCENT5,dff[dff["Flipped"]==0]),
                           ("Flipped",ACCENT2,dff[dff["Flipped"]==1])]:
        if len(sub)>0:
            fig.add_trace(go.Box(y=sub["OU_LOS_hrs"],name=lbl,boxmean="sd",
                marker=dict(color=color,size=4,opacity=0.5),
                line=dict(color=color,width=2),
                fillcolor="rgba(255,255,255,0.03)"))
    fig.add_hline(y=48,line_dash="dot",line_color="rgba(255,255,255,0.18)",
                  annotation_text="48-hr target",
                  annotation_font=dict(color=TEXT_SEC,size=10))
    fig.update_layout(**CL(
        yaxis=dict(title="Hours in OU",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=310,
        title=dict(text=title,font=dict(size=12,color=TEXT_SEC)),
    ))
    return fig

# =============================================================================
# UI HELPERS
# =============================================================================
def kpi_card(icon, value, label, sub, color):
    return html.Div([
        html.Span(icon, className="kpi-icon"),
        html.Div(value, className=f"kpi-value {color}"),
        html.Div(label, className="kpi-label"),
        html.Div(sub,   className="kpi-sub"),
    ], className=f"kpi-card {color}")

def glass(*children, style=None):
    s = {}
    if style: s.update(style)
    return html.Div(list(children), className="glass-card", style=s)

def ctitle(icon, text):
    return html.Div([html.Span(icon,style={"marginRight":"6px"}),html.Span(text)],
                    className="chart-title")

def vitals_panel():
    vitals_info = [
        ("BloodPressureUpper","Systolic BP","mmHg"),
        ("BloodPressureLower","Diastolic BP","mmHg"),
        ("Pulse","Heart Rate","bpm"),
        ("PulseOximetry","O2 Saturation","%"),
        ("Respirations","Respirations","br/min"),
        ("Temperature","Temperature","F"),
    ]
    items=[]
    for col,lbl,unit in vitals_info:
        sm = round(df[df["Flipped"]==0][col].mean(),1)
        fm = round(df[df["Flipped"]==1][col].mean(),1)
        diff = round(fm-sm,1)
        arrow = "up" if diff>0 else "dn"
        ac    = ACCENT2 if diff>0 else ACCENT3
        pct   = min(fm/df[col].max()*100,100)
        items.append(html.Div([
            html.Div([
                html.Span(lbl,style={"color":TEXT_PRI,"fontSize":"0.83rem","fontWeight":"600"}),
                html.Span(f"{'↑' if diff>0 else '↓'} {abs(diff)} {unit}",
                          style={"color":ac,"fontSize":"0.73rem","fontWeight":"600","marginLeft":"auto"}),
            ],style={"display":"flex","alignItems":"center","marginBottom":"3px"}),
            html.Div([
                html.Span(f"Stayed: {sm}",style={"color":TEXT_SEC,"fontSize":"0.73rem"}),
                html.Span("  ·  ",style={"color":BORDER}),
                html.Span(f"Flipped: {fm}",style={"color":TEXT_PRI,"fontSize":"0.73rem"}),
            ]),
            html.Div(html.Div(style={
                "width":f"{pct:.0f}%","height":"100%","borderRadius":"4px",
                "background":f"linear-gradient(90deg,{ac},{ac}88)",
            }),className="progress-bg",style={"marginTop":"5px"}),
        ],className="vitals-row-item"))
    return html.Div(items)

def sidebar_nav(active):
    items=[
        ("tab-overview",  "📊","Patient Overview"),
        ("tab-diagnosis", "🔬","Diagnosis Analysis"),
        ("tab-vitals",    "🩺","Vitals Analysis"),
        ("tab-model",     "🤖","Predictive Model"),
        ("tab-recommend", "💡","Exclusion List"),
    ]
    btns=[]
    for tid,icon,label in items:
        cls="nav-btn active" if tid==active else "nav-btn"
        btns.append(html.Button(
            [html.Span(icon,className="nav-icon"),html.Span(label)],
            id=f"nav-{tid}",className=cls,
        ))
    return html.Div([
        html.Div([
            html.Div("🏥",style={"fontSize":"1.9rem","marginBottom":"5px"}),
            html.H3("Montanaro OU"),
            html.P("Analytics Dashboard"),
        ],className="sidebar-logo"),
        html.Div([
            html.Div("NAVIGATION",className="nav-section-label"),
            *btns,
        ],className="sidebar-nav"),
        html.Div([
            html.P("BDA 640 Final Case"),
            html.P("Hospital OU Operations",style={"marginTop":"2px"}),
        ],className="sidebar-footer"),
    ],className="sidebar")

def topbar(title):
    return html.Div([
        html.Div(title,className="topbar-title"),
        html.Div([
            html.Span(className="live-dot"),
            html.Span("Live",style={"color":TEXT_SEC,"fontSize":"0.78rem","marginRight":"14px"}),
            html.Span("BDA 640",className="topbar-badge"),
        ],style={"display":"flex","alignItems":"center"}),
    ],className="topbar")

# =============================================================================
# PAGES
# =============================================================================
def page_overview():
    return html.Div([
        topbar("📊 Patient Population Overview"),
        html.Div([
            dbc.Row([
                dbc.Col(kpi_card("👥",f"{total_patients:,}","Total Patients","Medicine service","indigo"),md=2),
                dbc.Col(kpi_card("🔄",f"{flip_rate:.1f}%","Flip Rate",f"{flip_n} flipped","red"),md=2),
                dbc.Col(kpi_card("⏱",f"{avg_los_all:.1f}h","Avg LOS (All)","Mean OU stay","blue"),md=2),
                dbc.Col(kpi_card("📈",f"{avg_los_flipped:.1f}h","LOS (Flipped)","Converted pts","amber"),md=2),
                dbc.Col(kpi_card("📉",f"{avg_los_stayed:.1f}h","LOS (Stayed)","Obs discharge","green"),md=2),
                dbc.Col(kpi_card("🧑‍⚕",f"{avg_age:.0f}","Mean Age","Patient population","teal"),md=2),
            ],className="g-3",style={"marginBottom":"20px"}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div("🎛  Filters",style={"fontWeight":"700","color":TEXT_PRI,
                                                       "fontSize":"0.88rem","marginBottom":"14px"}),
                        html.Span("Insurance",className="filter-label"),
                        dcc.Checklist(id="filter-insurance",
                            options=[{"label":f"  {i}","value":i} for i in df["InsuranceGroup"].unique()],
                            value=df["InsuranceGroup"].unique().tolist(),
                            labelStyle={"display":"block","fontSize":"0.82rem","marginBottom":"5px","color":TEXT_SEC},
                            inputStyle={"marginRight":"7px","accentColor":ACCENT1}),
                        html.Hr(className="divider"),
                        html.Span("Gender",className="filter-label"),
                        dcc.Checklist(id="filter-gender",
                            options=[{"label":"  Male","value":"Male"},{"label":"  Female","value":"Female"}],
                            value=["Male","Female"],
                            labelStyle={"display":"block","fontSize":"0.82rem","marginBottom":"5px","color":TEXT_SEC},
                            inputStyle={"marginRight":"7px","accentColor":ACCENT1}),
                        html.Hr(className="divider"),
                        html.Span("Age Range",className="filter-label"),
                        dcc.RangeSlider(id="filter-age",min=19,max=89,step=1,value=[19,89],
                            marks={i:{"label":str(i),"style":{"color":TEXT_SEC,"fontSize":"0.68rem"}} for i in range(20,90,10)},
                            tooltip={"placement":"bottom","always_visible":True}),
                    ],className="filter-panel"),
                ],md=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(glass(ctitle("📈","Age Distribution"),
                            dcc.Graph(id="chart-age-hist",config={"displayModeBar":False})),md=6),
                        dbc.Col(glass(ctitle("🏥","Flip Rate by Insurance"),
                            dcc.Graph(id="chart-ins-bar",config={"displayModeBar":False})),md=6),
                    ],className="g-3"),
                    glass(ctitle("🎻","OU Length of Stay: Flipped vs. Stayed"),
                        dcc.Graph(id="chart-los",figure=make_los(),config={"displayModeBar":False})),
                ],md=10),
            ],className="g-3"),
        ],className="page-content"),
    ])

def page_diagnosis():
    return html.Div([
        topbar("🔬 Diagnosis Deep-Dive"),
        html.Div([
            dbc.Row([
                dbc.Col(glass(ctitle("📊","Flip Rate by Diagnosis"),
                    dcc.Graph(figure=make_flip_drg(),config={"displayModeBar":False})),md=6),
                dbc.Col(glass(ctitle("🔥","Flip Rate Heatmap: Diagnosis x Age Group"),
                    dcc.Graph(figure=make_heatmap(),config={"displayModeBar":False})),md=6),
            ],className="g-3"),
            glass(
                ctitle("📦","LOS Distribution — Select Diagnosis to Filter"),
                dcc.Dropdown(id="drg-selector",
                    options=[{"label":v,"value":v} for v in sorted(df["DiagnosisName"].unique())],
                    value=None, placeholder="Select a diagnosis...",
                    style={"backgroundColor":BG_CARD2,"color":TEXT_PRI,
                           "border":f"1px solid {BORDER}","borderRadius":"8px",
                           "marginBottom":"14px","maxWidth":"360px"}),
                dcc.Graph(id="chart-drg-detail",config={"displayModeBar":False}),
            ),
        ],className="page-content"),
    ])

def page_vitals():
    return html.Div([
        topbar("🩺 Vital Signs Analysis"),
        html.Div([
            dbc.Row([
                dbc.Col(glass(ctitle("🕸","Vitals Radar: Flipped vs. Stayed"),
                    dcc.Graph(figure=make_radar(),config={"displayModeBar":False})),md=5),
                dbc.Col(glass(ctitle("📋","Mean Vitals Comparison"),vitals_panel()),md=7),
            ],className="g-3"),
            glass(ctitle("🚨","Abnormal Vital Flags: Flipped vs. Stayed"),
                dcc.Graph(figure=make_flags(),config={"displayModeBar":False})),
        ],className="page-content"),
    ])

def page_model():
    banner=html.Div([
        html.Span("ℹ",style={"fontSize":"1rem","marginRight":"8px","color":ACCENT1}),
        html.Span("Three ML models trained to predict patient flip probability. "
                  "Run python3 predictive_model.py first for full results.",
                  style={"color":TEXT_SEC,"fontSize":"0.82rem"}),
    ],className="info-banner")
    return html.Div([
        topbar("🤖 Predictive Model Results"),
        html.Div([
            banner,
            dbc.Row([
                dbc.Col(glass(ctitle("📊","Model Performance (ROC-AUC)"),
                    dcc.Graph(figure=make_model_bar(),config={"displayModeBar":False})),md=6),
                dbc.Col(glass(
                    ctitle("⚠","Predicted Flip Risk by Diagnosis"),
                    html.Div([
                        html.Span("High Risk",className="risk-high",style={"marginRight":"8px"}),
                        html.Span("Medium Risk",className="risk-medium",style={"marginRight":"8px"}),
                        html.Span("Low Risk",className="risk-low"),
                    ],style={"marginBottom":"10px"}),
                    dcc.Graph(figure=make_drg_risk(),config={"displayModeBar":False})),md=6),
            ],className="g-3"),
            glass(ctitle("💡","How to Interpret"),
                dbc.Row([
                    dbc.Col([
                        html.Div("What is ROC-AUC?",
                            style={"color":TEXT_PRI,"fontWeight":"600","fontSize":"0.86rem","marginBottom":"6px"}),
                        html.P("Measures ability to distinguish flippers from non-flippers. "
                               "1.0 = perfect, 0.5 = random. Scores >0.70 are clinically meaningful.",
                               style={"color":TEXT_SEC,"fontSize":"0.8rem","lineHeight":"1.65"}),
                    ],md=4),
                    dbc.Col([
                        html.Div("Risk Tiers",
                            style={"color":TEXT_PRI,"fontWeight":"600","fontSize":"0.86rem","marginBottom":"6px"}),
                        html.P("Diagnoses with predicted flip probability >55% are High Risk — "
                               "prime candidates for the expanded exclusion list.",
                               style={"color":TEXT_SEC,"fontSize":"0.8rem","lineHeight":"1.65"}),
                    ],md=4),
                    dbc.Col([
                        html.Div("Which Model to Trust?",
                            style={"color":TEXT_PRI,"fontWeight":"600","fontSize":"0.86rem","marginBottom":"6px"}),
                        html.P("Gradient Boosting typically achieves highest AUC. "
                               "Logistic Regression is most interpretable for clinical staff.",
                               style={"color":TEXT_SEC,"fontSize":"0.8rem","lineHeight":"1.65"}),
                    ],md=4),
                ],className="g-3"),
            ),
        ],className="page-content"),
    ])

def page_recommend():
    return html.Div([
        topbar("💡 Exclusion List & Financial Impact"),
        html.Div([
            dbc.Row([
                dbc.Col([
                    glass(ctitle("📋","Current Exclusion List (6 Diagnoses)"),
                        html.Div([
                            html.Div([
                                html.Span("●",style={"color":"#64748B","marginRight":"10px","flexShrink":"0"}),
                                html.Span(item),
                            ],className="excl-item current")
                            for item in ["Alcohol Intoxication","Alcohol Withdrawal",
                                "Mental Health Disorder","Obstetrics Patients",
                                "Sickle Cell Anemia Crisis","Cerebrovascular Accident (Stroke)"]
                        ]),
                    ),
                    glass(ctitle("✅","Data-Supported Additions"),
                        html.Div([
                            html.Div([
                                html.Span("✓",style={"color":ACCENT3,"fontWeight":"700","marginRight":"10px","flexShrink":"0"}),
                                html.Span(item,style={"flex":"1"}),
                                html.Span(badge,style={"fontSize":"0.7rem","color":ACCENT4,"marginLeft":"8px","whiteSpace":"nowrap"}),
                            ],className="excl-item add")
                            for item,badge in [
                                ("Congestive Heart Failure","61%+ flip rate"),
                                ("Pancreatitis","70%+ flip rate"),
                                ("Urinary Tract Infection","66%+ flip rate"),
                                ("Pneumonia","57%+ flip rate"),
                                ("GI Bleeding","52%+ flip rate"),
                            ]
                        ]),
                    ),
                ],md=5),
                dbc.Col([
                    glass(ctitle("🎯","Capacity Impact Calculator"),
                        html.Span("Target Flip Rate After Intervention (%)",className="filter-label"),
                        dcc.Slider(id="flip-target-slider",min=10,max=45,step=5,value=20,
                            marks={i:{"label":f"{i}%","style":{"color":TEXT_SEC,"fontSize":"0.7rem"}}
                                   for i in range(10,50,5)},
                            tooltip={"placement":"bottom","always_visible":True}),
                        html.Div(id="impact-output",style={"marginTop":"18px"}),
                        dcc.Graph(id="waterfall-chart",config={"displayModeBar":False}),
                    ),
                ],md=7),
            ],className="g-3"),
            dbc.Row([
                dbc.Col(glass(
                    html.Div("🚨",style={"fontSize":"1.8rem","marginBottom":"7px"}),
                    html.Div("~1,900",className="impact-num",style={"color":ACCENT2}),
                    html.Div("LWBS Cases/Year",className="kpi-label",style={"marginTop":"4px"}),
                    html.Div("Leave-without-being-seen",className="kpi-sub"),
                ),md=3),
                dbc.Col(glass(
                    html.Div("💵",style={"fontSize":"1.8rem","marginBottom":"7px"}),
                    html.Div("$700",className="impact-num",style={"color":ACCENT5}),
                    html.Div("Revenue per ED Visit",className="kpi-label",style={"marginTop":"4px"}),
                    html.Div("Average reimbursement",className="kpi-sub"),
                ),md=3),
                dbc.Col(glass(
                    html.Div("📈",style={"fontSize":"1.8rem","marginBottom":"7px"}),
                    html.Div("$399K+",className="impact-num",style={"color":ACCENT3}),
                    html.Div("Potential Revenue Gain",className="kpi-label",style={"marginTop":"4px"}),
                    html.Div("30% LWBS reduction via throughput",className="kpi-sub"),
                ),md=3),
                dbc.Col(glass(
                    html.Div("🏥",style={"fontSize":"1.8rem","marginBottom":"7px"}),
                    html.Div("260-570",className="impact-num",style={"color":ACCENT4}),
                    html.Div("Extra Patients/Year",className="kpi-label",style={"marginTop":"4px"}),
                    html.Div("Reducing flip rate to 20-33%",className="kpi-sub"),
                ),md=3),
            ],className="g-3"),
        ],className="page-content"),
    ])

# =============================================================================
# APP
# =============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Montanaro OU Analytics", suppress_callback_exceptions=True)

app.index_string = f"""<!DOCTYPE html>
<html><head>{{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
<style>{CSS}</style></head>
<body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body></html>"""

app.layout = html.Div([
    dcc.Store(id="active-tab", data="tab-overview"),
    html.Div(id="sidebar-container"),
    html.Div(id="page-container", className="main-content"),
], style={"fontFamily":FONT,"backgroundColor":BG_DARK})

# Single unified callback — all 5 nav buttons in one
@app.callback(
    Output("active-tab","data"),
    [Input("nav-tab-overview","n_clicks"),
     Input("nav-tab-diagnosis","n_clicks"),
     Input("nav-tab-vitals","n_clicks"),
     Input("nav-tab-model","n_clicks"),
     Input("nav-tab-recommend","n_clicks")],
    prevent_initial_call=True,
)
def switch_tab(n1, n2, n3, n4, n5):
    triggered = dash.ctx.triggered_id
    if triggered:
        return triggered.replace("nav-", "")
    return dash.no_update

@app.callback(
    [Output("sidebar-container","children"), Output("page-container","children")],
    Input("active-tab","data"),
)
def render(tab):
    pages = {"tab-overview":page_overview,"tab-diagnosis":page_diagnosis,
             "tab-vitals":page_vitals,"tab-model":page_model,"tab-recommend":page_recommend}
    return sidebar_nav(tab), pages.get(tab, page_overview)()

@app.callback(
    [Output("chart-age-hist","figure"),
     Output("chart-ins-bar","figure"),
     Output("chart-los","figure")],
    [Input("filter-insurance","value"),
     Input("filter-gender","value"),
     Input("filter-age","value")],
)
def update_overview(ins_vals, gender_vals, age_range):
    dff = df[
        (df["InsuranceGroup"].isin(ins_vals or [])) &
        (df["Gender"].isin(gender_vals or [])) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]
    if len(dff)==0:
        empty=go.Figure()
        empty.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(255,255,255,0.02)")
        return empty,empty,empty

    fig_los=go.Figure()
    for lbl,color,fill,sub in [
        ("Stayed",ACCENT5,"rgba(59,130,246,0.15)",dff[dff["Flipped"]==0]),
        ("Flipped",ACCENT2,"rgba(236,72,153,0.15)",dff[dff["Flipped"]==1]),
    ]:
        if len(sub)>0:
            fig_los.add_trace(go.Violin(y=sub["OU_LOS_hrs"],name=lbl,box_visible=True,
                meanline_visible=True,fillcolor=fill,line_color=color,opacity=0.9,
                points="outliers",marker=dict(size=3,color=color,opacity=0.4)))
    fig_los.add_hline(y=48,line_dash="dot",line_color="rgba(255,255,255,0.18)")
    fig_los.update_layout(**CL(
        yaxis=dict(title="Hours in OU",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC),height=310,
        legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
    ))
    return make_age_hist_fig(dff), make_ins_bar_fig(dff), fig_los

@app.callback(Output("chart-drg-detail","figure"), Input("drg-selector","value"))
def drg_detail(sel):
    dff = df if not sel else df[df["DiagnosisName"]==sel]
    return make_drg_box_fig(dff, f"LOS — {sel or 'All Diagnoses'}")

@app.callback(
    [Output("waterfall-chart","figure"), Output("impact-output","children")],
    Input("flip-target-slider","value"),
)
def update_impact(target_pct):
    extra    = max(0,round(current_per_wk*(1-target_pct/100)/(1-0.46))-current_per_wk)
    extra_yr = extra*52
    rev      = round(extra_yr*0.2)*700
    impact   = dbc.Row([
        dbc.Col(html.Div([
            html.Div(f"+{extra}/wk",className="impact-num",style={"color":ACCENT3}),
            html.Div("Extra patients/week",className="kpi-sub",style={"marginTop":"4px"}),
        ],style={"textAlign":"center","padding":"10px 0"}),md=4),
        dbc.Col(html.Div([
            html.Div(f"+{extra_yr:,}/yr",className="impact-num",style={"color":ACCENT5}),
            html.Div("Additional patients/year",className="kpi-sub",style={"marginTop":"4px"}),
        ],style={"textAlign":"center","padding":"10px 0"}),md=4),
        dbc.Col(html.Div([
            html.Div(f"${rev:,.0f}",className="impact-num",style={"color":ACCENT4}),
            html.Div("Est. revenue gain",className="kpi-sub",style={"marginTop":"4px"}),
        ],style={"textAlign":"center","padding":"10px 0"}),md=4),
    ],className="g-2",style={"marginBottom":"14px"})
    return make_waterfall(target_pct), impact

# =============================================================================
# SERVER — expose for gunicorn (Render.com deployment)
# =============================================================================
server = app.server  # gunicorn needs this line

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
        use_reloader=False,    # prevents signal thread error
        dev_tools_hot_reload=False
    )
