# =============================================================================
# HOSPITAL OBSERVATION UNIT — Premium Analytics Dashboard v2.0
# BDA 640 Final Case Report
# Streamlit version — deploy on Streamlit Cloud
# Run locally: streamlit run dashboard.py
# =============================================================================

import os
import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Montanaro OU Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #0A0E1A;
    color: #F1F5F9;
}
.stApp { background-color: #0A0E1A; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1526 100%) !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #94A3B8 !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.85rem !important; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: rgba(17,24,39,0.9);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 16px 18px;
}
div[data-testid="metric-container"] label { color: #64748B !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.6px; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.85rem !important; font-weight: 800 !important; }

/* ── Cards ── */
.glass-card {
    background: rgba(17,24,39,0.85);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

/* ── Section headers ── */
h1, h2, h3 { color: #F1F5F9 !important; font-family: 'Inter', system-ui, sans-serif !important; }

/* ── Selectbox / Dropdown ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stMultiSelect"] > div {
    background: #1A2233 !important;
    border: 1px solid #1E293B !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] { padding: 4px 0; }

/* ── Checkbox / Radio ── */
.stCheckbox label, .stRadio label { color: #94A3B8 !important; font-size: 0.83rem !important; }

/* ── Divider ── */
hr { border-color: #1E293B !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #EC4899);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; padding: 8px 20px;
}

/* ── Plotly chart bg ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Risk badges ── */
.risk-high   { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3);  padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display:inline-block; }
.risk-medium { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display:inline-block; }
.risk-low    { background: rgba(16,185,129,0.15);  color: #34D399; border: 1px solid rgba(16,185,129,0.3); padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; display:inline-block; }

.excl-item { display:flex; align-items:center; gap:10px; padding:9px 13px; border-radius:10px; margin-bottom:5px; font-size:0.83rem; color:#CBD5E1; }
.excl-current { background:rgba(100,116,139,0.08); border:1px solid rgba(100,116,139,0.15); }
.excl-add     { background:rgba(16,185,129,0.07);  border:1px solid rgba(16,185,129,0.18);  color:#6EE7B7; }

.impact-box {
    background: rgba(17,24,39,0.9); border: 1px solid #1E293B;
    border-radius: 16px; padding: 20px; text-align: center;
}
.impact-num { font-size: 2rem; font-weight: 800; letter-spacing: -1px; line-height:1; }
.impact-label { font-size: 0.72rem; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.6px; margin-top:4px; }
.impact-sub   { font-size: 0.7rem; color:#475569; margin-top:3px; }

.topbar-badge {
    background: linear-gradient(135deg,#6366F1,#EC4899);
    color: white; font-size: 0.75rem; font-weight: 600;
    padding: 4px 14px; border-radius: 20px; display:inline-block;
}
.live-dot {
    display:inline-block; width:8px; height:8px;
    background:#10B981; border-radius:50%; margin-right:6px;
    animation: pulseDot 2s infinite;
}
@keyframes pulseDot {
    0%,100%{opacity:1;transform:scale(1);}
    50%{opacity:0.4;transform:scale(1.4);}
}
.info-banner {
    background:rgba(99,102,241,0.07); border:1px solid rgba(99,102,241,0.2);
    border-radius:12px; padding:12px 16px; margin-bottom:16px;
    color:#94A3B8; font-size:0.82rem;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(BASE_DIR, "data", "OUData_cleaned.csv")
    df = pd.read_csv(DATA_FILE)

    DRG_MAP = {
        276:"Dehydration",         428:"Congestive Heart Failure",
        486:"Pneumonia",           558:"Colitis",
        577:"Pancreatitis",        578:"GI Bleeding",
        599:"Urinary Tract Infection", 780:"Syncope",
        782:"Edema",               786:"Chest Pain",
        787:"Nausea",              789:"Abdominal Pain",
    }
    df["DiagnosisName"] = df["DRG01"].map(DRG_MAP)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,40,55,65,75,89],
                            labels=["18-40","41-55","56-65","66-75","76+"])
    return df

@st.cache_data
def load_model_outputs():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    try:
        drg_risk         = pd.read_csv(os.path.join(BASE_DIR, "model_outputs", "drg_risk_scores.csv"))
        model_comparison = pd.read_csv(os.path.join(BASE_DIR, "model_outputs", "model_comparison.csv"))
        return drg_risk, model_comparison, True
    except FileNotFoundError:
        return None, None, False

df = load_data()
drg_risk_raw, model_comparison, MODEL_RAN = load_model_outputs()

if not MODEL_RAN:
    drg_risk = (
        df.groupby(["DRG01","DiagnosisName"])["Flipped"]
        .agg(Avg_Prob="mean", Count="count").reset_index()
    )
    drg_risk["Actual_Rate"]  = drg_risk["Avg_Prob"]
    drg_risk["Avg_Prob_Pct"] = (drg_risk["Avg_Prob"] * 100).round(1)
else:
    drg_risk = drg_risk_raw

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
# CHART HELPERS
# =============================================================================
def CL(**kw):
    out = {**CHART_BASE}
    out.update(kw)
    return out

def make_flip_drg():
    data   = flip_by_drg.copy()
    colors = [ACCENT2 if p>60 else (ACCENT4 if p>45 else ACCENT5) for p in data["Pct"]]
    fig = go.Figure(go.Bar(
        x=data["Pct"], y=data["DiagnosisName"], orientation="h",
        marker=dict(color=colors),
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
        yaxis=dict(title="",color=TEXT_SEC), height=420,
    ))
    return fig

def make_los(dff=None):
    if dff is None: dff = df
    fig = go.Figure()
    for lbl,color,fill,sub in [
        ("Stayed",  ACCENT5,"rgba(59,130,246,0.15)",  dff[dff["Flipped"]==0]),
        ("Flipped", ACCENT2,"rgba(236,72,153,0.15)",  dff[dff["Flipped"]==1]),
    ]:
        if len(sub)>0:
            fig.add_trace(go.Violin(y=sub["OU_LOS_hrs"],name=lbl,box_visible=True,
                meanline_visible=True,fillcolor=fill,line_color=color,opacity=0.9,
                points="outliers",marker=dict(size=3,color=color,opacity=0.4)))
    fig.add_hline(y=48, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                  annotation_text="48-hr target",
                  annotation_font=dict(color=TEXT_SEC,size=10),
                  annotation_position="top right")
    fig.update_layout(**CL(
        yaxis=dict(title="Hours in OU",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC), height=360,
        legend=dict(orientation="h",y=1.08,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
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
        colorbar=dict(ticksuffix="%", tickfont=dict(color=TEXT_SEC),
                      title=dict(text="Flip %",font=dict(color=TEXT_SEC))),
        hovertemplate="<b>%{y}</b><br>Age: %{x}<br>Flip Rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(**CL(
        xaxis=dict(title="Age Group",color=TEXT_SEC),
        yaxis=dict(title="",color=TEXT_SEC), height=450,
    ))
    return fig

def make_radar():
    vitals = ["BloodPressureUpper","BloodPressureLower","Pulse","PulseOximetry","Respirations","Temperature"]
    labels = ["Systolic BP","Diastolic BP","Heart Rate","O2 Sat","Respirations","Temperature"]
    s  = df[df["Flipped"]==0][vitals].mean()
    f  = df[df["Flipped"]==1][vitals].mean()
    mn = df[vitals].min(); mx = df[vitals].max()
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
        margin=dict(l=46,r=46,t=26,b=46), height=400,
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
        marker=dict(color=ACCENT5), opacity=0.85))
    fig.add_trace(go.Bar(name="Flipped",x=labels, y=fp,
        marker=dict(color=ACCENT2), opacity=0.85))
    fig.update_layout(**CL(
        barmode="group",
        yaxis=dict(title="% with Flag",ticksuffix="%",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC), height=330,
        legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
    ))
    return fig

def make_model_bar():
    def blank(msg):
        fig = go.Figure()
        fig.add_annotation(text=msg,xref="paper",yref="paper",x=0.5,y=0.5,
            showarrow=False,font=dict(size=13,color=TEXT_SEC,family=FONT),align="center")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",height=320,
            xaxis=dict(visible=False),yaxis=dict(visible=False))
        return fig
    if not MODEL_RAN:
        return blank("Run python3 predictive_model.py<br>then redeploy")
    try:
        mc   = model_comparison.copy()
        aucs = mc["ROC-AUC"].astype(str).str.extract(r"([\d\.]+)")[0].astype(float)
        colors = [ACCENT5, ACCENT2, ACCENT3]
        fig = go.Figure()
        for m,a,c in zip(mc["Model"],aucs,colors):
            fig.add_trace(go.Bar(x=[m],y=[a],name=m,width=0.5,
                marker=dict(color=c),
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
        data = drg_risk.copy()
        if "Avg_Prob_Pct" not in data.columns:
            data["Avg_Prob_Pct"] = (data["Avg_Prob"]*100).round(1)
        if "DiagnosisName" not in data.columns and "DRG01" in data.columns:
            DRG_MAP = {276:"Dehydration",428:"Congestive Heart Failure",486:"Pneumonia",
                       558:"Colitis",577:"Pancreatitis",578:"GI Bleeding",
                       599:"Urinary Tract Infection",780:"Syncope",782:"Edema",
                       786:"Chest Pain",787:"Nausea",789:"Abdominal Pain"}
            data["DiagnosisName"] = data["DRG01"].map(DRG_MAP).fillna(data["DRG01"].astype(str))
        data   = data.sort_values("Avg_Prob_Pct", ascending=True)
        colors = [ACCENT2 if p>55 else (ACCENT4 if p>35 else ACCENT5) for p in data["Avg_Prob_Pct"]]
        fig = go.Figure(go.Bar(
            x=data["Avg_Prob_Pct"], y=data["DiagnosisName"], orientation="h",
            marker=dict(color=colors),
            text=[f"  {p:.0f}%  " for p in data["Avg_Prob_Pct"]],
            textposition="inside", textfont=dict(color="rgba(255,255,255,0.9)",size=11),
            hovertemplate="<b>%{y}</b><br>Flip Prob: %{x:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=55,line_dash="dash",line_color="rgba(255,255,255,0.18)",
                      annotation_text="Exclusion threshold",
                      annotation_font=dict(color=TEXT_SEC,size=10),
                      annotation_position="top right")
        fig.update_layout(**CL(
            xaxis=dict(title="Predicted Flip Probability (%)",ticksuffix="%",range=[0,100],
                       color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="",color=TEXT_SEC), height=440,
        ))
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error: {e}",xref="paper",yref="paper",x=0.5,y=0.5,
            showarrow=False,font=dict(size=12,color=ACCENT2,family=FONT))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=340,
            xaxis=dict(visible=False),yaxis=dict(visible=False))
        return fig

def make_waterfall(target_pct):
    extra = max(0, round(current_per_wk*(1-target_pct/100)/(1-0.46)) - current_per_wk)
    fig   = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute","relative","total"],
        x=["Current\nBaseline", f"Reduce flip\nto {target_pct}%", "Projected\nCapacity"],
        y=[current_per_wk, extra, 0],
        text=[f"44/wk", f"+{extra}/wk", f"{current_per_wk+extra}/wk"],
        textposition="outside", textfont=dict(color=TEXT_PRI,size=12),
        connector=dict(line=dict(color="rgba(255,255,255,0.08)",dash="dot")),
        increasing=dict(marker=dict(color=ACCENT3)),
        totals=dict(marker=dict(color=ACCENT1)),
    ))
    fig.update_layout(**CL(
        yaxis=dict(title="Patients / Week",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC), height=340,
    ))
    return fig

def make_age_hist(dff):
    fig = go.Figure()
    for lbl,color,sub in [("Stayed",ACCENT5,dff[dff["Flipped"]==0]),
                           ("Flipped",ACCENT2,dff[dff["Flipped"]==1])]:
        fig.add_trace(go.Histogram(x=sub["Age"],name=lbl,opacity=0.8,
            marker=dict(color=color),nbinsx=16,histnorm="percent"))
    fig.update_layout(**CL(
        barmode="overlay",
        xaxis=dict(title="Age (years)",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(title="%",ticksuffix="%",color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        height=290,
        legend=dict(orientation="h",y=1.1,bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT_SEC,size=11)),
    ))
    return fig

def make_ins_bar(dff):
    ins = dff.groupby("InsuranceGroup")["Flipped"].agg(["mean","count"]).reset_index()
    ins["Pct"] = (ins["mean"]*100).round(1)
    fig = go.Figure(go.Bar(
        x=ins["InsuranceGroup"], y=ins["Pct"], width=0.5,
        marker=dict(color=[ACCENT2,ACCENT5,ACCENT3][:len(ins)]),
        text=[f"{p}%" for p in ins["Pct"]], textposition="outside",
        textfont=dict(color=TEXT_PRI,size=12),
    ))
    fig.update_layout(**CL(
        yaxis=dict(title="Flip Rate (%)",ticksuffix="%",range=[0,70],
                   color=TEXT_SEC,gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(color=TEXT_SEC), height=290,
    ))
    return fig

def make_drg_box(dff, title="LOS"):
    fig = go.Figure()
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
        xaxis=dict(color=TEXT_SEC), height=320,
        title=dict(text=title,font=dict(size=12,color=TEXT_SEC)),
    ))
    return fig

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 18px; border-bottom:1px solid #1E293B; margin-bottom:16px;'>
        <div style='font-size:2rem; margin-bottom:6px;'>🏥</div>
        <div style='font-size:1rem; font-weight:700; color:#F1F5F9;'>Montanaro OU</div>
        <div style='font-size:0.75rem; color:#64748B;'>Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        options=["📊 Patient Overview",
                 "🔬 Diagnosis Analysis",
                 "🩺 Vitals Analysis",
                 "🤖 Predictive Model",
                 "💡 Exclusion List"],
        label_visibility="visible",
    )

    st.markdown("<hr style='border-color:#1E293B;margin:20px 0 14px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.7rem; color:#475569;'>
        BDA 640 Final Case<br>Hospital OU Operations
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# TOP BAR
# =============================================================================
page_titles = {
    "📊 Patient Overview":  "📊 Patient Population Overview",
    "🔬 Diagnosis Analysis":"🔬 Diagnosis Deep-Dive",
    "🩺 Vitals Analysis":   "🩺 Vital Signs Analysis",
    "🤖 Predictive Model":  "🤖 Predictive Model Results",
    "💡 Exclusion List":    "💡 Exclusion List & Financial Impact",
}
st.markdown(f"""
<div style='display:flex; align-items:center; justify-content:space-between;
     background:rgba(17,24,39,0.96); border-bottom:1px solid #1E293B;
     padding:14px 4px; margin-bottom:22px;'>
  <div style='font-size:1.1rem; font-weight:700; color:#F1F5F9;'>{page_titles[page]}</div>
  <div style='display:flex; align-items:center; gap:10px;'>
    <span class="live-dot"></span>
    <span style='color:#94A3B8; font-size:0.78rem;'>Live</span>
    <span class="topbar-badge">BDA 640</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# ── PAGE 1: PATIENT OVERVIEW ──
# =============================================================================
if page == "📊 Patient Overview":

    # KPI Row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("👥 Total Patients",  f"{total_patients:,}",    "Medicine service")
    c2.metric("🔄 Flip Rate",       f"{flip_rate:.1f}%",      f"{flip_n} flipped")
    c3.metric("⏱ Avg LOS (All)",   f"{avg_los_all:.1f}h",    "Mean OU stay")
    c4.metric("📈 LOS (Flipped)",   f"{avg_los_flipped:.1f}h","Converted pts")
    c5.metric("📉 LOS (Stayed)",    f"{avg_los_stayed:.1f}h", "Obs discharge")
    c6.metric("🧑‍⚕ Mean Age",        f"{avg_age:.0f}",         "Patient population")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters + Charts
    col_filt, col_charts = st.columns([1, 4])

    with col_filt:
        st.markdown("**🎛 Filters**")
        ins_options = df["InsuranceGroup"].unique().tolist()
        ins_vals    = st.multiselect("Insurance", ins_options, default=ins_options)
        gender_vals = st.multiselect("Gender", ["Male","Female"], default=["Male","Female"])
        age_range   = st.slider("Age Range", 19, 89, (19, 89))

    # Apply filters
    dff = df[
        (df["InsuranceGroup"].isin(ins_vals)) &
        (df["Gender"].isin(gender_vals)) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]

    with col_charts:
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.markdown("**📈 Age Distribution**")
            st.plotly_chart(make_age_hist(dff), use_container_width=True, config={"displayModeBar":False})
        with r1c2:
            st.markdown("**🏥 Flip Rate by Insurance**")
            st.plotly_chart(make_ins_bar(dff), use_container_width=True, config={"displayModeBar":False})

        st.markdown("**🎻 OU Length of Stay: Flipped vs. Stayed**")
        st.plotly_chart(make_los(dff), use_container_width=True, config={"displayModeBar":False})

# =============================================================================
# ── PAGE 2: DIAGNOSIS ANALYSIS ──
# =============================================================================
elif page == "🔬 Diagnosis Analysis":

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Flip Rate by Diagnosis**")
        st.plotly_chart(make_flip_drg(), use_container_width=True, config={"displayModeBar":False})
    with col2:
        st.markdown("**🔥 Flip Rate Heatmap: Diagnosis × Age Group**")
        st.plotly_chart(make_heatmap(), use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown("**📦 LOS Distribution — Select Diagnosis to Filter**")
    sel = st.selectbox("Select a diagnosis", ["All Diagnoses"] + sorted(df["DiagnosisName"].dropna().unique().tolist()))
    dff = df if sel == "All Diagnoses" else df[df["DiagnosisName"] == sel]
    st.plotly_chart(make_drg_box(dff, f"LOS — {sel}"), use_container_width=True, config={"displayModeBar":False})

# =============================================================================
# ── PAGE 3: VITALS ANALYSIS ──
# =============================================================================
elif page == "🩺 Vitals Analysis":

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("**🕸 Vitals Radar: Flipped vs. Stayed**")
        st.plotly_chart(make_radar(), use_container_width=True, config={"displayModeBar":False})

    with col2:
        st.markdown("**📋 Mean Vitals Comparison**")
        vitals_info = [
            ("BloodPressureUpper","Systolic BP","mmHg"),
            ("BloodPressureLower","Diastolic BP","mmHg"),
            ("Pulse","Heart Rate","bpm"),
            ("PulseOximetry","O2 Saturation","%"),
            ("Respirations","Respirations","br/min"),
            ("Temperature","Temperature","F"),
        ]
        for col_name, lbl, unit in vitals_info:
            sm   = round(df[df["Flipped"]==0][col_name].mean(), 1)
            fm   = round(df[df["Flipped"]==1][col_name].mean(), 1)
            diff = round(fm - sm, 1)
            arrow = "↑" if diff > 0 else "↓"
            color = "#F87171" if diff > 0 else "#34D399"
            pct   = min(fm / df[col_name].max() * 100, 100)
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid #1E293B;
                 border-radius:10px;padding:10px 14px;margin-bottom:7px;'>
              <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                <span style='color:#F1F5F9;font-size:0.83rem;font-weight:600;'>{lbl}</span>
                <span style='color:{color};font-size:0.73rem;font-weight:600;'>{arrow} {abs(diff)} {unit}</span>
              </div>
              <div style='font-size:0.73rem;color:#94A3B8;'>
                Stayed: {sm} &nbsp;·&nbsp; Flipped: {fm}
              </div>
              <div style='background:rgba(255,255,255,0.06);border-radius:4px;height:5px;
                   overflow:hidden;margin-top:5px;'>
                <div style='width:{pct:.0f}%;height:100%;border-radius:4px;
                     background:linear-gradient(90deg,{color},{color}88);'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🚨 Abnormal Vital Flags: Flipped vs. Stayed**")
    st.plotly_chart(make_flags(), use_container_width=True, config={"displayModeBar":False})

# =============================================================================
# ── PAGE 4: PREDICTIVE MODEL ──
# =============================================================================
elif page == "🤖 Predictive Model":

    st.markdown("""
    <div class="info-banner">
        ℹ️ &nbsp;Three ML models trained to predict patient flip probability.
        Run <code>python3 predictive_model.py</code> first for full results.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📊 Model Performance (ROC-AUC)**")
        st.plotly_chart(make_model_bar(), use_container_width=True, config={"displayModeBar":False})

    with col2:
        st.markdown("**⚠ Predicted Flip Risk by Diagnosis**")
        st.markdown("""
        <span class="risk-high">High Risk</span>&nbsp;
        <span class="risk-medium">Medium Risk</span>&nbsp;
        <span class="risk-low">Low Risk</span>
        <br><br>
        """, unsafe_allow_html=True)
        st.plotly_chart(make_drg_risk(), use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    st.markdown("**💡 How to Interpret**")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown("**What is ROC-AUC?**")
        st.markdown("<span style='color:#94A3B8;font-size:0.82rem;'>Measures ability to distinguish flippers from non-flippers. 1.0 = perfect, 0.5 = random. Scores >0.70 are clinically meaningful.</span>", unsafe_allow_html=True)
    with ic2:
        st.markdown("**Risk Tiers**")
        st.markdown("<span style='color:#94A3B8;font-size:0.82rem;'>Diagnoses with predicted flip probability >55% are High Risk — prime candidates for the expanded exclusion list.</span>", unsafe_allow_html=True)
    with ic3:
        st.markdown("**Which Model to Trust?**")
        st.markdown("<span style='color:#94A3B8;font-size:0.82rem;'>Gradient Boosting typically achieves highest AUC. Logistic Regression is most interpretable for clinical staff.</span>", unsafe_allow_html=True)

# =============================================================================
# ── PAGE 5: EXCLUSION LIST ──
# =============================================================================
elif page == "💡 Exclusion List":

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("**📋 Current Exclusion List (6 Diagnoses)**")
        current_list = ["Alcohol Intoxication","Alcohol Withdrawal",
                        "Mental Health Disorder","Obstetrics Patients",
                        "Sickle Cell Anemia Crisis","Cerebrovascular Accident (Stroke)"]
        for item in current_list:
            st.markdown(f"""
            <div class='excl-item excl-current'>
                <span style='color:#64748B;'>●</span>
                <span>{item}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>**✅ Data-Supported Additions**", unsafe_allow_html=True)
        additions = [
            ("Congestive Heart Failure","61%+ flip rate"),
            ("Pancreatitis",            "70%+ flip rate"),
            ("Urinary Tract Infection", "66%+ flip rate"),
            ("Pneumonia",               "57%+ flip rate"),
            ("GI Bleeding",             "52%+ flip rate"),
        ]
        for item, badge in additions:
            st.markdown(f"""
            <div class='excl-item excl-add'>
                <span style='color:#10B981;font-weight:700;'>✓</span>
                <span style='flex:1;'>{item}</span>
                <span style='font-size:0.7rem;color:#F59E0B;white-space:nowrap;'>{badge}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**🎯 Capacity Impact Calculator**")
        target_pct = st.slider("Target Flip Rate After Intervention (%)", 10, 45, 20, step=5)

        extra    = max(0, round(current_per_wk*(1-target_pct/100)/(1-0.46)) - current_per_wk)
        extra_yr = extra * 52
        rev      = round(extra_yr * 0.2) * 700

        mc1, mc2, mc3 = st.columns(3)
        mc1.markdown(f"""<div class='impact-box'>
            <div class='impact-num' style='color:#10B981;'>+{extra}/wk</div>
            <div class='impact-label'>Extra patients/week</div>
        </div>""", unsafe_allow_html=True)
        mc2.markdown(f"""<div class='impact-box'>
            <div class='impact-num' style='color:#3B82F6;'>+{extra_yr:,}/yr</div>
            <div class='impact-label'>Additional patients/year</div>
        </div>""", unsafe_allow_html=True)
        mc3.markdown(f"""<div class='impact-box'>
            <div class='impact-num' style='color:#F59E0B;'>${rev:,.0f}</div>
            <div class='impact-label'>Est. revenue gain</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(make_waterfall(target_pct), use_container_width=True, config={"displayModeBar":False})

    st.markdown("---")
    b1, b2, b3, b4 = st.columns(4)
    b1.markdown("""<div class='impact-box'>
        <div style='font-size:1.8rem;margin-bottom:7px;'>🚨</div>
        <div class='impact-num' style='color:#EC4899;'>~1,900</div>
        <div class='impact-label'>LWBS Cases/Year</div>
        <div class='impact-sub'>Leave-without-being-seen</div>
    </div>""", unsafe_allow_html=True)
    b2.markdown("""<div class='impact-box'>
        <div style='font-size:1.8rem;margin-bottom:7px;'>💵</div>
        <div class='impact-num' style='color:#3B82F6;'>$700</div>
        <div class='impact-label'>Revenue per ED Visit</div>
        <div class='impact-sub'>Average reimbursement</div>
    </div>""", unsafe_allow_html=True)
    b3.markdown("""<div class='impact-box'>
        <div style='font-size:1.8rem;margin-bottom:7px;'>📈</div>
        <div class='impact-num' style='color:#10B981;'>$399K+</div>
        <div class='impact-label'>Potential Revenue Gain</div>
        <div class='impact-sub'>30% LWBS reduction via throughput</div>
    </div>""", unsafe_allow_html=True)
    b4.markdown("""<div class='impact-box'>
        <div style='font-size:1.8rem;margin-bottom:7px;'>🏥</div>
        <div class='impact-num' style='color:#F59E0B;'>260-570</div>
        <div class='impact-label'>Extra Patients/Year</div>
        <div class='impact-sub'>Reducing flip rate to 20-33%</div>
    </div>""", unsafe_allow_html=True)
