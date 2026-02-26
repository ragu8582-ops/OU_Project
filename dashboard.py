# =============================================================================
# HOSPITAL OBSERVATION UNIT — Phase 4: Interactive Dashboard
# BDA 640 Final Case Report
# Run: python dashboard.py → open http://127.0.0.1:8050 in browser
#
# Install requirements first:
#   pip install dash plotly pandas numpy scikit-learn
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Dash ──────────────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# =============================================================================
# 0 — DATA LOADING & PREP
# =============================================================================
df = pd.read_csv("OUData_cleaned.csv")

DRG_MAP = {
    276: "Dehydration", 428: "Congestive Heart Failure", 486: "Pneumonia",
    558: "Colitis",     577: "Pancreatitis",             578: "GI Bleeding",
    599: "Urinary Tract Infection", 780: "Syncope",      782: "Edema",
    786: "Chest Pain",  787: "Nausea",                  789: "Abdominal Pain",
}
df["DiagnosisName"] = df["DRG01"].map(DRG_MAP)
df["AgeGroup"]      = pd.cut(df["Age"], bins=[0,40,55,65,75,89],
                             labels=["18–40","41–55","56–65","66–75","76+"])

# Try loading model outputs (if predictive_model.py was run first)
try:
    drg_risk = pd.read_csv("model_outputs/drg_risk_scores.csv")
    model_comparison = pd.read_csv("model_outputs/model_comparison.csv")
    MODEL_RAN = True
except FileNotFoundError:
    MODEL_RAN = False
    drg_risk = (
        df.groupby(["DRG01","DiagnosisName"])["Flipped"]
        .agg(Avg_Prob="mean", Count="count")
        .reset_index()
    )
    drg_risk["Actual_Rate"]  = drg_risk["Avg_Prob"]
    drg_risk["Avg_Prob_Pct"] = (drg_risk["Avg_Prob"] * 100).round(1)

# =============================================================================
# 1 — STYLE CONSTANTS
# =============================================================================
C_RED  = "#E63946"
C_BLUE = "#2196F3"
C_GREEN= "#2ECC71"
C_DARK = "#1D3557"
C_GOLD = "#F4A261"
C_BG   = "#F0F4F8"
FONT   = "Inter, Segoe UI, Arial, sans-serif"

CARD_STYLE = {
    "borderRadius": "12px",
    "padding": "20px 24px",
    "backgroundColor": "white",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
    "marginBottom": "16px",
}

# =============================================================================
# 2 — HELPER FUNCTIONS
# =============================================================================
def kpi_card(title, value, subtitle, color, icon):
    return html.Div([
        html.Div(icon, style={"fontSize":"2rem","marginBottom":"4px"}),
        html.Div(value, style={"fontSize":"2.2rem","fontWeight":"700","color":color,"lineHeight":"1.1"}),
        html.Div(title, style={"fontSize":"0.85rem","color":"#666","fontWeight":"600","textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.Div(subtitle, style={"fontSize":"0.78rem","color":"#999","marginTop":"4px"}),
    ], style={**CARD_STYLE, "textAlign":"center","minWidth":"170px"})

def section_header(title, subtitle=""):
    return html.Div([
        html.H4(title, style={"color":C_DARK,"fontWeight":"700","marginBottom":"4px","margin":"0"}),
        html.P(subtitle, style={"color":"#888","fontSize":"0.85rem","marginTop":"4px","marginBottom":"16px"}),
    ])

# =============================================================================
# 3 — KPI COMPUTATIONS
# =============================================================================
total_patients   = len(df)
flip_n           = df["Flipped"].sum()
flip_rate        = df["Flipped"].mean() * 100
avg_los_all      = df["OU_LOS_hrs"].mean()
avg_los_flipped  = df[df["Flipped"]==1]["OU_LOS_hrs"].mean()
avg_los_stayed   = df[df["Flipped"]==0]["OU_LOS_hrs"].mean()
avg_age          = df["Age"].mean()
current_per_wk   = 44

# Financial impact
LWBS_cases         = 1900
revenue_per_visit  = 700
extra_patients_20  = 570
extra_patients_33  = 260

# =============================================================================
# 4 — CHARTS (pre-built for performance)
# =============================================================================

# ── Chart A: Flip Rate by Diagnosis ──
flip_by_drg = (
    df.groupby("DiagnosisName")["Flipped"]
    .agg(Flipped_Count="sum", Total="count", Flip_Rate="mean")
    .reset_index()
    .sort_values("Flip_Rate", ascending=True)
)
flip_by_drg["Pct"] = (flip_by_drg["Flip_Rate"] * 100).round(1)

def make_flip_by_drg(data):
    fig = go.Figure(go.Bar(
        x=data["Pct"], y=data["DiagnosisName"],
        orientation="h",
        marker=dict(color=data["Pct"],
                    colorscale=[[0,C_BLUE],[0.5,C_GOLD],[1,C_RED]],
                    showscale=True,
                    colorbar=dict(title="Flip %",ticksuffix="%",len=0.8)),
        text=[f"  {p}%  (n={n})" for p,n in zip(data["Pct"],data["Total"])],
        textposition="inside",
        textfont=dict(color="white",size=11),
    ))
    fig.add_vline(x=flip_rate, line_dash="dash", line_color=C_DARK,
                  annotation_text=f"Avg {flip_rate:.1f}%",
                  annotation_font_color=C_DARK)
    fig.update_layout(
        xaxis=dict(title="Flip Rate (%)", ticksuffix="%", range=[0,100]),
        yaxis=dict(title=""),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        margin=dict(l=20,r=20,t=20,b=20),
        height=380,
    )
    return fig

# ── Chart B: LOS violin ──
def make_los_violin():
    fig = go.Figure()
    for label, color, subset in [
        ("Stayed (Observation)", C_BLUE, df[df["Flipped"]==0]),
        ("Flipped (Inpatient)",  C_RED,  df[df["Flipped"]==1]),
    ]:
        fig.add_trace(go.Violin(
            y=subset["OU_LOS_hrs"], name=label, box_visible=True,
            meanline_visible=True, fillcolor=color, opacity=0.6,
            line_color=color, points="outliers",
        ))
    fig.add_hline(y=48, line_dash="dash", line_color=C_DARK,
                  annotation_text="48-hr threshold", annotation_position="top right")
    fig.update_layout(
        yaxis=dict(title="Hours in OU"),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        margin=dict(l=20,r=20,t=20,b=20), height=360,
        legend=dict(orientation="h", y=1.06),
    )
    return fig

# ── Chart C: Heatmap ──
def make_heatmap():
    hm = (
        df.groupby(["DiagnosisName","AgeGroup"],observed=True)["Flipped"]
        .mean().unstack().fillna(0) * 100
    ).round(1)
    fig = go.Figure(go.Heatmap(
        z=hm.values, x=hm.columns.tolist(), y=hm.index.tolist(),
        colorscale=[[0,"#EEF4FF"],[0.5,C_GOLD],[1,C_RED]],
        text=hm.values, texttemplate="%{text:.0f}%",
        textfont=dict(size=10),
        colorbar=dict(title="Flip %", ticksuffix="%"),
        hovertemplate="Dx: %{y}<br>Age: %{x}<br>Flip Rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Age Group"),
        yaxis=dict(title=""),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=11),
        margin=dict(l=20,r=20,t=20,b=20), height=420,
    )
    return fig

# ── Chart D: Vitals radar ──
def make_radar():
    vitals = ["BloodPressureUpper","BloodPressureLower","Pulse","PulseOximetry","Respirations","Temperature"]
    labels = ["Systolic BP","Diastolic BP","Heart Rate","O₂ Saturation","Respirations","Temperature"]
    s_means = df[df["Flipped"]==0][vitals].mean()
    f_means = df[df["Flipped"]==1][vitals].mean()
    vmin, vmax = df[vitals].min(), df[vitals].max()
    s_norm = ((s_means - vmin)/(vmax - vmin)*100).round(1)
    f_norm = ((f_means - vmin)/(vmax - vmin)*100).round(1)

    fig = go.Figure()
    radar_colors = [("Stayed", s_norm, C_BLUE, "rgba(33,150,243,0.15)"),
                    ("Flipped", f_norm, C_RED,  "rgba(230,57,70,0.15)")]
    for name, vals, line_col, fill_col in radar_colors:
        fig.add_trace(go.Scatterpolar(
            r=list(vals)+[vals.iloc[0]], theta=labels+[labels[0]],
            fill="toself", name=name,
            line_color=line_col, fillcolor=fill_col,
            opacity=0.85,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100])),
        paper_bgcolor="white", font=dict(family=FONT, size=12),
        margin=dict(l=40,r=40,t=30,b=40), height=370,
        legend=dict(orientation="h", y=-0.12),
    )
    return fig

# ── Chart E: Capacity waterfall ──
def make_waterfall(target_pct):
    extra = round((current_per_wk * (1 - target_pct/100) / (1 - 0.46)) - current_per_wk)
    extra_year = extra * 52
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","total"],
        x=["Current Baseline", f"Flip Rate → {target_pct}%", "Projected Total"],
        y=[current_per_wk, extra, 0],
        text=[f"44/wk", f"+{extra}/wk\n(+{extra_year}/yr)", f"{current_per_wk+extra}/wk"],
        textposition="outside",
        connector={"line":{"color":"#DDD","dash":"dot"}},
        increasing={"marker":{"color":C_GREEN}},
        totals={"marker":{"color":C_DARK}},
    ))
    fig.update_layout(
        yaxis=dict(title="Medicine Service Patients/Week"),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT, size=12),
        margin=dict(l=20,r=20,t=20,b=30), height=360,
    )
    return fig

# ── Chart F: ROC placeholder or model comparison ──
def make_model_bar():
    def _placeholder(msg):
        fig = go.Figure()
        fig.add_annotation(
            text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color=C_DARK, family=FONT),
            align="center",
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor=C_BG,
            font=dict(family=FONT), height=360,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    if not MODEL_RAN:
        return _placeholder(
            "⚠️ Run <b>python3 predictive_model.py</b> in your terminal,<br>"
            "then <b>restart</b> the dashboard to see model results."
        )
    try:
        mc   = model_comparison.copy()
        # ROC-AUC column may be "0.732" or "0.732 ± 0.01" — extract first number
        aucs = mc["ROC-AUC"].astype(str).str.extract(r"([\d\.]+)")[0].astype(float)
        fig  = go.Figure(go.Bar(
            x=mc["Model"], y=aucs,
            marker_color=[C_BLUE, C_RED, C_GREEN],
            text=[f"AUC = {a:.3f}" for a in aucs],
            textposition="outside", width=0.5,
        ))
        fig.update_layout(
            yaxis=dict(title="ROC-AUC Score", range=[0.5, 1.0]),
            plot_bgcolor=C_BG, paper_bgcolor="white",
            font=dict(family=FONT, size=12),
            margin=dict(l=20, r=20, t=40, b=20), height=360,
        )
        return fig
    except Exception as e:
        return _placeholder(f"Could not load model results.<br>Error: {str(e)}")

# ── Chart G: DRG Risk (from model or actual) ──
def make_drg_risk():
    try:
        data = drg_risk.copy()
        if "Avg_Prob_Pct" not in data.columns:
            data["Avg_Prob_Pct"] = (data["Avg_Prob"] * 100).round(1)
        if "DiagnosisName" not in data.columns and "DRG01" in data.columns:
            data["DiagnosisName"] = data["DRG01"].map(DRG_MAP).fillna(data["DRG01"].astype(str))
        data   = data.sort_values("Avg_Prob_Pct", ascending=True)
        colors = [C_RED if p > 55 else (C_GOLD if p > 35 else C_BLUE) for p in data["Avg_Prob_Pct"]]
        fig    = go.Figure(go.Bar(
            x=data["Avg_Prob_Pct"], y=data["DiagnosisName"],
            orientation="h", marker_color=colors,
            text=[f"  {p:.0f}%  " for p in data["Avg_Prob_Pct"]],
            textposition="inside", textfont=dict(color="white", size=11),
        ))
        fig.add_vline(x=55, line_dash="dash", line_color=C_DARK,
                      annotation_text="Exclusion threshold (55%)",
                      annotation_position="top right")
        fig.update_layout(
            xaxis=dict(title="Flip Probability (%)", ticksuffix="%", range=[0, 100]),
            yaxis=dict(title=""),
            plot_bgcolor=C_BG, paper_bgcolor="white",
            font=dict(family=FONT, size=12),
            margin=dict(l=20, r=20, t=20, b=20), height=400,
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Chart error: {str(e)}", xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color=C_RED, family=FONT), align="center",
        )
        fig.update_layout(paper_bgcolor="white", height=400,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig

# =============================================================================
# 5 — APP LAYOUT
# =============================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Montanaro OU Analytics",
    suppress_callback_exceptions=True,
)

# ── Header ──
header = html.Div([
    html.Div([
        html.Div("🏥", style={"fontSize":"2.5rem"}),
        html.Div([
            html.H2("Montanaro Hospital — Observation Unit Analytics",
                    style={"color":"white","fontWeight":"700","margin":"0","fontSize":"1.5rem"}),
            html.P("BDA 640 Final Case | Data-Driven Approach to Improving OU Operations",
                   style={"color":"rgba(255,255,255,0.75)","margin":"0","fontSize":"0.85rem"}),
        ]),
    ], style={"display":"flex","alignItems":"center","gap":"16px"}),
], style={
    "background":f"linear-gradient(135deg, {C_DARK} 0%, #2D6A9F 100%)",
    "padding":"20px 32px","marginBottom":"0",
})

# ── KPI Row ──
kpi_row = html.Div([
    kpi_card("Total Patients",    f"{total_patients:,}",   "Medicine service patients", C_DARK, "👥"),
    kpi_card("Flip Rate",         f"{flip_rate:.1f}%",    f"{flip_n} of {total_patients} flipped", C_RED,  "🔄"),
    kpi_card("Avg LOS (All)",     f"{avg_los_all:.1f} hrs","Mean OU length of stay",   C_BLUE, "⏱️"),
    kpi_card("Avg LOS (Flipped)", f"{avg_los_flipped:.1f} hrs","Flipped patients",     C_GOLD, "📈"),
    kpi_card("Avg LOS (Stayed)",  f"{avg_los_stayed:.1f} hrs","Stayed in OU",         C_GREEN,"📉"),
    kpi_card("Avg Patient Age",   f"{avg_age:.0f} yrs",   "Mean age of OU patients",  C_DARK, "🧑‍⚕️"),
], style={
    "display":"flex","gap":"12px","flexWrap":"wrap",
    "padding":"20px 32px","backgroundColor":"#EEF4FF",
    "borderBottom":"2px solid #DDE4EE",
})

# ── Tabs ──
tab_style = {
    "borderRadius":"8px 8px 0 0",
    "fontWeight":"600","fontSize":"0.9rem",
    "color":"#555","padding":"10px 20px",
}
selected_tab_style = {**tab_style, "backgroundColor":C_DARK,"color":"white",
                      "borderTop":f"3px solid {C_GOLD}"}

tabs = dcc.Tabs(id="tabs", value="tab-overview", children=[
    dcc.Tab(label="📊 Patient Overview",       value="tab-overview",
            style=tab_style, selected_style=selected_tab_style),
    dcc.Tab(label="🔬 Diagnosis Deep-Dive",    value="tab-diagnosis",
            style=tab_style, selected_style=selected_tab_style),
    dcc.Tab(label="🩺 Vitals Analysis",        value="tab-vitals",
            style=tab_style, selected_style=selected_tab_style),
    dcc.Tab(label="🤖 Predictive Model",       value="tab-model",
            style=tab_style, selected_style=selected_tab_style),
    dcc.Tab(label="💡 Exclusion List & Impact",value="tab-recommend",
            style=tab_style, selected_style=selected_tab_style),
], style={"padding":"0 32px","backgroundColor":"#EEF4FF"})

app.layout = html.Div([
    header,
    kpi_row,
    tabs,
    html.Div(id="tab-content", style={"padding":"24px 32px","backgroundColor":"white","minHeight":"80vh"}),
], style={"fontFamily":FONT,"backgroundColor":"white","maxWidth":"1600px","margin":"0 auto"})

# =============================================================================
# 6 — TAB CALLBACKS
# =============================================================================

@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):

    # ── TAB 1: Patient Overview ──────────────────────────────────────────────
    if tab == "tab-overview":
        return html.Div([
            section_header("Patient Population Overview",
                           "Distribution of demographics and outcomes across all OU patients"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Filters", style={"fontWeight":"700","color":C_DARK}),
                        html.Label("Insurance Category", style={"fontSize":"0.82rem","color":"#666"}),
                        dcc.Checklist(
                            id="filter-insurance",
                            options=[{"label":f"  {i}","value":i} for i in df["InsuranceGroup"].unique()],
                            value=df["InsuranceGroup"].unique().tolist(),
                            labelStyle={"display":"block","fontSize":"0.85rem","marginBottom":"4px"},
                        ),
                        html.Hr(),
                        html.Label("Gender", style={"fontSize":"0.82rem","color":"#666"}),
                        dcc.Checklist(
                            id="filter-gender",
                            options=[{"label":f"  {g}","value":g} for g in ["Male","Female"]],
                            value=["Male","Female"],
                            labelStyle={"display":"block","fontSize":"0.85rem","marginBottom":"4px"},
                        ),
                        html.Hr(),
                        html.Label("Age Range", style={"fontSize":"0.82rem","color":"#666"}),
                        dcc.RangeSlider(
                            id="filter-age",
                            min=19, max=89, step=1,
                            value=[19,89],
                            marks={i:str(i) for i in range(20,90,10)},
                            tooltip={"placement":"bottom","always_visible":True},
                        ),
                    ], style={**CARD_STYLE,"backgroundColor":"#F8FBFF"}),
                ], width=3),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Age Distribution by Outcome", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                                dcc.Graph(id="chart-age-hist", config={"displayModeBar":False}),
                            ], style=CARD_STYLE),
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.H6("Outcome by Insurance Category", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                                dcc.Graph(id="chart-ins-bar", config={"displayModeBar":False}),
                            ], style=CARD_STYLE),
                        ], width=6),
                    ]),
                    html.Div([
                        html.H6("OU Length of Stay Distribution", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(id="chart-los", figure=make_los_violin(), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=9),
            ]),
        ])

    # ── TAB 2: Diagnosis Deep-Dive ───────────────────────────────────────────
    elif tab == "tab-diagnosis":
        return html.Div([
            section_header("Diagnosis-Level Analysis",
                           "Flip rate and length of stay broken down by DRG diagnosis code"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Flip Rate by Diagnosis", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(figure=make_flip_by_drg(flip_by_drg), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H6("Flip Rate Heatmap: Diagnosis × Age Group",
                                style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(figure=make_heatmap(), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=6),
            ]),
            html.Div([
                html.H6("LOS by Diagnosis — Click a Diagnosis to Explore",
                        style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                dcc.Dropdown(
                    id="drg-selector",
                    options=[{"label":v,"value":v} for v in sorted(df["DiagnosisName"].unique())],
                    value=None,
                    placeholder="Select a diagnosis...",
                    style={"marginBottom":"12px","maxWidth":"400px"},
                ),
                dcc.Graph(id="chart-drg-detail", config={"displayModeBar":False}),
            ], style=CARD_STYLE),
        ])

    # ── TAB 3: Vitals Analysis ───────────────────────────────────────────────
    elif tab == "tab-vitals":
        vitals_actual = pd.DataFrame({
            "Vital Sign":       ["Systolic BP","Diastolic BP","Heart Rate","O₂ Saturation","Respirations","Temperature"],
            "Stayed – Mean":    df[df["Flipped"]==0][["BloodPressureUpper","BloodPressureLower","Pulse","PulseOximetry","Respirations","Temperature"]].mean().round(1).values,
            "Flipped – Mean":   df[df["Flipped"]==1][["BloodPressureUpper","BloodPressureLower","Pulse","PulseOximetry","Respirations","Temperature"]].mean().round(1).values,
        })
        vitals_actual["Δ (Flipped−Stayed)"] = (vitals_actual["Flipped – Mean"] - vitals_actual["Stayed – Mean"]).round(1)
        vitals_actual["Direction"] = vitals_actual["Δ (Flipped−Stayed)"].apply(lambda x: "↑ Higher" if x>0 else "↓ Lower")

        tbl = go.Figure(go.Table(
            header=dict(
                values=["<b>"+c+"</b>" for c in vitals_actual.columns],
                fill_color=C_DARK, font=dict(color="white",size=12,family=FONT),
                align="center", height=34,
            ),
            cells=dict(
                values=[vitals_actual[c] for c in vitals_actual.columns],
                fill_color=[["white","#F8FBFF"]*6],
                align=["left","center","center","center","center"],
                font=dict(size=12,family=FONT), height=30,
            )
        ))
        tbl.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=280, paper_bgcolor="white")

        flag_cols = ["Flag_Tachycardia","Flag_Hypo_O2","Flag_Fever","Flag_Tachypnea","Flag_Hypertension","Flag_Hypotension"]
        flag_labels = ["Tachycardia\n(HR>100)","Low O₂\n(SpO₂<92%)","Fever\n(>100.4°F)","Tachypnea\n(RR>20)","Hypertension\n(SBP>140)","Hypotension\n(SBP<90)"]
        flag_flipped = df[df["Flipped"]==1][flag_cols].mean()*100
        flag_stayed  = df[df["Flipped"]==0][flag_cols].mean()*100

        fig_flags = go.Figure()
        fig_flags.add_trace(go.Bar(name="Stayed", x=flag_labels, y=flag_stayed.round(1),
                                    marker_color=C_BLUE, opacity=0.8))
        fig_flags.add_trace(go.Bar(name="Flipped", x=flag_labels, y=flag_flipped.round(1),
                                    marker_color=C_RED, opacity=0.8))
        fig_flags.update_layout(
            barmode="group", yaxis=dict(title="% of Patients with Flag",ticksuffix="%"),
            plot_bgcolor=C_BG, paper_bgcolor="white",
            font=dict(family=FONT,size=12),
            legend=dict(orientation="h",y=1.06),
            margin=dict(l=20,r=20,t=20,b=20), height=340,
        )

        return html.Div([
            section_header("Vital Signs Analysis",
                           "Comparing vitals between patients who stayed vs. were converted to inpatient"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Vital Signs Radar Profile", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(figure=make_radar(), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=5),
                dbc.Col([
                    html.Div([
                        html.H6("Mean Vitals Comparison Table", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(figure=tbl, config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=7),
            ]),
            html.Div([
                html.H6("Abnormal Vital Flags: Flipped vs. Stayed",
                        style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                dcc.Graph(figure=fig_flags, config={"displayModeBar":False}),
            ], style=CARD_STYLE),
        ])

    # ── TAB 4: Predictive Model ──────────────────────────────────────────────
    elif tab == "tab-model":
        if not MODEL_RAN:
            note = html.Div([
                html.H5("⚠️  Run predictive_model.py first", style={"color":C_GOLD}),
                html.P("Execute: python predictive_model.py  — then refresh this page to see model results.",
                       style={"color":"#666"}),
            ], style={**CARD_STYLE, "borderLeft":f"4px solid {C_GOLD}"})
        else:
            note = html.Div()

        return html.Div([
            section_header("Predictive Model Results",
                           "Three models trained to predict which OU patients will flip to Inpatient status"),
            note,
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Model Performance (ROC-AUC)", style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        dcc.Graph(figure=make_model_bar(), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H6("Predicted Flip Risk by Diagnosis",
                                style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                        html.P("Red = High Risk (>55%) → Recommend exclusion list",
                               style={"fontSize":"0.8rem","color":"#888","marginBottom":"8px"}),
                        dcc.Graph(figure=make_drg_risk(), config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=6),
            ]),
            html.Div([
                html.H6("💡 How to Read These Results",
                        style={"fontWeight":"700","color":C_DARK,"marginBottom":"8px"}),
                html.P(
                    "The predictive models use patient demographics, vital signs, and initial diagnosis to estimate the probability "
                    "that a patient will be converted from Observation to Inpatient status. "
                    "A diagnosis with a predicted flip probability >55% is a candidate for the OU Exclusion List — "
                    "meaning these patients should be admitted directly to inpatient beds rather than the OU. "
                    "ROC-AUC scores above 0.70 indicate good discriminatory power.",
                    style={"color":"#555","lineHeight":"1.7","marginBottom":"0"},
                ),
            ], style={**CARD_STYLE,"borderLeft":f"4px solid {C_BLUE}"}),
        ])

    # ── TAB 5: Exclusion List & Impact ──────────────────────────────────────
    elif tab == "tab-recommend":
        return html.Div([
            section_header("Exclusion List Recommendations & Financial Impact",
                           "Data-driven recommendations for updating the OU exclusion list"),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("📋 Current Exclusion List (6 diagnoses)",
                                style={"fontWeight":"700","color":C_DARK,"marginBottom":"10px"}),
                        html.Ol([html.Li(item, style={"marginBottom":"6px","color":"#555"}) for item in [
                            "Alcohol Intoxication",
                            "Alcohol Withdrawal",
                            "Mental Health Disorder w/ Behavioral Disturbance or Suicidality",
                            "Obstetrics Patients",
                            "Sickle Cell Anemia Crisis",
                            "Cerebrovascular Accident (Stroke)",
                        ]], style={"fontSize":"0.88rem","lineHeight":"1.8"}),
                    ], style={**CARD_STYLE,"borderLeft":f"4px solid #888"}),
                    html.Div([
                        html.H6("✅ Proposed Additions (Data-Supported)",
                                style={"fontWeight":"700","color":C_GREEN,"marginBottom":"10px"}),
                        html.Ol([html.Li(item, style={"marginBottom":"6px","color":"#555"}) for item in [
                            "Congestive Heart Failure (DRG 428) — 63%+ flip rate",
                            "Pneumonia (DRG 486) — 55%+ flip rate",
                            "Pancreatitis (DRG 577) — 52%+ flip rate",
                            "GI Bleeding (DRG 578) — 50%+ flip rate",
                            "Edema (DRG 782) — 48%+ flip rate",
                        ]], style={"fontSize":"0.88rem","lineHeight":"1.8"}),
                        html.P("★ CHF, Pancreatitis, UTI males, GI Bleed, Pneumonia were also in the interdisciplinary team's list.",
                               style={"fontSize":"0.78rem","color":"#888","marginTop":"8px"}),
                    ], style={**CARD_STYLE,"borderLeft":f"4px solid {C_GREEN}"}),
                ], width=5),

                dbc.Col([
                    html.Div([
                        html.H6("📊 Capacity Impact Calculator",
                                style={"fontWeight":"700","color":C_DARK,"marginBottom":"12px"}),
                        html.Label("Target Flip Rate After Intervention (%)",
                                   style={"fontSize":"0.85rem","color":"#666","fontWeight":"600"}),
                        dcc.Slider(
                            id="flip-target-slider",
                            min=10, max=45, step=5,
                            value=20,
                            marks={i:f"{i}%" for i in range(10,50,5)},
                            tooltip={"placement":"bottom","always_visible":True},
                        ),
                        html.Div(id="impact-output", style={"marginTop":"16px"}),
                        dcc.Graph(id="waterfall-chart", config={"displayModeBar":False}),
                    ], style=CARD_STYLE),
                ], width=7),
            ]),

            html.Div([
                html.H6("💰 Financial Impact Summary",
                        style={"fontWeight":"700","color":C_DARK,"marginBottom":"12px"}),
                dbc.Row([
                    dbc.Col(kpi_card("LWBS Cases/Year","~1,900","Leave-without-being-seen",C_RED,"🚨"), width=3),
                    dbc.Col(kpi_card("Revenue/ED Visit","$700","Avg per emergency visit",C_BLUE,"💵"), width=3),
                    dbc.Col(kpi_card("Potential Revenue","$399K+","If LWBS reduced by 30% via OU throughput",C_GREEN,"📈"), width=3),
                    dbc.Col(kpi_card("Extra Patients/yr","260–570","By reducing flip rate to 20–33%",C_GOLD,"🏥"), width=3),
                ], style={"gap":"0"}),
            ], style=CARD_STYLE),
        ])

    return html.Div("Tab not found")

# =============================================================================
# 7 — INTERACTIVE CALLBACKS
# =============================================================================

@app.callback(
    [Output("chart-age-hist","figure"),
     Output("chart-ins-bar","figure"),
     Output("chart-los","figure")],
    [Input("filter-insurance","value"),
     Input("filter-gender","value"),
     Input("filter-age","value")],
)
def update_overview(insurance_vals, gender_vals, age_range):
    dff = df[
        (df["InsuranceGroup"].isin(insurance_vals)) &
        (df["Gender"].isin(gender_vals)) &
        (df["Age"] >= age_range[0]) &
        (df["Age"] <= age_range[1])
    ]
    if len(dff) == 0:
        empty = go.Figure()
        return empty, empty, empty

    # Age histogram
    fig_age = go.Figure()
    for label, color, subset in [("Stayed",C_BLUE,dff[dff["Flipped"]==0]),
                                  ("Flipped",C_RED, dff[dff["Flipped"]==1])]:
        fig_age.add_trace(go.Histogram(
            x=subset["Age"], name=label, opacity=0.7,
            marker_color=color, nbinsx=18, histnorm="percent",
        ))
    fig_age.update_layout(
        barmode="overlay",
        xaxis=dict(title="Age"), yaxis=dict(title="%",ticksuffix="%"),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT,size=11),
        legend=dict(orientation="h",y=1.06),
        margin=dict(l=20,r=20,t=20,b=20), height=280,
    )

    # Insurance bar
    ins_flip = dff.groupby("InsuranceGroup")["Flipped"].agg(["mean","count"]).reset_index()
    ins_flip["Pct"] = (ins_flip["mean"]*100).round(1)
    fig_ins = go.Figure(go.Bar(
        x=ins_flip["InsuranceGroup"], y=ins_flip["Pct"],
        marker_color=[C_RED,C_BLUE,C_GREEN][:len(ins_flip)],
        text=[f"{p}%" for p in ins_flip["Pct"]], textposition="outside",
        width=0.45,
    ))
    fig_ins.update_layout(
        yaxis=dict(title="Flip Rate",ticksuffix="%",range=[0,70]),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT,size=11),
        margin=dict(l=20,r=20,t=20,b=20), height=280,
    )

    # LOS violin
    fig_los = go.Figure()
    for label, color, subset in [("Stayed",C_BLUE,dff[dff["Flipped"]==0]),
                                  ("Flipped",C_RED, dff[dff["Flipped"]==1])]:
        if len(subset) > 0:
            fig_los.add_trace(go.Violin(
                y=subset["OU_LOS_hrs"], name=label, box_visible=True,
                meanline_visible=True, fillcolor=color, opacity=0.6,
                line_color=color, points="outliers",
            ))
    fig_los.add_hline(y=48, line_dash="dash", line_color=C_DARK)
    fig_los.update_layout(
        yaxis=dict(title="Hours in OU"),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT,size=11),
        legend=dict(orientation="h",y=1.06),
        margin=dict(l=20,r=20,t=20,b=20), height=300,
    )

    return fig_age, fig_ins, fig_los


@app.callback(
    Output("chart-drg-detail","figure"),
    Input("drg-selector","value"),
)
def update_drg_detail(selected_drg):
    dff = df if not selected_drg else df[df["DiagnosisName"]==selected_drg]
    fig = go.Figure()
    for label, color, subset in [("Stayed",C_BLUE,dff[dff["Flipped"]==0]),
                                  ("Flipped",C_RED, dff[dff["Flipped"]==1])]:
        if len(subset) > 0:
            fig.add_trace(go.Box(
                y=subset["OU_LOS_hrs"], name=label, boxmean="sd",
                marker_color=color, line_color=color,
            ))
    fig.add_hline(y=48, line_dash="dash", line_color=C_DARK,
                  annotation_text="48-hr target")
    fig.update_layout(
        yaxis=dict(title="Hours in OU"),
        plot_bgcolor=C_BG, paper_bgcolor="white",
        font=dict(family=FONT,size=12),
        margin=dict(l=20,r=20,t=20,b=20), height=340,
        title=dict(text=f"LOS — {selected_drg or 'All Diagnoses'}",
                   font=dict(size=14,color=C_DARK)),
    )
    return fig


@app.callback(
    [Output("waterfall-chart","figure"),
     Output("impact-output","children")],
    Input("flip-target-slider","value"),
)
def update_waterfall(target_pct):
    current = 44
    extra   = max(0, round(current * (1 - target_pct/100) / (1 - 0.46)) - current)
    extra_yr = extra * 52
    lwbs_reduction = round(extra_yr * 0.2)
    revenue_gain   = lwbs_reduction * 700

    fig = make_waterfall(target_pct)

    impact_cards = html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(f"+{extra}/wk", style={"fontSize":"1.8rem","fontWeight":"700","color":C_GREEN}),
                html.Div("Extra patients/week", style={"fontSize":"0.78rem","color":"#666"}),
            ], style={**CARD_STYLE,"textAlign":"center","padding":"12px"}), width=4),
            dbc.Col(html.Div([
                html.Div(f"+{extra_yr:,}/yr", style={"fontSize":"1.8rem","fontWeight":"700","color":C_BLUE}),
                html.Div("Additional patients/year", style={"fontSize":"0.78rem","color":"#666"}),
            ], style={**CARD_STYLE,"textAlign":"center","padding":"12px"}), width=4),
            dbc.Col(html.Div([
                html.Div(f"${revenue_gain:,.0f}", style={"fontSize":"1.8rem","fontWeight":"700","color":C_GOLD}),
                html.Div("Est. revenue gain (LWBS reduction)", style={"fontSize":"0.78rem","color":"#666"}),
            ], style={**CARD_STYLE,"textAlign":"center","padding":"12px"}), width=4),
        ]),
    ])
    return fig, impact_cards


# =============================================================================
# 8 — RUN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("🏥 Montanaro OU Analytics Dashboard")
    print("="*60)
    print("→ Opening at: http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop")
    print("="*60)
    app.run(debug=True, port=8050)
