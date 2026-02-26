# =============================================================================
# HOSPITAL OBSERVATION UNIT — Phase 3: Predictive Modeling
# BDA 640 Final Case Report
# Models: Logistic Regression, Random Forest, Gradient Boosting
# Goal: Predict which patients will "flip" from Observation → Inpatient
# =============================================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)

os.makedirs("charts", exist_ok=True)
os.makedirs("model_outputs", exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────
COLOR_FLIPPED = "#E63946"
COLOR_STAYED  = "#2196F3"
COLOR_GB      = "#2ECC71"
FONT          = "Inter, Arial, sans-serif"
COLOR_BG      = "#F8F9FA"

# =============================================================================
# STEP 1 — Load & Prepare Features
# =============================================================================
df = pd.read_csv("OUData_cleaned.csv")

# Feature set (available at time of OU admission)
FEATURES = [
    "Age",
    "GenderBinary",                 # 1=Male, 0=Female
    "BloodPressureUpper",
    "BloodPressureLower",
    "BloodPressureDiff",
    "Pulse",
    "PulseOximetry",
    "Respirations",
    "Temperature",
    "AbnormalVitalCount",
    "Flag_Tachycardia",
    "Flag_Hypo_O2",
    "Flag_Fever",
    "Flag_Tachypnea",
    "Flag_Hypertension",
    "Flag_Hypotension",
]

# One-hot encode Insurance and DRG
df_model = df.copy()

# Insurance dummies
ins_dummies = pd.get_dummies(df_model["InsuranceGroup"], prefix="Ins", drop_first=True)
df_model = pd.concat([df_model, ins_dummies], axis=1)
FEATURES += list(ins_dummies.columns)

# DRG dummies
drg_dummies = pd.get_dummies(df_model["DRG01"], prefix="DRG", drop_first=True)
df_model = pd.concat([df_model, drg_dummies], axis=1)
FEATURES += list(drg_dummies.columns)

X = df_model[FEATURES].astype(float)
y = df_model["Flipped"]

print(f"✓ Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
print(f"  Class balance — Flipped: {y.mean()*100:.1f}%  |  Stayed: {(1-y.mean())*100:.1f}%")

# =============================================================================
# STEP 2 — Train / Test Split (stratified 80/20)
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features (required for Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"✓ Train set: {len(X_train)} | Test set: {len(X_test)}")

# =============================================================================
# STEP 3 — Train Three Models
# =============================================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   random_state=42, class_weight="balanced"),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                        learning_rate=0.05, random_state=42),
}

results    = {}
thresholds = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold (maximize F1 for positive class)
    precisions, recalls, thresh = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    opt_idx   = np.argmax(f1_scores)
    opt_thr   = thresh[opt_idx] if opt_idx < len(thresh) else 0.5
    thresholds[name] = opt_thr

    y_pred = (y_prob >= opt_thr).astype(int)

    auc   = roc_auc_score(y_test, y_prob)
    ap    = average_precision_score(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)
    cr    = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validation AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if name == "Logistic Regression":
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="roc_auc")
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    results[name] = {
        "model":      model,
        "y_prob":     y_prob,
        "y_pred":     y_pred,
        "auc":        auc,
        "ap":         ap,
        "cm":         cm,
        "cr":         cr,
        "cv_mean":    cv_scores.mean(),
        "cv_std":     cv_scores.std(),
        "threshold":  opt_thr,
    }
    print(f"✓ {name:<25} AUC={auc:.3f}  CV-AUC={cv_scores.mean():.3f}±{cv_scores.std():.3f}  Threshold={opt_thr:.2f}")

# =============================================================================
# STEP 4 — Model Comparison Summary Table
# =============================================================================
summary_rows = []
for name, r in results.items():
    cr = r["cr"]
    summary_rows.append({
        "Model":           name,
        "ROC-AUC":         f"{r['auc']:.3f}",
        "CV-AUC (5-fold)": f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}",
        "Precision (Flip)":f"{cr['1']['precision']:.3f}",
        "Recall (Flip)":   f"{cr['1']['recall']:.3f}",
        "F1 (Flip)":       f"{cr['1']['f1-score']:.3f}",
        "Accuracy":        f"{cr['accuracy']:.3f}",
        "Threshold":       f"{r['threshold']:.2f}",
    })

summary_df = pd.DataFrame(summary_rows)

fig_table = go.Figure(go.Table(
    header=dict(
        values=[f"<b>{c}</b>" for c in summary_df.columns],
        fill_color="#1D3557",
        font=dict(color="white", size=12, family=FONT),
        align="center", height=35,
    ),
    cells=dict(
        values=[summary_df[c] for c in summary_df.columns],
        fill_color=[["white", "#EEF4FF", "white"]],
        align="center",
        font=dict(size=12, family=FONT),
        height=32,
    )
))
fig_table.update_layout(
    title=dict(text="<b>Model Performance Comparison</b>"),
    height=250,
    paper_bgcolor="white",
    font=dict(family=FONT),
)
fig_table.write_html("charts/13_model_comparison_table.html")
print("✓ Chart 13: Model Comparison Table")

# =============================================================================
# STEP 5 — ROC Curves (all 3 models)
# =============================================================================
fig_roc = go.Figure()
colors = [COLOR_STAYED, COLOR_FLIPPED, COLOR_GB]
for (name, r), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f"{name} (AUC={r['auc']:.3f})",
        mode="lines",
        line=dict(color=color, width=2.5),
    ))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    line=dict(dash="dash", color="gray", width=1),
    name="Random Classifier",
    showlegend=True,
))
fig_roc.update_layout(
    title=dict(text="<b>ROC Curves — All Models</b><br><sup>Higher AUC = better discrimination of flippers vs. non-flippers</sup>"),
    xaxis=dict(title="False Positive Rate", range=[0, 1]),
    yaxis=dict(title="True Positive Rate", range=[0, 1.02]),
    height=500,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    legend=dict(x=0.55, y=0.05, bgcolor="white", bordercolor="#DDD", borderwidth=1),
)
fig_roc.write_html("charts/14_roc_curves.html")
print("✓ Chart 14: ROC Curves")

# =============================================================================
# STEP 6 — Confusion Matrices (best model = Gradient Boosting or Random Forest)
# =============================================================================
best_name = max(results, key=lambda n: results[n]["auc"])
best_r    = results[best_name]

fig_cm = make_subplots(rows=1, cols=3, subplot_titles=list(results.keys()))
for col, (name, r) in enumerate(results.items(), 1):
    cm = r["cm"]
    labels = [["TN", "FP"], ["FN", "TP"]]
    annotations = [[f"<b>{v}</b><br>{l}" for v, l in zip(row_v, row_l)]
                   for row_v, row_l in zip(cm, labels)]
    fig_cm.add_trace(
        go.Heatmap(
            z=cm,
            x=["Predicted: Stayed", "Predicted: Flipped"],
            y=["Actual: Stayed", "Actual: Flipped"],
            colorscale=[[0, "#EEF4FF"], [1, COLOR_FLIPPED]],
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>",
            showscale=(col == 3),
            hovertemplate="Count: %{text}<extra></extra>",
        ),
        row=1, col=col
    )
fig_cm.update_layout(
    title=dict(text="<b>Confusion Matrices — All Models</b><br><sup>At optimized decision threshold</sup>"),
    height=420,
    paper_bgcolor="white",
    font=dict(family=FONT),
)
fig_cm.write_html("charts/15_confusion_matrices.html")
print("✓ Chart 15: Confusion Matrices")

# =============================================================================
# STEP 7 — Feature Importance (Random Forest + Gradient Boosting)
# =============================================================================
for model_name in ["Random Forest", "Gradient Boosting"]:
    model_obj  = results[model_name]["model"]
    importance = pd.Series(model_obj.feature_importances_, index=FEATURES)
    top20      = importance.nlargest(20).sort_values(ascending=True)

    # Map DRG feature names to diagnosis names
    DRG_MAP = {
        276: "Dehydration", 428: "Congestive Heart Failure", 486: "Pneumonia",
        558: "Colitis", 577: "Pancreatitis", 578: "GI Bleeding",
        599: "Urinary Tract Infection", 780: "Syncope", 782: "Edema",
        786: "Chest Pain", 787: "Nausea", 789: "Abdominal Pain",
    }
    nice_labels = []
    for feat in top20.index:
        if feat.startswith("DRG_"):
            code = int(feat.split("_")[1])
            nice_labels.append(f"DRG: {DRG_MAP.get(code, feat)}")
        elif feat.startswith("Ins_"):
            nice_labels.append(f"Insurance: {feat.split('_')[1]}")
        else:
            label_map = {
                "Age": "Patient Age",
                "GenderBinary": "Gender (Male)",
                "BloodPressureUpper": "Systolic BP",
                "BloodPressureLower": "Diastolic BP",
                "BloodPressureDiff": "Pulse Pressure",
                "Pulse": "Heart Rate",
                "PulseOximetry": "O₂ Saturation",
                "Respirations": "Respiratory Rate",
                "Temperature": "Temperature",
                "AbnormalVitalCount": "# Abnormal Vitals",
                "Flag_Tachycardia": "Flag: Tachycardia",
                "Flag_Hypo_O2": "Flag: Low O₂",
                "Flag_Fever": "Flag: Fever",
                "Flag_Tachypnea": "Flag: Tachypnea",
                "Flag_Hypertension": "Flag: Hypertension",
                "Flag_Hypotension": "Flag: Hypotension",
            }
            nice_labels.append(label_map.get(feat, feat))

    chart_num = "16" if model_name == "Random Forest" else "17"
    fig_fi = go.Figure(go.Bar(
        x=top20.values * 100,
        y=nice_labels,
        orientation="h",
        marker=dict(
            color=top20.values,
            colorscale=[[0, "#EEF4FF"], [1, COLOR_FLIPPED]],
        ),
        text=[f"{v*100:.2f}%" for v in top20.values],
        textposition="outside",
    ))
    fig_fi.update_layout(
        title=dict(text=f"<b>Top 20 Feature Importances — {model_name}</b><br><sup>% contribution to prediction accuracy</sup>"),
        xaxis=dict(title="Importance (%)", ticksuffix="%"),
        yaxis=dict(title=""),
        height=580,
        paper_bgcolor="white", plot_bgcolor=COLOR_BG,
        font=dict(family=FONT),
    )
    fig_fi.write_html(f"charts/{chart_num}_feature_importance_{model_name.replace(' ','_').lower()}.html")
    print(f"✓ Chart {chart_num}: Feature Importance ({model_name})")

# =============================================================================
# STEP 8 — DRG-level Flip Probability from Best Model
# =============================================================================
best_model_name = max(results, key=lambda n: results[n]["auc"])
print(f"\n★ Best model: {best_model_name} (AUC={results[best_model_name]['auc']:.3f})")

# Average predicted probability of flipping per DRG
if best_model_name == "Logistic Regression":
    all_probs = results[best_model_name]["model"].predict_proba(X_test_sc)[:, 1]
else:
    all_probs = results[best_model_name]["model"].predict_proba(X_test)[:, 1]

df_test_preds = X_test.copy()
df_test_preds["Predicted_Flip_Prob"] = all_probs
df_test_preds["Actual_Flipped"]      = y_test.values
df_test_preds["DRG01"]               = df.loc[X_test.index, "DRG01"].values

DRG_MAP = {
    276: "Dehydration", 428: "Congestive Heart Failure", 486: "Pneumonia",
    558: "Colitis", 577: "Pancreatitis", 578: "GI Bleeding",
    599: "Urinary Tract Infection", 780: "Syncope", 782: "Edema",
    786: "Chest Pain", 787: "Nausea", 789: "Abdominal Pain",
}

drg_risk = (
    df_test_preds.groupby("DRG01")
    .agg(Avg_Prob=("Predicted_Flip_Prob", "mean"),
         Actual_Rate=("Actual_Flipped", "mean"),
         Count=("DRG01", "count"))
    .reset_index()
)
drg_risk["DiagnosisName"] = drg_risk["DRG01"].map(DRG_MAP)
drg_risk["Avg_Prob_Pct"]   = (drg_risk["Avg_Prob"] * 100).round(1)
drg_risk["Actual_Pct"]     = (drg_risk["Actual_Rate"] * 100).round(1)
drg_risk["Risk_Tier"] = pd.cut(
    drg_risk["Avg_Prob"],
    bins=[0, 0.35, 0.55, 1.0],
    labels=["Low Risk (<35%)", "Medium Risk (35–55%)", "High Risk (>55%)"]
)
drg_risk = drg_risk.sort_values("Avg_Prob_Pct", ascending=True)

color_map = {
    "Low Risk (<35%)":     COLOR_STAYED,
    "Medium Risk (35–55%)": COLOR_GB,
    "High Risk (>55%)":    COLOR_FLIPPED,
}

fig18 = go.Figure()
for tier, color in color_map.items():
    sub = drg_risk[drg_risk["Risk_Tier"] == tier]
    if len(sub) == 0:
        continue
    fig18.add_trace(go.Bar(
        x=sub["Avg_Prob_Pct"],
        y=sub["DiagnosisName"],
        orientation="h",
        name=tier,
        marker_color=color,
        text=[f"{p}%  (n={n})" for p, n in zip(sub["Avg_Prob_Pct"], sub["Count"])],
        textposition="inside",
        textfont=dict(color="white", size=11),
        hovertemplate="<b>%{y}</b><br>Predicted Flip Prob: %{x:.1f}%<extra></extra>",
    ))
fig18.add_vline(x=55, line_dash="dash", line_color="#333",
                annotation_text="High-Risk threshold (55%)",
                annotation_position="top right")
fig18.update_layout(
    title=dict(text=f"<b>Predicted Flip Probability by Diagnosis</b><br><sup>{best_model_name} — risk tier classification for exclusion list</sup>"),
    xaxis=dict(title="Predicted Flip Probability (%)", ticksuffix="%", range=[0, 100]),
    yaxis=dict(title=""),
    barmode="stack",
    height=520,
    paper_bgcolor="white", plot_bgcolor=COLOR_BG,
    font=dict(family=FONT),
    legend=dict(orientation="h", y=1.08),
)
fig18.write_html("charts/18_drg_risk_tiers.html")
print("✓ Chart 18: DRG Risk Tiers (Exclusion List Recommendations)")

# =============================================================================
# STEP 9 — Save model outputs for dashboard
# =============================================================================
drg_risk.to_csv("model_outputs/drg_risk_scores.csv", index=False)
summary_df.to_csv("model_outputs/model_comparison.csv", index=False)

# Save test predictions
df_test_preds["DiagnosisName"] = df_test_preds["DRG01"].map(DRG_MAP)
df_test_preds[["DRG01", "DiagnosisName", "Age", "Predicted_Flip_Prob", "Actual_Flipped"]].to_csv(
    "model_outputs/test_predictions.csv", index=False
)

print("\n" + "="*60)
print("MODELING COMPLETE")
print("="*60)
print(summary_df.to_string(index=False))
print("\n★ Best Model:", best_model_name)
print(f"  AUC:  {results[best_model_name]['auc']:.3f}")
print(f"  CV-AUC: {results[best_model_name]['cv_mean']:.3f}")
print("\n→ Run next: python dashboard.py")
print("="*60)
