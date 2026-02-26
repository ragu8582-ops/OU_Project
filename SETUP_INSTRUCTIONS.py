# ================================================
# HOSPITAL OU PROJECT — Setup & Run Instructions
# BDA 640 Final Case | Python / VS Code
# ================================================

# STEP 1 — Install required libraries
# Open VS Code Terminal (Ctrl+`) and run:

pip install pandas numpy scikit-learn plotly dash dash-bootstrap-components

# ================================================
# STEP 2 — Folder Setup
# ================================================
# Place these files in ONE folder (e.g., "OU_Project"):
#
#   OU_Project/
#   ├── OUData.csv               ← your data file (rename to exactly this)
#   ├── data_cleaning.py
#   ├── eda_visualizations.py
#   ├── predictive_model.py
#   └── dashboard.py
#
# Open VS Code → File → Open Folder → select OU_Project

# ================================================
# STEP 3 — Run in Order (in VS Code Terminal)
# ================================================

# 1. Clean the data:
python data_cleaning.py
# → Creates: OUData_cleaned.csv

# 2. Generate all EDA charts:
python eda_visualizations.py
# → Creates: charts/ folder with 12 interactive HTML charts
# → Open any .html file in a browser to view charts for your report

# 3. Train predictive models:
python predictive_model.py
# → Creates: model_outputs/ folder with results
# → Prints model comparison table in terminal

# 4. Launch the interactive dashboard:
python dashboard.py
# → Open browser at: http://127.0.0.1:8050
# → 5 tabs: Overview | Diagnosis | Vitals | Model | Recommendations

# ================================================
# CHARTS CREATED (for your report)
# ================================================
# charts/01_flip_rate_by_diagnosis.html
# charts/02_volume_and_flip_rate.html
# charts/03_los_distribution.html
# charts/04_age_distribution.html
# charts/05_flip_by_insurance.html
# charts/06_vitals_radar.html
# charts/07_vitals_table.html
# charts/08_heatmap_diagnosis_age.html
# charts/09_flip_by_gender.html
# charts/10_abnormal_vitals_flip.html
# charts/11_los_by_diagnosis.html
# charts/12_capacity_waterfall.html
# charts/13_model_comparison_table.html   (after predictive_model.py)
# charts/14_roc_curves.html
# charts/15_confusion_matrices.html
# charts/16_feature_importance_random_forest.html
# charts/17_feature_importance_gradient_boosting.html
# charts/18_drg_risk_tiers.html

# ================================================
# DASHBOARD TABS
# ================================================
# Tab 1: Patient Overview         → filtered demographics & LOS
# Tab 2: Diagnosis Deep-Dive      → flip rate by DRG, heatmap
# Tab 3: Vitals Analysis          → radar chart, flag comparison
# Tab 4: Predictive Model         → AUC comparison, DRG risk tiers
# Tab 5: Exclusion List & Impact  → recommendations + financial slider

# ================================================
# REPORT STRUCTURE MAPPING
# ================================================
# Section 1 (Executive Summary)   → KPIs from data_cleaning.py output
# Section 2 (Problem Description) → Charts 01, 02, 03, 12
# Section 3 (Methodology)         → data_cleaning.py + predictive_model.py logic
# Section 4 (Results)             → Charts 13–18, model terminal output
# Section 5 (Recommendations)     → Chart 18 + dashboard Tab 5

# ================================================
# COMMON ISSUES
# ================================================
# Issue: "Module not found"
# Fix:   pip install <module_name>

# Issue: "OUData.csv not found"
# Fix:   Make sure OUData.csv is in the SAME folder as the .py files

# Issue: Dashboard not loading
# Fix:   Wait 3–5 seconds after running dashboard.py, then open browser

# Issue: model_outputs/ not found when running dashboard
# Fix:   Run predictive_model.py first, then restart dashboard.py
