# =============================================================================
# HOSPITAL OBSERVATION UNIT — Phase 1: Data Cleaning & Preprocessing
# BDA 640 Final Case Report
# =============================================================================

import pandas as pd
import numpy as np
import os

# ── Load raw data ─────────────────────────────────────────────────────────────
df = pd.read_csv("OUData.csv")
print(f"✓ Loaded {len(df)} rows × {len(df.columns)} columns")

# ── DRG Code → Diagnosis Name mapping ────────────────────────────────────────
DRG_MAP = {
    276: "Dehydration",
    428: "Congestive Heart Failure",
    486: "Pneumonia",
    558: "Colitis",
    577: "Pancreatitis",
    578: "GI Bleeding",
    599: "Urinary Tract Infection",
    780: "Syncope",
    782: "Edema",
    786: "Chest Pain",
    787: "Nausea",
    789: "Abdominal Pain",
}

# =============================================================================
# STEP 1 — Fix BloodPressureDiff (#VALUE! errors from Excel)
# =============================================================================
# 4 rows have Excel formula errors → recalculate as Upper - Lower
df["BloodPressureDiff"] = pd.to_numeric(df["BloodPressureDiff"], errors="coerce")
mask_bp_null = df["BloodPressureDiff"].isna()
df.loc[mask_bp_null, "BloodPressureDiff"] = (
    df.loc[mask_bp_null, "BloodPressureUpper"] - df.loc[mask_bp_null, "BloodPressureLower"]
)
print(f"✓ Fixed {mask_bp_null.sum()} BloodPressureDiff #VALUE! errors")

# =============================================================================
# STEP 2 — Handle BloodPressureLower = 0 (missing data, BP upper also NaN)
# =============================================================================
# 3 rows where BloodPressureLower = 0 AND BloodPressureUpper = NaN → treat as missing
df.loc[df["BloodPressureLower"] == 0, "BloodPressureLower"] = np.nan
df.loc[df["BloodPressureUpper"].isna() & (df["BloodPressureLower"].isna()), "BloodPressureDiff"] = np.nan
print(f"✓ Set 3 invalid BloodPressureLower=0 rows to NaN")

# =============================================================================
# STEP 3 — Fix Respirations outlier (73 breaths/min is clinically impossible)
# =============================================================================
# 1 row with Respirations = 73 (normal is 12–20, critical is up to ~40)
df.loc[df["Respirations"] > 60, "Respirations"] = np.nan
print(f"✓ Removed clinically impossible Respirations value (73 → NaN)")

# =============================================================================
# STEP 4 — Median imputation for missing vitals
# =============================================================================
vitals_cols = ["BloodPressureUpper", "BloodPressureLower", "BloodPressureDiff",
               "Pulse", "PulseOximetry", "Respirations", "Temperature"]

missing_before = df[vitals_cols].isna().sum()
for col in vitals_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

missing_after = df[vitals_cols].isna().sum()
print(f"✓ Imputed missing vitals with median values:")
for col in vitals_cols:
    if missing_before[col] > 0:
        print(f"   {col}: {missing_before[col]} values → median = {df[col].median():.1f}")

# =============================================================================
# STEP 5 — Add Diagnosis Name column
# =============================================================================
df["DiagnosisName"] = df["DRG01"].map(DRG_MAP)
print(f"✓ Mapped DRG codes to diagnosis names")

# =============================================================================
# STEP 6 — Consolidate Insurance Categories
# =============================================================================
# MEDICARE + MEDICARE OTHER → "Medicare"
# MEDICAID STATE + MEDICAID OTHER → "Medicaid"
# Private → "Private"
insurance_map = {
    "MEDICARE":       "Medicare",
    "MEDICARE OTHER": "Medicare",
    "MEDICAID STATE": "Medicaid",
    "MEDICAID OTHER": "Medicaid",
    "Private":        "Private",
}
df["InsuranceGroup"] = df["PrimaryInsuranceCategory"].map(insurance_map)
print(f"✓ Consolidated insurance into 3 groups: {df['InsuranceGroup'].value_counts().to_dict()}")

# =============================================================================
# STEP 7 — Feature Engineering
# =============================================================================

# Age Groups (clinically meaningful buckets)
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 40, 55, 65, 75, 89],
    labels=["18–40", "41–55", "56–65", "66–75", "76+"]
)

# Abnormal Vital Flags (clinical thresholds)
df["Flag_Tachycardia"]    = (df["Pulse"] > 100).astype(int)           # HR > 100 bpm
df["Flag_Hypo_O2"]        = (df["PulseOximetry"] < 92).astype(int)    # SpO2 < 92%
df["Flag_Fever"]          = (df["Temperature"] > 100.4).astype(int)   # Temp > 100.4°F
df["Flag_Tachypnea"]      = (df["Respirations"] > 20).astype(int)     # RR > 20
df["Flag_Hypertension"]   = (df["BloodPressureUpper"] > 140).astype(int)  # SBP > 140
df["Flag_Hypotension"]    = (df["BloodPressureUpper"] < 90).astype(int)   # SBP < 90
df["AbnormalVitalCount"]  = (
    df["Flag_Tachycardia"] + df["Flag_Hypo_O2"] + df["Flag_Fever"] +
    df["Flag_Tachypnea"]   + df["Flag_Hypertension"] + df["Flag_Hypotension"]
)

# Diagnosis on Expanded Exclusion List (CHF=428, Pancreatitis=577, Pneumonia=486, GI Bleeding=578)
EXPANDED_EXCLUSION_DRGS = {428, 486, 577, 578}
df["OnExpandedExclusionList"] = df["DRG01"].isin(EXPANDED_EXCLUSION_DRGS).astype(int)

# Binary encode Gender
df["GenderBinary"] = (df["Gender"] == "Male").astype(int)  # 1=Male, 0=Female

print(f"✓ Engineered features: AgeGroup, 6 abnormal vital flags, AbnormalVitalCount, OnExpandedExclusionList")

# =============================================================================
# STEP 8 — Final summary & save
# =============================================================================
print("\n" + "="*60)
print("CLEANED DATASET SUMMARY")
print("="*60)
print(f"Total patients:        {len(df)}")
print(f"Flipped to Inpatient:  {df['Flipped'].sum()} ({df['Flipped'].mean()*100:.1f}%)")
print(f"Stayed Observation:    {(df['Flipped']==0).sum()} ({(df['Flipped']==0).mean()*100:.1f}%)")
print(f"Avg OU LOS (all):      {df['OU_LOS_hrs'].mean():.1f} hrs")
print(f"Avg OU LOS (Flipped):  {df[df['Flipped']==1]['OU_LOS_hrs'].mean():.1f} hrs")
print(f"Avg OU LOS (Stayed):   {df[df['Flipped']==0]['OU_LOS_hrs'].mean():.1f} hrs")
print(f"Remaining missing:     {df.isnull().sum().sum()}")
print("="*60)

# Save cleaned dataset
df.to_csv("OUData_cleaned.csv", index=False)
print("\n✓ Saved: OUData_cleaned.csv")
print("→ Run next: python eda_visualizations.py")
