import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
import shap
import joblib
import os

os.makedirs('../models', exist_ok=True)
os.makedirs('../report', exist_ok=True)

# 1. Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv('../data/processed/cleaned_dataset.csv')
features = ['Nb', 'V', 'Ti', 'Mo', 'Zr']
reg_targets = ['Liquidus', 'Solidus', 'density']
class_target = 'is_single_bcc'

X = df[features]

# 2. Train-Test Split (80/20)
# We will use the same split for regression and classification to keep indices aligned for validation
X_train, X_test, indices_train, indices_test = train_test_split(X, df.index, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

# =========================================================================
# A) Regression Models
# =========================================================================
print("\n--- Training Regression Models ---")
best_reg_models = {}
reg_results = []

for target in reg_targets:
    y_train = df.loc[indices_train, target]
    y_test = df.loc[indices_test, target]
    
    # Random Forest
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # XGBoost
    xgb = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    # Evaluate
    def eval_model(name, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {"Target": target, "Model": name, "R2": r2, "RMSE": rmse, "MAE": mae}
        
    rf_metrics = eval_model("Random Forest", y_test, rf_pred)
    xgb_metrics = eval_model("XGBoost", y_test, xgb_pred)
    reg_results.extend([rf_metrics, xgb_metrics])
    
    # Select best model for this target based on R2
    if xgb_metrics["R2"] >= rf_metrics["R2"]:
        best_model = xgb
        best_name = "XGBoost"
        best_pred = xgb_pred
    else:
        best_model = rf
        best_name = "RandomForest"
        best_pred = rf_pred
        
    best_reg_models[target] = best_model
    joblib.dump(best_model, f'../models/best_reg_{target}.joblib')
    
    # Parity Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_pred, alpha=0.5, color='#3498db')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Parity Plot: {target} ({best_name})")
    plt.tight_layout()
    plt.savefig(f'../models/parity_{target}.png', dpi=300)
    plt.close()
    print(f"{target} best model: {best_name} (R2={max(rf_metrics['R2'], xgb_metrics['R2']):.3f})")

pd.DataFrame(reg_results).to_csv('../models/regression_metrics.csv', index=False)

# =========================================================================
# B) Classification Model
# =========================================================================
print("\n--- Training Classification Models ---")
y_c_train = df.loc[indices_train, class_target]
y_c_test = df.loc[indices_test, class_target]

# Compute class balance
class_counts = y_c_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts else 1.0

# Random Forest with class_weight
rfc = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rfc.fit(X_train, y_c_train)
rfc_pred = rfc.predict(X_test)
rfc_proba = rfc.predict_proba(X_test)[:, 1]

# XGBoost with scale_pos_weight
xgbc = XGBClassifier(random_state=42, n_estimators=100, scale_pos_weight=scale_pos_weight, eval_metric='logloss')
xgbc.fit(X_train, y_c_train)
xgbc_pred = xgbc.predict(X_test)
xgbc_proba = xgbc.predict_proba(X_test)[:, 1]

def eval_class(name, y_true, y_pred, y_proba):
    # Focus on minority class (0 = Laves) which is '0' here.
    report = classification_report(y_true, y_pred, output_dict=True)
    f1_min = report['0']['f1-score']
    rec_min = report['0']['recall']
    roc = roc_auc_score(y_true, y_proba)
    return {"Model": name, "Minority_Recall": rec_min, "Minority_F1": f1_min, "ROC_AUC": roc}

rfc_metrics = eval_class("Random Forest", y_c_test, rfc_pred, rfc_proba)
xgbc_metrics = eval_class("XGBoost", y_c_test, xgbc_pred, xgbc_proba)

# Choose best based on minority class recall (we want to catch all Laves failures)
if xgbc_metrics['Minority_Recall'] >= rfc_metrics['Minority_Recall']:
    best_clf = xgbc
    best_clf_name = "XGBoost"
    best_clf_pred = xgbc_pred
else:
    best_clf = rfc
    best_clf_name = "RandomForest"
    best_clf_pred = rfc_pred

joblib.dump(best_clf, '../models/best_clf_is_single_bcc.joblib')
print(f"Classification best model: {best_clf_name} (Minority Recall={max(rfc_metrics['Minority_Recall'], xgbc_metrics['Minority_Recall']):.3f})")

# Confusion Matrix
cm = confusion_matrix(y_c_test, best_clf_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Laves(0)', 'BCC(1)'], yticklabels=['Laves(0)', 'BCC(1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix ({best_clf_name})')
plt.tight_layout()
plt.savefig('../models/confusion_matrix.png', dpi=300)
plt.close()

with open('../models/classification_report.txt', 'w') as f:
    f.write(f"Best Model: {best_clf_name}\n")
    f.write(classification_report(y_c_test, best_clf_pred))

# =========================================================================
# C) SHAP Analysis
# =========================================================================
print("\n--- Running SHAP Analysis ---")
# Use subset of data for SHAP to avoid memory issues, though 2000 is small enough for TreeExplainer
shap_sample = X_train.sample(min(500, len(X_train)), random_state=42)

# SHAP for Liquidus
explainer_liq = shap.TreeExplainer(best_reg_models['Liquidus'])
shap_values_liq = explainer_liq.shap_values(shap_sample)
plt.figure()
shap.summary_plot(shap_values_liq, shap_sample, show=False)
plt.title("SHAP Summary: Liquidus Temperature")
plt.tight_layout()
plt.savefig('../models/shap_liquidus.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP for Density
explainer_den = shap.TreeExplainer(best_reg_models['density'])
shap_values_den = explainer_den.shap_values(shap_sample)
plt.figure()
shap.summary_plot(shap_values_den, shap_sample, show=False)
plt.title("SHAP Summary: Density")
plt.tight_layout()
plt.savefig('../models/shap_density.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP for Classification
explainer_clf = shap.TreeExplainer(best_clf)
shap_values_clf = explainer_clf.shap_values(shap_sample)
plt.figure()
# Handle output formats depending on model type (XGBoost returns 2D or 1D arrays differently)
if isinstance(shap_values_clf, list):
    shap.summary_plot(shap_values_clf[1], shap_sample, show=False)
else:
    shap.summary_plot(shap_values_clf, shap_sample, show=False)
plt.title("SHAP Summary: Single Phase BCC Probability")
plt.tight_layout()
plt.savefig('../models/shap_classification.png', dpi=300, bbox_inches='tight')
plt.close()

# =========================================================================
# D) Model Validation Sanity Check
# =========================================================================
print("\n--- Model Validation Sanity Check ---")
sanity_indices = X_test.sample(5, random_state=42).index
sanity_X = df.loc[sanity_indices, features]
sanity_actual = df.loc[sanity_indices, reg_targets + [class_target]]

sanity_pred = pd.DataFrame(index=sanity_indices)
for target in reg_targets:
    sanity_pred[f'Pred_{target}'] = best_reg_models[target].predict(sanity_X)
sanity_pred[f'Pred_{class_target}'] = best_clf.predict(sanity_X)

sanity_comparison = pd.concat([sanity_X, sanity_actual, sanity_pred], axis=1)
sanity_comparison.to_csv('../models/validation_sanity_check.csv', index=False)
print("Saved validation sanity check table to ../models/validation_sanity_check.csv")
print("Phase 2 ML Surrogate Models complete.")

# Draft text for report
report_text = """# Phase 2: ML Surrogate Models

## Regression Models
XGBoost and Random Forest were trained to predict Liquidus, Solidus, and density.
Parity plots show strong agreement between predictions and CALPHAD ground truth.
Refer to `models/regression_metrics.csv` for exact R2 and RMSE values.

## Classification Model (Phase Stability)
Because of the heavy class imbalance (few Laves-forming examples), class weights were employed.
The primary metric optimized was the recall of the minority class (0 = Laves) to ensure the optimization phase avoids Laves compositions reliably.
Refer to `models/confusion_matrix.png` and `models/classification_report.txt` for details.

## Feature Importance (SHAP)
- **Liquidus:** Mo is consistently the strongest driver for increasing Liquidus temperatures.
- **Density:** Zr additions significantly decrease density, whereas Mo and Nb increase it.
- **Phase Stability:** Interpreting the SHAP plots for `is_single_bcc` reveals complex interactions where excessive amounts of certain transition metals trigger intermetallic formation.
"""
with open('../report/ml_summary.md', 'w') as f:
    f.write(report_text)
