"""
Phase 2 - FULL Comprehensive ML Pipeline (Both Datasets Combined)
==================================================================
Datasets:
  - dataset1_calphad_original.xlsx  (2000 rows, original CALPHAD)
  - dataset2_calphad_augmented.csv  (2000 rows, augmented BTP4output)
  Combined → ~3981 clean rows after dedup/NaN removal

Models from BTP Notes:
  REGRESSION (Basic):
    1. Simple Linear Regression
    2. Multiple Linear Regression
    3. Polynomial Regression (degree=2)
  REGRESSION (Advanced):
    4. Ridge
    5. Lasso
    6. ElasticNet
    7. Decision Tree Regressor
    8. Random Forest Regressor
    9. XGBoost Regressor

  CLASSIFICATION (Basic):
    1. Logistic Regression
    2. KNN
    3. Naive Bayes (Gaussian)
    4. Decision Tree Classifier
  CLASSIFICATION (Advanced):
    5. Random Forest Classifier
    6. SVM
    7. AdaBoost
    8. Gradient Boosting
    9. XGBoost Classifier
   10. LightGBM  (skipped if not installed)
   11. CatBoost  (skipped if not installed)

Metrics:
  Regression:  MAE, MSE, RMSE, R2, Adjusted-R2, MAPE
  Classification: Accuracy, Precision, Recall, F1, AUC-ROC,
                  Log Loss, Specificity, Confusion Matrix, ROC Curve

Model Evaluation:
  - 80/20 stratified train-test split
  - 5-fold Cross-Validation (StratifiedKFold for classification)
  - GridSearchCV  for smaller grids (LR, Ridge, Lasso, ElasticNet, KNN, SVM)
  - RandomizedSearchCV for larger spaces (DT, RF, XGB, ADA, GB, LGB, CAT)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, ast
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                  LogisticRegression)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    roc_curve, auc
)
from xgboost import XGBRegressor, XGBClassifier
import joblib

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[INFO] LightGBM not installed - skipping.")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("[INFO] CatBoost not installed - skipping.")

# ── Directories ──────────────────────────────────────────────────────────────
OUT_DIR  = '../models_comprehensive'
PLOT_DIR = f'{OUT_DIR}/plots'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs('../report', exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD & COMBINE BOTH DATASETS
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  STEP 1: Loading & Combining Both Datasets")
print("=" * 65)

cols_needed = ['Nb', 'V', 'Ti', 'Mo', 'Zr',
               'Liquidus', 'Solidus', 'density', 'phases', 'number_phases']

df1 = pd.read_excel('../data/raw/dataset1_calphad_original.xlsx')[cols_needed]
df2 = pd.read_csv('../data/raw/dataset2_calphad_augmented.csv')[cols_needed]

print(f"Dataset 1 (original CALPHAD): {len(df1)} rows")
print(f"Dataset 2 (augmented BTP4):   {len(df2)} rows")

df = pd.concat([df1, df2], ignore_index=True)
print(f"Combined (before cleaning):   {len(df)} rows")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CLEAN COMBINED DATA
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STEP 2: Cleaning Data")
print("=" * 65)

before = len(df)
df = df.dropna(subset=['Liquidus', 'Solidus', 'density', 'phases'])
print(f"Dropped {before - len(df)} rows with NaN → {len(df)} rows remain")

before = len(df)
df = df[df['Liquidus'] < 4000]
print(f"Dropped {before - len(df)} impossible Liquidus (>4000K) → {len(df)} rows remain")

# Parse phases → binary is_single_bcc
def parse_bcc(phases_val):
    try:
        if isinstance(phases_val, str):
            phases_list = ast.literal_eval(phases_val)
        else:
            phases_list = phases_val
        # Single BCC only: exactly 1 phase AND it contains 'BCC'
        if len(phases_list) == 1 and 'BCC' in str(phases_list[0]).upper():
            return 1
        # Also accept 2-phase BCC (BCC_B2#1 + BCC_B2#2) as "effectively BCC" (original logic)
        # But check for Laves
        phases_str = ' '.join(str(p).upper() for p in phases_list)
        if 'C15_LAVES' in phases_str:
            return 0
        if 'BCC_B2' in phases_str:
            return 1
        return 0
    except:
        return 0

df['is_single_bcc'] = df['phases'].apply(parse_bcc)

print(f"\nFinal cleaned dataset: {len(df)} rows")
print(f"Single-BCC (1): {df['is_single_bcc'].sum()}  |  "
      f"Non-BCC Laves (0): {(df['is_single_bcc']==0).sum()}")
print(f"Class ratio:  {df['is_single_bcc'].mean()*100:.1f}% BCC")

# Save combined cleaned dataset
df.to_csv('../data/processed/combined_cleaned_dataset.csv', index=False)
print("Saved → data/processed/combined_cleaned_dataset.csv")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: PREPARE FEATURES & SPLIT
# ════════════════════════════════════════════════════════════════════════════
features    = ['Nb', 'V', 'Ti', 'Mo', 'Zr']
reg_targets = ['Liquidus', 'Solidus', 'density']
cls_target  = 'is_single_bcc'

X     = df[features].values
y_cls = df[cls_target].values
n_feat = len(features)

X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index, test_size=0.2, random_state=42,
    stratify=y_cls
)
y_cls_train = df.loc[idx_train, cls_target].values
y_cls_test  = df.loc[idx_test,  cls_target].values

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Scaled versions for linear/SVM/KNN models
scaler_reg = StandardScaler()
X_train_sc = scaler_reg.fit_transform(X_train)
X_test_sc  = scaler_reg.transform(X_test)

scaler_cls = StandardScaler()
X_train_cls = scaler_cls.fit_transform(X_train)
X_test_cls  = scaler_cls.transform(X_test)

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_sc)
X_test_poly  = poly.transform(X_test_sc)
n_poly_feat  = X_train_poly.shape[1]

cv5  = 5
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Class imbalance helpers
class_counts     = np.bincount(y_cls_train)
scale_pos_weight = class_counts[0] / max(class_counts[1], 1)
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
def adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def mape_score(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def eval_regression(name, target, y_true, y_pred, k=n_feat):
    r2  = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'Model'  : name,  'Target': target,
        'MAE'    : round(mean_absolute_error(y_true, y_pred), 4),
        'MSE'    : round(mse, 4),
        'RMSE'   : round(np.sqrt(mse), 4),
        'R2'     : round(r2, 4),
        'Adj_R2' : round(adj_r2(r2, len(y_true), k), 4),
        'MAPE_%' : round(mape_score(y_true, y_pred), 4),
    }

def eval_classification(name, y_true, y_pred, y_proba):
    cm_arr = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm_arr.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'Model'      : name,
        'Accuracy'   : round(accuracy_score(y_true, y_pred), 4),
        'Precision'  : round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall'     : round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1'         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'AUC_ROC'    : round(roc_auc_score(y_true, y_proba), 4),
        'Log_Loss'   : round(log_loss(y_true, y_proba), 4),
        'Specificity': round(spec, 4),
    }

def parity_plot(y_true, y_pred, name, target):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color='steelblue')
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel(f'Actual {target}')
    ax.set_ylabel(f'Predicted {target}')
    ax.set_title(f'{name}\n{target}  (R²={r2:.4f})')
    fig.tight_layout()
    fname = f'{PLOT_DIR}/parity_{target}_{name.replace(" ","_")}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname

def cm_plot(y_true, y_pred, name):
    cm_arr = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-BCC(0)', 'BCC(1)'],
                yticklabels=['Non-BCC(0)', 'BCC(1)'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f'CM – {name}\nAcc={acc:.3f}')
    fig.tight_layout()
    fname = f'{PLOT_DIR}/cm_{name.replace(" ","_")}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# A) REGRESSION MODELS  (all 9)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  A) REGRESSION MODELS")
print("=" * 65)

all_reg  = []
cv_reg   = {}   # {target: {model_name: cv_score}}

for target in reg_targets:
    print(f"\n--- Target: {target} ---")
    y_tr = df.loc[idx_train, target].values
    y_te = df.loc[idx_test,  target].values
    cv_reg[target] = {}

    # ── 1. Simple Linear Regression (Nb only as single feature baseline) ──
    # Use just one feature (Nb) for "simple" regression
    lr_simple = LinearRegression()
    lr_simple.fit(X_train_sc[:, :1], y_tr)
    pred = lr_simple.predict(X_test_sc[:, :1])
    res  = eval_regression("Simple Linear Regression", target, y_te, pred, k=1)
    all_reg.append(res); parity_plot(y_te, pred, "Simple Linear Regression", target)
    cv_simple = cross_val_score(lr_simple, X_train_sc[:, :1], y_tr, cv=cv5, scoring='r2').mean()
    cv_reg[target]["Simple Linear Regression"] = round(cv_simple, 4)
    print(f"  Simple LR       R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  CV-R2={cv_simple:.4f}")

    # ── 2. Multiple Linear Regression (all 5 features) ──
    lr_multi = LinearRegression()
    lr_multi.fit(X_train_sc, y_tr)
    pred = lr_multi.predict(X_test_sc)
    res  = eval_regression("Multiple Linear Regression", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "Multiple Linear Regression", target)
    cv_multi = cross_val_score(lr_multi, X_train_sc, y_tr, cv=cv5, scoring='r2').mean()
    cv_reg[target]["Multiple Linear Regression"] = round(cv_multi, 4)
    print(f"  Multiple LR     R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  CV-R2={cv_multi:.4f}")

    # ── 3. Polynomial Regression (degree=2) ──
    pr = LinearRegression()
    pr.fit(X_train_poly, y_tr)
    pred = pr.predict(X_test_poly)
    res  = eval_regression("Polynomial Regression", target, y_te, pred, k=n_poly_feat)
    all_reg.append(res); parity_plot(y_te, pred, "Polynomial Regression", target)
    cv_poly = cross_val_score(pr, X_train_poly, y_tr, cv=cv5, scoring='r2').mean()
    cv_reg[target]["Polynomial Regression"] = round(cv_poly, 4)
    print(f"  Polynomial LR   R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  CV-R2={cv_poly:.4f}")

    # ── 4. Ridge (GridSearchCV) ──
    gs = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1, 10, 100]},
                      cv=cv5, scoring='r2')
    gs.fit(X_train_sc, y_tr)
    pred = gs.best_estimator_.predict(X_test_sc)
    res  = eval_regression("Ridge", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "Ridge", target)
    cv_reg[target]["Ridge"] = round(gs.best_score_, 4)
    print(f"  Ridge           R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"best_alpha={gs.best_params_['alpha']}  CV-R2={gs.best_score_:.4f}")

    # ── 5. Lasso (GridSearchCV) ──
    gs = GridSearchCV(Lasso(max_iter=5000),
                      {'alpha': [0.0001, 0.001, 0.01, 0.1, 1]},
                      cv=cv5, scoring='r2')
    gs.fit(X_train_sc, y_tr)
    pred = gs.best_estimator_.predict(X_test_sc)
    res  = eval_regression("Lasso", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "Lasso", target)
    cv_reg[target]["Lasso"] = round(gs.best_score_, 4)
    print(f"  Lasso           R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"best_alpha={gs.best_params_['alpha']}  CV-R2={gs.best_score_:.4f}")

    # ── 6. ElasticNet (GridSearchCV) ──
    gs = GridSearchCV(ElasticNet(max_iter=5000),
                      {'alpha': [0.001, 0.01, 0.1, 1],
                       'l1_ratio': [0.2, 0.5, 0.8]},
                      cv=cv5, scoring='r2')
    gs.fit(X_train_sc, y_tr)
    pred = gs.best_estimator_.predict(X_test_sc)
    res  = eval_regression("ElasticNet", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "ElasticNet", target)
    cv_reg[target]["ElasticNet"] = round(gs.best_score_, 4)
    print(f"  ElasticNet      R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"params={gs.best_params_}  CV-R2={gs.best_score_:.4f}")

    # ── 7. Decision Tree (RandomizedSearchCV) ──
    rs = RandomizedSearchCV(
        DecisionTreeRegressor(random_state=42),
        {'max_depth': [3, 5, 7, 10, None],
         'min_samples_leaf': [1, 5, 10, 20],
         'max_features': ['sqrt', 'log2', None]},
        n_iter=15, cv=cv5, scoring='r2', random_state=42)
    rs.fit(X_train, y_tr)
    pred = rs.best_estimator_.predict(X_test)
    res  = eval_regression("Decision Tree", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "Decision Tree", target)
    cv_reg[target]["Decision Tree"] = round(rs.best_score_, 4)
    print(f"  Decision Tree   R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"params={rs.best_params_}  CV-R2={rs.best_score_:.4f}")

    # ── 8. Random Forest (RandomizedSearchCV) ──
    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {'n_estimators': [100, 200, 300],
         'max_depth': [None, 5, 10, 15],
         'min_samples_leaf': [1, 3, 5]},
        n_iter=15, cv=cv5, scoring='r2', random_state=42)
    rs.fit(X_train, y_tr)
    pred = rs.best_estimator_.predict(X_test)
    res  = eval_regression("Random Forest", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "Random Forest", target)
    cv_reg[target]["Random Forest"] = round(rs.best_score_, 4)
    print(f"  Random Forest   R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"params={rs.best_params_}  CV-R2={rs.best_score_:.4f}")
    # Save best RF for this target
    joblib.dump(rs.best_estimator_, f'../models/best_reg_{target}.joblib')

    # ── 9. XGBoost (RandomizedSearchCV) ──
    rs = RandomizedSearchCV(
        XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        {'n_estimators': [200, 300, 500],
         'learning_rate': [0.03, 0.05, 0.1],
         'max_depth': [4, 6, 8],
         'subsample': [0.7, 0.8, 1.0],
         'colsample_bytree': [0.7, 0.8, 1.0]},
        n_iter=15, cv=cv5, scoring='r2', random_state=42)
    rs.fit(X_train, y_tr)
    pred = rs.best_estimator_.predict(X_test)
    res  = eval_regression("XGBoost", target, y_te, pred)
    all_reg.append(res); parity_plot(y_te, pred, "XGBoost", target)
    cv_reg[target]["XGBoost"] = round(rs.best_score_, 4)
    print(f"  XGBoost         R2={res['R2']:.4f}  RMSE={res['RMSE']:.2f}  "
          f"params={rs.best_params_}  CV-R2={rs.best_score_:.4f}")

    # Compare RF vs XGB, save better one
    rf_r2 = next(r['R2'] for r in all_reg
                 if r['Target']==target and r['Model']=='Random Forest')
    xg_r2 = next(r['R2'] for r in all_reg
                 if r['Target']==target and r['Model']=='XGBoost')
    winner = "Random Forest" if rf_r2 >= xg_r2 else "XGBoost"
    print(f"  >>> Best for {target}: {winner}")

reg_df = pd.DataFrame(all_reg)
reg_df.to_csv(f'{OUT_DIR}/all_regression_metrics.csv', index=False)
pd.DataFrame(cv_reg).to_csv(f'{OUT_DIR}/cv_regression_scores.csv')
print(f"\nSaved regression metrics → {OUT_DIR}/all_regression_metrics.csv")

# Comparison bar charts per target
for target in reg_targets:
    sub = reg_df[reg_df['Target'] == target].sort_values('R2', ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, metric in zip(axes, ['R2', 'RMSE', 'MAE']):
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(sub))]
        bars = ax.barh(sub['Model'], sub[metric], color=colors)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} – {target}', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        for bar, val in zip(bars, sub[metric]):
            ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='center', fontsize=8, color='white', fontweight='bold')
    fig.suptitle(f'Regression Model Comparison – {target}', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{PLOT_DIR}/reg_comparison_{target}.png', dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart for {target}")

# ════════════════════════════════════════════════════════════════════════════
# B) CLASSIFICATION MODELS  (all 11)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  B) CLASSIFICATION MODELS")
print("=" * 65)
print(f"Class 0 (Non-BCC/Laves): {class_counts[0]}  |  Class 1 (BCC): {class_counts[1]}")
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

all_cls  = []
roc_data = {}

def run_clf(name, clf, X_tr, X_te, fit_needs_scaler=True):
    clf.fit(X_tr, y_cls_train)
    pred = clf.predict(X_te)
    if hasattr(clf, 'predict_proba'):
        proba = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, 'decision_function'):
        d = clf.decision_function(X_te)
        proba = (d - d.min()) / (d.max() - d.min() + 1e-9)
    else:
        proba = pred.astype(float)

    res = eval_classification(name, y_cls_test, pred, proba)
    all_cls.append(res)
    cm_plot(y_cls_test, pred, name)

    fpr, tpr, _ = roc_curve(y_cls_test, proba)
    roc_data[name] = (fpr, tpr, auc(fpr, tpr))

    cv_auc = cross_val_score(clf, X_tr, y_cls_train, cv=skf, scoring='roc_auc')
    print(f"  {name:28s}  Acc={res['Accuracy']:.4f}  "
          f"F1={res['F1']:.4f}  AUC={res['AUC_ROC']:.4f}  "
          f"Spec={res['Specificity']:.4f}  "
          f"CV-AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}")
    return clf

# 1. Logistic Regression (GridSearchCV)
gs = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=2000),
    {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']},
    cv=skf, scoring='roc_auc')
gs.fit(X_train_cls, y_cls_train)
print(f"\n  [LogReg] best: {gs.best_params_}  CV-AUC={gs.best_score_:.4f}")
run_clf("Logistic Regression", gs.best_estimator_, X_train_cls, X_test_cls)

# 2. KNN (GridSearchCV)
gs = GridSearchCV(
    KNeighborsClassifier(),
    {'n_neighbors': [3, 5, 7, 11, 15, 21],
     'weights': ['uniform', 'distance'],
     'metric': ['euclidean', 'minkowski']},
    cv=skf, scoring='roc_auc')
gs.fit(X_train_cls, y_cls_train)
print(f"\n  [KNN] best: {gs.best_params_}  CV-AUC={gs.best_score_:.4f}")
run_clf("KNN", gs.best_estimator_, X_train_cls, X_test_cls)

# 3. Naive Bayes
run_clf("Naive Bayes", GaussianNB(), X_train_cls, X_test_cls)

# 4. Decision Tree (RandomizedSearchCV)
rs = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    {'max_depth': [3, 5, 7, 10, None],
     'min_samples_leaf': [1, 5, 10, 20],
     'class_weight': ['balanced'],
     'criterion': ['gini', 'entropy']},
    n_iter=15, cv=skf, scoring='roc_auc', random_state=42)
rs.fit(X_train_cls, y_cls_train)
print(f"\n  [DT] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
run_clf("Decision Tree", rs.best_estimator_, X_train_cls, X_test_cls)

# 5. Random Forest (RandomizedSearchCV)
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [100, 200, 300],
     'max_depth': [None, 5, 10, 15],
     'min_samples_leaf': [1, 3, 5],
     'class_weight': ['balanced', 'balanced_subsample']},
    n_iter=15, cv=skf, scoring='roc_auc', random_state=42)
rs.fit(X_train_cls, y_cls_train)
print(f"\n  [RF] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
run_clf("Random Forest", rs.best_estimator_, X_train_cls, X_test_cls)
joblib.dump(rs.best_estimator_, '../models/best_clf_RF.joblib')

# 6. SVM (GridSearchCV)
gs = GridSearchCV(
    SVC(class_weight='balanced', probability=True, random_state=42),
    {'C': [0.1, 1, 10, 100],
     'kernel': ['rbf', 'linear', 'poly'],
     'gamma': ['scale', 'auto']},
    cv=skf, scoring='roc_auc')
gs.fit(X_train_cls, y_cls_train)
print(f"\n  [SVM] best: {gs.best_params_}  CV-AUC={gs.best_score_:.4f}")
run_clf("SVM", gs.best_estimator_, X_train_cls, X_test_cls)

# 7. AdaBoost (RandomizedSearchCV)
rs = RandomizedSearchCV(
    AdaBoostClassifier(random_state=42, algorithm='SAMME'),
    {'n_estimators': [50, 100, 200, 300],
     'learning_rate': [0.3, 0.5, 1.0, 1.5]},
    n_iter=12, cv=skf, scoring='roc_auc', random_state=42)
rs.fit(X_train_cls, y_cls_train)
print(f"\n  [AdaBoost] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
run_clf("AdaBoost", rs.best_estimator_, X_train_cls, X_test_cls)

# 8. Gradient Boosting (RandomizedSearchCV)
rs = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    {'n_estimators': [100, 200, 300],
     'learning_rate': [0.03, 0.05, 0.1, 0.2],
     'max_depth': [3, 4, 5],
     'subsample': [0.7, 0.8, 1.0]},
    n_iter=15, cv=skf, scoring='roc_auc', random_state=42)
rs.fit(X_train_cls, y_cls_train)
print(f"\n  [GradientBoosting] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
run_clf("Gradient Boosting", rs.best_estimator_, X_train_cls, X_test_cls)

# 9. XGBoost (RandomizedSearchCV)
rs = RandomizedSearchCV(
    XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss',
                  n_jobs=-1, scale_pos_weight=scale_pos_weight),
    {'n_estimators': [200, 300, 500],
     'learning_rate': [0.03, 0.05, 0.1],
     'max_depth': [3, 5, 6, 8],
     'subsample': [0.7, 0.8, 1.0],
     'colsample_bytree': [0.7, 0.8, 1.0]},
    n_iter=15, cv=skf, scoring='roc_auc', random_state=42)
rs.fit(X_train_cls, y_cls_train)
print(f"\n  [XGBoost] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
run_clf("XGBoost", rs.best_estimator_, X_train_cls, X_test_cls)
joblib.dump(rs.best_estimator_, '../models/best_clf_is_single_bcc.joblib')

# 10. LightGBM
# Skipped due to Windows C-runtime DLL access violation in python 3.12 wheel
print("\n  LightGBM: SKIPPED (Windows DLL runtime conflict)")

# 11. CatBoost
if HAS_CAT:
    rs = RandomizedSearchCV(
        CatBoostClassifier(random_state=42, verbose=0,
                           class_weights=[1, scale_pos_weight]),
        {'iterations': [200, 300, 500],
         'learning_rate': [0.03, 0.05, 0.1],
         'depth': [4, 6, 8]},
        n_iter=12, cv=skf, scoring='roc_auc', random_state=42)
    rs.fit(X_train_cls, y_cls_train)
    print(f"\n  [CatBoost] best: {rs.best_params_}  CV-AUC={rs.best_score_:.4f}")
    run_clf("CatBoost", rs.best_estimator_, X_train_cls, X_test_cls)
else:
    print("\n  CatBoost: SKIPPED")

# ── Save classification results ────────────────────────────────────────────
cls_df = pd.DataFrame(all_cls)
cls_df.to_csv(f'{OUT_DIR}/all_classification_metrics.csv', index=False)
print(f"\nSaved classification metrics → {OUT_DIR}/all_classification_metrics.csv")
print("\nAll Classification Results:")
print(cls_df.sort_values('AUC_ROC', ascending=False).to_string(index=False))

# ── Combined ROC Curve ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
cmap_colors = plt.cm.tab20(np.linspace(0, 1, len(roc_data)))
for (name, (fpr, tpr, auc_val)), col in zip(roc_data.items(), cmap_colors):
    ax.plot(fpr, tpr, lw=2.0, color=col, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0,1],[0,1],'k--',lw=1.2, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves – All Classifiers\n(Nb-V-Ti-Mo-Zr RHEA Phase Prediction)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/roc_all_classifiers.png', dpi=150)
plt.close(fig)
print(f"Saved ROC curves → {PLOT_DIR}/roc_all_classifiers.png")

# ── Classification comparison radar/bar chart ──────────────────────────────
metrics_plot = ['Accuracy','Precision','Recall','F1','AUC_ROC','Specificity']
cls_sorted   = cls_df.sort_values('AUC_ROC', ascending=False)
fig, axes    = plt.subplots(2, 3, figsize=(18, 11))
for ax, metric in zip(axes.flat, metrics_plot):
    colors = ['#2ecc71' if i == 0 else ('#e74c3c' if i == len(cls_sorted)-1 else '#3498db')
              for i in range(len(cls_sorted))]
    bars = ax.barh(cls_sorted['Model'], cls_sorted[metric], color=colors)
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(metric, fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, cls_sorted[metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8)
fig.suptitle('Classification Model Comparison – All Metrics\n(sorted by AUC-ROC)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/cls_comparison_all_metrics.png', dpi=150)
plt.close(fig)
print(f"Saved classification comparison → {PLOT_DIR}/cls_comparison_all_metrics.png")

# ── Log Loss comparison ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ll_sorted = cls_df.sort_values('Log_Loss')
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(ll_sorted))]
ax.barh(ll_sorted['Model'], ll_sorted['Log_Loss'], color=colors)
ax.set_xlabel('Log Loss (lower = better)', fontsize=12)
ax.set_title('Log Loss – All Classifiers', fontweight='bold', fontsize=13)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/cls_log_loss.png', dpi=150)
plt.close(fig)

# ── Summary JSON ──────────────────────────────────────────────────────────
best_clf = cls_df.sort_values('AUC_ROC', ascending=False).iloc[0]
summary  = {
    'dataset_info': {
        'dataset1_rows': len(df1), 'dataset2_rows': len(df2),
        'combined_cleaned': len(df),
        'bcc_count': int(df['is_single_bcc'].sum()),
        'non_bcc_count': int((df['is_single_bcc']==0).sum()),
    },
    'best_regression': {},
    'best_classifier': best_clf.to_dict(),
    'all_regression': all_reg,
    'all_classification': all_cls,
}
for t in reg_targets:
    best = reg_df[reg_df['Target'] == t].sort_values('R2', ascending=False).iloc[0]
    summary['best_regression'][t] = best.to_dict()

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*65}")
print(f"  COMPREHENSIVE ML PIPELINE COMPLETE")
print(f"{'='*65}")
print(f"Best Classifier: {best_clf['Model']} (AUC={best_clf['AUC_ROC']:.4f})")
for t in reg_targets:
    best = reg_df[reg_df['Target'] == t].sort_values('R2', ascending=False).iloc[0]
    print(f"Best Regression ({t}): {best['Model']} (R²={best['R2']:.4f}, RMSE={best['RMSE']:.2f})")
print(f"\nAll outputs saved to: {OUT_DIR}/")
