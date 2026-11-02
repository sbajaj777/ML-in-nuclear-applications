"""
Phase Retrain: Combined Dataset ML + NSGA-II Optimization
Combines original CALPHAD dataset + new BTP4output dataset,
retrains all ML surrogate models, then runs NSGA-II to find
the 2 best compositions (Highest Melting Point + Lowest Density)
that are confirmed single-phase BCC.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import os
import ast

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

FEATURES = ['Nb', 'V', 'Ti', 'Mo', 'Zr']
MODEL_DIR = '../models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: LOAD & COMBINE DATASETS
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading and combining datasets...")
print("=" * 60)

df_orig = pd.read_excel('../data/raw/dataset1_calphad_original.xlsx')
df_new  = pd.read_csv('../data/raw/dataset2_calphad_augmented.csv')

print(f"Original dataset: {len(df_orig)} rows")
print(f"New dataset:      {len(df_new)} rows")

# Align columns - keep only what we need
cols_needed = ['Nb', 'V', 'Ti', 'Mo', 'Zr', 'Liquidus', 'Solidus', 'density', 'phases', 'number_phases']
df_orig = df_orig[cols_needed]
df_new  = df_new[cols_needed]

df = pd.concat([df_orig, df_new], ignore_index=True)
print(f"Combined dataset: {len(df)} rows")

# ─────────────────────────────────────────────
# STEP 2: CLEAN DATA
# ─────────────────────────────────────────────
print("\nSTEP 2: Cleaning data...")

# Drop rows missing critical values
df = df.dropna(subset=['Liquidus', 'Solidus', 'density', 'phases'])
print(f"After dropping NaN rows: {len(df)} rows")

# Drop physically impossible Liquidus values (CALPHAD convergence failures)
# Max possible melting point for this alloy system is well below 4000 K
before = len(df)
df = df[df['Liquidus'] < 4000]
print(f"After removing impossible Liquidus (>4000K): removed {before - len(df)} bad rows, {len(df)} remaining")

# Parse phases column to determine single-phase BCC
def is_single_bcc(phases_val):
    try:
        if isinstance(phases_val, str):
            phases_list = ast.literal_eval(phases_val)
        else:
            phases_list = phases_val
        # Single phase and it is a BCC phase
        if len(phases_list) == 1 and 'BCC' in str(phases_list[0]).upper():
            return 1
        return 0
    except:
        return 0

df['is_single_bcc'] = df['phases'].apply(is_single_bcc)
print(f"Single-phase BCC samples: {df['is_single_bcc'].sum()} / {len(df)}")

# Get training bounds from combined cleaned data
bounds = {}
for col in FEATURES:
    bounds[col] = (df[col].min(), df[col].max())
    print(f"  {col}: [{bounds[col][0]:.4f}, {bounds[col][1]:.4f}]")

# ─────────────────────────────────────────────
# STEP 3: TRAIN ML MODELS
# ─────────────────────────────────────────────
print("\nSTEP 3: Training ML surrogate models...")

X = df[FEATURES]
targets = {
    'Liquidus': df['Liquidus'],
    'Solidus':  df['Solidus'],
    'density':  df['density'],
}

def train_best_regressor(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    rf  = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)

    xgb.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    r2_xgb = r2_score(y_te, xgb.predict(X_te))
    r2_rf  = r2_score(y_te, rf.predict(X_te))

    if r2_xgb >= r2_rf:
        best, best_name = xgb, 'XGBoost'
        r2_best = r2_xgb
    else:
        best, best_name = rf, 'RandomForest'
        r2_best = r2_rf

    rmse = np.sqrt(mean_squared_error(y_te, best.predict(X_te)))
    print(f"  {name}: {best_name} | R2={r2_best:.4f} | RMSE={rmse:.2f}")
    path = os.path.join(MODEL_DIR, f'best_reg_{name}.joblib')
    joblib.dump(best, path)
    return best

models = {}
for target_name, y in targets.items():
    models[target_name] = train_best_regressor(X, y, target_name)

# Classifier
print("  Training phase stability classifier...")
X_tr, X_te, y_tr, y_te = train_test_split(X, df['is_single_bcc'], test_size=0.2, random_state=42)
scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
clf = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                    scale_pos_weight=scale_pos, random_state=42, n_jobs=-1,
                    use_label_encoder=False, eval_metric='logloss')
clf.fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te), target_names=['Multi-phase', 'Single BCC']))
joblib.dump(clf, os.path.join(MODEL_DIR, 'best_clf_is_single_bcc.joblib'))
models['clf'] = clf

# ─────────────────────────────────────────────
# STEP 4: NSGA-II OPTIMIZATION
# ─────────────────────────────────────────────
print("\nSTEP 4: Running NSGA-II optimization...")

lb = np.array([bounds[f][0] for f in FEATURES])
ub = np.array([bounds[f][1] for f in FEATURES])

class SimplexBoundsRepair(Repair):
    def _do(self, problem, X, **kwargs):
        X = np.clip(X, lb, ub)
        row_sums = X.sum(axis=1, keepdims=True)
        X = X / row_sums
        X = np.clip(X, lb, ub)
        row_sums = X.sum(axis=1, keepdims=True)
        X = X / row_sums
        return X

class SimplexSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var))
        for i in range(n_samples):
            while True:
                sample = np.random.uniform(lb, ub)
                if sample.sum() > 0:
                    sample = sample / sample.sum()
                    if np.all(sample >= lb) and np.all(sample <= ub):
                        X[i] = sample
                        break
        return X

class RHEAProblem(Problem):
    def __init__(self):
        super().__init__(n_var=5, n_obj=3, xl=lb, xu=ub)

    def _evaluate(self, X, out, *args, **kwargs):
        df_X = pd.DataFrame(X, columns=FEATURES)
        liq      = models['Liquidus'].predict(df_X)
        density  = models['density'].predict(df_X)
        prob_bcc = models['clf'].predict_proba(df_X)[:, 1]
        # Minimize: -Liquidus, +density, -Prob_BCC
        out['F'] = np.column_stack([-liq, density, -prob_bcc])

algorithm = NSGA2(
    pop_size=200,
    sampling=SimplexSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    repair=SimplexBoundsRepair(),
    eliminate_duplicates=True
)

res = minimize(
    RHEAProblem(),
    algorithm,
    ('n_gen', 300),
    seed=42,
    verbose=True
)

print(f"\nNSGA-II complete. Pareto front size: {len(res.X)}")

# ─────────────────────────────────────────────
# STEP 5: EXTRACT 2 BEST COMPOSITIONS
# ─────────────────────────────────────────────
print("\nSTEP 5: Selecting 2 best compositions...")

pareto_df = pd.DataFrame(res.X, columns=FEATURES)
pareto_df['Liquidus']   = -res.F[:, 0]
pareto_df['density']    = res.F[:, 1]
pareto_df['Prob_BCC']   = -res.F[:, 2]

# Save full new Pareto front
pareto_df.to_csv('../optimization/retrain_pareto_front.csv', index=False)
print('Full Pareto front saved to optimization/retrain_pareto_front.csv')

# Filter to only those with Prob_BCC > 0.5
good = pareto_df[pareto_df['Prob_BCC'] > 0.5].copy()
print(f"Candidates with Prob(BCC) > 50%: {len(good)}")

# Composition 1: Highest Melting Point
best_mp  = good.sort_values('Liquidus', ascending=False).iloc[0]

# Composition 2: Lowest Density
best_den = good.sort_values('density', ascending=True).iloc[0]

# Predict Solidus for both
s1 = models['Solidus'].predict(pd.DataFrame([best_mp[FEATURES].values], columns=FEATURES))[0]
s2 = models['Solidus'].predict(pd.DataFrame([best_den[FEATURES].values], columns=FEATURES))[0]

print("\n" + "=" * 60)
print("RESULT 1: HIGHEST MELTING POINT")
print("=" * 60)
for f in FEATURES:
    print(f"  {f}: {best_mp[f]:.4f}")
print(f"  ML Liquidus:  {best_mp['Liquidus']:.2f} K")
print(f"  ML Solidus:   {s1:.2f} K")
print(f"  ML Density:   {best_mp['density']:.2f} kg/m³")
print(f"  Prob(BCC):    {best_mp['Prob_BCC']*100:.2f}%")

print("\n" + "=" * 60)
print("RESULT 2: LOWEST DENSITY")
print("=" * 60)
for f in FEATURES:
    print(f"  {f}: {best_den[f]:.4f}")
print(f"  ML Liquidus:  {best_den['Liquidus']:.2f} K")
print(f"  ML Solidus:   {s2:.2f} K")
print(f"  ML Density:   {best_den['density']:.2f} kg/m³")
print(f"  Prob(BCC):    {best_den['Prob_BCC']*100:.2f}%")

# Save
out_df = pd.DataFrame([
    {'Rationale': 'Highest Melting Point', **{f: best_mp[f] for f in FEATURES},
     'ML_Liquidus': best_mp['Liquidus'], 'ML_Solidus': s1,
     'ML_Density': best_mp['density'], 'ML_Prob_BCC': best_mp['Prob_BCC']},
    {'Rationale': 'Lowest Density', **{f: best_den[f] for f in FEATURES},
     'ML_Liquidus': best_den['Liquidus'], 'ML_Solidus': s2,
     'ML_Density': best_den['density'], 'ML_Prob_BCC': best_den['Prob_BCC']},
])
out_df.to_csv('../optimization/retrain_best_compositions.csv', index=False)
print("\nSaved to optimization/retrain_best_compositions.csv")
print("\nDone!")
