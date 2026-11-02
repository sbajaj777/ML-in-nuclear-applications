import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

FEATURES = ['Nb', 'V', 'Ti', 'Mo', 'Zr']

# Load existing Pareto front and re-score with new retrained models
pf = pd.read_csv('../optimization/retrain_pareto_front.csv')

m_liq = joblib.load('../models/best_reg_Liquidus.joblib')
m_sol = joblib.load('../models/best_reg_Solidus.joblib')
m_den = joblib.load('../models/best_reg_density.joblib')
m_clf = joblib.load('../models/best_clf_is_single_bcc.joblib')

pf['Liquidus']  = m_liq.predict(pf[FEATURES])
pf['Solidus']   = m_sol.predict(pf[FEATURES])
pf['Density']   = m_den.predict(pf[FEATURES])
pf['Prob_BCC']  = m_clf.predict_proba(pf[FEATURES])[:, 1]

# Filter: only BCC > 50%
good = pf[pf['Prob_BCC'] > 0.5].copy()

# HEA constraint: each element must be 5%-35% (0.05-0.35)
HEA_MIN = 0.05
HEA_MAX = 0.35
before = len(good)
hea_mask = (
    (good['Nb'] >= HEA_MIN) & (good['Nb'] <= HEA_MAX) &
    (good['V']  >= HEA_MIN) & (good['V']  <= HEA_MAX) &
    (good['Ti'] >= HEA_MIN) & (good['Ti'] <= HEA_MAX) &
    (good['Mo'] >= HEA_MIN) & (good['Mo'] <= HEA_MAX) &
    (good['Zr'] >= HEA_MIN) & (good['Zr'] <= HEA_MAX)
)
good = good[hea_mask].copy()
print("After HEA constraint (5-35% each): {} / {} candidates remain".format(len(good), before))
print('Candidates with Prob(BCC) > 50%:', len(good))

# Normalize all 3 objectives to [0,1]
good['liq_norm'] = (good['Liquidus'] - good['Liquidus'].min()) / (good['Liquidus'].max() - good['Liquidus'].min())
good['den_norm'] = 1.0 - (good['Density'] - good['Density'].min()) / (good['Density'].max() - good['Density'].min())
good['bcc_norm'] = (good['Prob_BCC'] - good['Prob_BCC'].min()) / (good['Prob_BCC'].max() - good['Prob_BCC'].min())

# Equal-weight balanced score (higher is better)
good['balance_score'] = (good['liq_norm'] + good['den_norm'] + good['bcc_norm']) / 3.0
good = good.sort_values('balance_score', ascending=False).reset_index(drop=True)

print()
print("=" * 55)
print("TOP 2 MOST BALANCED COMPOSITIONS")
print("=" * 55)

for i in range(2):
    row = good.iloc[i]
    print()
    print("BALANCED #{} (Score = {:.4f})".format(i+1, row['balance_score']))
    print("-" * 40)
    for f in FEATURES:
        print("  {}: {:.4f}".format(f, row[f]))
    print("  Liquidus:  {:.2f} K".format(row['Liquidus']))
    print("  Solidus:   {:.2f} K".format(row['Solidus']))
    print("  Density:   {:.2f} kg/m3".format(row['Density']))
    print("  Prob(BCC): {:.2f}%".format(row['Prob_BCC'] * 100))

# Save
out = good.head(2)[FEATURES + ['Liquidus', 'Solidus', 'Density', 'Prob_BCC', 'balance_score']].copy()
out.insert(0, 'Rationale', ['Most Balanced #1', 'Most Balanced #2'])
out.to_csv('../optimization/retrain_best_compositions.csv', index=False)
print()
print("Saved to optimization/retrain_best_compositions.csv")
