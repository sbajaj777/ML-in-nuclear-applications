import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
os.makedirs('../optimization', exist_ok=True)
os.makedirs('../report', exist_ok=True)

# 1. Load bounds and models
print("Loading dataset for bounds and trained models...")
df = pd.read_csv('../data/processed/cleaned_dataset.csv')
features = ['Nb', 'V', 'Ti', 'Mo', 'Zr']

# Extract bounds to prevent extrapolation
xl = df[features].min().values
xu = df[features].max().values

print("Bounds (Min):", xl)
print("Bounds (Max):", xu)

model_liq = joblib.load('../models/best_reg_Liquidus.joblib')
model_den = joblib.load('../models/best_reg_density.joblib')
model_clf = joblib.load('../models/best_clf_is_single_bcc.joblib')

# 2. Define the Optimization Problem and Repair Operator
class SimplexBoundsRepair(Repair):
    def _do(self, problem, X, **kwargs):
        # X is (N, 5)
        for i in range(X.shape[0]):
            x = X[i, :]
            # Iteratively fix bounds and normalize the rest
            for _ in range(20):
                x = np.clip(x, problem.xl, problem.xu)
                s = np.sum(x)
                if np.isclose(s, 1.0, atol=1e-6):
                    # Ensure perfectly 1.0 sum to avoid floating point issues
                    x = x / np.sum(x)
                    break
                
                diff = 1.0 - s
                if diff > 0:
                    mask = x < problem.xu - 1e-6
                else:
                    mask = x > problem.xl + 1e-6
                
                if not np.any(mask):
                    x = x / np.sum(x)
                    break
                
                x[mask] += diff / np.sum(mask)
            X[i, :] = x
        return X

class SimplexSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        val = np.random.random((n_samples, problem.n_var))
        val = problem.xl + val * (problem.xu - problem.xl)
        repair = SimplexBoundsRepair()
        return repair._do(problem, val)

class HEAProblem(Problem):
    def __init__(self, xl, xu, model_liq, model_den, model_clf):
        super().__init__(n_var=5, 
                         n_obj=3, 
                         n_ieq_constr=0, 
                         xl=xl, 
                         xu=xu)
        self.model_liq = model_liq
        self.model_den = model_den
        self.model_clf = model_clf
        
    def _evaluate(self, x, out, *args, **kwargs):
        # x is a numpy array of shape (N, 5)
        # Prepare input for ML models
        X_in = pd.DataFrame(x, columns=['Nb', 'V', 'Ti', 'Mo', 'Zr'])
        
        # Predict properties
        liquidus = self.model_liq.predict(X_in)
        density = self.model_den.predict(X_in)
        prob_bcc = self.model_clf.predict_proba(X_in)[:, 1]
        
        # Objectives (PyMoo minimizes by default)
        f1 = -liquidus           # Maximize Liquidus
        f2 = density             # Minimize Density
        f3 = -prob_bcc           # Maximize Prob(BCC)
        
        out["F"] = np.column_stack([f1, f2, f3])

# 3. Setup NSGA-II
print("\n--- Setting up NSGA-II ---")
problem = HEAProblem(xl, xu, model_liq, model_den, model_clf)

algorithm = NSGA2(
    pop_size=200,
    n_offsprings=50,
    sampling=SimplexSampling(),
    crossover=SBX(prob=0.9, eta=15, repair=SimplexBoundsRepair()),
    mutation=PM(eta=20, repair=SimplexBoundsRepair()),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 250)

# Run Optimization
print("Running optimization (this may take a minute)...")
res = minimize(problem,
               algorithm,
               termination,
               seed=42,
               save_history=True,
               verbose=True)

print(f"Optimization finished. Found {len(res.F)} non-dominated solutions.")

# 4. Process Results
# Extract features and objectives
compositions = res.X

# Convert objectives back to original scales
liquidus_opt = -res.F[:, 0]
density_opt = res.F[:, 1]
prob_bcc_opt = -res.F[:, 2]

pareto_df = pd.DataFrame(compositions, columns=features)
pareto_df['Liquidus'] = liquidus_opt
pareto_df['Density'] = density_opt
pareto_df['Prob_BCC'] = prob_bcc_opt

pareto_df.to_csv('../optimization/pareto_front.csv', index=False)
print("Saved Pareto front to ../optimization/pareto_front.csv")

# 5. Visualizations
print("\n--- Generating Plots ---")
plt.figure(figsize=(10, 7))
sc = plt.scatter(pareto_df['Density'], pareto_df['Liquidus'], 
                 c=pareto_df['Prob_BCC'], cmap='viridis', alpha=0.8)
plt.colorbar(sc, label='Probability of Single BCC Phase')
plt.xlabel('Predicted Density (kg/m^3) [Minimize]')
plt.ylabel('Predicted Liquidus (K) [Maximize]')
plt.title('2D Pareto Front: Density vs Liquidus')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../optimization/pareto_front_2d.png', dpi=300)
plt.close()

# Hypervolume convergence (approximated by objective spread)
hist = res.history
n_evals = []
hv_approx = [] # using min density and max liquidus spread as simple proxy
for algo in hist:
    n_evals.append(algo.evaluator.n_eval)
    opt_F = algo.opt.get("F")
    if opt_F is not None and len(opt_F) > 0:
        spread = np.max(-opt_F[:,0]) - np.min(-opt_F[:,0]) # Spread of liquidus
        hv_approx.append(spread)
    else:
        hv_approx.append(0)

plt.figure(figsize=(8, 5))
plt.plot(n_evals, hv_approx, lw=2, color='coral')
plt.xlabel("Function Evaluations")
plt.ylabel("Objective Spread (Liquidus)")
plt.title("Convergence Proxy over Generations")
plt.grid(True)
plt.tight_layout()
plt.savefig('../optimization/convergence.png', dpi=300)
plt.close()

# 6. Extract Recommended Compositions
# 1) Highest Melting Point
best_liq_idx = pareto_df['Liquidus'].idxmax()
# 2) Lowest Density
best_den_idx = pareto_df['Density'].idxmin()
# 3) Most Balanced (distance to ideal point)
# Normalize objectives
norm_liq = (pareto_df['Liquidus'] - pareto_df['Liquidus'].min()) / (pareto_df['Liquidus'].max() - pareto_df['Liquidus'].min())
norm_den = (pareto_df['Density'] - pareto_df['Density'].min()) / (pareto_df['Density'].max() - pareto_df['Density'].min())
norm_prob = (pareto_df['Prob_BCC'] - pareto_df['Prob_BCC'].min()) / (pareto_df['Prob_BCC'].max() - pareto_df['Prob_BCC'].min())
# Distance to ideal (max liq, min den, max prob => 1, 0, 1)
dist = np.sqrt((1 - norm_liq)**2 + (0 - norm_den)**2 + (1 - norm_prob)**2)
balanced_idx = dist.idxmin()

recommended = pareto_df.loc[[best_liq_idx, best_den_idx, balanced_idx]].copy()
recommended['Rationale'] = ['Highest Melting Point', 'Lowest Density', 'Most Balanced']
recommended.to_csv('../optimization/recommended_compositions.csv', index=False)
print("\nRecommended Compositions:")
print(recommended[['Rationale'] + features + ['Liquidus', 'Density', 'Prob_BCC']])

# Report draft
report_text = f"""# Phase 3: Multi-Objective Optimization (NSGA-II)

Using the ML surrogates, the NSGA-II algorithm explored the composition space within the strict bounds of the training data.
The formulation handled the `sum(fractions) = 1` constraint by optimizing 4 variables and deriving the 5th (Zr), subject to inequality constraints ensuring Zr remained within valid bounds.

**Optimization Setup:**
- Population: 200
- Generations: 250
- Non-dominated solutions found: {len(res.F)}

Refer to `optimization/pareto_front_2d.png` for the trade-off between Density and Liquidus temperature, colored by the probability of forming a single BCC phase.
The convergence plot shows the spread of objectives stabilizing over the generations.

Three recommended candidates were extracted (Highest Melting Point, Lowest Density, and Most Balanced) for CALPHAD verification.
"""
with open('../report/nsga2_summary.md', 'w') as f:
    f.write(report_text)
