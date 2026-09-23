# Phase 3: Multi-Objective Optimization (NSGA-II)

Using the ML surrogates, the NSGA-II algorithm explored the composition space within the strict bounds of the training data.
The formulation handled the `sum(fractions) = 1` constraint by optimizing 4 variables and deriving the 5th (Zr), subject to inequality constraints ensuring Zr remained within valid bounds.

**Optimization Setup:**
- Population: 200
- Generations: 250
- Non-dominated solutions found: 200

Refer to `optimization/pareto_front_2d.png` for the trade-off between Density and Liquidus temperature, colored by the probability of forming a single BCC phase.
The convergence plot shows the spread of objectives stabilizing over the generations.

Three recommended candidates were extracted (Highest Melting Point, Lowest Density, and Most Balanced) for CALPHAD verification.
