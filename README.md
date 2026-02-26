# Project 34: QP Solver on Cartesian Product of Simplices

Implementation of Algorithm 1: Feasible-Start, Fixed σ Interior Point Method for solving convex QP problems on a Cartesian product of simplices.

## Problem Formulation

Solve:

```
min  (1/2) x^T Q x + q^T x
s.t. Ex = 1,  x >= 0
```

where:

- Q is a symmetric positive semidefinite matrix
- Blocks {I_k} partition indices (simplex constraint per block)
- E is the block-sum operator (never formed explicitly)

## Project Structure

```
main/
  algo1_final_solver.py    # Final solver (FeasibleStartIPM) — H formulation, Cholesky
  algo1_baseline_solver.py # Baseline solver (BaselineIPM) — H formulation, np.linalg.solve
  baseline_solvers.py      # CVXPY and SciPy reference solvers
  helper/
    block_ops.py           # Block operators (apply_E, apply_E^T), example problem generator
    benchmark.py           # Benchmark helpers (timing, feasibility, suite generator)
    utils.py               # Numerics helpers (fraction-to-boundary, norms)
benchmark_runner.py        # CLI benchmark script (single problem or full suite)
feedback2.tex              # Paper / theoretical specification
explanation/               # Detailed markdown explanations of each module
changes/                   # Changelog for major refactors
```

## Solvers

### Final Solver (`algo1_final_solver.py` — `FeasibleStartIPM`)

The production solver. Uses the symmetric **H formulation** (H = Q + X⁻¹Z) with Cholesky factorization, operator-form Schur complement, and automatic dense/sparse dispatch.

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 0.1 | Centering parameter σ ∈ (0, 0.5). Lower = more aggressive |
| `max_iter` | 100 | Maximum IPM iterations |
| `eps_feas` | 1e-8 | Feasibility tolerance (‖r_P‖∞, ‖r_D‖∞) |
| `eps_comp` | 1e-8 | Complementarity tolerance (μ) |
| `eps_delta` | 1e-8 | Minimum δ for initialization |
| `tau_delta` | 1e-2 | δ scaling factor for range-based rule |
| `tau_reg` | None | Regularization: None = adaptive, or fixed float |
| `K_th` | 200 | Block count threshold: ≤K_th assembles Schur, >K_th uses PCG |
| `pcg_tol` | 1e-10 | PCG solver tolerance (when |K| > K_th) |
| `pcg_maxit` | 1000 | PCG max iterations |
| `verbosity` | 1 | 0=silent, 1=summary, 2=per-iter, 3=debug |
| `gamma` | 0.99 | Fraction-to-boundary safety factor |

### Baseline Solver (`algo1_baseline_solver.py` — `BaselineIPM`)

A simpler reference implementation of the same algorithm. Uses the H formulation with `np.linalg.solve` (no Cholesky), explicitly forms the full E matrix, and assembles the Schur complement directly. Meant for validation, not performance.

**Configuration parameters:**

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 0.01 | Centering parameter (more conservative default than final) |
| `max_iter` | 100000 | Maximum iterations (very high to ensure convergence) |
| `eps_feas` | 1e-8 | Feasibility tolerance |
| `eps_comp` | 1e-8 | Complementarity tolerance |
| `eps` | 1e-10 | Minimum ε for δ-rule |
| `tau` | 1e-3 | δ scaling factor |

### Reference Solvers (`baseline_solvers.py`)

- **CVXPY** — Calls OSQP (compiled C solver) via CVXPY. Used as the performance baseline.
- **SciPy** — Uses `scipy.optimize.minimize` with SLSQP. Very slow for QPs; included for completeness.

## Usage

### Basic Usage

```python
from main.algo1_final_solver import FeasibleStartIPM

solver = FeasibleStartIPM(Q, q, blocks, cfg={'sigma': 0.1, 'verbosity': 2})
result = solver.solve()

x = result['x']   # primal solution
y = result['y']   # dual variables (per-block)
z = result['z']   # dual slack
```

### Example Problem Generation

```python
from main.helper.block_ops import create_example_problem

# Dense, n=500, 50 blocks
Q, q, blocks = create_example_problem(n=500, n_blocks=50, seed=42, density=1.0)

# Sparse, n=1000, 5% density
Q, q, blocks = create_example_problem(n=1000, n_blocks=50, seed=42, density=0.05)
```

## Benchmarking

### Single Problem

```bash
uv run benchmark_runner.py --n 500 --n-blocks 50
uv run benchmark_runner.py --n 1000 --n-blocks 100 --density 0.1
uv run benchmark_runner.py --n 100 --n-blocks 10 --debug
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--n` | 20 | Problem size |
| `--n-blocks` | 3 | Number of simplex blocks |
| `--density` | 1.0 | Q matrix density (1.0=dense, <1.0=sparse) |
| `--n-runs` | 1 | Number of timing runs (for averaging) |
| `--seed` | 42 | Random seed |
| `--sigma` | 0.1 | Centering parameter for Algorithm 1 |
| `--max-iter` | 200 | Max iterations for Algorithm 1 |
| `--verbosity` | 0 | Verbosity level |
| `--debug` | off | Print detailed solution statistics |
| `--suite` | off | Run the full benchmark suite (see below) |

### Benchmark Suite

Runs a systematic grid of problems covering three axes:

```bash
uv run benchmark_runner.py --suite
```

The suite includes:
- **Scaling tests** (n = 50, 100, 500, 1000, 2000) — how the solver scales with problem size
- **Sparsity tests** (density = 1.0, 0.5, 0.1, 0.01 at n=1000) — dense vs sparse performance
- **Block structure tests** (|K| = 2, 10, 50, 100, 250 at n=500) — effect of block count

At the end, prints a compact summary table comparing CVXPY vs Final Solver across all configurations.

## Mathematical Details

See `feedback2.tex` for the full theoretical specification. Key points:

1. **Initialization (δ-rule)**: Uniform primal x⁰_i = 1/|I_k|, adaptive δ per block ensuring z⁰ > 0.

2. **Newton system**: Eliminates Δz from the KKT system to get (Q + X⁻¹Z)Δx + E^T Δy = rhs, then uses the Schur complement S = E H⁻¹ E^T to solve for Δy first.

3. **H formulation**: H = Q + X⁻¹Z is symmetric positive-definite → Cholesky factorization. The alternative M = Z + XQ is used only for the invertibility proof (M is not symmetric).

4. **Step size**: Fraction-to-boundary rule with safety factor γ = 0.99.

5. **Convergence**: ‖r_P‖∞ ≤ ε_feas, ‖r_D‖∞ ≤ ε_feas, μ ≤ ε_comp.

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- cvxpy (for baseline comparison)

Install with:

```bash
pip install -r requirements.txt
```
