# Project 34: QP Solver on Cartesian Product of Simplices

Implementation of Algorithm 1: Feasible-Start, Fixed σ Interior Point Method for solving convex QP problems on a Cartesian product of simplices.

## Problem Formulation

Solve:

```bash
min  (1/2) x^T Q x + q^T x
s.t. Ex = 1,  x >= 0
```

where:

- Q is a symmetric positive semidefinite matrix
- Blocks {I_k} partition indices (simplex constraint per block)
- E is the block-sum operator (never formed explicitly)

## Implementation

### Files

- `run_p34.py`: Main entry point - loads problem data and calls solver
- `solver_algo1.py`: Core Algorithm 1 implementation (FeasibleStartIPM class)
- `block_ops.py`: Block operators (apply_E, apply_E^T) - never forms full E matrix
- `utils.py`: Numerics helpers (fraction-to-boundary, norms, rcond check)

### Algorithm Features

- **Feasible-start initialization**: δ-rule for strictly interior starting point
- **Fixed centering parameter**: σ ∈ (0, 0.5)
- **Newton backend**: Uses M = Z + XQ (SPD) with optional regularization
- **Schur system**: Operator form S v = E(M^{-1}(X(E^T v)))
  - Assembles full matrix if |K| ≤ K_th (default 200)
  - Uses PCG on operator if |K| > K_th
- **Dense/sparse support**: Automatically handles both dense and sparse Q matrices

## Usage

### Basic Usage

```python
from solver_algo1 import FeasibleStartIPM
import numpy as np

# Problem data
Q = np.array([[2, 1], [1, 2]])  # PSD matrix
q = np.array([-1, -1])
blocks = [[0], [1]]  # Two blocks, one index each

# Configuration
cfg = {
    'sigma': 0.1,
    'max_iter': 100,
    'eps_feas': 1e-8,
    'eps_comp': 1e-8,
    'verbosity': 2,
}

# Solve
solver = FeasibleStartIPM(Q, q, blocks, cfg=cfg)
result = solver.solve()

# Access solution
x = result['x']
y = result['y']
z = result['z']
```

### Run Example

```bash
python run_p34.py
```

## Configuration Parameters

- `sigma`: Centering parameter in (0, 0.5), default 0.1
- `max_iter`: Maximum iterations, default 100
- `eps_feas`: Feasibility tolerance, default 1e-8
- `eps_comp`: Complementarity tolerance, default 1e-8
- `eps_delta`: Minimum δ for initialization, default 1e-8
- `tau_delta`: δ scaling factor, default 1e-2
- `tau_reg`: Regularization parameter (None for adaptive), default None
- `K_th`: Threshold for assembling Schur matrix, default 200
- `pcg_tol`: PCG tolerance, default 1e-10
- `pcg_maxit`: PCG max iterations, default 1000
- `verbosity`: 0=silent, 1=summary, 2=verbose, default 1
- `gamma`: Fraction-to-boundary safety factor, default 0.99

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0

Install with:

```bash
pip install -r requirements.txt
```

## Mathematical Details

See the LaTeX specification document for full mathematical details. Key points:

1. **Initialization (δ-rule)**:
   - Uniform primal: x^0_i = 1/|I_k| for i in I_k
   - Compute w = Qx^0 + q
   - Per-block δ_k = max(ε, τ · (max w_i - min w_i))
   - Dual: y^0_k = -min w_i + δ_k, z^0_i = w_i + y^0_k

2. **Newton Step**:
   - M = Z + XQ (SPD)
   - Schur system: S Δy = b where S = E M^{-1} X E^T
   - Back-substitute: Δx = M^{-1}(-r_C - X E^T Δy), Δz = Q Δx + E^T Δy

3. **Step Size**:
   - Fraction-to-boundary: α = min(α_pri, α_dual) with safety factor γ

4. **Convergence**:
   - ||r_P||∞ ≤ ε_feas
   - ||r_D||∞ ≤ ε_feas
   - μ ≤ ε_comp
