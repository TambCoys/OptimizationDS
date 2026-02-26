# Refactor: M → H Formulation

## Why the change was needed

The original final solver (`algo1_final_solver.py`) used the **M formulation**:

$$M = Z + XQ$$

to build and factorize the reduced Newton system. The code applied **Cholesky factorization** to $M$, but $M$ is **not symmetric**: $(XQ)^\top = QX \neq XQ$. Cholesky silently reads only one triangle and mirrors it, effectively factorizing $\frac{1}{2}(M + M^\top)$ — a different matrix. This produced incorrect Newton directions and caused the solver to drift.

Meanwhile, the **baseline solver** (`algo1_baseline_solver.py`) had always used the correct **H formulation**:

$$H = Q + X^{-1}Z + \delta I$$

which is symmetric (sum of symmetric $Q$ + diagonal $X^{-1}Z$ + scalar $\delta I$), and worked correctly.

## What changed conceptually

1. **The Schur complement elimination naturally gives H, not M.** Eliminating $\Delta z$ from the Newton system yields $(Q + X^{-1}Z)\Delta x + E^\top \Delta y = \text{rhs}$. This is $H$. The $M$ formulation is obtained by multiplying by $X$, which breaks symmetry.

2. **M is kept only as a proof tool.** In the paper, $M = Z + XQ$ is used to prove the system is invertible ($v^\top M v > 0$). But it is not what we factorize.

3. **H is what we factorize and solve.** Since $H$ is SPD, we use Cholesky (dense: `cho_factor`; sparse: `splu` with symmetric permutation), which is ~2× faster and more stable than LU.

4. **$X^{-1}$ is safe in IPMs.** On the central path, $x_i z_i = \mu$, so $z_i/x_i = \mu/x_i^2$ — large but finite for $\mu > 0$. A minor clamp `np.maximum(x, 1e-14)` handles edge cases.

## Code changes (algo1_final_solver.py)

| Component | Before (M) | After (H) |
|---|---|---|
| **Matrix build** | `M = diag(z) + diag(x) @ Q` | `H = Q + diag(z/x)` |
| **Factorization** | `lu_factor(M)` / `splu(permc='COLAMD')` | `cho_factor(H)` / `splu(permc='MMD_AT_PLUS_A')` |
| **Solve** | `lu_solve(M_factor, rhs)` | `cho_solve(H_factor, rhs)` |
| **Schur operator** | $Sv = E\,M^{-1}(X\,E^\top v)$ | $Sv = E\,H^{-1}(E^\top v)$ |
| **Schur RHS** | $b = r_P - E\,M^{-1}(X\,r_D + r_C)$ | $b = r_P - E\,H^{-1}(r_D + X^{-1}r_C)$ |
| **Back-sub Δx** | $M^{-1}(-r_C - X r_D - X E^\top \Delta y)$ | $H^{-1}(-r_D - X^{-1}r_C - E^\top \Delta y)$ |
| **Back-sub Δz** | $Q\Delta x + E^\top\Delta y + r_D$ | $Q\Delta x + E^\top\Delta y + r_D$ (unchanged) |

## Paper changes (feedback2.tex)

- The **"Invertibility of the Schur matrix"** section now clarifies that $M$ is used only for the positive-definiteness proof, and explicitly notes $M$ is not symmetric.
- A new **"Implementation via the H formulation"** section explains why $H$ is symmetric, why Cholesky applies, and why $X^{-1}$ is safe in IPMs.
- The **"Operational solution"** section now uses $H$-based formulas (Schur operator, back-substitution).
- The final Algorithm 2 (Mehrotra) was already written in terms of $H$ — no change needed there.

## What did NOT change

- `algo1_baseline_solver.py` — already used $H$ from the start.
- `baseline_solvers.py` (CVXPY/SciPy wrappers) — unrelated.
- Initialization (δ-rule) — identical for both formulations.
- Step-size selection (fraction-to-boundary) — unchanged.
- Convergence criteria — unchanged.
