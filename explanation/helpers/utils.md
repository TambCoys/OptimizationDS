# Helper Functions Analysis: `utils.py`

Based on the theoretical background from `feedback2.tex` and the lessons learned from the naive `algo1_baseline_solver.py`, this document analyzes the helper functions located in `helper/utils.py`. 

These utility functions provide essential numerical and algorithmic support for the Interior-Point Method, handling step size calculations, norm evaluations, and matrix conditioning checks.

---

## 1. Walkthrough: `utils.py`

### 1.1 `fraction_to_boundary(x, d_x, z, d_z, gamma=0.99)`
```python
def fraction_to_boundary(x, d_x, z, d_z, gamma=0.99):
    # Primal: find largest alpha such that x + alpha * d_x > 0
    alpha_pri = 1.0
    neg_dx = d_x < 0
    if np.any(neg_dx):
        ratios = -x[neg_dx] / d_x[neg_dx]
        alpha_pri = min(1.0, gamma * np.min(ratios))
    
    # Dual: find largest alpha such that z + alpha * d_z > 0
    alpha_dual = 1.0
    neg_dz = d_z < 0
    if np.any(neg_dz):
        ratios = -z[neg_dz] / d_z[neg_dz]
        alpha_dual = min(1.0, gamma * np.min(ratios))
    
    alpha = min(alpha_pri, alpha_dual)
    return alpha_pri, alpha_dual, alpha
```
*   **Mathematical Meaning:** As defined in `feedback2.tex`, interior-point methods must maintain strict positivity ($x > 0, z > 0$). If a Newton step direction ($\Delta x$ or $\Delta z$) is negative for a specific coordinate, taking a full step ($\alpha = 1$) might push the variable below zero. This function calculates the maximum step size $\alpha \in (0, 1]$ that keeps all variables strictly positive, scaled back slightly by a safety factor $\gamma$ (usually 0.99) so the iterates don't get stuck exactly on the boundary.
*   **Computational Choice:** 
    *   It uses boolean masking (`neg_dx = d_x < 0`) to only compute ratios for variables moving towards zero. This avoids division-by-zero errors.
    *   Unlike the naive baseline which used a single `alpha` calculation, this function explicitly separates `alpha_pri` and `alpha_dual`. While `algo1_final_solver.py` ultimately uses the minimum of the two (`alpha = min(alpha_pri, alpha_dual)`) to update the variables, returning both allows the solver to log them separately for debugging. If `alpha_pri` is tiny but `alpha_dual` is 1.0, it tells you the primal variables are hitting the boundary, which is crucial diagnostic information.

### 1.2 `norm_inf(a)`
```python
def norm_inf(a):
    """Compute infinity norm (max absolute value)."""
    return np.linalg.norm(a, ord=np.inf)
```
*   **Mathematical Meaning:** The infinity norm $||a||_\infty$ is simply the maximum absolute value of the vector $a$.
*   **Computational Choice:** Strictly speaking, this function is not strictly necessary, as one could just write `np.linalg.norm(a, ord=np.inf)` everywhere. However, in `algo1_final_solver.py`, the infinity norm is used constantly to check convergence criteria ($||r_P||_\infty \le \epsilon_{feas}$ and $||r_D||_\infty \le \epsilon_{feas}$). Wrapping it in a tiny helper function makes the main solver code much cleaner and closer to the mathematical notation in `feedback2.tex`.