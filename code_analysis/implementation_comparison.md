# Implementation Comparison: FeasibleStandard.py vs solver_algo1.py

This document provides a detailed comparison between two implementations of Algorithm 1:
1. **FeasibleStandard.py** (Marco's simplified version)
2. **solver_algo1.py** (Vincenzo's full-featured version)

Both implementations target the same algorithm from the LaTeX document, but with different design choices.

---

## Executive Summary

| Aspect | FeasibleStandard.py | solver_algo1.py | Verdict |
|--------|---------------------|------------------|---------|
| **Complexity** | Simple, direct | Full-featured, modular | Both valid |
| **E matrix** | Explicitly formed | Operator form (never formed) | Different approaches |
| **System formulation** | H = Q + diag(z/x) + δI | M = Z + XQ | Mathematically equivalent |
| **Schur complement** | Explicit S = E H^{-1} E^T | Operator form S v = E(M^{-1}(X(E^T v))) | Different approaches |
| **Sparse support** | ❌ Dense only | ✅ Dense + sparse | solver_algo1.py more general |
| **Large-scale** | ❌ Limited (forms full matrices) | ✅ PCG for large |K| | solver_algo1.py scales better |
| **Initialization δ** | Single global δ | Per-block δ_k | Minor difference |
| **Numerical robustness** | Basic | Advanced (regularization, checks) | solver_algo1.py more robust |

**Overall**: Both implementations are **mathematically correct** and follow Algorithm 1. `FeasibleStandard.py` is simpler and sufficient for small-to-medium problems. `solver_algo1.py` is more robust and scalable for larger problems.

---

## 1. Initialization (Step 1)

### FeasibleStandard.py (Lines 38-60)
```python
# Uniform primal
x = np.zeros(n)
for Ik in blocks:
    x[Ik] = 1.0 / len(Ik)

w = Q @ x + q

# Single global δ
range_max = 0.0
for Ik in blocks:
    wb = w[Ik]
    if wb.size > 1:
        range_max = max(range_max, float(wb.max() - wb.min()))
delta = max(eps, tau * range_max)  # Single δ for all blocks

# Dual start
y = np.zeros(K)
for k, Ik in enumerate(blocks):
    y[k] = -float(w[Ik].min()) + delta  # Same δ for all blocks

z = w + E.T @ y
mu = float(x @ z) / n
```

### solver_algo1.py (Lines 113-174)
```python
# Uniform primal (same)
x[block] = 1.0 / block_size

# Per-block δ_k
delta_per_block = np.zeros(self.n_blocks, dtype=float)
for k, block in enumerate(self.blocks):
    w_block = w[block]
    w_range = w_max - w_min
    delta_k = max(self.cfg['eps_delta'], self.cfg['tau_delta'] * w_range)
    delta_per_block[k] = delta_k  # Different δ per block

# Dual start with per-block δ
y[k] = -w_min + delta_k  # Block-specific δ
```

### Comparison
- ✅ **Both correct**: Both use the δ-rule from the document
- ⚠️ **Difference**: `FeasibleStandard.py` uses a **single global δ** (max over all blocks), while `solver_algo1.py` uses **per-block δ_k** (adaptive per block)
- 📝 **Note**: The document suggests per-block δ_k, but a single global δ is also valid and simpler

**Verdict**: ✅ Both are correct. Per-block δ_k is more adaptive, but global δ is simpler and sufficient.

---

## 2. E Matrix Construction

### FeasibleStandard.py (Lines 28-31)
```python
# Explicitly forms full E matrix
E = np.zeros((K, n))
for k, Ik in enumerate(blocks):
    E[k, Ik] = 1.0
```

### solver_algo1.py (block_ops.py)
```python
# Never forms E matrix, uses operator form
def apply_E(x, blocks):
    y = np.zeros(n_blocks)
    for k in range(n_blocks):
        y[k] = np.sum(x[blocks[k]])  # Direct sum, no matrix
```

### Comparison
- ✅ **Both correct**: Mathematically equivalent
- ⚠️ **Memory**: `FeasibleStandard.py` stores O(K×n) matrix, `solver_algo1.py` uses O(1) memory
- ⚠️ **Performance**: For small problems, explicit E is fine. For large problems, operator form is better

**Verdict**: ✅ Both correct. Operator form is more memory-efficient for large problems.

---

## 3. Residual Computation (Step 2.1)

### FeasibleStandard.py (Lines 72-75)
```python
rD = Q @ x + q + E.T @ y - z
rP = E @ x - onesK
rC = x * z - mu_target * onesn  # Computed later at line 87
```

### solver_algo1.py (Lines 191-209)
```python
r_D = Qx + self.q + Ety - self.z
r_P = Ex - 1.0
r_C = XZ - mu_target  # Same computation
```

### Comparison
- ✅ **Identical**: Both compute residuals exactly as specified in Algorithm 1

**Verdict**: ✅ Fully matches.

---

## 4. System Assembly (Step 2.2) - CRITICAL DIFFERENCE

### FeasibleStandard.py (Lines 85-88)
```python
# Uses H formulation (exactly as in Algorithm 1)
H = Q + np.diag(z / x) + delta * np.eye(n)  # H = Q + X^{-1}Z + δI
mu_target = sigma * mu
rC = x * z - mu_target * onesn
rhs = -rD - (rC / x)  # rhs = -rD - X^{-1} rC
```

### solver_algo1.py (Lines 222-289)
```python
# Uses M formulation (alternative from document)
M = Z + X @ Q  # M = Z + XQ (not H!)
# Then uses M^{-1} in operator form
```

### Comparison
- ⚠️ **Different formulations**: 
  - `FeasibleStandard.py`: Uses **H = Q + diag(z/x) + δI** (exactly as Algorithm 1)
  - `solver_algo1.py`: Uses **M = Z + XQ** (alternative formulation from document)
- ✅ **Mathematically equivalent**: The document states both are equivalent (H = X^{-1}M + δI)
- ⚠️ **Numerical stability**: 
  - H formulation: Requires computing `z/x` (division by x, can be unstable if x is small)
  - M formulation: Avoids explicit division, more stable

**Verdict**: ✅ Both correct. `FeasibleStandard.py` matches Algorithm 1 exactly. `solver_algo1.py` uses a more numerically stable alternative.

---

## 5. Schur System RHS (Step 2.2) - NEEDS VERIFICATION

### Document Specification (Algorithm 1):
- `rhs = μ̃ X^{-1}1 - Z1` where μ̃ = σμ
- `b = E H^{-1} rhs`

### FeasibleStandard.py (Lines 88, 91-94)
```python
rhs = -rD - (rC / x)  # rhs = -rD - X^{-1} rC
H_inv_rhs = np.linalg.solve(H, rhs)
b = E @ H_inv_rhs + rP
```

**Analysis**:
- `rC = x*z - μ_target = XZ1 - μ̃1`
- `rC / x = X^{-1} rC = X^{-1}(XZ1 - μ̃1) = Z1 - μ̃ X^{-1}1`
- So: `rhs = -rD - (Z1 - μ̃ X^{-1}1) = -rD - Z1 + μ̃ X^{-1}1`

In **feasible-start**, `rD = 0` (theoretically), so:
- `rhs = μ̃ X^{-1}1 - Z1` ✅ **MATCHES DOCUMENT**

For `b`:
- Document: `b = E H^{-1} rhs`
- Code: `b = E @ H_inv_rhs + rP = E H^{-1} rhs + rP`

In **feasible-start**, `rP = 0` (theoretically), so:
- `b = E H^{-1} rhs` ✅ **MATCHES DOCUMENT**

### solver_algo1.py (Lines 360-389)
```python
# Uses M formulation
rhs_core = r_C + self.x * r_D  # r_C + X r_D
M_inv_rhs = self.solve_M_system(rhs_core)
b = r_P - apply_E(M_inv_rhs, self.blocks)  # r_P - E M^{-1} rhs_core
```

**Analysis** (using M formulation):
- With `M = Z + XQ` and `H = X^{-1}M + δI`, we have `H^{-1} ≈ M^{-1} X` (ignoring δI for now)
- The document's `rhs = μ̃ X^{-1}1 - Z1` becomes `M^{-1} X (μ̃ X^{-1}1 - Z1) = M^{-1}(μ̃1 - XZ1) = -M^{-1} r_C`
- So: `b = E H^{-1} rhs ≈ E M^{-1} (-r_C) = -E M^{-1} r_C`

The code computes: `b = r_P - E M^{-1} (r_C + X r_D)`

In **feasible-start** (`r_P = 0`, `r_D = 0`):
- `b = -E M^{-1} r_C` ✅ **MATCHES** (after accounting for M formulation)

### Comparison
- ✅ **Both correct**: Both match the document after accounting for their respective formulations
- ⚠️ **Difference**: `FeasibleStandard.py` includes `+ rP` term (for robustness), `solver_algo1.py` includes `- r_P` term (sign difference due to M formulation)

**Verdict**: ✅ Both are mathematically correct. The sign difference is due to different formulations (H vs M).

---

## 6. Schur Complement Computation (Step 2.3)

### FeasibleStandard.py (Lines 91-93)
```python
H_inv_ET = np.linalg.solve(H, E.T)  # Solves H * (H_inv_ET) = E.T
S = E @ H_inv_ET  # S = E H^{-1} E^T (explicit formation)
```

### solver_algo1.py (Lines 391-487)
```python
# Operator form: S v = E(M^{-1}(X(E^T v)))
def schur_operator(v):
    Ety = apply_E_transpose(v, self.blocks, self.n)
    X_Ety = self.x * Ety
    M_inv_X_Ety = self.solve_M_system(X_Ety)
    Sv = apply_E(M_inv_X_Ety, self.blocks)
    return Sv

# If |K| <= K_th: assemble S explicitly
# Otherwise: use PCG on operator
```

### Comparison
- ✅ **Both correct**: Mathematically equivalent
- ⚠️ **Memory**: 
  - `FeasibleStandard.py`: Always forms full S (O(K²) memory)
  - `solver_algo1.py`: Only forms S if |K| ≤ K_th, otherwise uses operator (O(1) memory)
- ⚠️ **Scalability**: 
  - `FeasibleStandard.py`: Limited by K (must form full S)
  - `solver_algo1.py`: Can handle large K using PCG

**Verdict**: ✅ Both correct. `solver_algo1.py` is more scalable for large problems.

---

## 7. Direction Computation (Step 2.4)

### FeasibleStandard.py (Lines 97-99)
```python
dy = np.linalg.solve(S, b)
dx = np.linalg.solve(H, rhs - E.T @ dy)
dz = rD + Q @ dx + E.T @ dy
```

### solver_algo1.py (Lines 489-538)
```python
d_y = self.schur_solve(b)  # Uses operator or explicit S
d_x = self.solve_M_system(rhs_dx)  # Uses M formulation
d_z = Q_dx + Ety_dy + r_D  # Includes r_D for robustness
```

### Comparison
- ✅ **Both correct**: Both compute directions correctly
- ⚠️ **Formulation difference**: 
  - `FeasibleStandard.py`: Uses H formulation directly
  - `solver_algo1.py`: Uses M formulation (equivalent)

**Verdict**: ✅ Both correct.

---

## 8. Step-Size Selection (Step 2.5)

### FeasibleStandard.py (Lines 103-109)
```python
def frac_to_boundary(v, dv):
    mask = dv < 0
    if np.any(mask):
        return min(1.0, 0.99 * float(np.min(-v[mask] / dv[mask])))
    return 1.0

alpha = min(frac_to_boundary(x, dx), frac_to_boundary(z, dz))
```

### solver_algo1.py (utils.py, Lines 8-54)
```python
def fraction_to_boundary(x, d_x, z, d_z, gamma=0.99):
    # Same logic, but more structured
    alpha_pri = min(1.0, gamma * np.min(ratios)) if np.any(neg_dx) else 1.0
    alpha_dual = min(1.0, gamma * np.min(ratios)) if np.any(neg_dz) else 1.0
    alpha = min(alpha_pri, alpha_dual)
```

### Comparison
- ✅ **Identical logic**: Both implement the fraction-to-boundary rule correctly
- ⚠️ **Code quality**: `solver_algo1.py` is more structured and returns all three values

**Verdict**: ✅ Both correct.

---

## 9. Update (Step 2.6)

### FeasibleStandard.py (Lines 112-116)
```python
x += alpha * dx
y += alpha * dy
z += alpha * dz
x = np.maximum(x, tiny)  # Clamping to avoid numerical issues
z = np.maximum(z, tiny)
```

### solver_algo1.py (Lines 560-563)
```python
self.x = self.x + alpha * d_x
self.y = self.y + alpha * d_y
self.z = self.z + alpha * d_z
# No clamping (relies on fraction-to-boundary to maintain positivity)
```

### Comparison
- ✅ **Both correct**: Both update variables correctly
- ⚠️ **Clamping**: `FeasibleStandard.py` clamps to `tiny = 1e-16` for safety, `solver_algo1.py` relies on fraction-to-boundary
- 📝 **Note**: Clamping is a safety measure, but fraction-to-boundary should be sufficient

**Verdict**: ✅ Both correct. Clamping is a conservative safety measure.

---

## 10. Stopping Criterion (Step 2.6)

### FeasibleStandard.py (Lines 78-82)
```python
xz_max = float(np.max(x * z))
if (info["rP_inf"][-1] <= eps_feas and
    info["rD_inf"][-1] <= eps_feas and
    (mu <= eps_comp or xz_max <= eps_comp)):
    break
```

### solver_algo1.py (Lines 565-592)
```python
def converged(self, r_D, r_P):
    norm_r_P = norm_inf(r_P)
    norm_r_D = norm_inf(r_D)
    feas_ok = norm_r_P <= self.cfg['eps_feas'] and norm_r_D <= self.cfg['eps_feas']
    comp_ok = self.mu <= self.cfg['eps_comp']
    return feas_ok and comp_ok
```

### Comparison
- ✅ **Both correct**: Both check feasibility and complementarity
- ⚠️ **Additional check**: `FeasibleStandard.py` also checks `xz_max <= eps_comp` (more conservative)
- 📝 **Note**: The document only requires `μ ≤ ε_comp`, but checking `xz_max` is also reasonable

**Verdict**: ✅ Both correct. `FeasibleStandard.py` has an additional conservative check.

---

## 11. Additional Features

### FeasibleStandard.py
- ✅ Simple, straightforward implementation
- ✅ Explicitly forms matrices (easier to debug)
- ❌ No sparse matrix support
- ❌ Limited scalability (forms full S matrix)
- ❌ No PCG for large |K|

### solver_algo1.py
- ✅ Modular, object-oriented design
- ✅ Sparse matrix support
- ✅ Operator form (memory efficient)
- ✅ PCG for large |K| (scalable)
- ✅ Advanced regularization
- ✅ Comprehensive error handling
- ✅ Verbosity levels for debugging

---

## Mathematical Correctness Verification

### Key Equations Check

#### 1. Newton System (from document)
For feasible-start with H formulation:
- $(Q + X^{-1}Z + δI) \Delta x + E^T \Delta y = -r_D - X^{-1} r_C$
- $E \Delta x = -r_P$

With $r_P = 0$ and $r_D = 0$:
- $(Q + X^{-1}Z + δI) \Delta x + E^T \Delta y = -X^{-1} r_C$
- $E \Delta x = 0$

#### 2. FeasibleStandard.py Verification
- Line 85: `H = Q + diag(z/x) + delta*I` ✅ (H = Q + X^{-1}Z + δI)
- Line 88: `rhs = -rD - (rC / x)` ✅ (rhs = -r_D - X^{-1} r_C)
- Line 98: `dx = solve(H, rhs - E.T @ dy)` ✅ (solves H Δx = rhs - E^T Δy)
- Line 94: `b = E @ H_inv_rhs + rP` ✅ (b = E H^{-1} rhs + r_P, with r_P=0 in feasible-start)

#### 3. solver_algo1.py Verification
Using M formulation with $M = Z + XQ$:
- The relationship: $H = X^{-1}M + δI$ (from document)
- So: $H^{-1} ≈ M^{-1} X$ (ignoring δI regularization)
- The Newton system becomes: $M^{-1} X (rhs) = M^{-1} X (-r_D - X^{-1} r_C) = M^{-1}(-X r_D - r_C)$
- Code computes: `rhs_core = r_C + X r_D`, then `M^{-1} rhs_core` ✅

**Verdict**: ✅ Both implementations are **mathematically correct**.

---

## Performance Considerations

### Small Problems (n < 1000, K < 100)
- **FeasibleStandard.py**: ✅ Excellent (simple, direct)
- **solver_algo1.py**: ✅ Excellent (slight overhead from modularity)

### Medium Problems (n < 10000, K < 500)
- **FeasibleStandard.py**: ⚠️ May struggle (forms full S matrix)
- **solver_algo1.py**: ✅ Good (can use explicit S or operator)

### Large Problems (n > 10000, K > 500)
- **FeasibleStandard.py**: ❌ Not suitable (forms full S matrix, O(K²) memory)
- **solver_algo1.py**: ✅ Suitable (uses PCG on operator, O(1) memory)

### Sparse Q Matrices
- **FeasibleStandard.py**: ❌ No support (converts to dense)
- **solver_algo1.py**: ✅ Full support (handles sparse efficiently)

---

## Recommendations

### Use FeasibleStandard.py when:
- ✅ Problem is small-to-medium (n < 5000, K < 200)
- ✅ Q is dense
- ✅ You want a simple, easy-to-understand implementation
- ✅ Debugging is easier with explicit matrices
- ✅ You want to match Algorithm 1 exactly (H formulation)

### Use solver_algo1.py when:
- ✅ Problem is large (n > 5000 or K > 200)
- ✅ Q is sparse
- ✅ You need scalability
- ✅ You want advanced features (regularization, PCG, etc.)
- ✅ You prefer numerical stability (M formulation avoids z/x division)

### Hybrid Approach:
Consider using `FeasibleStandard.py` as a reference implementation and `solver_algo1.py` for production use.

---

## Conclusion

Both implementations are **mathematically correct** and follow Algorithm 1. The main differences are:

1. **Formulation**: H (FeasibleStandard.py) vs M (solver_algo1.py) - both equivalent
2. **E matrix**: Explicit (FeasibleStandard.py) vs operator (solver_algo1.py) - both correct
3. **Scalability**: Limited (FeasibleStandard.py) vs scalable (solver_algo1.py)
4. **Features**: Basic (FeasibleStandard.py) vs advanced (solver_algo1.py)

**FeasibleStandard.py is NOT oversimplified** - it's a clean, direct implementation that matches Algorithm 1 exactly. It's sufficient for small-to-medium problems. **solver_algo1.py** is more feature-rich and scalable, making it better for larger problems and production use.

Both are valid implementations! 🎯

