# Code vs. Theory Comparison Report
## Algorithm 1: Feasible-Start Primal-Dual Interior-Point Method

This document provides a systematic comparison between the implementation in `solver_algo1.py` and Algorithm 1 from the LaTeX document.

---

## 1. Initialization (Step 1) ✓ MATCHES

### Document Specification:
- For each block $I^k$: set $x_i = 1/|I^k|$ for $i \in I^k$
- Compute $w = Qx + q$
- For each block $k$: $y_k = -\min_{i \in I^k} w_i + \delta$ (with adaptive $\delta$)
- Compute $z = w + E^T y$
- Compute $\mu = (x^T z)/n$

### Code Implementation (`initialize_delta_rule()`):
- ✅ Lines 114-119: Sets $x_i = 1/|I^k|$ for each block
- ✅ Lines 122-125: Computes $w = Qx + q$ (handles both sparse and dense)
- ✅ Lines 128-135: Computes adaptive $\delta_k$ per block using $\max(\varepsilon, \tau \cdot \text{range}(w_{I^k}))$
- ✅ Lines 138-146: Sets $y_k = -\min_{i \in I^k} w_i + \delta_k$ and $z = w + E^T y$
- ✅ Line 149: Computes $\mu = (x^T z)/n$
- ✅ Lines 152-167: Validates strict positivity and feasibility

**Verdict**: ✅ **FULLY MATCHES** - The initialization follows the δ-rule exactly as specified.

---

## 2. Residual Computation (Step 2.1) ✓ MATCHES

### Document Specification:
- $r_D = Qx + q + E^T y - z$
- $r_P = Ex - \mathbf{1}$
- $r_C = XZ\mathbf{1} - \mu\mathbf{1}$ where $\mu_{\text{target}} = \sigma \mu$

### Code Implementation (`compute_residuals()`):
- ✅ Lines 192-197: Computes $r_D = Qx + q + E^T y - z$
- ✅ Lines 199-201: Computes $r_P = Ex - \mathbf{1}$
- ✅ Lines 205-207: Computes $\mu_{\text{target}} = \sigma \mu$ and $r_C = XZ - \mu_{\text{target}}$

**Verdict**: ✅ **FULLY MATCHES** - Residual computation is correct.

---

## 3. System Assembly (Step 2.2) ⚠️ PARTIAL MATCH - Different Formulation

### Document Specification:
The document presents **two formulations**:

**Formulation H (explicit in Algorithm 1):**
- $H = Q + X^{-1}Z + \delta I$ (regularized)
- $\tilde{\mu} = \sigma \mu$
- $\text{rhs} = \tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1}$
- $S = E H^{-1} E^T$
- $b = E H^{-1} \text{rhs}$

**Formulation M (mentioned in theory section):**
- $M = Z + XQ$
- $S = E M^{-1} X E^T$

The document notes: *"Both formulations are connected by $H = X^{-1}M + \delta I$, ensuring consistency between theory and implementation."*

### Code Implementation (`build_M_and_factorize()`):
- ✅ Lines 222-289: Uses **M formulation**: $M = Z + XQ$
- ✅ Lines 234-247 (sparse) / 270-279 (dense): Applies adaptive regularization $\tau$ to $M$ (not $\delta I$ to $H$)
- ⚠️ **Note**: The code uses $M = Z + XQ + \tau I$ (regularized M), which is equivalent to the H formulation when $\delta = \tau$ and the relationship $H = X^{-1}M + \delta I$ is considered.

**Verdict**: ⚠️ **CONCEPTUALLY MATCHES** - The code uses the M formulation (which is mentioned in the document as an alternative), but Algorithm 1 explicitly shows the H formulation. The document acknowledges both are equivalent.

**Recommendation**: The code should add a comment explaining that it uses the M formulation for numerical stability (avoids explicit $X^{-1}$), which is equivalent to the H formulation shown in Algorithm 1.

---

## 4. Schur System RHS (Step 2.2) ⚠️ NEEDS VERIFICATION

### Document Specification (H formulation):
- $\text{rhs} = \tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1}$
- $b = E H^{-1} \text{rhs}$

### Document Specification (M formulation, from theory):
For feasible-start with $r_P = 0$ and $r_D = 0$:
- The reduced system is: $(Q + X^{-1}Z) \Delta x + E^T \Delta y = -X^{-1} r_C$
- With $E \Delta x = 0$

Using $M = Z + XQ$ and the relationship $H = X^{-1}M$:
- $H^{-1} = M^{-1} X$
- So: $H^{-1} \text{rhs} = M^{-1} X (\tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1}) = M^{-1} (\tilde{\mu}\mathbf{1} - XZ\mathbf{1})$

But $r_C = XZ\mathbf{1} - \tilde{\mu}\mathbf{1}$, so:
- $H^{-1} \text{rhs} = M^{-1} (-r_C)$

Therefore: $b = E H^{-1} \text{rhs} = E M^{-1} (-r_C)$

### Code Implementation (`schur_rhs()`):
- ✅ Lines 379-383: Computes `rhs_core = r_C + X * r_D` (includes $X r_D$ term for robustness)
- ✅ Line 385: Computes `M_inv_rhs = M^{-1} * rhs_core`
- ✅ Line 388: Computes `b = r_P - E * M_inv_rhs`

**Analysis**:
- In **feasible-start**, $r_P = 0$ and $r_D = 0$ (theoretically), so:
  - `rhs_core = r_C`
  - `b = 0 - E M^{-1} (-r_C) = E M^{-1} r_C` ❌ **SIGN ERROR**

Wait, let me recalculate. The document says:
- $r_C = XZ\mathbf{1} - \tilde{\mu}\mathbf{1}$
- $\text{rhs} = \tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1}$

We have:
- $\text{rhs} = \tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1} = X^{-1}(\tilde{\mu}\mathbf{1} - XZ\mathbf{1}) = -X^{-1} r_C$

So: $H^{-1} \text{rhs} = H^{-1} (-X^{-1} r_C)$

With $H = X^{-1}M$, we get $H^{-1} = M^{-1} X$, so:
- $H^{-1} \text{rhs} = M^{-1} X (-X^{-1} r_C) = -M^{-1} r_C$

Therefore: $b = E H^{-1} \text{rhs} = E (-M^{-1} r_C) = -E M^{-1} r_C$

But the code computes: `b = r_P - E M^{-1} rhs_core = 0 - E M^{-1} r_C = -E M^{-1} r_C` ✅

**Verdict**: ✅ **MATCHES** (when $r_P = 0$ and $r_D = 0$). The code includes $r_D$ term for numerical robustness (which is good practice).

---

## 5. Schur Operator (Step 2.2) ✓ MATCHES

### Document Specification:
- $S = E M^{-1} X E^T$
- Operator form: $S v = E(M^{-1}(X(E^T v)))$

### Code Implementation (`schur_operator()`):
- ✅ Lines 408-417: Implements exactly $S v = E(M^{-1}(X(E^T v)))$

**Verdict**: ✅ **FULLY MATCHES** - The operator form is correctly implemented.

---

## 6. Schur System Solution (Step 2.3) ✓ MATCHES

### Document Specification:
- If $|K| \leq K_{\text{th}}$: Assemble $S$ explicitly and solve via Cholesky
- Otherwise: Use PCG on the operator form

### Code Implementation (`schur_solve()`):
- ✅ Lines 442-458: Assembles $S$ explicitly if $|K| \leq K_{\text{th}}$ and solves via Cholesky
- ✅ Lines 459-485: Uses PCG on operator form otherwise

**Verdict**: ✅ **FULLY MATCHES** - Matches the document's specification.

---

## 7. Back-Substitution (Step 2.3) ✓ MATCHES

### Document Specification (from theory section):
From the Newton system with $M = Z + XQ$:
- $\Delta x = M^{-1}(-r_C - X E^T \Delta y)$
- $\Delta z = Q \Delta x + E^T \Delta y$ (in feasible-start, $r_D = 0$)

### Code Implementation (`back_substitute_dx_dz()`):
- ✅ Lines 518-527: Computes $\Delta x = M^{-1}(-r_C - X r_D - X E^T \Delta y)$
  - Includes $X r_D$ term for robustness (good practice)
- ✅ Lines 530-536: Computes $\Delta z = Q \Delta x + E^T \Delta y + r_D$
  - Includes $r_D$ term for robustness

**Verdict**: ✅ **MATCHES** (with robustness terms that are zero in exact feasible-start).

---

## 8. Step-Size Selection (Step 2.4) ✓ MATCHES

### Document Specification:
- $\alpha_{\text{pri}} = \min(1, 0.99 \cdot \min_{i: \Delta x_i < 0} -x_i/\Delta x_i)$
- $\alpha_{\text{dual}} = \min(1, 0.99 \cdot \min_{i: \Delta z_i < 0} -z_i/\Delta z_i)$
- $\alpha = \min(\alpha_{\text{pri}}, \alpha_{\text{dual}})$

### Code Implementation (`fraction_to_boundary()` in `utils.py`):
- ✅ Lines 40-44: Computes $\alpha_{\text{pri}}$ exactly as specified
- ✅ Lines 47-51: Computes $\alpha_{\text{dual}}$ exactly as specified
- ✅ Line 53: Computes $\alpha = \min(\alpha_{\text{pri}}, \alpha_{\text{dual}})$
- ✅ Uses configurable `gamma` parameter (default 0.99) instead of hardcoded 0.99

**Verdict**: ✅ **FULLY MATCHES** - Implementation is correct and more flexible.

---

## 9. Update (Step 2.5) ✓ MATCHES

### Document Specification:
- $x \leftarrow x + \alpha \Delta x$
- $y \leftarrow y + \alpha \Delta y$
- $z \leftarrow z + \alpha \Delta z$
- $\mu \leftarrow (x^T z)/n$

### Code Implementation (`update_and_mu()`):
- ✅ Lines 560-563: Updates all variables exactly as specified

**Verdict**: ✅ **FULLY MATCHES** - Correct implementation.

---

## 10. Stopping Criterion (Step 2.6) ✓ MATCHES

### Document Specification:
Stop if:
- $\|r_P\|_\infty \leq \epsilon_{\text{feas}}$
- $\|r_D\|_\infty \leq \epsilon_{\text{feas}}$
- $\mu \leq \epsilon_{\text{comp}}$

### Code Implementation (`converged()`):
- ✅ Lines 586-592: Checks all three conditions exactly as specified

**Verdict**: ✅ **FULLY MATCHES** - Correct implementation.

---

## Summary

### ✅ **FULLY MATCHING COMPONENTS:**
1. Initialization (Step 1) - δ-rule implementation
2. Residual computation (Step 2.1)
3. Schur operator (Step 2.2)
4. Schur system solution (Step 2.3)
5. Step-size selection (Step 2.4)
6. Update (Step 2.5)
7. Stopping criterion (Step 2.6)

### ⚠️ **PARTIAL MATCHES (Conceptually Correct, Different Formulation):**
1. **System Assembly (Step 2.2)**: Code uses M formulation ($M = Z + XQ$) instead of H formulation ($H = Q + X^{-1}Z + \delta I$) shown in Algorithm 1. However, the document acknowledges both formulations are equivalent, and the M formulation is preferred for numerical stability (avoids explicit $X^{-1}$).

2. **Schur RHS**: The code's implementation is correct when accounting for the M formulation. The sign and terms match after algebraic manipulation.

### 📝 **RECOMMENDATIONS:**

1. **Add documentation comment** in `build_M_and_factorize()` explaining that the M formulation is used instead of the H formulation for numerical stability, and that they are equivalent (as noted in the document).

2. **Add documentation comment** in `schur_rhs()` explaining the relationship between the H-formulation RHS ($\tilde{\mu} X^{-1}\mathbf{1} - Z\mathbf{1}$) and the M-formulation RHS ($r_C + X r_D$).

3. **Consider adding** a note in the code that in exact feasible-start, $r_P = 0$ and $r_D = 0$, but the code includes these terms for numerical robustness.

---

## Overall Verdict

✅ **The code methodically and conceptually matches Algorithm 1 from the document.**

The implementation correctly follows the feasible-start primal-dual interior-point method. The main difference is the use of the M formulation instead of the H formulation, which is:
- Mathematically equivalent (as acknowledged in the document)
- Numerically more stable (avoids explicit $X^{-1}$)
- Consistent with the document's theory section

The code includes additional robustness features (handling small $r_P$ and $r_D$ residuals) which are good practice for numerical implementations.

