"""
Baseline QP solver using CVXPY for comparison.
Solves:  min (1/2) x^T Q x + q^T x   s.t.  sum_{i in I_k} x_i = 1  (for each block k),  x >= 0
"""

import time
import numpy as np
import scipy.sparse as sp

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not available. Install with: pip install cvxpy")


def _to_dense(Q, q):
    """Converte Q in denso (se necess.) e q in vettore 1D float."""
    if sp.issparse(Q):
        Q = Q.toarray()
    else:
        Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    return Q, q


def _make_psd(Q, ridge=1e-10):
    """Simmetrizza Q e aggiunge un ridge minimo; restituisce anche la versione psd_wrap per CVXPY."""
    Qsym = 0.5 * (Q + Q.T)
    if ridge and ridge > 0.0:
        Qsym = Qsym + ridge * np.eye(Qsym.shape[0], dtype=Qsym.dtype)
    return Qsym


def solve_baseline_cvxpy(Q, q, blocks, solver="OSQP", solver_opts=None):
    """
    Solve QP using CVXPY.

    Parameters
    ----------
    Q : array-like or sparse, shape (n, n), PSD (o quasi-PSD numericamente)
    q : array-like, shape (n,)
    blocks : list[list[int]]
    solver : str, optional ("OSQP", "ECOS", "CLARABEL", "SCS")
    solver_opts : dict, optional, solver-specific options

    Returns
    -------
    dict: { x, status, value, solve_time }
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY not available")

    # Normalizza input per CVXPY
    Q, q = _to_dense(Q, q)
    n = q.size

    # Rendi Q sicura per quad_form
    Qsym = _make_psd(Q, ridge=1e-10)
    Q_psd = cp.psd_wrap(Qsym)

    # Variabile
    x = cp.Variable(n, nonneg=True)

    # vincoli di blocco: sum_{i in I_k} x_i = 1
    constraints = [cp.sum(x[block]) == 1.0 for block in blocks]

    # obiettivo
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q_psd) + q @ x)
    problem = cp.Problem(objective, constraints)

    # Opzioni default (stabili) per OSQP
    if solver_opts is None:
        solver_opts = {}
    default_opts = dict(eps_abs=1e-6, eps_rel=1e-6, max_iter=200000, polish=True)
    # Unisci senza sovrascrivere user opts
    for k, v in default_opts.items():
        solver_opts.setdefault(k, v)

    # Prova solver scelto; se fallisce, fallback automatico
    tried = []
    t0 = time.time()
    try:
        problem.solve(solver=getattr(cp, solver), verbose=False, **solver_opts)
        tried.append(solver)
    except Exception:
        pass

    if problem.status not in ("optimal", "optimal_inaccurate"):
        for alt in ("ECOS", "CLARABEL", "SCS", "OSQP"):
            if alt in tried:
                continue
            try:
                problem.solve(solver=getattr(cp, alt), verbose=False)
                tried.append(alt)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    break
            except Exception:
                continue

    solve_time = time.time() - t0

    return {
        "x": x.value if x.value is not None else np.full(n, np.nan),
        "status": problem.status,
        "value": float(problem.value) if problem.value is not None else np.nan,
        "solve_time": solve_time,
    }


def solve_baseline_scipy(Q, q, blocks):
    """
    Baseline con scipy.optimize.minimize (SLSQP).

    Utile come confronto, ma più lenta e meno robusta su istanze grandi.
    """
    from scipy.optimize import minimize

    Q, q = _to_dense(Q, q)
    n = q.size

    def objective(x):
        return 0.5 * x.T @ Q @ x + q @ x

    def gradient(x):
        return Q @ x + q

    # Vincoli: per ogni blocco sum(x[block]) == 1
    constraints = []
    for block in blocks:
        idx = np.array(block, dtype=int)

        def constraint_eq(x, idx=idx):
            return np.sum(x[idx]) - 1.0

        constraints.append({"type": "eq", "fun": constraint_eq})

    # Bounds: x >= 0
    bounds = [(0.0, None) for _ in range(n)]

    # Starting point: uniforme per blocco
    x0 = np.zeros(n)
    for block in blocks:
        if len(block) > 0:
            x0[block] = 1.0 / len(block)

    t0 = time.time()
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    solve_time = time.time() - t0

    return {
        "x": res.x if res.success else np.full(n, np.nan),
        "status": "optimal" if res.success else str(res.message),
        "value": float(res.fun) if res.success else np.nan,
        "solve_time": solve_time,
    }

