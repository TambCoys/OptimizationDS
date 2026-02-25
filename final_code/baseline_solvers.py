"""
Baseline QP solvers using CVXPY and SciPy for comparison.
Solves the same problem: min (1/2) x^T Q x + q^T x  s.t.  Ex = 1, x >= 0
"""

import numpy as np
import scipy.sparse

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not available")


def solve_baseline_cvxpy(Q, q, blocks):
    """
    Solve QP using CVXPY.
    
    Parameters:
    -----------
    Q : array-like or sparse matrix, shape (n, n)
        Quadratic matrix (symmetric, Q >= 0).
    q : array-like, shape (n,)
        Linear term.
    blocks : list of lists
        blocks[k] contains indices in block k (0-indexed).
    
    Returns:
    --------
    result : dict
        Dictionary with solution:
        - x: primal solution
        - status: solver status
        - value: objective value
        - solve_time: solve time
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY not available")
    
    Q = np.asarray(Q, dtype=float)
    if scipy.sparse.issparse(Q):
        Q = Q.toarray() # type: ignore
    
    # Symmetrize if needed
    if not np.allclose(Q, Q.T):
        Q = (Q + Q.T) / 2
    
    q = np.asarray(q, dtype=float)
    n = len(q)
    
    # Create variable
    x = cp.Variable(n, nonneg=True)
    
    # Build constraints: Ex = 1 (one constraint per block)
    constraints = []
    for k, block in enumerate(blocks):
        constraints.append(cp.sum(x[block]) == 1.0)
    
    # Objective: (1/2) x^T Q x + q^T x
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + q @ x)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    import time
    start_time = time.time()
    problem.solve(solver=cp.OSQP, verbose=False)
    solve_time = time.time() - start_time
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        # Try with different solver
        problem.solve(solver=cp.ECOS, verbose=False)
        solve_time = time.time() - start_time
    
    result = {
        'x': x.value if x.value is not None else np.full(n, np.nan),
        'status': problem.status,
        'value': problem.value if problem.value is not None else np.nan,
        'solve_time': solve_time,
    }
    
    return result


def solve_baseline_scipy(Q, q, blocks):
    """
    Solve QP using scipy.optimize.minimize (SLSQP).
    
    Parameters:
    -----------
    Q : array-like or sparse matrix, shape (n, n)
        Quadratic matrix (symmetric, Q >= 0).
    q : array-like, shape (n,)
        Linear term.
    blocks : list of lists
        blocks[k] contains indices in block k (0-indexed).
    
    Returns:
    --------
    result : dict
        Dictionary with solution:
        - x: primal solution
        - status: solver status
        - value: objective value
        - solve_time: solve time
    """
    from scipy.optimize import minimize

    Q = np.asarray(Q, dtype=float)
    if scipy.sparse.issparse(Q):
        Q = Q.toarray() # type: ignore
    
    # Symmetrize if needed
    if not np.allclose(Q, Q.T):
        Q = (Q + Q.T) / 2
    
    q = np.asarray(q, dtype=float)
    n = len(q)
    
    # Objective function
    def objective(x):
        return 0.5 * x.T @ Q @ x + q @ x
    
    # Gradient
    def gradient(x):
        return Q @ x + q
    
    
    # OLD APPROACH (Nested closures - commented out for clarity)
    # constraints = []
    # for k, block in enumerate(blocks):
    #     # Create constraint: sum(x[block]) == 1
    #     def make_constraint(block_indices):
    #         def constraint_eq(x):
    #             return np.sum(x[block_indices]) - 1.0
    #         return {'type': 'eq', 'fun': constraint_eq}
    #     constraints.append(make_constraint(block))


    # Constraints: Ex = 1 (equality constraints)
    def eq_constraints(x):
        return np.array([np.sum(x[block]) - 1.0 for block in blocks])
    
    constraints = [{'type': 'eq', 'fun': eq_constraints}]
    
    # Bounds: x >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Initial guess: uniform per block
    x0 = np.zeros(n)
    for k, block in enumerate(blocks):
        x0[block] = 1.0 / len(block)
    
    import time
    start_time = time.time()
    res = minimize(objective, x0, method='SLSQP', jac=gradient,
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-9})
    solve_time = time.time() - start_time
    
    result = {
        'x': res.x if res.success else np.full(n, np.nan),
        'status': 'optimal' if res.success else res.message,
        'value': res.fun if res.success else np.nan,
        'solve_time': solve_time,
    }
    
    return result
