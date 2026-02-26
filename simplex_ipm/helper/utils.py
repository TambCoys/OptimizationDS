"""
Numerics helpers: fraction-to-boundary, norm infinity.
"""

import numpy as np


def fraction_to_boundary(x, d_x, z, d_z, gamma=0.99):
    """
    Compute fraction-to-boundary step sizes to maintain strict positivity.
    
    Parameters:
    -----------
    x : array-like, shape (n,)
        Current primal iterate (must be > 0).
    d_x : array-like, shape (n,)
        Primal step direction.
    z : array-like, shape (n,)
        Current dual slack (must be > 0).
    d_z : array-like, shape (n,)
        Dual slack step direction.
    gamma : float, default=0.99
        Safety factor (typically 0.99).
    
    Returns:
    --------
    alpha_pri : float
        Primal step size in [0, 1].
    alpha_dual : float
        Dual step size in [0, 1].
    alpha : float
        Overall step size = min(alpha_pri, alpha_dual).
    """
    x = np.asarray(x, dtype=float)
    d_x = np.asarray(d_x, dtype=float)
    z = np.asarray(z, dtype=float)
    d_z = np.asarray(d_z, dtype=float)
    
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


def norm_inf(a):
    """Compute infinity norm (max absolute value)."""
    return np.linalg.norm(a, ord=np.inf)