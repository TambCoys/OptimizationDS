"""Numerics helpers."""

import numpy as np


def norm_inf(a):
    """Compute infinity norm (max absolute value)."""
    return np.linalg.norm(a, ord=np.inf)
