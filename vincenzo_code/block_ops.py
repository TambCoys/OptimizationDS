"""
Block operators: apply_E, apply_E^T, block utilities.
Never form the full E matrix explicitly.
"""

import numpy as np


def apply_E(x, blocks):
    """
    Apply E operator: [E x]_k = sum_{i in I_k} x_i.
    
    Parameters:
    -----------
    x : array-like, shape (n,)
        Vector to apply E to.
    blocks : list of lists
        blocks[k] contains indices in block k (0-indexed).
    
    Returns:
    --------
    y : ndarray, shape (|K|,)
        Result E x, where y[k] = sum_{i in I_k} x_i.
    """
    x = np.asarray(x, dtype=float)
    n_blocks = len(blocks)
    y = np.zeros(n_blocks, dtype=float)
    
    for k in range(n_blocks):
        y[k] = np.sum(x[blocks[k]])
    
    return y


def apply_E_transpose(y, blocks, n=None):
    """
    Apply E^T operator: [E^T y]_i = y_k where i in I_k.
    
    Parameters:
    -----------
    y : array-like, shape (|K|,)
        Vector to apply E^T to.
    blocks : list of lists
        blocks[k] contains indices in block k (0-indexed).
    n : int, optional
        Total dimension. If None, inferred from blocks.
    
    Returns:
    --------
    x : ndarray, shape (n,)
        Result E^T y, where x[i] = y[k] for i in I_k.
    """
    y = np.asarray(y, dtype=float)
    
    if n is None:
        # Infer n from maximum index in blocks
        n = max(max(block) for block in blocks if len(block) > 0) + 1
    
    x = np.zeros(n, dtype=float)
    
    for k in range(len(blocks)):
        x[blocks[k]] = y[k]
    
    return x


def validate_blocks(blocks, n):
    """
    Validate that blocks form a disjoint partition of {0, ..., n-1}.
    
    Parameters:
    -----------
    blocks : list of lists
        blocks[k] contains indices in block k (0-indexed).
    n : int
        Total dimension.
    
    Returns:
    --------
    is_valid : bool
        True if blocks are valid.
    error_msg : str or None
        Error message if invalid, None otherwise.
    """
    all_indices = set()
    
    for k, block in enumerate(blocks):
        if len(block) == 0:
            return False, f"Block {k} is empty"
        
        for i in block:
            if i < 0 or i >= n:
                return False, f"Index {i} in block {k} is out of range [0, {n-1}]"
            if i in all_indices:
                return False, f"Index {i} appears in multiple blocks"
            all_indices.add(i)
    
    if len(all_indices) != n:
        missing = set(range(n)) - all_indices
        return False, f"Missing indices: {missing}"
    
    return True, None

