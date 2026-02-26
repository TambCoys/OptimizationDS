"""
Block operators: apply_E, apply_E^T, block utilities and example problem generator.
Never form the full E matrix explicitly.
"""

import numpy as np
import scipy.sparse as sp


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



def create_example_problem(n=10, n_blocks=3, seed=42, density=1.0):
    """
    Create a randomly generated example problem with a block partition.
    
    Parameters:
    -----------
    n : int, optional
        Total dimension of the problem (default is 10).
    n_blocks : int, optional
        Number of blocks to partition the indices into (default is 3).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    density : float, optional
        Density of the Q matrix (1.0 = dense, < 1.0 = sparse).
    
    Returns:
    --------
    Q : numpy.ndarray or scipy.sparse.csc_matrix
        A strictly positive definite symmetric matrix of shape (n, n).
    q : numpy.ndarray
        A random 1D array (vector) of length n.
    blocks : list of lists
        A disjoint partition of indices {0, ..., n-1} into n_blocks blocks.
    """

    # Simple 2-block example
    # Block 0: indices [0, 1, 2]
    # Block 1: indices [3, 4]
    # n = 5
    # blocks = [[0, 1, 2], [3, 4]]

    np.random.seed(seed)
    
    # Create blocks (roughly equal size)
    block_size = n // n_blocks
    blocks = []
    for k in range(n_blocks):
        start_idx = k * block_size
        if k == n_blocks - 1:
            end_idx = n
        else:
            end_idx = (k + 1) * block_size
        blocks.append(list(range(start_idx, end_idx)))
    
    # Create positive semidefinite Q
    if density >= 1.0:
        # Dense Q
        Q = np.random.randn(n, n)
        Q = Q.T @ Q  # Make it PSD
        Q = Q + 0.1 * np.eye(n)  # Make it strictly positive definite
    else:
        # Sparse Q
        # Generate a random sparse matrix
        A = sp.random(n, n, density=density, format='csc', random_state=seed)
        # Make it symmetric and PSD: Q = A^T * A + 0.1 * I
        Q = A.T @ A
        Q = Q + 0.1 * sp.eye(n, format='csc')
    
    # Create q
    q = np.random.randn(n)
    
    return Q, q, blocks