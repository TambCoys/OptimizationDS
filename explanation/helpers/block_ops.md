# Helper Functions Analysis: `block_ops.py`

Based on the theoretical background from `feedback2.tex` and the lessons learned from the naive `algo1_baseline_solver.py`, this document analyzes the helper functions located in `helper/block_ops.py`. 

Chronologically, these functions represent the first major step towards an optimized solver. After realizing that building dense matrices (like $E$) and performing naive linear algebra operations ($O(n^3)$) prevented the baseline algorithm from scaling, these helpers were created to exploit the specific mathematical structure of the problem.

---

## 1. The Goal of the Helper Functions
The core problem involves the constraint matrix $E$, which enforces the simplex condition $\sum_{i \in I^k} x_i = 1$ for each block $k$. 
In the naive baseline, $E$ was built as a dense $K \times n$ matrix. If $n = 10,000$ and $K = 100$, $E$ contains 1,000,000 elements, but only 10,000 of them are non-zero (exactly one `1` per column). 

The primary goal of `block_ops.py` is to **never form the full $E$ matrix explicitly**. Instead, it provides functions that compute the *action* of $E$ and $E^T$ on vectors in $O(n)$ time.

---

## 2. Walkthrough: `block_ops.py`

### 2.1 `apply_E(x, blocks)`
```python
def apply_E(x, blocks):
    x = np.asarray(x, dtype=float)
    n_blocks = len(blocks)
    y = np.zeros(n_blocks, dtype=float)
    
    for k in range(n_blocks):
        y[k] = np.sum(x[blocks[k]])
    
    return y
```
*   **Mathematical Meaning:** This computes the matrix-vector product $y = Ex$. Since $E$ is the block-summation matrix, the $k$-th element of the result is simply the sum of the variables in the $k$-th block: $y_k = \sum_{i \in I^k} x_i$.
*   **Computational Choice:** Instead of doing a dense matrix multiplication `E @ x` (which takes $O(K \cdot n)$ operations), this function loops through the blocks and sums the relevant slices of $x$. This takes exactly $O(n)$ operations. It completely eliminates the need to store $E$ in memory. This is used constantly, for example, to compute the primal residual $r_P = Ex - \mathbf{1}$.

### 2.2 `apply_E_transpose(y, blocks, n=None)`
```python
def apply_E_transpose(y, blocks, n=None):
    y = np.asarray(y, dtype=float)
    if n is None:
        n = max(max(block) for block in blocks if len(block) > 0) + 1
    
    x = np.zeros(n, dtype=float)
    for k in range(len(blocks)):
        x[blocks[k]] = y[k]
    
    return x
```
*   **Mathematical Meaning:** This computes the matrix-vector product $x = E^T y$. The transpose of the block-summation matrix acts as a "broadcast" operator. If $y$ is a vector of length $K$ (one value per block), $E^T y$ creates a vector of length $n$ where every variable in block $k$ receives the value $y_k$.
*   **Computational Choice:** Again, instead of a dense multiplication `E.T @ y`, this function uses NumPy's advanced indexing (`x[blocks[k]] = y[k]`) to broadcast the values in $O(n)$ time. This is heavily used in the dual residual calculation ($r_D = Qx + q + E^Ty - z$) and when back-substituting to find $\Delta z$.

### 2.3 `validate_blocks(blocks, n)`
```python
def validate_blocks(blocks, n):
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
```
*   **Mathematical Meaning:** This enforces the strict definition of a partition as defined in `feedback2.tex`: $\bigcup_{k \in K} I^k = \{1,\ldots,n\}$ and $I^h \cap I^k = \emptyset$ for all $h \neq k$.
*   **Computational Choice:** It uses a Python `set` to track seen indices. It ensures no block is empty, no index is out of bounds, no index is duplicated (disjoint sets), and no index is missing (complete coverage). This is crucial defensive programming; if the blocks are malformed, the $O(n)$ operators `apply_E` and `apply_E_transpose` will silently produce mathematically incorrect results. The solver would compute invalid Newton steps and fail to converge, but it wouldn't throw a Python error. This function acts as a mathematical firewall, ensuring the fundamental assumptions of Project 34 hold true before any computation begins.

### 2.4 `create_example_problem(n=10, n_blocks=3, seed=42)`
```python
def create_example_problem(n=10, n_blocks=3, seed=42):
    # ... block generation ...
    Q = np.random.randn(n, n)
    Q = Q.T @ Q  # Make it PSD
    Q = Q + 0.1 * np.eye(n)  # Make it strictly positive definite
    q = np.random.randn(n)
    return Q, q, blocks
```
*   **Mathematical Meaning:** This generates a synthetic, mathematically sound instance of your specific QP problem to test the solvers.
*   **Computational Choice:** 
    *   **Generating $Q$:** A random matrix is not necessarily positive semi-definite (PSD). By computing `Q.T @ Q`, it guarantees the matrix is PSD (all eigenvalues $\ge 0$), which makes the objective function convex.
    *   **Strict Convexity:** It adds `0.1 * np.eye(n)` to the diagonal. This shifts all eigenvalues up by 0.1, guaranteeing the matrix is *strictly* positive definite ($Q \succ 0$). As noted in `feedback2.tex`, this ensures the quadratic program is strictly convex and has a **unique global minimum**. This is vital for benchmarking: if the problem had multiple optimal solutions, comparing the custom solver's output to SciPy/CVXPY would be ambiguous.