# generate_problem.py
import numpy as np
from scipy.sparse import random as sprand, csr_matrix
import os

def generate_problem(n, K, density=0.01, seed=42, save_dir="problems"):
    """
    Genera un problema QP valido:
    - Q: PSD, sparsa
    - q: normale
    - blocks: partizione VALIDA di {0,...,n-1} → NO ripetizioni, NO vuoti
    """
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. BLOCKS: partizione sicura ---
    if K > n:
        raise ValueError(f"K={K} > n={n}: non puoi avere più blocchi che variabili")

    # Dividi n in K parti, ogni parte almeno 1
    block_sizes = np.full(K, n // K, dtype=int)
    remainder = n % K
    for i in range(remainder):
        block_sizes[i] += 1

    # Mescola gli indici
    indices = np.random.permutation(n)
    blocks = []
    start = 0
    for size in block_sizes:
        if size == 0:
            raise RuntimeError("Blocco vuoto generato!")
        block = indices[start:start + size].tolist()
        blocks.append(block)
        start += size

    # --- Verifica finale ---
    all_indices = set(i for block in blocks for i in block)
    if len(all_indices) != n or any(len(b) == 0 for b in blocks):
        raise RuntimeError("Partizione non valida!")

    # --- 2. Q: PSD sparsa ---
    L = sprand(n, n, density=density, format='csr', random_state=seed)
    L = (L + L.T) / 2
    L.data = np.abs(L.data)  # solo valori positivi
    Q = L @ L.T
    Q = Q + 1e-6 * csr_matrix(np.eye(n))
    Q = Q.tocsr()

    # --- 3. q: normale ---
    q = np.random.randn(n)

    # --- Salva ---
    path = f"{save_dir}/prob_n{n}_k{K}_seed{seed}"
    np.save(f"{path}_Q.npy", Q)
    np.save(f"{path}_q.npy", q)
    np.save(f"{path}_blocks.npy", np.array(blocks, dtype=object))

    return Q, q, blocks
