# run_experiments.py
import os
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp

from vincenzo_code.solver_algo1 import FeasibleStartIPM
from vincenzo_code.baseline_solver import solve_baseline_cvxpy
from generate_datas import generate_problem

np.random.seed(42)


def to_dense_for_cvxpy(Q, q):
    """Converti solo per la baseline CVXPY (denso + q vettore 1D)."""
    if sp.issparse(Q):
        Qd = Q.toarray()
    else:
        Qd = np.asarray(Q, dtype=float)
    qd = np.asarray(q, dtype=float).reshape(-1)
    return Qd, qd


# SEQUENZA CRESCENTE
configs = [
    {"n": 100, "K": 5},
    {"n": 500, "K": 10},
    {"n": 1000, "K": 20},
    {"n": 2000, "K": 30},
    {"n": 3000, "K": 40},

]

# cartella risultati
os.makedirs("results", exist_ok=True)

results = []

for cfg in configs:
    n, K = cfg["n"], cfg["K"]
    print(f"\nRunning n={n}, K={K}...")

    # Generazione problema (Q può essere sparsa)
    Q, q, blocks = generate_problem(n, K, seed=42, density=0.005)

    # Config IPM
    solver_cfg = {
        "sigma": 0.1,
        "max_iter": 300,
        "eps_feas": 1e-8,
        "eps_comp": 1e-8,
        "verbosity": 0,
    }

    # --- IPM (usa Q/q così come sono; l'algoritmo dovrebbe gestire sparse) ---
    t0 = time.time()
    solver = FeasibleStartIPM(Q, q, blocks, cfg=solver_cfg)
    res_ipm = solver.solve()
    t_ipm = time.time() - t0

    # --- Baseline CVXPY: converti a denso per evitare errori ---
    Qd, qd = to_dense_for_cvxpy(Q, q)
    t0 = time.time()
    res_cvx = solve_baseline_cvxpy(Qd, qd, blocks)  # usa Qd/qd!
    t_cvx = time.time() - t0

    # Estraggo iterazioni IPM se presenti
    ipm_iters = 0
    if isinstance(res_ipm, dict):
        # tenta più chiavi plausibili
        ipm_iters = res_ipm.get("iterations") or res_ipm.get("k") or res_ipm.get("n_iter") or 0

    # Salvo riga risultati
    results.append({
        "n": n,
        "K": K,
        "IPM_time": round(t_ipm, 3),
        "CVXPY_time": round(t_cvx, 3),
        "speedup": round(t_cvx / t_ipm, 2) if t_ipm > 0 else None,
        "IPM_iters": ipm_iters,
    })

# Salva CSV + stampa tabella riepilogo
df = pd.DataFrame(results)
df.to_csv("results/performance_full.csv", index=False)
print("\n", df[["n", "K", "IPM_time", "CVXPY_time", "speedup"]])
