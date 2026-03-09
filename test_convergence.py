"""Quick convergence test for all 18 benchmark configs (IPM only, no CVXPY)."""
from simplex_ipm.solver import SimplifiedIPM
from simplex_ipm.helper.benchmark import create_example_problem
import time

configs = [
    ("scale_n50",        50,    5, 1.0),
    ("scale_n100",      100,   10, 1.0),
    ("scale_n500",      500,   50, 1.0),
    ("scale_n1000",    1000,  100, 1.0),
    ("scale_n2000",    2000,  200, 1.0),
    ("sparse_100pct",  1000,   50, 1.0),
    ("sparse_50pct",   1000,   50, 0.5),
    ("sparse_10pct",   1000,   50, 0.1),
    ("sparse_1pct",    1000,   50, 0.01),
    ("blocks_k2",       500,    2, 1.0),
    ("blocks_k10",      500,   10, 1.0),
    ("blocks_k50",      500,   50, 1.0),
    ("blocks_k100",     500,  100, 1.0),
    ("blocks_k250",     500,  250, 1.0),
    ("lg_n5000_d1pct",  5000, 100, 0.01),
    ("lg_n5000_d.1pct", 5000, 100, 0.001),
    ("lg_n10k_d1pct",  10000, 200, 0.01),
    ("lg_n10k_d.1pct", 10000, 200, 0.001),
]

hdr = f"{'Name':<22} {'n':>6} {'|K|':>5} {'dens':>6}  {'Time':>10} {'Iter':>5} {'Conv':>5} {'mu':>10}"
print(hdr)
print("-" * len(hdr))

all_pass = True
for name, n, nb, d in configs:
    Q, q, blocks = create_example_problem(n=n, n_blocks=nb, seed=42, density=d)
    t0 = time.perf_counter()
    r = SimplifiedIPM(Q, q, blocks, cfg={"verbosity": 0, "max_iter": 200}).solve()
    dt = time.perf_counter() - t0
    c = "Y" if r["converged"] else "N"
    if not r["converged"]:
        all_pass = False
    print(f"{name:<22} {n:>6} {nb:>5} {d:>6.3f}  {dt:>10.4f} {r['iter']:>5} {c:>5} {r['mu']:>10.2e}")

print()
print("ALL CONVERGED" if all_pass else "*** SOME FAILED ***")
