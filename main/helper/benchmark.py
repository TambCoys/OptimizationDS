"""
Benchmark library: all logic for timing, comparing, and reporting solver results.

Provides:
  - benchmark_solver:       time a single solver on a problem
  - print_debug_summary:    detailed solution dump
  - print_comparison_table: formatted results + speedup + accuracy
  - run_single_benchmark:   run all 4 solvers on one problem instance
  - run_suite:              run the full systematic benchmark grid
  - generate_benchmark_suite: build the grid of problem configs
  - compute_objective / check_feasibility / format_seconds: small helpers
"""

import numpy as np
import time
import traceback

from main.algo1_baseline_solver import BaselineIPM
from main.algo1_final_solver import FeasibleStartIPM
from main.baseline_solvers import solve_baseline_cvxpy, solve_baseline_scipy
from main.helper.utils import norm_inf
from main.helper.block_ops import create_example_problem, apply_E


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def format_seconds(seconds):
    """Format timing values with enough precision for very fast runs."""
    if seconds < 1e-3:
        return f"{seconds:.3e}"
    if seconds < 1:
        return f"{seconds:.6f}"
    return f"{seconds:.4f}"


def compute_objective(Q, q, x):
    """Compute objective value: (1/2) x^T Q x + q^T x."""
    if hasattr(Q, 'dot'):
        Qx = Q.dot(x)
    else:
        Qx = Q @ x
    return 0.5 * np.dot(x, Qx) + np.dot(q, x)


def check_feasibility(x, blocks, tol=1e-6):
    """Check if x satisfies Ex = 1 and x >= 0."""
    Ex = apply_E(x, blocks)
    primal_feas = np.allclose(Ex, 1.0, atol=tol)
    nonneg = np.all(x >= -tol)
    return primal_feas and nonneg, Ex


# ---------------------------------------------------------------------------
# Suite generator
# ---------------------------------------------------------------------------

def generate_benchmark_suite(seed=42):
    """
    Generate a systematic grid of test problems for evaluation.

    Returns a list of dicts with keys: name, n, n_blocks, density, seed.

    Axes covered:
      - Scaling:   vary n with fixed density=1.0
      - Sparsity:  vary density with fixed n=1000
      - Block structure: vary n_blocks with fixed n=500
    """
    suite = []

    for n in [50, 100, 500, 1000, 2000]:
        n_blocks = max(2, n // 10)
        suite.append({
            'name': f'scale_n{n}',
            'n': n,
            'n_blocks': n_blocks,
            'density': 1.0,
            'seed': seed,
        })

    for density in [1.0, 0.5, 0.1, 0.01]:
        suite.append({
            'name': f'sparse_d{density}',
            'n': 1000,
            'n_blocks': 50,
            'density': density,
            'seed': seed,
        })

    for n_blocks in [2, 10, 50, 100, 250]:
        suite.append({
            'name': f'blocks_k{n_blocks}',
            'n': 500,
            'n_blocks': n_blocks,
            'density': 1.0,
            'seed': seed,
        })

    return suite


# ---------------------------------------------------------------------------
# Core benchmarking
# ---------------------------------------------------------------------------

def benchmark_solver(solver_func, Q, q, blocks, name, n_runs=1):
    """
    Benchmark a single solver function.

    Parameters
    ----------
    solver_func : callable(Q, q, blocks) -> dict
    Q, q, blocks : problem data
    name : str
    n_runs : int   (>1 averages timing over multiple runs)

    Returns
    -------
    dict with keys: name, success, x, obj_value, feasible, solve_time, ...
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*70}")

    try:
        start_time = time.perf_counter()
        result = solver_func(Q, q, blocks)
        solve_time = time.perf_counter() - start_time

        if n_runs > 1:
            times = [solve_time]
            for _ in range(n_runs - 1):
                t0 = time.perf_counter()
                _ = solver_func(Q, q, blocks)
                times.append(time.perf_counter() - t0)
            solve_time = np.mean(times)
            solve_time_std = np.std(times)
        else:
            solve_time_std = 0.0

        x = result.get('x', None)

        if x is not None and not np.any(np.isnan(x)):
            obj_value = compute_objective(Q, q, x)
            feas_ok, Ex = check_feasibility(x, blocks)
        else:
            obj_value = np.nan
            feas_ok = False
            Ex = None

        converged = result.get('converged',
                               result.get('status', 'unknown') == 'optimal')
        n_iter = result.get('iter', result.get('iters', None))

        return {
            'name': name,
            'success': True,
            'x': x,
            'obj_value': obj_value,
            'feasible': feas_ok,
            'Ex': Ex,
            'solve_time': solve_time,
            'solve_time_std': solve_time_std,
            'converged': converged,
            'n_iter': n_iter,
            'result': result,
        }
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {
            'name': name,
            'success': False,
            'error': str(e),
        }


# ---------------------------------------------------------------------------
# Printing / reporting
# ---------------------------------------------------------------------------

def print_debug_summary(result, name):
    """Print detailed solution summary for debugging."""
    print("\n" + "=" * 70)
    print(f"DEBUG SUMMARY: {name}")
    print("=" * 70)

    if not result.get('success', False):
        print(f"Solver failed: {result.get('error', 'Unknown error')}")
        return

    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['n_iter']}")

    if 'result' in result and 'mu' in result['result']:
        print(f"Final complementarity gap mu: {result['result']['mu']:.6e}")

    if result['converged']:
        print("\n[OK] Solution found!")
    else:
        print("\n[FAIL] Did not converge within iteration limit")

    x = result['x']
    if x is not None:
        print(f"\nPrimal solution x:")
        print(f"  Min: {np.min(x):.6e}, Max: {np.max(x):.6e}, Mean: {np.mean(x):.6e}")
        print(f"  Non-zero entries: {np.sum(x > 1e-10)} / {len(x)}")

    if 'result' in result:
        res = result['result']
        if 'y' in res and res['y'] is not None:
            y = res['y']
            print(f"\nDual solution y:")
            print(f"  Min: {np.min(y):.6e}, Max: {np.max(y):.6e}, Mean: {np.mean(y):.6e}")

        if 'z' in res and res['z'] is not None:
            z = res['z']
            print(f"\nDual slack z:")
            print(f"  Min: {np.min(z):.6e}, Max: {np.max(z):.6e}, Mean: {np.mean(z):.6e}")

        if 'history' in res and len(res['history']) > 0:
            last = res['history'][-1]
            print(f"\nFinal residuals:")
            print(f"  ||r_P||_inf = {last.get('norm_r_P', 'N/A')}")
            print(f"  ||r_D||_inf = {last.get('norm_r_D', 'N/A')}")

    print("=" * 70)


def print_comparison_table(results):
    """Print formatted results table, speedup, and accuracy comparison."""
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}")

    print(f"\n{'Solver':<25} {'Time (s)':<15} {'Objective':<15} "
          f"{'Feasible':<10} {'Iter':<8} {'Status':<15}")
    print("-" * 100)

    for r in results:
        if not r['success']:
            print(f"{r['name']:<25} {'ERROR':<15} {'-':<15} "
                  f"{'-':<10} {'-':<8} {'FAILED':<15}")
            continue
        time_str = format_seconds(r['solve_time'])
        obj_str = f"{r['obj_value']:.6e}" if not np.isnan(r['obj_value']) else "N/A"
        feas_str = "✓" if r['feasible'] else "✗"
        iter_str = str(r['n_iter']) if r['n_iter'] is not None else "-"
        status_str = "✓ Converged" if r['converged'] else "✗ Failed"
        print(f"{r['name']:<25} {time_str:<15} {obj_str:<15} "
              f"{feas_str:<10} {iter_str:<8} {status_str:<15}")

    print("-" * 100)

    successful = [r for r in results if r['success']]
    if len(successful) < 2:
        return

    baseline = successful[0]
    min_time = 1e-12
    baseline_time = max(baseline['solve_time'], min_time)

    # Speedup
    print(f"\n{'='*100}")
    print("SPEEDUP COMPARISON (relative to baseline)")
    print(f"{'='*100}")
    print(f"Baseline: {baseline['name']} ({format_seconds(baseline['solve_time'])} s)\n")

    for r in successful[1:]:
        run_time = max(r['solve_time'], min_time)
        speedup = baseline_time / run_time
        if speedup > 1:
            print(f"  {r['name']:<25}: {speedup:.2f}x faster")
        else:
            print(f"  {r['name']:<25}: {1/speedup:.2f}x slower")

    # Accuracy
    print(f"\n{'='*100}")
    print("ACCURACY COMPARISON")
    print(f"{'='*100}")

    baseline_obj = baseline['obj_value']
    if not np.isnan(baseline_obj):
        print(f"Baseline objective: {baseline_obj:.10e}\n")
        for r in successful[1:]:
            if not np.isnan(r['obj_value']):
                obj_diff = abs(r['obj_value'] - baseline_obj)
                obj_rel = obj_diff / abs(baseline_obj) if baseline_obj != 0 else obj_diff
                print(f"  {r['name']:<25}: obj = {r['obj_value']:.10e}, "
                      f"diff = {obj_diff:.6e} ({obj_rel*100:.4f}%)")
                if r['x'] is not None and baseline['x'] is not None:
                    x_diff = norm_inf(r['x'] - baseline['x'])
                    print(f"    Solution difference "
                          f"(||x - x_baseline||_inf): {x_diff:.6e}")


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_single_benchmark(Q, q, blocks, sigma, max_iter, verbosity,
                         n_runs, debug):
    """Run all 4 solvers on one problem instance and return results list."""

    def solve_final(Q, q, blocks):
        cfg = {
            'sigma': sigma,
            'max_iter': max_iter,
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
            'verbosity': verbosity,
        }
        return FeasibleStartIPM(Q, q, blocks, cfg=cfg).solve()

    def solve_baseline(Q, q, blocks):
        cfg = {
            'sigma': sigma,
            'max_iter': max_iter,
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
        }
        return BaselineIPM(Q, q, blocks, cfg=cfg).solve()

    results = [
        benchmark_solver(solve_baseline_cvxpy, Q, q, blocks,
                         "Baseline (CVXPY)", n_runs=n_runs),
        benchmark_solver(solve_baseline_scipy, Q, q, blocks,
                         "Baseline (SciPy)", n_runs=n_runs),
        benchmark_solver(solve_final, Q, q, blocks,
                         "Algorithm 1 (Final)", n_runs=n_runs),
        benchmark_solver(solve_baseline, Q, q, blocks,
                         "Algorithm 1 (baseline)", n_runs=n_runs),
    ]

    if debug:
        for r in results:
            if r['success']:
                print_debug_summary(r, r['name'])

    return results


def run_suite(sigma, max_iter, verbosity, n_runs, debug, seed):
    """
    Run the full benchmark suite and print a compact summary at the end.
    """
    suite = generate_benchmark_suite(seed=seed)

    print("=" * 100)
    print("BENCHMARK SUITE")
    print("=" * 100)
    print(f"  {len(suite)} problem configurations  |  seed = {seed}")
    print(f"  Solvers: CVXPY, SciPy, Final, Baseline")
    print("=" * 100)

    summary_rows = []

    for i, cfg in enumerate(suite, 1):
        name = cfg['name']
        n = cfg['n']
        n_blocks = cfg['n_blocks']
        density = cfg['density']

        print(f"\n{'#'*100}")
        print(f"  [{i}/{len(suite)}]  {name}  |  n={n}  |K|={n_blocks}  "
              f"density={density}")
        print(f"{'#'*100}")

        Q, q, blocks = create_example_problem(
            n=n, n_blocks=n_blocks, seed=cfg['seed'], density=density)

        results = run_single_benchmark(
            Q, q, blocks,
            sigma=sigma, max_iter=max_iter,
            verbosity=verbosity, n_runs=n_runs, debug=debug,
        )
        print_comparison_table(results)

        def _get(name_prefix):
            for r in results:
                if r['success'] and r['name'].startswith(name_prefix):
                    return r
            return None

        r_cvxpy = _get("Baseline (CVXPY)")
        r_final = _get("Algorithm 1 (Final)")

        cvxpy_t = r_cvxpy['solve_time'] if r_cvxpy else float('nan')
        final_t = r_final['solve_time'] if r_final else float('nan')
        ratio = (cvxpy_t / max(final_t, 1e-15)
                 if r_cvxpy and r_final else float('nan'))

        obj_match = ""
        if r_cvxpy and r_final:
            diff = abs(r_cvxpy['obj_value'] - r_final['obj_value'])
            rel = diff / max(abs(r_cvxpy['obj_value']), 1e-15)
            obj_match = f"{rel:.2e}"

        summary_rows.append((
            name, n, n_blocks, density, cvxpy_t, final_t, ratio,
            obj_match, r_final['converged'] if r_final else False,
        ))

    # Final compact summary
    print(f"\n\n{'='*110}")
    print("SUITE SUMMARY: CVXPY vs Algorithm 1 (Final)")
    print(f"{'='*110}")
    print(f"{'Problem':<18} {'n':>6} {'|K|':>5} {'dens':>5}  "
          f"{'CVXPY (s)':>12} {'Final (s)':>12} {'Ratio':>8}  "
          f"{'Obj rel err':>12}  {'Conv':>5}")
    print("-" * 110)
    for (name, n, K, density, ct, ft, ratio, obj_m, conv) in summary_rows:
        ct_s = format_seconds(ct) if not np.isnan(ct) else "FAIL"
        ft_s = format_seconds(ft) if not np.isnan(ft) else "FAIL"
        if np.isnan(ratio):
            ratio_s = "-"
        elif ratio >= 1:
            ratio_s = f"{ratio:.2f}x \u2191"
        else:
            ratio_s = f"{1/ratio:.2f}x \u2193"
        conv_s = "\u2713" if conv else "\u2717"
        print(f"{name:<18} {n:>6} {K:>5} {density:>5.2f}  "
              f"{ct_s:>12} {ft_s:>12} {ratio_s:>8}  "
              f"{obj_m:>12}  {conv_s:>5}")
    print("-" * 110)
    print("Ratio: \u2191 = Final faster than CVXPY, "
          "\u2193 = Final slower\n")


