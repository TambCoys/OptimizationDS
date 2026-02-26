"""
Benchmark script: Compare Algorithm 1 implementations vs Baseline solvers.

Compares:
- Algorithm 1 (Final implementation)
- Algorithm 1 (Baseline implementation)
- Baseline solvers (CVXPY, SciPy)

Measures:
- Solve time (performance)
- Solution accuracy (objective value, feasibility)
- Convergence statistics
"""

import numpy as np
import time
import sys
import os
from main.algo1_baseline_solver import ipm_simplex_qp
from main.algo1_final_solver import FeasibleStartIPM
from main.baseline_solvers import solve_baseline_cvxpy, solve_baseline_scipy
from main.helper.block_ops import apply_E, create_example_problem
from main.helper.formatting import format_seconds


def compute_objective(Q, q, x):
    """Compute objective value: (1/2) x^T Q x + q^T x"""
    if hasattr(Q, 'dot'):
        Qx = Q.dot(x)
    else:
        Qx = Q @ x
    return 0.5 * np.dot(x, Qx) + np.dot(q, x)


def check_feasibility(x, blocks, tol=1e-6):
    """Check if x satisfies Ex = 1 and x >= 0"""
    Ex = apply_E(x, blocks)
    primal_feas = np.allclose(Ex, 1.0, atol=tol)
    nonneg = np.all(x >= -tol)  # Allow small negative due to numerical errors
    return primal_feas and nonneg, Ex


def benchmark_solver(solver_func, Q, q, blocks, name, n_runs=1):
    """
    Benchmark a solver function.
    
    Parameters:
    -----------
    solver_func : callable
        Function that takes (Q, q, blocks) and returns a result dict.
    Q, q, blocks : problem data
    name : str
        Name of the solver.
    n_runs : int
        Number of runs for timing (for statistical significance).
    
    Returns:
    --------
    result : dict
        Dictionary with timing and solution information.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {name}")
    print(f"{'='*70}")
    
    # First run (for solution)
    try:
        start_time = time.perf_counter()
        result = solver_func(Q, q, blocks)
        solve_time = time.perf_counter() - start_time
        
        # Additional runs for timing (if n_runs > 1)
        if n_runs > 1:
            times = [solve_time]
            for _ in range(n_runs - 1):
                start_time = time.perf_counter()
                _ = solver_func(Q, q, blocks)
                times.append(time.perf_counter() - start_time)
            solve_time = np.mean(times)
            solve_time_std = np.std(times)
        else:
            solve_time_std = 0.0
        
        # Extract solution
        if 'x' in result:
            x = result['x']
        else:
            x = None
        
        # Compute objective
        if x is not None and not np.any(np.isnan(x)):
            obj_value = compute_objective(Q, q, x)
            feas_ok, Ex = check_feasibility(x, blocks)
        else:
            obj_value = np.nan
            feas_ok = False
            Ex = None
        
        # Extract convergence info
        converged = result.get('converged', result.get('status', 'unknown') == 'optimal')
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
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'success': False,
            'error': str(e),
        }


def print_comparison_table(results):
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Header
    print(f"\n{'Solver':<25} {'Time (s)':<15} {'Objective':<15} {'Feasible':<10} {'Iter':<8} {'Status':<15}")
    print("-" * 100)
    
    # Results
    for r in results:
        if not r['success']:
            print(f"{r['name']:<25} {'ERROR':<15} {'-':<15} {'-':<10} {'-':<8} {'FAILED':<15}")
            continue
        
        time_str = format_seconds(r['solve_time'])
        
        obj_str = f"{r['obj_value']:.6e}" if not np.isnan(r['obj_value']) else "N/A"
        feas_str = "✓" if r['feasible'] else "✗"
        iter_str = str(r['n_iter']) if r['n_iter'] is not None else "-"
        status_str = "✓ Converged" if r['converged'] else "✗ Failed"
        
        print(f"{r['name']:<25} {time_str:<15} {obj_str:<15} {feas_str:<10} {iter_str:<8} {status_str:<15}")
    
    print("-" * 100)
    
    # Speedup comparison
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        # Find baseline (first successful result)
        baseline = successful_results[0]
        print(f"\n{'='*100}")
        print("SPEEDUP COMPARISON (relative to baseline)")
        print(f"{'='*100}")
        print(f"Baseline: {baseline['name']} ({format_seconds(baseline['solve_time'])} s)\n")
        
        min_time = 1e-12
        baseline_time = max(baseline['solve_time'], min_time)
        for r in successful_results[1:]:
            run_time = max(r['solve_time'], min_time)
            speedup = baseline_time / run_time
            if speedup > 1:
                print(f"  {r['name']:<25}: {speedup:.2f}x faster")
            else:
                print(f"  {r['name']:<25}: {1/speedup:.2f}x slower")
        
        # Accuracy comparison
        print(f"\n{'='*100}")
        print("ACCURACY COMPARISON")
        print(f"{'='*100}")
        
        baseline_obj = baseline['obj_value']
        if not np.isnan(baseline_obj):
            print(f"Baseline objective: {baseline_obj:.10e}\n")
            
            for r in successful_results[1:]:
                if not np.isnan(r['obj_value']):
                    obj_diff = abs(r['obj_value'] - baseline_obj)
                    obj_rel_diff = obj_diff / abs(baseline_obj) if baseline_obj != 0 else obj_diff
                    print(f"  {r['name']:<25}: obj = {r['obj_value']:.10e}, diff = {obj_diff:.6e} ({obj_rel_diff*100:.4f}%)")
                    
                    # Solution difference
                    if r['x'] is not None and baseline['x'] is not None:
                        x_diff = np.linalg.norm(r['x'] - baseline['x'], ord=np.inf)
                        print(f"    Solution difference (||x - x_baseline||_inf): {x_diff:.6e}")


def main():
    """Main benchmarking function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark Algorithm 1 vs Baseline solvers')
    parser.add_argument('--n', type=int, default=20, help='Problem size (default: 20)')
    parser.add_argument('--n-blocks', type=int, default=3, help='Number of blocks (default: 3)')
    parser.add_argument('--n-runs', type=int, default=1, help='Number of runs for timing (default: 1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--sigma', type=float, default=0.1, help='Centering parameter (default: 0.1)')
    parser.add_argument('--max-iter', type=int, default=200, help='Max iterations (default: 200)')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level (default: 0) minimal verbosity for benchmarking')
    
    args = parser.parse_args()
    
    print("="*100)
    print("BENCHMARK: Algorithm 1 Implementations vs Baseline Solvers")
    print("="*100)
    print(f"\nProblem configuration:")
    print(f"  n = {args.n}")
    print(f"  |K| = {args.n_blocks}")
    print(f"  n_runs = {args.n_runs}")
    print(f"  seed = {args.seed}")
    
    # Create problem
    Q, q, blocks = create_example_problem(n=args.n, n_blocks=args.n_blocks, seed=args.seed)
    
    print(f"\nBlocks: {blocks}")
    
    # Wrapper for math modules baseline solvers
    def solve_baseline_cvxpy_wrapper(Q, q, blocks):
        return solve_baseline_cvxpy(Q, q, blocks)
    
    def solve_baseline_scipy_wrapper(Q, q, blocks):
        return solve_baseline_scipy(Q, q, blocks)
    
    # Wrapper for Algorithm 1
    def solve_final_wrapper(Q, q, blocks):
        cfg = {
            'sigma': args.sigma,
            'max_iter': args.max_iter,
            'eps_feas': 1e-8,
            'eps_comp': 1e-8,
            'verbosity': args.verbosity  # Default to 0 (Silent) for benchmarking
        }
        solver = FeasibleStartIPM(Q, q, blocks, cfg=cfg)
        result = solver.solve()
        return result
    
    # Wrapper for our baseline algorithm
    def solve_our_baseline_wrapper(Q, q, blocks):
        """Wrapper for our baseline algorithm to match expected format."""
        x, y, z, info = ipm_simplex_qp(
            Q, q, blocks,
            eps_feas=1e-8,
            eps_comp=1e-8,
            sigma=args.sigma,
            eps=1e-10,
            tau=1e-3,
            max_iter=args.max_iter
        )
        # Convert to expected format
        # Check convergence based on last iteration
        if len(info["rP_inf"]) > 0 and len(info["rD_inf"]) > 0 and len(info["mu"]) > 0:
            converged = (info["rP_inf"][-1] <= 1e-8 and 
                        info["rD_inf"][-1] <= 1e-8 and 
                        info["mu"][-1] <= 1e-8)
            mu = info['mu'][-1]
        else:
            converged = False
            mu = np.nan
        
        return {
            'x': x,
            'y': y,
            'z': z,
            'converged': converged,
            'iter': info['iters'],
            'mu': mu,
        }
    
    # Run benchmarks
    results = []
    
    # 1. Baseline CVXPY
    results.append(benchmark_solver(
        solve_baseline_cvxpy_wrapper, Q, q, blocks, 
        "Baseline (CVXPY)", n_runs=args.n_runs
    ))
    
    # 2. Baseline SciPy
    results.append(benchmark_solver(
        solve_baseline_scipy_wrapper, Q, q, blocks,
        "Baseline (SciPy)", n_runs=args.n_runs
    ))
    
    # 3. Algorithm 1 (Final implementation)
    results.append(benchmark_solver(
        solve_final_wrapper, Q, q, blocks,
        "Algorithm 1 (Final)", n_runs=args.n_runs
    ))
    
    # 4. Algorithm 1 (our baseline implementation)
    results.append(benchmark_solver(
        solve_our_baseline_wrapper, Q, q, blocks,
        "Algorithm 1 (baseline)", n_runs=args.n_runs
        ))
    
    # Print comparison
    print_comparison_table(results)
    
    print(f"\n{'='*100}")
    print("Benchmark complete!")
    print(f"{'='*100}\n")
    
    return results


if __name__ == '__main__':
    results = main()

