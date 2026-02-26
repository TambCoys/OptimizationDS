"""
Main entry point for Project 34: QP Solver on Cartesian Product of Simplices.

Loads problem data (Q, q, blocks) and runs Algorithm 1.
"""

import numpy as np
import scipy.sparse
import sys
import time
from simplex_ipm.algo1_final_solver import FeasibleStartIPM
from simplex_ipm.baseline_solvers import solve_baseline_cvxpy, solve_baseline_scipy
from simplex_ipm.helper.block_ops import create_example_problem
from simplex_ipm.helper.benchmark import compute_objective, check_feasibility


def print_summary(result):
    """
    Print solution summary.
    
    Parameters:
    -----------
    result : dict
        Result dictionary from solver.
    """
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iter']}")
    print(f"Final complementarity gap mu: {result['mu']:.6e}")
    
    if result['converged']:
        print("\n[OK] Solution found!")
    else:
        print("\n[FAIL] Did not converge within iteration limit")
    
    # Print some solution statistics
    x = result['x']
    y = result['y']
    z = result['z']
    
    print(f"\nPrimal solution x:")
    print(f"  Min: {np.min(x):.6e}, Max: {np.max(x):.6e}, Mean: {np.mean(x):.6e}")
    print(f"  Non-zero entries: {np.sum(x > 1e-10)} / {len(x)}")
    
    print(f"\nDual solution y:")
    print(f"  Min: {np.min(y):.6e}, Max: {np.max(y):.6e}, Mean: {np.mean(y):.6e}")
    
    print(f"\nDual slack z:")
    print(f"  Min: {np.min(z):.6e}, Max: {np.max(z):.6e}, Mean: {np.mean(z):.6e}")
    
    # Print final residuals from history
    if len(result['history']) > 0:
        last = result['history'][-1]
        print(f"\nFinal residuals:")
        print(f"  ||r_P||_inf = {last['norm_r_P']:.6e}")
        print(f"  ||r_D||_inf = {last['norm_r_D']:.6e}")
    
    print("=" * 70)


def main():
    """
    Main function: load problem and solve.
    """
    # Configuration
    cfg = {
        'sigma': 0.1,  # Centering parameter
        'max_iter': 200,  # Increased for better convergence
        'eps_feas': 1e-8,
        'eps_comp': 1e-8,
        'verbosity': 3,  # Very verbose output for debugging
    }
    
    # Load problem data
    # For now, use example problem
    # In practice, you would load from file or pass as arguments
    print("Loading problem data...")
    Q, q, blocks = create_example_problem()
    
    print(f"Problem size: n = {len(q)}, |K| = {len(blocks)}")
    print(f"Blocks: {blocks}")
    
    # Solve with baseline solver first
    print("\n" + "=" * 70)
    print("BASELINE SOLVER (CVXPY)")
    print("=" * 70)
    try:
        baseline_result = solve_baseline_cvxpy(Q, q, blocks)
        print(f"Status: {baseline_result['status']}")
        print(f"Objective value: {baseline_result['value']:.6e}")
        print(f"Solve time: {baseline_result['solve_time']:.4f} seconds")
        print(f"Solution x: {baseline_result['x']}")
        
        # Check feasibility
        feas_ok, Ex_baseline = check_feasibility(baseline_result['x'], blocks)
        print(f"Feasibility check (Ex=1, x>=0): {feas_ok}")
        print(f"Ex = {Ex_baseline}")
    except Exception as e:
        print(f"Baseline solver failed: {e}")
        import traceback
        traceback.print_exc()
        baseline_result = None
    
    # Try scipy baseline
    print("\n" + "=" * 70)
    print("BASELINE SOLVER (SciPy)")
    print("=" * 70)
    try:
        scipy_result = solve_baseline_scipy(Q, q, blocks)
        print(f"Status: {scipy_result['status']}")
        print(f"Objective value: {scipy_result['value']:.6e}")
        print(f"Solve time: {scipy_result['solve_time']:.4f} seconds")
        print(f"Solution x: {scipy_result['x']}")
        
        # Check feasibility
        feas_ok, Ex_scipy = check_feasibility(scipy_result['x'], blocks)
        print(f"Feasibility check (Ex=1, x>=0): {feas_ok}")
        print(f"Ex = {Ex_scipy}")
    except Exception as e:
        print(f"SciPy baseline solver failed: {e}")
        import traceback
        traceback.print_exc()
        scipy_result = None
    
    # Create solver
    print("\n" + "=" * 70)
    print("ALGORITHM 1 (OUR IMPLEMENTATION)")
    print("=" * 70)
    print("\nInitializing solver...")
    solver = FeasibleStartIPM(Q, q, blocks, cfg=cfg)
    
    # Solve with timing
    print("\nRunning Algorithm 1...")
    start_time = time.time()
    result = solver.solve()
    solve_time = time.time() - start_time
    
    # Add timing to result
    result['solve_time'] = solve_time
    
    # Print summary
    print_summary(result)
    print(f"\nSolve time: {solve_time:.4f} seconds")
    
    # Compare with baseline
    if baseline_result is not None and baseline_result['x'] is not None:
        print("\n" + "=" * 70)
        print("COMPARISON WITH BASELINE")
        print("=" * 70)
        x_ours = result['x']
        x_baseline = baseline_result['x']
        
        # Compute objective values
        obj_baseline = compute_objective(Q, q, x_baseline)
        obj_ours = compute_objective(Q, q, x_ours)
        
        print(f"Baseline objective: {obj_baseline:.6e}")
        print(f"Our objective:      {obj_ours:.6e}")
        print(f"Difference:        {abs(obj_ours - obj_baseline):.6e}")
        print(f"Solution difference (||x_ours - x_baseline||_inf): {np.linalg.norm(x_ours - x_baseline, ord=np.inf):.6e}")
        
        # Time comparison
        if 'solve_time' in baseline_result:
            baseline_time = baseline_result['solve_time']
            speedup = baseline_time / solve_time if solve_time > 0 else float('inf')
            print(f"\nTiming comparison:")
            print(f"  Baseline time: {baseline_time:.4f} seconds")
            print(f"  Our time:      {solve_time:.4f} seconds")
            if speedup > 1:
                print(f"  Speedup:       {speedup:.2f}x faster")
            else:
                print(f"  Slowdown:      {1/speedup:.2f}x slower")
        
        print("=" * 70)
    
    return result


if __name__ == '__main__':
    result = main()

