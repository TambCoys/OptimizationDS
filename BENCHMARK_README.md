# Benchmarking Guide

This guide explains how to benchmark Algorithm 1 implementations (Vincenzo's and Marco's) against baseline solvers.

## Quick Start

### Option 1: Use the dedicated benchmark script (Recommended)

```bash
# Basic benchmark (default: n=20, 3 blocks, 1 run)
python benchmark_solver.py

# Custom problem size
python benchmark_solver.py --n 50 --n-blocks 5

# Multiple runs for statistical significance
python benchmark_solver.py --n 30 --n-runs 5

# Custom solver parameters
python benchmark_solver.py --sigma 0.1 --max-iter 200
```

### Option 2: Use the existing run script (now with timing)

```bash
python run_p34.py
```

The `run_p34.py` script now includes timing information and compares the algorithm with baseline solvers.

## Benchmark Script Features

The `benchmark_solver.py` script provides:

1. **Timing Comparison**: Measures solve time for all solvers
2. **Accuracy Comparison**: Compares objective values and solution differences
3. **Feasibility Check**: Verifies that solutions satisfy constraints
4. **Speedup Analysis**: Shows how much faster/slower each algorithm is
5. **Statistical Analysis**: Can run multiple times for average/std deviation
6. **Algorithm Comparison**: Compares Vincenzo's vs Marco's implementations

## Command-Line Options

```bash
--n           Problem size (number of variables, default: 20)
--n-blocks    Number of blocks (default: 3)
--n-runs      Number of runs for timing (default: 1, use >1 for statistics)
--seed        Random seed (default: 42)
--sigma       Centering parameter (default: 0.1)
--max-iter    Maximum iterations (default: 200)
```

## Example Output

```bash
====================================================================================================
BENCHMARK RESULTS SUMMARY
====================================================================================================

Solver                    Time (s)        Objective       Feasible   Iter     Status        
----------------------------------------------------------------------------------------------------
Baseline (CVXPY)          0.0234         1.234567e+00    ✓           -        ✓ Converged   
Baseline (SciPy)          0.0156         1.234567e+00    ✓           -        ✓ Converged   
Algorithm 1 (Vincenzo)    0.0123         1.234567e+00    ✓           15       ✓ Converged   
Algorithm 1 (Marco)       0.0118         1.234567e+00    ✓           14       ✓ Converged   
----------------------------------------------------------------------------------------------------

====================================================================================================
SPEEDUP COMPARISON (relative to baseline)
====================================================================================================
Baseline: Baseline (CVXPY) (0.0234 s)

  Algorithm 1 (Vincenzo): 1.90x faster
  Algorithm 1 (Marco): 1.98x faster

====================================================================================================
ACCURACY COMPARISON
====================================================================================================
Baseline objective: 1.2345678901e+00

  Algorithm 1 (Vincenzo): obj = 1.2345678901e+00, diff = 0.000000e+00 (0.0000%)
    Solution difference (||x - x_baseline||_inf): 1.234567e-08
  Algorithm 1 (Marco): obj = 1.2345678901e+00, diff = 0.000000e+00 (0.0000%)
    Solution difference (||x - x_baseline||_inf): 1.234567e-08
```

## Tips

1. **For small problems** (n < 50): Use `--n-runs 1` (single run is fine)
2. **For larger problems** (n > 100): Use `--n-runs 3` or more for reliable timing
3. **For debugging**: Use `run_p34.py` with `verbosity: 3` in the config
4. **For production**: Use `benchmark_solver.py` with multiple runs

## Troubleshooting

- **If baseline solvers fail**: Check that CVXPY and SciPy are installed
- **If Marco's algorithm is skipped**: Make sure `FeasibleStandard.py` is in `marco_code/` directory
- **If timing seems off**: Use `--n-runs 5` for more reliable averages
- **If solutions differ**: Check feasibility tolerances (default: 1e-8)

## Comparing Implementations

The benchmark now includes both Algorithm 1 implementations:

- **Vincenzo's**: Uses M formulation, operator form, supports sparse matrices
- **Marco's**: Uses H formulation, explicit matrices, simpler implementation

Both should produce similar results, but may differ in:

- **Performance**: Different formulations may have different computational costs
- **Numerical stability**: M formulation (Vincenzo) avoids explicit division by x
- **Scalability**: Operator form (Vincenzo) is better for large problems
