"""Thin CLI entrypoint — all logic lives in main.helper.benchmark."""

import argparse
from simplex_ipm.helper.block_ops import create_example_problem
from simplex_ipm.helper.benchmark import (
    run_single_benchmark,
    run_suite,
    print_comparison_table,
)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Algorithm 1 vs Baseline solvers')
    parser.add_argument('--n', type=int, default=20,
                        help='Problem size (default: 20)')
    parser.add_argument('--n-blocks', type=int, default=3,
                        help='Number of blocks (default: 3)')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Density of Q matrix (1.0=dense, <1.0=sparse)')
    parser.add_argument('--n-runs', type=int, default=1,
                        help='Number of runs for timing (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Centering parameter (default: 0.1)')
    parser.add_argument('--max-iter', type=int, default=200,
                        help='Max iterations (default: 200)')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level (default: 0)')
    parser.add_argument('--debug', action='store_true',
                        help='Print detailed solution statistics')
    parser.add_argument('--suite', action='store_true',
                        help='Run full benchmark suite '
                             '(ignores --n, --n-blocks, --density)')
    args = parser.parse_args()

    if args.debug and args.verbosity == 0:
        args.verbosity = 3

    # ---- Suite mode ----
    if args.suite:
        run_suite(
            sigma=args.sigma, max_iter=args.max_iter,
            verbosity=args.verbosity, n_runs=args.n_runs,
            debug=args.debug, seed=args.seed,
        )
        return

    # ---- Single problem mode ----
    print("=" * 100)
    print("BENCHMARK: Algorithm 1 Implementations vs Baseline Solvers")
    print("=" * 100)
    print(f"\nProblem configuration:")
    print(f"  n = {args.n}")
    print(f"  |K| = {args.n_blocks}")
    print(f"  density = {args.density}")
    print(f"  n_runs = {args.n_runs}")
    print(f"  seed = {args.seed}")

    Q, q, blocks = create_example_problem(
        n=args.n, n_blocks=args.n_blocks,
        seed=args.seed, density=args.density)

    print(f"\nBlocks: {blocks}")

    results = run_single_benchmark(
        Q, q, blocks,
        sigma=args.sigma, max_iter=args.max_iter,
        verbosity=args.verbosity, n_runs=args.n_runs,
        debug=args.debug,
    )

    print_comparison_table(results)

    print(f"\n{'='*100}")
    print("Benchmark complete!")
    print(f"{'='*100}\n")

    return results


if __name__ == '__main__':
    main()

