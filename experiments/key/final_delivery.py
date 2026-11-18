"""Minimal, reproducible final deliverables for the Numerical Practicum.

Only two core benchmarks are required: sine_dirac and no_exact.
All adaptive histories use the existing L2-Riesz nested initialization,
PDAS relative indicator tolerance 1e-3 with KKT safeguard, and NO outer
estimator stopping tolerance. No FEM, PDAS, or refinement algorithms change.

Examples (from the repository root):
    python -m experiments.key.final_delivery check
    python -m experiments.key.final_delivery plots
    python -m experiments.key.final_delivery warm
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.key.final_study import ROOT, PAIRS, solve, write_rows
from experiments.key.inner_study import verify, transferred_start

CASES = ('sine_dirac', 'no_exact')
INITIALIZATIONS = ('cold', 'primal', 'riesz')


def checked_histories(args):
    """Validate provenance and terminal stops before publishing figures."""
    verify(SimpleNamespace(
        cases=CASES, max_dofs=args.max_dofs,
        inner_tol=args.inner_tol, min_h=args.min_h,
    ))
    folder = ROOT / 'results/key/inner_study' / f'budget_{args.max_dofs}'
    histories = {}
    for case in CASES:
        for degree, method in PAIRS:
            path = folder / case / f'{case}_p{degree}_{method}.csv'
            with path.open(newline='', encoding='utf-8') as handle:
                histories[case, degree, method] = list(csv.DictReader(handle))
    return folder, histories


def plot_histories(args):
    _, histories = checked_histories(args)
    out = ROOT / 'figures/key/inner_study' / f'budget_{args.max_dofs}' / 'final'
    out.mkdir(parents=True, exist_ok=True)
    metrics = (
        ('h1_error', r'$|u-u_h|_{H^1}$', 'h1_convergence.png'),
        ('l2_error', r'$\|u-u_h\|_{L^2}$', 'l2_convergence.png'),
        ('eta', r'Residual estimator $\eta$', 'estimator_convergence.png'),
    )
    for metric, ylabel, filename in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
        for ax, case in zip(axes, CASES):
            for degree, method in PAIRS:
                rows = histories[case, degree, method]
                x = np.array([int(r['dofs']) for r in rows])
                y = np.array([float(r[metric]) for r in rows])
                good = np.isfinite(y) & (y > 0)
                ax.loglog(x[good], y[good], '--s' if method == 'uniform' else '-o',
                          markersize=3, linewidth=1.3, label=f'P{degree} {method}')
            ax.set_title({'sine_dirac': 'Sine–Dirac', 'no_exact': 'Regular load'}[case])
            ax.set_xlabel('Interior degrees of freedom')
            ax.set_ylabel(ylabel)
            ax.grid(which='both', alpha=0.2)
            ax.legend(fontsize=8)
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
        print('Saved:', out / filename, flush=True)
    print('Summary:', ROOT / 'results/key/inner_study' /
          f'budget_{args.max_dofs}/verified_summary.csv', flush=True)


def warm_starts(args):
    """Compare initializations on the SAME fine mesh with the SAME PDAS rule.

    Median wall time includes the transfer and solver, but EXCLUDES the
    common coarse-grid solve. This is a solver-stage comparison, not an
    end-to-end adaptive runtime comparison.
    """
    folder = ROOT / 'results/key/inner_study' / f'budget_{args.max_dofs}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'warm_starts_inner.csv'
    rows = []
    for case in CASES:
        for degree in (1, 2):
            for elements in args.elements:
                coarse_nodes = np.linspace(0., 1., elements // 2 + 1)
                fine_nodes = np.linspace(0., 1., elements + 1)
                coarse = solve(case, coarse_nodes, degree, 5000,
                               None, None, inner=args.inner_tol, measure=True)
                reference = None
                for method in INITIALIZATIONS:
                    times = []
                    trial = None
                    for _ in range(args.repetitions):
                        start = perf_counter()
                        initial_solution, initial_multiplier = transferred_start(
                            coarse, fine_nodes, degree, method)
                        trial = solve(case, fine_nodes, degree, 5000,
                                      initial_solution, initial_multiplier,
                                      inner=args.inner_tol, measure=True)
                        times.append(perf_counter() - start)
                    if reference is None:
                        reference = trial['U'].copy()
                    difference = float(np.max(np.abs(trial['U'] - reference)))
                    if difference > 1e-8:
                        raise RuntimeError(f'Inconsistent solutions: {case}, '
                                           f'P{degree}, {method}: {difference:g}')
                    info = trial['info']
                    rows.append(dict(
                        case=case, degree=degree, elements=elements,
                        dofs=degree * elements - 1, initialization=method,
                        inner_tol=args.inner_tol, outer_tol='disabled',
                        iterations=info['iterations'],
                        stop_reason=info['pdas_stop_reason'],
                        max_difference_to_cold=difference,
                        median_wall_seconds=float(np.median(times)),
                        repetitions=args.repetitions,
                    ))
                    write_rows(output, rows)
                    print('Warm start:', rows[-1], flush=True)
    out = ROOT / 'figures/key/inner_study' / f'budget_{args.max_dofs}' / 'final'
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, case in enumerate(CASES):
        for j, degree in enumerate((1, 2)):
            ax = axes[i, j]
            for method in INITIALIZATIONS:
                selected = [r for r in rows if r['case'] == case
                            and r['degree'] == degree and r['initialization'] == method]
                ax.semilogx([r['dofs'] for r in selected],
                            [r['iterations'] for r in selected], '-o', label=method)
            ax.set(title=f'{case}, P{degree}', xlabel='Interior DOFs',
                   ylabel='PDAS iterations')
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    figure = out / 'initialization_comparison.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    print('Saved:', output, '\nSaved:', figure, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'plots', 'warm'))
    parser.add_argument('--max-dofs', type=int, default=300000)
    parser.add_argument('--inner-tol', type=float, default=1e-3)
    parser.add_argument('--min-h', type=float, default=1e-12)
    parser.add_argument('--elements', nargs='+', type=int, default=[80, 320, 1280],
                        help='Fine-mesh element counts for fixed-mesh warm starts')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if not (0 < args.inner_tol < 1 and args.min_h > 0 and args.max_dofs >= 39):
        parser.error('Invalid numerical protocol parameters')
    if args.repetitions < 1 or any(n < 4 or n % 2 for n in args.elements):
        parser.error('Warm-start elements must be even >=4; repetitions >=1')
    if args.action == 'check':
        checked_histories(args)
    elif args.action == 'plots':
        plot_histories(args)
    else:
        warm_starts(args)


if __name__ == '__main__':
    main()
