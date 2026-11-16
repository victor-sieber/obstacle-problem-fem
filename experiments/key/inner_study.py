"""Inner stopping study. Run with python -m experiments.key.inner_study ACTION."""
import argparse
import csv
import hashlib
import json
import platform
from functools import partial
from time import perf_counter

import numpy as np
import scipy
from experiments.key.final_study import (
    CASES, PAIRS, ROOT, solve, compute_errors, reference_errors, write_rows,
    prolongate_p1, prolongate_p2,
    transfer_l2_multiplier_p1, transfer_l2_multiplier_p2,
)
from experiments.key.sine_dirac.convergence import run_series

OUT = ROOT / 'results/key/inner_study'
VALID = ROOT / 'results/validation/inner_tolerance'
FIG = ROOT / 'figures/key/inner_study'


def save_config(folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    sources = sorted(list((ROOT/'src').glob('*.py')) + list((ROOT/'experiments/key').rglob('*.py')))
    config = dict(parameters=vars(args), python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources})
    (folder/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')


def transferred_start(coarse, nodes, degree, method):
    if method == 'cold':
        return None, None
    prolongate = prolongate_p1 if degree == 1 else prolongate_p2
    transfer = transfer_l2_multiplier_p1 if degree == 1 else transfer_l2_multiplier_p2
    primal = prolongate(coarse['nodes'], coarse['W'], nodes)
    dual = transfer(coarse['nodes'], coarse['Lambda'], nodes) if method == 'riesz' else None
    return primal, dual


def run_convergence(args):
    folder = OUT / f'budget_{args.max_dofs}'
    save_config(folder, args)
    summary = []
    for case in args.cases:
        for degree, method in PAIRS:
            errors = compute_errors if case == 'sine_dirac' else partial(reference_errors, case)
            rows = run_series(
                degree, method, partial(solve, case, inner=args.inner_tol),
                error_function=errors, benchmark=case, max_levels=args.max_levels,
                max_dofs=args.max_dofs, maxiter=2000, output_dir=folder/case,
                mesh_rel_tol=None, min_element_size=args.min_h)
            for row in rows:
                row.pop('eta_ratio', None)
                row.pop('mesh_rel_tol', None)
            write_rows(folder/case/f'{case}_p{degree}_{method}.csv', rows)
            last = rows[-1]
            summary.append(dict(
                case=case, degree=degree, method=method, dofs=last['dofs'],
                eta=last['eta'], h1_error=last['h1_error'], min_h=last['min_h'],
                mesh_stop=last['stop_reason'], meshes=len(rows),
                inner_relative_exits=sum(r['pdas_stop_reason']=='relative_indicator' for r in rows),
                relative_exits_before_stability=sum(
                    r['pdas_stop_reason']=='relative_indicator' and not r['active_set_stable'] for r in rows),
                min_element_gap=last['min_element_gap'],
                scaled_free_residual=last['scaled_free_residual']))
            write_rows(folder/'summary.csv', summary)
    print('SUMMARY:', folder/'summary.csv', flush=True)


def audit(args):
    save_config(VALID, args)
    rows = []
    for case in ('sine_dirac', 'no_exact', 'split40'):
        for degree in (1, 2):
            for elements in args.elements:
                nodes = np.linspace(0, 1, elements+1)
                coarse = solve(case, np.linspace(0, 1, elements//2+1), degree,
                               5000, None, None, measure=False)
                for method in ('cold', 'primal', 'riesz'):
                    primal, dual = transferred_start(coarse, nodes, degree, method)
                    baseline = solve(case, nodes, degree, 5000, primal, dual, measure=True)
                    for tolerance in args.tolerances:
                        start = perf_counter()
                        trial = solve(case, nodes, degree, 5000, primal, dual, inner=tolerance)
                        info = trial['info']
                        difference = float(np.max(np.abs(trial['U']-baseline['U'])))
                        assert difference <= 1e-8, 'Early exit differs from fully solved reference'
                        rows.append(dict(
                            case=case, degree=degree, elements=elements, method=method,
                            tolerance=tolerance, baseline_iterations=baseline['info']['iterations'],
                            iterations=info['iterations'], saved=baseline['info']['iterations']-info['iterations'],
                            stop=info['pdas_stop_reason'], active_set_stable=info['active_set_stable'],
                            inner_ratio=info['pdas_eta_ratio'],
                            rejected_candidates=len(info['pdas_rejected_indicator_steps']),
                            max_solution_difference=difference, seconds=perf_counter()-start))
                        write_rows(VALID/'sweep.csv', rows)
    for tolerance in args.tolerances:
        subset = [r for r in rows if r['tolerance']==tolerance]
        print(dict(tolerance=tolerance, solves=len(subset),
                   relative_exits=sum(r['stop']=='relative_indicator' for r in subset),
                   exits_before_stability=sum(r['stop']=='relative_indicator' and not r['active_set_stable'] for r in subset),
                   iterations_saved=sum(r['saved'] for r in subset)), flush=True)


def scaling(args):
    folder = OUT/f'scaling_{args.max_dofs}'
    save_config(folder, args)
    rows = []
    for degree in (1, 2):
        previous = None
        elements = 20
        while degree*elements-1 <= args.max_dofs:
            nodes = np.linspace(0, 1, elements+1)
            primal, dual = (None, None) if previous is None else transferred_start(previous, nodes, degree, 'riesz')
            print(f'Starting P{degree}, {degree*elements-1:,} DOFs', flush=True)
            start = perf_counter()
            result = solve('sine_dirac', nodes, degree, 2000, primal, dual, inner=args.inner_tol)
            info = result['info']
            rows.append(dict(degree=degree, dofs=degree*elements-1,
                             iterations=info['iterations'], seconds=perf_counter()-start,
                             pdas_stop=info['pdas_stop_reason'], inner_ratio=info['pdas_eta_ratio'],
                             active_set_stable=info['active_set_stable'],
                             scaled_free_residual=info['scaled_free_residual']))
            write_rows(folder/'scaling.csv', rows)
            print(rows[-1], flush=True)
            previous = result
            elements *= 2


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = OUT/f'budget_{args.max_dofs}'
    target = FIG/f'budget_{args.max_dofs}'
    target.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for degree, method in PAIRS:
            with (folder/case/f'{case}_p{degree}_{method}.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            dofs = [int(r['dofs']) for r in rows]
            label = f'P{degree} {method}'
            axes[0].loglog(dofs, [float(r['h1_error']) for r in rows], '.-', label=label)
            axes[1].semilogx(dofs, [float(r['pdas_eta_ratio']) for r in rows], '.-', label=label)
        axes[0].set_ylabel('H1-seminorm error')
        axes[1].set_ylabel('Final inner indicator ratio on each mesh')
        axes[1].axhline(args.inner_tol, color='black', linestyle=':', label='Inner tolerance')
        for axis in axes:
            axis.set_xlabel('Interior degrees of freedom')
            axis.grid(True, which='both', alpha=.2)
            axis.legend()
        fig.suptitle(case)
        fig.tight_layout()
        fig.savefig(target/f'{case}.png', dpi=180)
        plt.close(fig)
    print('Figures:', target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'audit', 'scaling', 'plot'))
    parser.add_argument('--max-dofs', type=int, default=300000)
    parser.add_argument('--max-levels', type=int, default=60)
    parser.add_argument('--inner-tol', type=float, default=1e-3)
    parser.add_argument('--min-h', type=float, default=64*np.finfo(float).eps)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--elements', nargs='+', type=int, default=[40, 160, 640])
    parser.add_argument('--tolerances', nargs='+', type=float, default=[.001, .1, .5, .9, .99])
    args = parser.parse_args()
    if args.max_dofs < 39 or args.max_levels < 0 or args.min_h <= 0:
        parser.error('Invalid mesh budget or resolution threshold')
    if not 0 < args.inner_tol < 1 or any(not 0<t<1 for t in args.tolerances):
        parser.error('Relative tolerances must lie between zero and one')
    if any(n < 4 or n%2 for n in args.elements):
        parser.error('Audit element counts must be even and at least four')
    {'run': run_convergence, 'audit': audit, 'scaling': scaling, 'plot': plot}[args.action](args)


if __name__ == '__main__':
    main()
