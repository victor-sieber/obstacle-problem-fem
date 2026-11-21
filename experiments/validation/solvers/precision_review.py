"""Isolated precision checks; never writes canonical study CSVs.

Default: small P2 manufactured-solution check using the retained binary64
physical-coordinate solver and the current gap solver on identical meshes.
This is a controlled reproduction, NOT the missing historical obstacle run.
--p1-tail additionally replays the regular-load P1 adaptive meshes and checks
the final active sets with the existing 45-digit solver. This takes longer.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

from src.fem_p2 import assemble_load_vector, create_p2_nodes
from src.p2_nested_large import solve_on_active_set, solve_gap_p2_on_active_set, _p2_gap_load
from src.p1_nested_large import _solve_p1_gap_high_precision
from src.adaptive import mark_by_maximum, bisect_marked_elements
from experiments.key.final_study import ROOT, solve, reference, reference_errors, write_rows
from experiments.key.inner_study import transferred_start

OUT = ROOT / 'results/validation/submission'


def polynomial_check():
    q, weights = leggauss(4)
    rows = []
    for small in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        nodes = np.sort(np.unique(np.r_[np.linspace(0, 1, 21), .426,
                                        .426 + small, .426 + 2*small]))
        m = len(nodes)-1
        f = assemble_load_vector(nodes, lambda x: np.ones_like(x), 3)
        fv = np.zeros(m+1)
        fv[1:-1] = f[1::2]
        fm = f[0::2]
        active = np.zeros(2*m-1, dtype=bool)
        old = solve_on_active_set(nodes, fv, fm, active)
        fvg, fmg = _p2_gap_load(nodes, fv, fm)
        corrected = solve_gap_p2_on_active_set(nodes, fvg, fmg, active)
        h = np.diff(nodes)
        x = nodes[:-1, None] + h[:, None]*(q+1)/2
        exact_derivative = (1-2*x)/2
        for method, values, boundary in (
                ('Retained binary64 solver', old, 0.),
                ('Current gap/precision solver', corrected, 1.)):
            full = np.r_[boundary, values, boundary]
            left, mid, right = full[:-2:2], full[1::2], full[2::2]
            derivative = ((right-left)[:, None]
                          + 2*((left-mid)+(right-mid))[:, None]*q)/h[:, None]
            h1 = np.sqrt(np.sum(h[:, None]*weights/2*(derivative-exact_derivative)**2))
            rows.append(dict(min_h=h.min(), method=method, h1_error=h1, elements=m))
    OUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUT/'p2_precision_controlled.csv', rows)
    fig, ax = plt.subplots(figsize=(7.7, 4.3), layout='constrained')
    for name in dict.fromkeys(r['method'] for r in rows):
        data = [r for r in rows if r['method'] == name]
        ax.loglog([r['min_h'] for r in data], [r['h1_error'] for r in data], '-o', label=name)
    ax.invert_xaxis()
    ax.set(xlabel='Smallest element length', ylabel=r'$H^1$-seminorm error',
           title='Controlled P2 precision check: exact quadratic solution')
    ax.grid(which='both', alpha=.2)
    ax.legend(fontsize=9)
    fig.text(.5, -.035, 'Identical meshes; f = 1; u(x) = x(1-x)/2. Not a historical convergence run.',
             ha='center', fontsize=9)
    target = ROOT/'figures/presentation'
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target/'precision_controlled.png', dpi=190, bbox_inches='tight')
    plt.close(fig)
    print('P2 controlled check:', rows[-2:], flush=True)


def p1_tail():
    canonical = ROOT/'results/key/inner_study/budget_300000/no_exact/no_exact_p1_adaptive.csv'
    expected = list(csv.DictReader(canonical.open()))
    nodes = np.linspace(0, 1, 21)
    previous = None
    rows = []
    for level, stored in enumerate(expected):
        initial, dual = ((None, None) if previous is None
                         else transferred_start(previous, nodes, 1, 'riesz'))
        r = solve('no_exact', nodes, 1, 2000, initial, dual, inner=1e-3)
        if len(r['U']) != int(stored['dofs']):
            raise RuntimeError(f'Mesh replay differs at level {level}; no same-mesh claim possible')
        if level >= len(expected)-3:
            old_l2, old_h1 = reference_errors('no_exact', r)
            active = r['gap_coefficients'] == 0
            accurate_gap = _solve_p1_gap_high_precision(nodes, r['F'], active)
            if np.min(accurate_gap) < -1e-9:
                raise RuntimeError('Higher-precision solution violates primal feasibility')
            candidate = dict(r, gap_coefficients=accurate_gap, U=accurate_gap-1)
            new_l2, new_h1 = reference_errors('no_exact', candidate)
            row = dict(level=level, dofs=len(r['U']), stored_l2=stored['l2_error'],
                       replay_l2=old_l2, precise_l2=new_l2, replay_h1=old_h1,
                       precise_h1=new_h1, max_gap_change=float(np.max(abs(accurate_gap-r['gap_coefficients']))))
            rows.append(row)
            write_rows(OUT/'p1_tail_precision.csv', rows)
            print('P1 same-mesh fixed-active-set check:', row, flush=True)
        previous = r
        nodes = bisect_marked_elements(nodes, mark_by_maximum(r['indicator']['eta'], .5))


def p2_uniform_recheck():
    """Recheck the retained L2 wiggle on identical uniform fine problems.

    Reference values are used ONLY to initialize this diagnostic solve;
    final-study initializations and canonical CSVs are never changed.
    """
    exact, boundaries = reference('no_exact')
    rows = []
    for elements in (20480, 40960, 81920):
        nodes = np.linspace(0, 1, elements+1)
        initial = exact(create_p2_nodes(nodes)[1:-1])[0]
        result = solve('no_exact', nodes, 2, 2000, initial, None, inner=1e-3)
        l2, h1 = reference_errors('no_exact', result)
        rows.append(dict(elements=elements, dofs=2*elements-1,
                         l2_error=l2, h1_error=h1, iterations=result['info']['iterations'],
                         contact_boundary_fraction=(boundaries[0]*elements) % 1))
    write_rows(OUT/'p2_uniform_recheck.csv', rows)
    print('P2 uniform recheck:', rows, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p1-tail', action='store_true')
    parser.add_argument('--p2-uniform', action='store_true')
    args = parser.parse_args()
    polynomial_check()
    if args.p1_tail:
        p1_tail()
    if args.p2_uniform:
        p2_uniform_recheck()
