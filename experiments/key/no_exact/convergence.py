"""Non-manufactured obstacle problem: f=-12+sin(pi*x), psi=-1.

No Dirac loads or prescribed solution. The unconstrained Dirichlet solution
is -6*x*(1-x)+sin(pi*x)/pi**2, which is below -1 at x=1/2.
"""
import argparse
from functools import partial
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from experiments.key.sine_dirac.convergence import run_series
from src.fem import assemble_load_vector as load_p1
from src.fem_p2 import assemble_load_vector as load_p2
from src.pdas_indicator import PDASIndicator
from src.p1_nested_large import assemble_p1_dirac, matvec, solve_pdas_p1_gap
from src.p2_nested_large import p2_matvec, solve_pdas_p2_gap
from src.error_indicators import (compute_p1_residual_indicator,
                                  compute_p2_residual_indicator)


def regular_load(x):
    return -12.0 + np.sin(np.pi*np.asarray(x))


def _gap_derivatives(nodes, degree, gap):
    h = np.diff(nodes)
    g = np.concatenate(([1.0], gap, [1.0]))
    if degree == 1:
        return np.diff(g)/h, None
    gl, gm, gr = g[0:-2:2], g[1::2], g[2::2]
    slope = (gr-gl)/h
    curvature = 4.0*((gl-gm)+(gr-gm))/h**2
    return None, (slope-0.5*h*curvature, slope+0.5*h*curvature, curvature)


def _intermesh_difference(nodes, degree, gap, previous_on_new):
    """Exact FE L2 and H1-seminorm differences; NOT true-error bounds."""
    delta = np.asarray(gap) - (np.asarray(previous_on_new)+1.0)
    h = np.diff(nodes)
    d = np.concatenate(([0.0], delta, [0.0]))
    if degree == 1:
        a,b = d[:-1],d[1:]
        h1sq = np.sum((b-a)**2/h)
        l2sq = np.sum(h*(a*a+a*b+b*b)/3.0)
    else:
        a,b,c = d[0:-2:2],d[1::2],d[2::2]
        left = (-3*a+4*b-c)/h
        right = (a-4*b+3*c)/h
        h1sq = np.sum(h*(left**2+left*right+right**2)/3.0)
        pts,ww = leggauss(3)
        l2sq = 0.0
        for t,w in zip(0.5*(pts+1.0),0.5*ww):
            v = 2*(t-0.5)*(t-1)*a+4*t*(1-t)*b+2*t*(t-0.5)*c
            l2sq += w*np.sum(h*v*v)
    return float(np.sqrt(max(0.0,l2sq))), float(np.sqrt(max(0.0,h1sq)))


def solve_on_mesh(nodes, degree, maxiter=2000,
                  initial_solution=None, initial_multiplier=None,
                  *, measure_pdas=False, pdas_rel_tol=None, load=regular_load):
    nodes = np.asarray(nodes, dtype=float)
    if (nodes.ndim != 1 or len(nodes) < 3 or not np.all(np.isfinite(nodes))
            or np.any(np.diff(nodes) <= 0)):
        raise ValueError("Invalid mesh")

    if degree == 1:
        low, diag, up, _ = assemble_p1_dirac(nodes)  # stiffness only!
        F = load_p1(nodes, load, quadrature_order=8)
        action = lambda x: matvec(low, diag, up, x)
        solve_gap = partial(solve_pdas_p1_gap, nodes, low, diag, up, F)
    elif degree == 2:
        F = load_p2(nodes, load, quadrature_order=8)
        M = len(nodes)-1
        Fv = np.zeros(M+1)
        Fv[1:-1] = F[1::2]
        Fm = F[0::2]
        action = lambda x: p2_matvec(nodes,x)
        solve_gap = partial(solve_pdas_p2_gap, nodes, Fv, Fm)
    else:
        raise ValueError("degree must be 1 or 2")

    def evaluate_indicator(gap, multiplier):
        physical = gap - 1.0
        slopes, derivatives = _gap_derivatives(nodes, degree, gap)
        if degree == 1:
            return compute_p1_residual_indicator(
                nodes, physical, action, F, load,
                Lambda_override=multiplier, slopes_override=slopes,
            )
        return compute_p2_residual_indicator(
            nodes, physical, action, F, load,
            Lambda_override=multiplier, derivatives_override=derivatives,
        )

    monitor = (PDASIndicator(evaluate_indicator, pdas_rel_tol)
               if measure_pdas or pdas_rel_tol is not None else None)
    U, info = solve_gap(
        initial_solution=initial_solution,
        initial_multiplier=initial_multiplier, maxiter=maxiter, stop_if=monitor,
    )
    gap = np.asarray(info["gap_coefficients"])
    Lambda = np.asarray(info["dual_coefficients"])
    indicator = (monitor.indicator if monitor is not None
                 else evaluate_indicator(gap, Lambda))
    info = dict(info)
    if monitor is not None:
        info["pdas_eta_history"] = monitor.history
        info["pdas_eta_ratio"] = monitor.ratio
        info["pdas_rel_tol"] = monitor.relative_tolerance
    info.update({
        "min_gap_physical":float(np.min(gap)),
        "min_multiplier_physical":float(np.min(Lambda)),
        "max_complementarity_physical":float(np.max(np.abs(gap*Lambda))),
    })
    if initial_solution is not None:
        d_l2,d_h1 = _intermesh_difference(nodes, degree, gap, initial_solution)
        info["delta_l2_previous"] = d_l2
        info["delta_h1_previous"] = d_h1
    return {"nodes":nodes,"degree":degree,"U":U,"W":U,
            "Lambda":Lambda,"gap_coefficients":gap,
            "F":F,"info":info,"indicator":indicator}


def _solve(nodes, degree, maxiter, initial_solution, initial_multiplier,
           *, pdas_rel_tol=None):
    return solve_on_mesh(
        nodes, degree, maxiter, initial_solution, initial_multiplier,
        measure_pdas=True, pdas_rel_tol=pdas_rel_tol,
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--degree",type=int,choices=(1,2),required=True)
    parser.add_argument("--method",choices=("uniform","adaptive"),required=True)
    parser.add_argument("--max-levels",type=int,default=12)
    parser.add_argument("--max-dofs",type=int,default=20000)
    parser.add_argument("--mesh-rel-tol", type=float)
    parser.add_argument("--pdas-rel-tol", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maxiter",type=int,default=2000)
    a=parser.parse_args()
    rows=run_series(degree=a.degree,method=a.method,solver=partial(_solve, pdas_rel_tol=a.pdas_rel_tol),
                    benchmark="no_exact",max_levels=a.max_levels,
                    max_dofs=a.max_dofs, maxiter=a.maxiter, output_dir=a.output_dir,
                    mesh_rel_tol=a.mesh_rel_tol)
    print(f"Final residual indicator: {rows[-1]['eta']:.6e}")


if __name__=="__main__":
    main()
