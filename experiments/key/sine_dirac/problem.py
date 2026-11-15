"""Sine-perturbed Dirac obstacle benchmark, epsilon=1 by default.

Use the existing P1/P2 structured PDAS solvers on the shifted
unknown W = U - I_h(sin(pi*x)), with constant obstacle -1.
"""

import argparse
from functools import partial
import numpy as np
from numpy.polynomial.legendre import leggauss

from experiments.validation.benchmarks.dirac_example import (
    ALPHA, BETA, GAMMA,
    exact_solution_dirac,
    exact_derivative_dirac,
)
from src.fem import assemble_load_vector as assemble_load_p1
from src.fem import evaluate_fem_solution as eval_p1
from src.fem import evaluate_fem_derivative as grad_p1
from src.fem_p2 import assemble_load_vector as assemble_load_p2
from src.fem_p2 import evaluate_fem_solution as eval_p2
from src.fem_p2 import evaluate_fem_derivative as grad_p2
from src.fem_p2 import create_p2_nodes
from src.error_indicators import (
    compute_p1_residual_indicator,
    compute_p2_residual_indicator,
)
from src.pdas_indicator import PDASIndicator
from src.p1_nested_large import (
    assemble_p1_dirac, matvec, solve_pdas_p1_gap,
)
from src.p2_nested_large import (
    assemble_p2_dirac, join_p2,
    p2_matvec, solve_pdas_p2_gap,
)


def sine(x, epsilon=1.0):
    return epsilon * np.sin(np.pi * np.asarray(x))


def regular_load(x, epsilon=1.0):
    return -6.0 + epsilon * np.pi**2 * np.sin(np.pi * np.asarray(x))


def exact_solution(x, epsilon=1.0):
    return exact_solution_dirac(x) + sine(x, epsilon)


def exact_derivative(x, epsilon=1.0):
    return (
        exact_derivative_dirac(x)
        + epsilon * np.pi * np.cos(np.pi * np.asarray(x))
    )


def obstacle(x, epsilon=1.0):
    return -1.0 + sine(x, epsilon)


def _sine_slopes(nodes, epsilon):
    """Exact first divided differences of epsilon*sin(pi*x), without cancellation."""
    h = np.diff(nodes)
    mid = nodes[:-1] + 0.5 * h
    return (2.0 * epsilon * np.sin(0.5 * np.pi * h) / h
            * np.cos(np.pi * mid))


def _sine_p2_curvature(nodes, epsilon):
    """Curvature of the quadratic sine interpolant, using sin(a)-2sin(m)+sin(b)."""
    h = np.diff(nodes)
    mid = nodes[:-1] + 0.5 * h
    return (-16.0 * epsilon * np.sin(0.25 * np.pi * h)**2 / h**2
            * np.sin(np.pi * mid))


def _sine_p2_stiffness_action(nodes, epsilon):
    """A_h I_h(epsilon*sin(pi*x)) from stable local trigonometric identities.

    Direct multiplication by A_h loses up to O(1/h) absolute accuracy
    when h is very small, because adjacent sine values are almost equal.
    """
    h = np.diff(nodes)
    mid = nodes[:-1] + 0.5 * h
    z = 0.5 * np.pi * h
    s = epsilon * np.sin(np.pi * mid)
    c = epsilon * np.cos(np.pi * mid)
    shared = -(16.0 / (3.0 * h)) * s * np.sin(0.5 * z)**2
    side = 2.0 * c * np.sin(z) / h
    left = shared - side
    right = shared + side
    middle = -2.0 * shared
    fv = np.zeros(len(nodes))
    fv[:-1] += left
    fv[1:] += right
    return join_p2(fv, middle)


def _stable_derivatives(nodes, degree, gap, epsilon):
    """Physical derivatives of g_h+I_h(sin)-1 without subtracting near -1."""
    h = np.diff(nodes)
    sine_slopes = _sine_slopes(nodes, epsilon)
    if degree == 1:
        g = np.concatenate(([1.0], gap, [1.0]))
        return (np.diff(g) / h + sine_slopes), None
    g = np.concatenate(([1.0], gap, [1.0]))
    gl, gm, gr = g[0:-2:2], g[1::2], g[2::2]
    slope_mid = (gr - gl) / h
    curvature_gap = 4.0 * (gl - 2.0 * gm + gr) / h**2
    curvature_sine = _sine_p2_curvature(nodes, epsilon)
    curvature = curvature_gap + curvature_sine
    left = slope_mid - 0.5 * h * curvature_gap
    right = slope_mid + 0.5 * h * curvature_gap
    left += sine_slopes - 0.5 * h * curvature_sine
    right += sine_slopes + 0.5 * h * curvature_sine
    return None, (left, right, curvature)


def solve_on_mesh(nodes, degree, epsilon=1.0, maxiter=1000,
                  initial_solution=None, initial_multiplier=None,
                  *, measure_pdas=False, pdas_rel_tol=None):
    """Solve physical obstacle u >= -1+I_h(epsilon*sin(pi*x)).

    Compute the physical KKT multiplier in the stable gap system, then
    reconstruct its L2 Riesz representative for the unchanged indicator.
    """
    nodes = np.asarray(nodes, dtype=float)
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    if (nodes.ndim != 1 or len(nodes) < 3 or
            not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0)):
        raise ValueError("Invalid mesh")

    if degree == 1:
        lower, diagonal, upper, F0 = assemble_p1_dirac(nodes)
        x_dof = nodes[1:-1]
        G = sine(x_dof, epsilon)
        F = F0 + assemble_load_p1(
            nodes, lambda x: epsilon * np.pi**2 * np.sin(np.pi*x),
            quadrature_order=8,
        )
        action = lambda v: matvec(lower, diagonal, upper, v)
        # Exact P1 Ritz identity: A_h I_h sin = (pi^2 sin,phi_h).
        # Hence the corrected RHS is F0, without unstable subtraction.
        F_shift = F0.copy()
        solve_gap = partial(solve_pdas_p1_gap, nodes, lower, diagonal, upper, F_shift)
        evaluate, derivative = eval_p1, grad_p1
    else:
        F0v, F0m = assemble_p2_dirac(nodes)
        F0 = join_p2(F0v, F0m)
        x_dof = create_p2_nodes(nodes)[1:-1]
        G = sine(x_dof, epsilon)
        F = F0 + assemble_load_p2(
            nodes, lambda x: epsilon * np.pi**2 * np.sin(np.pi*x),
            quadrature_order=8,
        )
        action = lambda v: p2_matvec(nodes, v)
        F_shift = F - _sine_p2_stiffness_action(nodes, epsilon)
        M = len(nodes) - 1
        Fv_shift = np.zeros(M + 1)
        Fv_shift[1:-1] = F_shift[1::2]
        Fm_shift = F_shift[0::2]
        solve_gap = partial(solve_pdas_p2_gap, nodes, Fv_shift, Fm_shift)
        evaluate, derivative = eval_p2, grad_p2

    def evaluate_indicator(gap, multiplier):
        physical = (gap - 1.0) + G
        slopes, derivatives = _stable_derivatives(nodes, degree, gap, epsilon)
        load = partial(regular_load, epsilon=epsilon)
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
    W, info = solve_gap(
        initial_solution=initial_solution,
        initial_multiplier=initial_multiplier, maxiter=maxiter, stop_if=monitor,
    )
    gap_coefficients = np.asarray(info["gap_coefficients"], dtype=float)
    Lambda = np.asarray(info["dual_coefficients"], dtype=float)
    U = W + G
    indicator = (monitor.indicator if monitor is not None
                 else evaluate_indicator(gap_coefficients, Lambda))

    info = dict(info)
    if monitor is not None:
        info["pdas_eta_history"] = monitor.history
        info["pdas_eta_ratio"] = monitor.ratio
        info["pdas_rel_tol"] = monitor.relative_tolerance
    info.update({
        "min_gap_physical": float(np.min(gap_coefficients)),
        "min_multiplier_physical": float(np.min(Lambda)),
        "max_complementarity_physical": float(
            np.max(np.abs(gap_coefficients * Lambda))
        ),
    })
    return {
        "nodes": nodes,
        "degree": degree,
        "epsilon": epsilon,
        "U": U,
        "W": W,
        "gap_coefficients": gap_coefficients,
        "shift": G,
        "F": F,
        "F_shift": F_shift,
        "Lambda": Lambda,
        "info": info,
        "evaluate": evaluate,
        "derivative": derivative,
        "matvec": action,
        "indicator": indicator,
    }


def compute_errors(result, quadrature_order=12):
    """True L2/H1 errors, evaluated in stable gap-plus-sine coordinates.

    Integrate separately across the three known nonsmooth locations.
    """
    nodes = result["nodes"]
    epsilon = result["epsilon"]
    degree = result["degree"]
    gap = result["gap_coefficients"]
    gfull = np.concatenate(([1.0], gap, [1.0]))
    G = result["shift"]
    sfull = np.concatenate(([0.0], G, [0.0]))
    slopes, derivatives = _stable_derivatives(nodes, degree, gap, epsilon)
    gslope = np.diff(gfull) / np.diff(nodes) if degree == 1 else None
    sslope = _sine_slopes(nodes, epsilon)
    g, w = leggauss(quadrature_order)
    breaks = (ALPHA, GAMMA, BETA)
    err2_l2 = 0.0
    err2_h1 = 0.0

    for element, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
        h = b-a
        bounds = [a] + [z for z in breaks if a < z < b] + [b]
        if degree == 2:
            gv = gfull[2*element:2*element+3]
            sv = sfull[2*element:2*element+3]
        for left, right in zip(bounds[:-1], bounds[1:]):
            x = 0.5*(left+right) + 0.5*(right-left)*g
            xi = (x-a)/h
            weights = 0.5*(right-left)*w
            if degree == 1:
                g_h = (1.0-xi)*gfull[element] + xi*gfull[element+1]
                s_h = (1.0-xi)*sfull[element] + xi*sfull[element+1]
                derivative_h = slopes[element]
            else:
                phi0 = 2.0*(xi-0.5)*(xi-1.0)
                phi1 = 4.0*xi*(1.0-xi)
                phi2 = 2.0*xi*(xi-0.5)
                g_h = gv[0]*phi0 + gv[1]*phi1 + gv[2]*phi2
                s_h = sv[0]*phi0 + sv[1]*phi1 + sv[2]*phi2
                # Quadratic derivative varies linearly across the element.
                l, r, _ = derivatives
                derivative_h = l[element] + (r[element]-l[element])*xi
            # Keep differences separate instead of subtracting values near -1.
            e = (g_h-(exact_solution_dirac(x)+1.0)) + (s_h-sine(x,epsilon))
            ep = derivative_h - (
                exact_derivative_dirac(x) + epsilon*np.pi*np.cos(np.pi*x)
            )
            err2_l2 += float(np.dot(weights, e**2))
            err2_h1 += float(np.dot(weights, ep**2))
    return float(np.sqrt(err2_l2)), float(np.sqrt(err2_h1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=1.0)
    args = parser.parse_args()

    for degree in (1, 2):
        nodes = np.linspace(0.0, 1.0, args.elements + 1)
        result = solve_on_mesh(nodes, degree, args.epsilon)
        l2, h1 = compute_errors(result)

        info = result["info"]
        eta = float(np.sqrt(np.sum(
            result["indicator"]["eta_squared"]
        )))

        print(
            f"P{degree}, elements={args.elements}, "
            f"epsilon={args.epsilon}, "
            f"DOFs={len(result['U'])}, "
            f"iterations={info['iterations']}, "
            f"L2={l2:.6e}, H1-semi={h1:.6e}, "
            f"eta={eta:.6e}, "
            f"min-gap={info['min_gap_physical']:.3e}, "
            f"min-dual={info['min_multiplier_physical']:.3e}, "
            f"complementarity="
            f"{info['max_complementarity_physical']:.3e}"
        )


if __name__ == "__main__":
    main()
