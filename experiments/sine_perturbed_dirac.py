"""Sine-perturbed Dirac obstacle benchmark, epsilon=1 by default.

Use the existing P1/P2 structured PDAS solvers on the shifted
unknown W = U - I_h(sin(pi*x)), with constant obstacle -1.
"""

import argparse
import numpy as np
from numpy.polynomial.legendre import leggauss

from experiments.dirac_example import (
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
from src.p1_nested_large import (
    assemble_p1_dirac, matvec, solve_pdas_p1,
)
from src.p2_nested_large import (
    assemble_p2_dirac, join_p2,
    p2_matvec, solve_pdas_p2,
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


def solve_on_mesh(nodes, degree, epsilon=1.0, maxiter=1000):
    nodes = np.asarray(nodes, dtype=float)

    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")

    if (
        nodes.ndim != 1
        or len(nodes) < 3
        or np.any(np.diff(nodes) <= 0)
    ):
        raise ValueError("Invalid mesh")

    if degree == 1:
        lower, diagonal, upper, F0 = assemble_p1_dirac(nodes)

        x_dof = nodes[1:-1]
        G = sine(x_dof, epsilon)

        F = F0 + assemble_load_p1(
            nodes,
            lambda x: epsilon * np.pi**2 * np.sin(np.pi*x),
            quadrature_order=8,
        )

        action = lambda v: matvec(lower, diagonal, upper, v)
        F_shift = F - action(G)

        W, info = solve_pdas_p1(
            nodes, lower, diagonal, upper,
            F_shift, maxiter=maxiter,
        )

        indicator = compute_p1_residual_indicator(
            nodes, W + G, action, F,
            lambda x: regular_load(x, epsilon),
        )

        evaluate, derivative = eval_p1, grad_p1

    else:
        F0v, F0m = assemble_p2_dirac(nodes)
        F0 = join_p2(F0v, F0m)

        x_dof = create_p2_nodes(nodes)[1:-1]
        G = sine(x_dof, epsilon)

        F = F0 + assemble_load_p2(
            nodes,
            lambda x: epsilon * np.pi**2 * np.sin(np.pi*x),
            quadrature_order=8,
        )

        action = lambda v: p2_matvec(nodes, v)
        F_shift = F - action(G)

        M = len(nodes) - 1
        Fv_shift = np.zeros(M + 1)
        Fv_shift[1:-1] = F_shift[1::2]
        Fm_shift = F_shift[0::2]

        W, info = solve_pdas_p2(
            nodes, Fv_shift, Fm_shift,
            maxiter=maxiter,
        )

        indicator = compute_p2_residual_indicator(
            nodes, W + G, action, F,
            lambda x: regular_load(x, epsilon),
        )

        evaluate, derivative = eval_p2, grad_p2

    U = W + G
    Lambda = action(U) - F
    gap = U - obstacle(x_dof, epsilon)

    info = dict(info)
    info.update({
        "min_gap_physical": float(np.min(gap)),
        "min_multiplier_physical": float(np.min(Lambda)),
        "max_complementarity_physical":
            float(np.max(np.abs(gap * Lambda))),
    })

    return {
        "nodes": nodes,
        "degree": degree,
        "epsilon": epsilon,
        "U": U,
        "W": W,
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
    """Compute true L2 and H1-seminorm errors.

    Split integration intervals at alpha, gamma and beta.
    """
    nodes = result["nodes"]
    U = result["U"]
    eps = result["epsilon"]

    evaluate = result["evaluate"]
    derivative = result["derivative"]

    g, w = leggauss(quadrature_order)
    breaks = (ALPHA, GAMMA, BETA)

    err2_l2 = 0.0
    err2_h1 = 0.0

    for a, b in zip(nodes[:-1], nodes[1:]):
        bounds = [a] + [z for z in breaks if a < z < b] + [b]

        for left, right in zip(bounds[:-1], bounds[1:]):
            x = 0.5*(left+right) + 0.5*(right-left)*g
            weights = 0.5*(right-left)*w

            err2_l2 += np.dot(
                weights,
                (evaluate(nodes, U, x) - exact_solution(x, eps))**2,
            )

            err2_h1 += np.dot(
                weights,
                (derivative(nodes, U, x) - exact_derivative(x, eps))**2,
            )

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
