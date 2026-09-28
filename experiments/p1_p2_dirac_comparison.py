from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.fem import (
    assemble_load_vector as assemble_load_vector_p1,
    assemble_stiffness_matrix as assemble_stiffness_matrix_p1,
    create_mesh,
    evaluate_fem_derivative as evaluate_fem_derivative_p1,
    evaluate_fem_solution as evaluate_fem_solution_p1,
    p1_basis,
)
from src.fem_p2 import (
    assemble_load_vector as assemble_load_vector_p2,
    assemble_stiffness_matrix as assemble_stiffness_matrix_p2,
    create_p2_nodes,
    evaluate_fem_derivative as evaluate_fem_derivative_p2,
    evaluate_fem_solution as evaluate_fem_solution_p2,
    p2_basis,
)
from src.solver import (
    solve_obstacle_problem,
    solve_obstacle_problem_pdas,
)


ALPHA = 1.0 / 3.0
BETA = 2.0 / 3.0
GAMMA = 1.0 - np.sqrt(2.0) / 3.0

ELEMENT_COUNTS = np.array([20, 40, 80, 160, 320])
SOLUTION_ELEMENTS = 20
ERROR_QUADRATURE_ORDER = 12


def obstacle_dirac(x):
    x = np.asarray(x, dtype=float)
    return -np.ones_like(x)


def smooth_load(x):
    x = np.asarray(x, dtype=float)
    return -6.0 * np.ones_like(x)


def _containing_element(nodes, point):
    element = np.searchsorted(
        nodes,
        point,
        side="right",
    ) - 1

    return int(
        np.clip(
            element,
            0,
            len(nodes) - 2,
        )
    )


def add_point_load_p1(F, nodes, point, weight):
    """
    Add weight * delta_point to the P1 load vector.

    Since <delta_point, v_h> = v_h(point), the contribution to the
    local load vector is weight times the P1 basis evaluated at point.
    """
    element = _containing_element(
        nodes,
        point,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    xi = (point - x_left) / h
    local_values = p1_basis(xi)

    global_dofs = [
        element,
        element + 1,
    ]

    n_elements = len(nodes) - 1

    for i_local, i_global in enumerate(global_dofs):
        if 1 <= i_global <= n_elements - 1:
            F[i_global - 1] += (
                weight * local_values[i_local]
            )


def add_point_load_p2(F, nodes, point, weight):
    """
    Add weight * delta_point to the P2 load vector.

    The Dirac distribution acts exactly by evaluating the three local
    P2 basis functions at the point. No spike approximation is used.
    """
    element = _containing_element(
        nodes,
        point,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    xi = (point - x_left) / h
    local_values = p2_basis(xi)

    global_dofs = [
        2 * element,
        2 * element + 1,
        2 * element + 2,
    ]

    n_elements = len(nodes) - 1

    for i_local, i_global in enumerate(global_dofs):
        if 1 <= i_global <= 2 * n_elements - 1:
            F[i_global - 1] += (
                weight * local_values[i_local]
            )


def assemble_dirac_load_p1(nodes):
    F = assemble_load_vector_p1(
        nodes,
        smooth_load,
    )

    add_point_load_p1(
        F,
        nodes,
        ALPHA,
        -2.0,
    )

    add_point_load_p1(
        F,
        nodes,
        BETA,
        -1.0,
    )

    return F


def assemble_dirac_load_p2(nodes):
    F = assemble_load_vector_p2(
        nodes,
        smooth_load,
    )

    add_point_load_p2(
        F,
        nodes,
        ALPHA,
        -2.0,
    )

    add_point_load_p2(
        F,
        nodes,
        BETA,
        -1.0,
    )

    return F


def exact_solution_dirac(x):
    x = np.asarray(x, dtype=float)
    u = np.empty_like(x)

    region_1 = x <= ALPHA
    region_2 = (
        (x > ALPHA)
        & (x <= GAMMA)
    )
    region_3 = (
        (x > GAMMA)
        & (x <= BETA)
    )
    region_4 = x > BETA

    u[region_1] = (
        3.0 * x[region_1] ** 2
        - 4.0 * x[region_1]
    )

    u[region_2] = -1.0

    u[region_3] = (
        -1.0
        + 3.0
        * (x[region_3] - GAMMA) ** 2
    )

    u[region_4] = (
        3.0 * x[region_4] ** 2
        + (2.0 * np.sqrt(2.0) - 5.0)
        * x[region_4]
        + 2.0
        - 2.0 * np.sqrt(2.0)
    )

    return u


def exact_derivative_dirac(x):
    x = np.asarray(x, dtype=float)
    du = np.empty_like(x)

    region_1 = x < ALPHA
    region_2 = (
        (x >= ALPHA)
        & (x < GAMMA)
    )
    region_3 = (
        (x >= GAMMA)
        & (x < BETA)
    )
    region_4 = x >= BETA

    du[region_1] = (
        6.0 * x[region_1]
        - 4.0
    )

    du[region_2] = 0.0

    du[region_3] = (
        6.0
        * (x[region_3] - GAMMA)
    )

    du[region_4] = (
        6.0 * x[region_4]
        + 2.0 * np.sqrt(2.0)
        - 5.0
    )

    return du


def compute_errors(
    nodes,
    U,
    evaluate_solution,
    evaluate_derivative,
):
    """
    Compute L2 and H1-seminorm errors by Gauss quadrature.

    Elements containing alpha, gamma, or beta are split at those exact
    breakpoints before quadrature, so the nonsmooth exact solution is
    integrated on smooth subintervals.
    """
    gauss_points, gauss_weights = leggauss(
        ERROR_QUADRATURE_ORDER
    )

    breakpoints = np.array([
        ALPHA,
        GAMMA,
        BETA,
    ])

    l2_squared = 0.0
    h1_squared = 0.0

    for element in range(len(nodes) - 1):
        x_left = nodes[element]
        x_right = nodes[element + 1]

        local_breakpoints = breakpoints[
            (breakpoints > x_left)
            & (breakpoints < x_right)
        ]

        subinterval_nodes = np.concatenate((
            [x_left],
            local_breakpoints,
            [x_right],
        ))

        for j in range(len(subinterval_nodes) - 1):
            a = subinterval_nodes[j]
            b = subinterval_nodes[j + 1]
            h_sub = b - a

            x_quad = (
                0.5 * (a + b)
                + 0.5 * h_sub * gauss_points
            )

            w_quad = (
                0.5 * h_sub * gauss_weights
            )

            u_h = evaluate_solution(
                nodes,
                U,
                x_quad,
            )

            du_h = evaluate_derivative(
                nodes,
                U,
                x_quad,
            )

            u_exact = exact_solution_dirac(
                x_quad
            )

            du_exact = exact_derivative_dirac(
                x_quad
            )

            l2_squared += np.sum(
                w_quad
                * (u_h - u_exact) ** 2
            )

            h1_squared += np.sum(
                w_quad
                * (du_h - du_exact) ** 2
            )

    return (
        float(np.sqrt(l2_squared)),
        float(np.sqrt(h1_squared)),
    )


def convergence_rates(errors):
    h = 1.0 / ELEMENT_COUNTS

    rates = np.full(
        len(errors),
        np.nan,
        dtype=float,
    )

    for i in range(1, len(errors)):
        rates[i] = np.log(
            errors[i - 1] / errors[i]
        ) / np.log(
            h[i - 1] / h[i]
        )

    return rates


def solve_p1(nodes):
    A = assemble_stiffness_matrix_p1(nodes)
    F = assemble_dirac_load_p1(nodes)

    U, psi, result = solve_obstacle_problem_pdas(
        nodes,
        A,
        F,
        obstacle_dirac,
        maxiter=300,
    )

    return U, psi, result, A, F


def solve_p2(nodes):
    p2_nodes = create_p2_nodes(nodes)

    A = assemble_stiffness_matrix_p2(nodes)
    F = assemble_dirac_load_p2(nodes)

    U, psi, result = solve_obstacle_problem_pdas(
        p2_nodes,
        A,
        F,
        obstacle_dirac,
        maxiter=300,
    )

    return U, psi, result, p2_nodes, A, F


def kkt_check(U, psi, A, F, label):
    gap = U - psi
    multiplier = A @ U - F

    if np.min(gap) < -1e-10:
        raise RuntimeError(
            f"Obstacle violation in {label}."
        )

    if np.min(multiplier) < -1e-10:
        raise RuntimeError(
            f"Negative multiplier in {label}."
        )

    if np.max(np.abs(gap * multiplier)) > 1e-10:
        raise RuntimeError(
            f"Complementarity failure in {label}."
        )


def solver_cross_check_p2():
    n_elements = 40
    nodes = create_mesh(n_elements)

    (
        U_pdas,
        _,
        result_pdas,
        p2_nodes,
        A,
        F,
    ) = solve_p2(nodes)

    U_lbfgsb, _, result_lbfgsb = (
        solve_obstacle_problem(
            p2_nodes,
            A,
            F,
            obstacle_dirac,
        )
    )

    difference = float(
        np.max(
            np.abs(
                U_lbfgsb - U_pdas
            )
        )
    )

    print()
    print("P2 Dirac solver cross-check: M = 40")
    print("-" * 46)

    print(
        "max |PDAS - L-BFGS-B| = "
        f"{difference:.6e}"
    )

    print(
        f"PDAS iterations: {result_pdas.nit}"
    )

    print(
        f"L-BFGS-B iterations: {result_lbfgsb.nit}"
    )


def main():
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    p1_l2 = []
    p1_h1 = []
    p2_l2 = []
    p2_h1 = []
    p1_iterations = []
    p2_iterations = []
    p2_min_pointwise_gap = []

    solution_data = None

    print(
        "M     P1 L2        P1 H1        "
        "P2 L2        P2 H1        "
        "P1 it   P2 it   P2 min gap"
    )

    print("-" * 103)

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(
            int(n_elements)
        )

        (
            U_p1,
            psi_p1,
            result_p1,
            A_p1,
            F_p1,
        ) = solve_p1(nodes)

        (
            U_p2,
            psi_p2,
            result_p2,
            _,
            A_p2,
            F_p2,
        ) = solve_p2(nodes)

        kkt_check(
            U_p1,
            psi_p1,
            A_p1,
            F_p1,
            f"P1, M={n_elements}",
        )

        kkt_check(
            U_p2,
            psi_p2,
            A_p2,
            F_p2,
            f"P2, M={n_elements}",
        )

        p1_errors = compute_errors(
            nodes,
            U_p1,
            evaluate_fem_solution_p1,
            evaluate_fem_derivative_p1,
        )

        p2_errors = compute_errors(
            nodes,
            U_p2,
            evaluate_fem_solution_p2,
            evaluate_fem_derivative_p2,
        )

        p1_l2.append(p1_errors[0])
        p1_h1.append(p1_errors[1])
        p2_l2.append(p2_errors[0])
        p2_h1.append(p2_errors[1])

        p1_iterations.append(result_p1.nit)
        p2_iterations.append(result_p2.nit)

        x_dense = np.linspace(
            0.0,
            1.0,
            100001,
        )

        p2_gap = (
            evaluate_fem_solution_p2(
                nodes,
                U_p2,
                x_dense,
            )
            - obstacle_dirac(x_dense)
        )

        min_gap = float(
            np.min(p2_gap)
        )

        p2_min_pointwise_gap.append(
            min_gap
        )

        print(
            f"{n_elements:<5d} "
            f"{p1_errors[0]:<12.6e} "
            f"{p1_errors[1]:<12.6e} "
            f"{p2_errors[0]:<12.6e} "
            f"{p2_errors[1]:<12.6e} "
            f"{result_p1.nit:<7d} "
            f"{result_p2.nit:<7d} "
            f"{min_gap:.3e}"
        )

        if int(n_elements) == SOLUTION_ELEMENTS:
            solution_data = (
                nodes,
                U_p1,
                U_p2,
            )

    p1_l2 = np.array(p1_l2)
    p1_h1 = np.array(p1_h1)
    p2_l2 = np.array(p2_l2)
    p2_h1 = np.array(p2_h1)

    p1_l2_rates = convergence_rates(p1_l2)
    p1_h1_rates = convergence_rates(p1_h1)
    p2_l2_rates = convergence_rates(p2_l2)
    p2_h1_rates = convergence_rates(p2_h1)

    print()
    print("Observed convergence rates")

    print(
        "M     P1 L2 rate   P1 H1 rate   "
        "P2 L2 rate   P2 H1 rate"
    )

    print("-" * 68)

    for i, n_elements in enumerate(ELEMENT_COUNTS):
        print(
            f"{n_elements:<5d} "
            f"{p1_l2_rates[i]:<12.3f} "
            f"{p1_h1_rates[i]:<12.3f} "
            f"{p2_l2_rates[i]:<12.3f} "
            f"{p2_h1_rates[i]:<12.3f}"
        )

    if solution_data is None:
        raise RuntimeError(
            "Solution comparison data missing."
        )

    nodes, U_p1, U_p2 = solution_data

    x_plot = np.linspace(
        0.0,
        1.0,
        4001,
    )

    u_exact = exact_solution_dirac(x_plot)

    u_p1 = evaluate_fem_solution_p1(
        nodes,
        U_p1,
        x_plot,
    )

    u_p2 = evaluate_fem_solution_p2(
        nodes,
        U_p2,
        x_plot,
    )

    psi = obstacle_dirac(x_plot)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    axes[0].plot(
        x_plot,
        u_exact,
        label="Exact solution",
        linewidth=2.0,
    )

    axes[0].plot(
        x_plot,
        u_p1,
        "--",
        label="P1 FEM",
    )

    axes[0].plot(
        x_plot,
        u_p2,
        ":",
        label="P2 FEM",
        linewidth=2.0,
    )

    axes[0].plot(
        x_plot,
        psi,
        label="Obstacle",
        alpha=0.8,
    )

    axes[0].axvline(
        ALPHA,
        linestyle=":",
        alpha=0.7,
        label=r"$\alpha=1/3$",
    )

    axes[0].axvline(
        BETA,
        linestyle=":",
        alpha=0.7,
        label=r"$\beta=2/3$",
    )

    axes[0].axvline(
        GAMMA,
        linestyle="--",
        alpha=0.7,
        label=r"$\gamma=1-\sqrt{2}/3$",
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title("Solution comparison")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    zoom_left = (
        ALPHA
        - 2.0 / SOLUTION_ELEMENTS
    )

    zoom_right = (
        ALPHA
        + 2.0 / SOLUTION_ELEMENTS
    )

    zoom = (
        (x_plot >= zoom_left)
        & (x_plot <= zoom_right)
    )

    axes[1].plot(
        x_plot[zoom],
        u_exact[zoom],
        label="Exact solution",
        linewidth=2.0,
    )

    axes[1].plot(
        x_plot[zoom],
        u_p1[zoom],
        "--",
        label="P1 FEM",
    )

    axes[1].plot(
        x_plot[zoom],
        u_p2[zoom],
        ":",
        label="P2 FEM",
        linewidth=2.0,
    )

    axes[1].axvline(
        ALPHA,
        linestyle="--",
        alpha=0.6,
        label=r"Dirac point $\alpha$",
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")
    axes[1].set_title(
        "Zoom near the derivative jump"
    )

    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        r"P1 vs P2 with Dirac loads, $M=20$"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        figures_dir
        / "p1_p2_dirac_solution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    h = 1.0 / ELEMENT_COUNTS

    l2_reference = (
        p1_l2[0]
        * (h / h[0]) ** 1.5
    )

    h1_reference = (
        p1_h1[0]
        * (h / h[0]) ** 0.5
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    axes[0].loglog(
        h,
        p1_l2,
        "o-",
        label="P1",
    )

    axes[0].loglog(
        h,
        p2_l2,
        "s-",
        label="P2",
    )

    axes[0].loglog(
        h,
        l2_reference,
        "--",
        label=r"$O(h^{3/2})$ reference",
    )

    axes[0].set_xlabel("Mesh size h")
    axes[0].set_ylabel(r"$L^2$ error")
    axes[0].set_title(r"$L^2$ error")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].loglog(
        h,
        p1_h1,
        "o-",
        label="P1",
    )

    axes[1].loglog(
        h,
        p2_h1,
        "s-",
        label="P2",
    )

    axes[1].loglog(
        h,
        h1_reference,
        "--",
        label=r"$O(h^{1/2})$ reference",
    )

    axes[1].set_xlabel("Mesh size h")
    axes[1].set_ylabel(
        r"$H^1$ seminorm error"
    )

    axes[1].set_title(
        r"$H^1$ seminorm error"
    )

    axes[1].grid(True, which="both")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(h)
        ax.set_xticklabels([
            r"$1/20$",
            r"$1/40$",
            r"$1/80$",
            r"$1/160$",
            r"$1/320$",
        ])
        ax.tick_params(
            axis="x",
            which="minor",
            labelbottom=False,
        )

    fig.suptitle(
        "P1 vs P2 convergence with Dirac loads"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        figures_dir
        / "p1_p2_dirac_convergence.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    solver_cross_check_p2()

    print()
    print("Generated figures:")
    print(
        "  figures/p1_p2_dirac_solution.png"
    )
    print(
        "  figures/p1_p2_dirac_convergence.png"
    )


if __name__ == "__main__":
    main()