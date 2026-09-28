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
)
from src.fem_p2 import (
    assemble_load_vector as assemble_load_vector_p2,
    assemble_stiffness_matrix as assemble_stiffness_matrix_p2,
    create_p2_nodes,
    evaluate_fem_derivative as evaluate_fem_derivative_p2,
    evaluate_fem_solution as evaluate_fem_solution_p2,
    p2_basis,
    p2_basis_derivative,
)
from src.problem import (
    exact_derivative,
    exact_solution,
    load,
    obstacle,
)
from src.solver import (
    solve_obstacle_problem,
    solve_obstacle_problem_pdas,
)


ALPHA_ALIGNED = 0.2
ALPHA_NONALIGNED = 1.0 / 3.0

# Refinement by a factor of four keeps the free boundary at the same
# relative position inside the cut element for alpha = 1/3.
ELEMENT_COUNTS = np.array([20, 80, 320])

SOLUTION_ELEMENTS = 20
ERROR_QUADRATURE_ORDER = 32


def compute_errors(
    nodes,
    interior_values,
    alpha,
    evaluate_solution,
    evaluate_derivative,
):
    """
    Compute L2 and H1-seminorm errors by Gauss quadrature.

    Elements containing a free boundary are split there first, so the
    exact solution is smooth on every integration subinterval.
    """
    gauss_points, gauss_weights = leggauss(
        ERROR_QUADRATURE_ORDER
    )

    breakpoints = np.array([
        alpha,
        1.0 - alpha,
    ])

    l2_error_squared = 0.0
    h1_error_squared = 0.0

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
                interior_values,
                x_quad,
            )

            du_h = evaluate_derivative(
                nodes,
                interior_values,
                x_quad,
            )

            u_exact = exact_solution(
                x_quad,
                alpha,
            )

            du_exact = exact_derivative(
                x_quad,
                alpha,
            )

            l2_error_squared += np.sum(
                w_quad
                * (u_exact - u_h) ** 2
            )

            h1_error_squared += np.sum(
                w_quad
                * (du_exact - du_h) ** 2
            )

    return (
        np.sqrt(l2_error_squared),
        np.sqrt(h1_error_squared),
    )


def convergence_rates(errors, mesh_sizes):
    rates = np.full_like(
        errors,
        np.nan,
        dtype=float,
    )

    for i in range(1, len(errors)):
        rates[i] = np.log(
            errors[i - 1] / errors[i]
        ) / np.log(
            mesh_sizes[i - 1] / mesh_sizes[i]
        )

    return rates


def solve_p1(nodes, alpha):
    A = assemble_stiffness_matrix_p1(nodes)
    F = assemble_load_vector_p1(nodes, load)

    obstacle_function = (
        lambda x: obstacle(x, alpha)
    )

    U, psi, result = solve_obstacle_problem_pdas(
        nodes,
        A,
        F,
        obstacle_function,
        maxiter=300,
    )

    return U, psi, result


def solve_p2(nodes, alpha):
    p2_nodes = create_p2_nodes(nodes)

    A = assemble_stiffness_matrix_p2(nodes)
    F = assemble_load_vector_p2(nodes, load)

    obstacle_function = (
        lambda x: obstacle(x, alpha)
    )

    U, psi, result = solve_obstacle_problem_pdas(
        p2_nodes,
        A,
        F,
        obstacle_function,
        maxiter=300,
    )

    return U, psi, result, p2_nodes, A, F


def check_p2_basis_and_stiffness():
    """
    Verify the defining P2 interpolation property and the local stiffness matrix.
    """
    interpolation_points = np.array(
        [0.0, 0.5, 1.0]
    )

    basis_matrix = p2_basis(
        interpolation_points
    )

    if not np.allclose(
        basis_matrix,
        np.eye(3),
        atol=1e-14,
    ):
        raise RuntimeError(
            "P2 basis failed the nodal interpolation test."
        )

    gauss_points, gauss_weights = leggauss(2)

    xi = 0.5 * (gauss_points + 1.0)
    weights = 0.5 * gauss_weights

    basis_derivatives = p2_basis_derivative(xi)

    local_from_quadrature = (
        (basis_derivatives * weights)
        @ basis_derivatives.T
    )

    local_exact = (
        np.array(
            [
                [7.0, -8.0, 1.0],
                [-8.0, 16.0, -8.0],
                [1.0, -8.0, 7.0],
            ]
        )
        / 3.0
    )

    if not np.allclose(
        local_from_quadrature,
        local_exact,
        atol=1e-14,
    ):
        raise RuntimeError(
            "P2 stiffness quadrature failed the exact local-matrix test."
        )

    print("P2 basis test: passed")
    print("P2 local stiffness test: passed")


def aligned_sanity_check(figures_dir):
    alpha = ALPHA_ALIGNED
    nodes = create_mesh(SOLUTION_ELEMENTS)

    U_p1, _, _ = solve_p1(
        nodes,
        alpha,
    )

    U_p2, _, _, _, _, _ = solve_p2(
        nodes,
        alpha,
    )

    p1_l2, p1_h1 = compute_errors(
        nodes,
        U_p1,
        alpha,
        evaluate_fem_solution_p1,
        evaluate_fem_derivative_p1,
    )

    p2_l2, p2_h1 = compute_errors(
        nodes,
        U_p2,
        alpha,
        evaluate_fem_solution_p2,
        evaluate_fem_derivative_p2,
    )

    print()
    print("Aligned benchmark: alpha = 0.2, M = 20")
    print("--------------------------------------------------------")

    print(
        f"P1: L2 = {p1_l2:.6e}, "
        f"H1-semi = {p1_h1:.6e}"
    )

    print(
        f"P2: L2 = {p2_l2:.6e}, "
        f"H1-semi = {p2_h1:.6e}"
    )

    print(
        "P2 is essentially exact here because the free boundaries "
        "align with the mesh and the exact pieces are quadratic."
    )

    x_plot = np.linspace(
        0.0,
        1.0,
        2001,
    )

    u_exact = exact_solution(
        x_plot,
        alpha,
    )

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

    psi = obstacle(
        x_plot,
        alpha,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_plot,
        u_exact,
        label="Exact solution",
        linewidth=2.0,
    )

    plt.plot(
        x_plot,
        u_p1,
        "--",
        label="P1 FEM",
    )

    plt.plot(
        x_plot,
        u_p2,
        ":",
        label="P2 FEM",
        linewidth=2.0,
    )

    plt.plot(
        x_plot,
        psi,
        label="Obstacle",
        alpha=0.8,
    )

    plt.axvline(
        alpha,
        linestyle="--",
        alpha=0.5,
    )

    plt.axvline(
        1.0 - alpha,
        linestyle="--",
        alpha=0.5,
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")

    plt.title(
        "P1 vs P2 on the aligned benchmark, "
        r"$\alpha=0.2$, $M=20$"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "p1_p2_alpha_0p2_solution.png",
        dpi=200,
    )

    plt.close()


def nonaligned_comparison(figures_dir):
    alpha = ALPHA_NONALIGNED

    mesh_sizes = 1.0 / ELEMENT_COUNTS

    p1_l2 = []
    p1_h1 = []

    p2_l2 = []
    p2_h1 = []

    p2_min_pointwise_gap = []
    p2_iterations = []

    solution_data = None

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(
            int(n_elements)
        )

        U_p1, _, _ = solve_p1(
            nodes,
            alpha,
        )

        (
            U_p2,
            psi_p2,
            result_p2,
            _,
            _,
            _,
        ) = solve_p2(
            nodes,
            alpha,
        )

        l2, h1 = compute_errors(
            nodes,
            U_p1,
            alpha,
            evaluate_fem_solution_p1,
            evaluate_fem_derivative_p1,
        )

        p1_l2.append(l2)
        p1_h1.append(h1)

        l2, h1 = compute_errors(
            nodes,
            U_p2,
            alpha,
            evaluate_fem_solution_p2,
            evaluate_fem_derivative_p2,
        )

        p2_l2.append(l2)
        p2_h1.append(h1)

        # P2 is constrained at its interpolation nodes.
        # Between those nodes, a quadratic can still dip below the obstacle.
        x_dense = np.linspace(
            0.0,
            1.0,
            100001,
        )

        pointwise_gap = (
            evaluate_fem_solution_p2(
                nodes,
                U_p2,
                x_dense,
            )
            - obstacle(
                x_dense,
                alpha,
            )
        )

        p2_min_pointwise_gap.append(
            float(np.min(pointwise_gap))
        )

        p2_iterations.append(
            result_p2.nit
        )

        if int(n_elements) == SOLUTION_ELEMENTS:
            solution_data = (
                nodes,
                U_p1,
                U_p2,
                psi_p2,
            )

    p1_l2 = np.array(p1_l2)
    p1_h1 = np.array(p1_h1)

    p2_l2 = np.array(p2_l2)
    p2_h1 = np.array(p2_h1)

    p2_min_pointwise_gap = np.array(
        p2_min_pointwise_gap
    )

    p1_l2_rates = convergence_rates(
        p1_l2,
        mesh_sizes,
    )

    p1_h1_rates = convergence_rates(
        p1_h1,
        mesh_sizes,
    )

    p2_l2_rates = convergence_rates(
        p2_l2,
        mesh_sizes,
    )

    p2_h1_rates = convergence_rates(
        p2_h1,
        mesh_sizes,
    )

    print()
    print("Nonaligned benchmark: alpha = 1/3")

    print(
        "M     P1 L2        P1 H1        "
        "P2 L2        P2 H1        P2 min gap     P2 it"
    )

    print("-" * 96)

    for i, n_elements in enumerate(ELEMENT_COUNTS):
        print(
            f"{n_elements:<5d} "
            f"{p1_l2[i]:<12.6e} "
            f"{p1_h1[i]:<12.6e} "
            f"{p2_l2[i]:<12.6e} "
            f"{p2_h1[i]:<12.6e} "
            f"{p2_min_pointwise_gap[i]:<14.6e} "
            f"{p2_iterations[i]}"
        )

    print()
    print("Observed rates under factor-four refinement")

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

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
    )

    axes[0].loglog(
        mesh_sizes,
        p1_l2,
        "o-",
        label="P1",
    )

    axes[0].loglog(
        mesh_sizes,
        p2_l2,
        "s-",
        label="P2",
    )

    l2_reference = (
        p1_l2[0]
        * (mesh_sizes / mesh_sizes[0]) ** 2
    )

    axes[0].loglog(
        mesh_sizes,
        l2_reference,
        "--",
        label=r"$O(h^2)$ reference",
    )

    axes[0].set_xlabel("Mesh size h")
    axes[0].set_ylabel(r"$L^2$ error")
    axes[0].set_title(r"$L^2$ error")
    axes[0].grid(True, which="both")
    axes[0].legend()

    axes[1].loglog(
        mesh_sizes,
        p1_h1,
        "o-",
        label="P1",
    )

    axes[1].loglog(
        mesh_sizes,
        p2_h1,
        "s-",
        label="P2",
    )

    h1_reference_p1 = (
        p1_h1[0]
        * (mesh_sizes / mesh_sizes[0])
    )

    h1_reference_p2 = (
        p2_h1[0]
        * (mesh_sizes / mesh_sizes[0]) ** 1.5
    )

    axes[1].loglog(
        mesh_sizes,
        h1_reference_p1,
        "--",
        label=r"$O(h)$ reference",
    )

    axes[1].loglog(
        mesh_sizes,
        h1_reference_p2,
        ":",
        label=r"$O(h^{3/2})$ reference",
    )

    axes[1].set_xlabel("Mesh size h")
    axes[1].set_ylabel(r"$H^1$ seminorm error")
    axes[1].set_title(r"$H^1$ seminorm error")
    axes[1].grid(True, which="both")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(mesh_sizes)
        ax.set_xticklabels([
            r"$1/20$",
            r"$1/80$",
            r"$1/320$",
        ])
        ax.tick_params(
            axis="x",
            which="minor",
            labelbottom=False,
        )

    fig.suptitle(
        r"P1 vs P2 convergence, nonaligned free boundary "
        r"$\alpha=1/3$"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        figures_dir
        / "p1_p2_alpha_one_third_convergence.png",
        dpi=200,
    )

    plt.close(fig)

    if solution_data is None:
        raise RuntimeError(
            "Solution comparison mesh was not generated."
        )

    nodes, U_p1, U_p2, _ = solution_data

    x_plot = np.linspace(
        0.0,
        1.0,
        4001,
    )

    u_exact = exact_solution(
        x_plot,
        alpha,
    )

    psi = obstacle(
        x_plot,
        alpha,
    )

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
        alpha,
        linestyle="--",
        alpha=0.5,
    )

    axes[0].axvline(
        1.0 - alpha,
        linestyle="--",
        alpha=0.5,
    )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].set_title("Solution comparison")
    axes[0].grid(True)
    axes[0].legend()

    gap_p2 = u_p2 - psi

    h = 1.0 / SOLUTION_ELEMENTS

    zoom_left = alpha - 2.0 * h
    zoom_right = alpha + 2.0 * h

    zoom = (
        (x_plot >= zoom_left)
        & (x_plot <= zoom_right)
    )

    axes[1].plot(
        x_plot[zoom],
        gap_p2[zoom],
        label=r"$u_h^{P2}-\psi$",
    )

    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].axvline(
        alpha,
        linestyle="--",
        alpha=0.5,
        label="Exact free boundary",
    )

    p2_nodes = create_p2_nodes(nodes)

    p2_gap_nodes = (
        np.concatenate(
            ([0.0], U_p2, [0.0])
        )
        - obstacle(
            p2_nodes,
            alpha,
        )
    )

    p2_zoom = (
        (p2_nodes >= zoom_left)
        & (p2_nodes <= zoom_right)
    )

    axes[1].plot(
        p2_nodes[p2_zoom],
        p2_gap_nodes[p2_zoom],
        "o",
        label="P2 constraint nodes",
    )

    axes[1].set_xlabel("x")
    axes[1].set_ylabel(
        r"$u_h^{P2}(x)-\psi(x)$"
    )

    axes[1].set_title(
        "Nodal P2 obstacle constraint"
    )

    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(
        r"P1 vs P2, nonaligned free boundary "
        r"$\alpha=1/3$, $M=20$"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    fig.savefig(
        figures_dir
        / "p1_p2_alpha_one_third_solution.png",
        dpi=200,
    )

    plt.close(fig)


def p2_solver_cross_check():
    """
    Check that L-BFGS-B and PDAS solve the same P2 quadratic program.
    """
    alpha = ALPHA_NONALIGNED
    n_elements = 40

    nodes = create_mesh(n_elements)
    p2_nodes = create_p2_nodes(nodes)

    A = assemble_stiffness_matrix_p2(nodes)
    F = assemble_load_vector_p2(
        nodes,
        load,
    )

    obstacle_function = (
        lambda x: obstacle(x, alpha)
    )

    U_pdas, _, result_pdas = (
        solve_obstacle_problem_pdas(
            p2_nodes,
            A,
            F,
            obstacle_function,
            maxiter=300,
        )
    )

    U_lbfgsb, _, result_lbfgsb = (
        solve_obstacle_problem(
            p2_nodes,
            A,
            F,
            obstacle_function,
        )
    )

    max_difference = np.max(
        np.abs(
            U_pdas - U_lbfgsb
        )
    )

    print()
    print(
        "P2 solver cross-check: alpha = 1/3, M = 40"
    )

    print("-" * 50)

    print(
        "max |PDAS - L-BFGS-B| = "
        f"{max_difference:.6e}"
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

    check_p2_basis_and_stiffness()

    aligned_sanity_check(
        figures_dir
    )

    nonaligned_comparison(
        figures_dir
    )

    p2_solver_cross_check()

    print()
    print("Generated figures:")
    print(
        "  figures/p1_p2_alpha_0p2_solution.png"
    )
    print(
        "  figures/p1_p2_alpha_one_third_solution.png"
    )
    print(
        "  figures/p1_p2_alpha_one_third_convergence.png"
    )


if __name__ == "__main__":
    main()