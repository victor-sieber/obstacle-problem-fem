from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.fem import (
    assemble_load_vector,
    assemble_stiffness_matrix,
    create_mesh,
    interpolate_fem_solution,
)
from src.problem import exact_solution, load, obstacle
from src.solver import solve_obstacle_problem, solve_obstacle_problem_pdas


ALPHA = 1.0 / 3.0
ELEMENT_COUNTS = [20, 40, 80, 160, 320]


def main():
    max_differences = []
    pdas_iterations = []
    lbfgsb_iterations = []

    comparison_data = None

    print(
        "M     L-BFGS-B it   PDAS it   "
        "max |U_LBFGSB-U_PDAS|"
    )
    print("-" * 60)

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(n_elements)
        A = assemble_stiffness_matrix(nodes)
        F = assemble_load_vector(nodes, load)

        obstacle_function = (
            lambda x: obstacle(x, ALPHA)
        )

        U_lbfgsb, psi, result_lbfgsb = (
            solve_obstacle_problem(
                nodes,
                A,
                F,
                obstacle_function,
            )
        )

        U_pdas, _, result_pdas = (
            solve_obstacle_problem_pdas(
                nodes,
                A,
                F,
                obstacle_function,
            )
        )

        difference = float(
            np.max(
                np.abs(
                    U_lbfgsb - U_pdas
                )
            )
        )

        gap = U_pdas - psi
        multiplier = A @ U_pdas - F
        complementarity = gap * multiplier

        if np.min(gap) < -1e-10:
            raise RuntimeError(
                f"Obstacle violation for M={n_elements}"
            )

        if np.min(multiplier) < -1e-10:
            raise RuntimeError(
                f"Negative multiplier for M={n_elements}"
            )

        if np.max(np.abs(complementarity)) > 1e-10:
            raise RuntimeError(
                f"Complementarity failure for M={n_elements}"
            )

        max_differences.append(difference)
        lbfgsb_iterations.append(
            result_lbfgsb.nit
        )
        pdas_iterations.append(
            result_pdas.nit
        )

        print(
            f"{n_elements:<5d} "
            f"{result_lbfgsb.nit:<13d} "
            f"{result_pdas.nit:<9d} "
            f"{difference:.3e}"
        )

        if n_elements == 20:
            comparison_data = (
                nodes,
                U_lbfgsb,
                U_pdas,
            )

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    element_counts = np.array(
        ELEMENT_COUNTS
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
    )

    axes[0].plot(
        element_counts,
        lbfgsb_iterations,
        "o-",
        label="L-BFGS-B",
    )

    axes[0].plot(
        element_counts,
        pdas_iterations,
        "s-",
        label="PDAS",
    )

    axes[0].set_xlabel(
        "Number of elements"
    )
    axes[0].set_ylabel("Iterations")
    axes[0].set_title("Iteration count")
    axes[0].set_xticks(element_counts)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].semilogy(
        element_counts,
        max_differences,
        "o-",
    )

    axes[1].set_xlabel(
        "Number of elements"
    )
    axes[1].set_ylabel(
        r"$\|U_{\mathrm{LBFGS}}"
        r"-U_{\mathrm{PDAS}}\|_\infty$"
    )
    axes[1].set_title(
        "Agreement of discrete solutions"
    )
    axes[1].set_xticks(element_counts)
    axes[1].grid(
        True,
        which="both",
    )

    fig.suptitle(
        r"P1 stress test, $\alpha=1/3$"
    )
    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    fig.savefig(
        figures_dir
        / "p1_alpha_one_third_solver_test.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    if comparison_data is None:
        raise RuntimeError(
            "Comparison data missing."
        )

    nodes, U_lbfgsb, U_pdas = (
        comparison_data
    )

    x_plot = np.linspace(
        0.0,
        1.0,
        1000,
    )

    exact = exact_solution(
        x_plot,
        ALPHA,
    )

    psi_plot = obstacle(
        x_plot,
        ALPHA,
    )

    lbfgsb_plot = (
        interpolate_fem_solution(
            nodes,
            U_lbfgsb,
            x_plot,
        )
    )

    pdas_plot = (
        interpolate_fem_solution(
            nodes,
            U_pdas,
            x_plot,
        )
    )

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x_plot,
        exact,
        label="Exact solution",
    )

    plt.plot(
        x_plot,
        lbfgsb_plot,
        "--",
        label="L-BFGS-B FEM",
    )

    plt.plot(
        x_plot,
        pdas_plot,
        ":",
        label="PDAS FEM",
    )

    plt.plot(
        x_plot,
        psi_plot,
        label="Obstacle",
    )

    plt.axvline(
        ALPHA,
        linestyle=":",
        label="Exact free boundary",
    )

    plt.axvline(
        1.0 - ALPHA,
        linestyle=":",
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        r"P1 obstacle problem: "
        r"$\alpha=1/3$, 20 elements"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig.savefig(
        figures_dir
        / "p1_alpha_one_third_solution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()