from pathlib import Path
import csv
import sys
import time

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
from src.problem import (
    exact_solution,
    load,
    obstacle,
)
from src.solver import (
    solve_obstacle_problem,
    solve_obstacle_problem_pdas,
)


ALPHA = 0.2

ELEMENT_COUNTS = np.array(
    [20, 40, 80, 160, 320]
)

TIMING_REPEATS = 7

COMPARISON_ELEMENTS = 20


def kkt_diagnostics(U, psi, A, F):
    gap = U - psi
    multiplier = A @ U - F

    return {
        "min_gap": float(
            np.min(gap)
        ),
        "min_multiplier": float(
            np.min(multiplier)
        ),
        "max_complementarity": float(
            np.max(
                np.abs(
                    gap * multiplier
                )
            )
        ),
    }


def median_solver_time(
    solver,
    nodes,
    A,
    F,
    obstacle_function,
    repeats,
):
    times = []
    result_tuple = None

    for _ in range(repeats):
        start = time.perf_counter()

        result_tuple = solver(
            nodes,
            A,
            F,
            obstacle_function,
        )

        times.append(
            time.perf_counter() - start
        )

    return (
        float(np.median(times)),
        result_tuple,
    )


def main():
    figures_dir = ROOT / "figures"
    results_dir = ROOT / "results"

    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    rows = []
    comparison_data = None

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(
            int(n_elements)
        )

        A = assemble_stiffness_matrix(
            nodes
        )

        F = assemble_load_vector(
            nodes,
            load,
        )

        obstacle_function = (
            lambda x: obstacle(
                x,
                ALPHA,
            )
        )

        # Warm up each solver once so one-time
        # setup overhead does not dominate timing.
        solve_obstacle_problem(
            nodes,
            A,
            F,
            obstacle_function,
        )

        solve_obstacle_problem_pdas(
            nodes,
            A,
            F,
            obstacle_function,
        )

        (
            lbfgsb_time,
            lbfgsb_output,
        ) = median_solver_time(
            solve_obstacle_problem,
            nodes,
            A,
            F,
            obstacle_function,
            TIMING_REPEATS,
        )

        (
            pdas_time,
            pdas_output,
        ) = median_solver_time(
            solve_obstacle_problem_pdas,
            nodes,
            A,
            F,
            obstacle_function,
            TIMING_REPEATS,
        )

        (
            U_lbfgsb,
            psi_lbfgsb,
            result_lbfgsb,
        ) = lbfgsb_output

        (
            U_pdas,
            psi_pdas,
            result_pdas,
        ) = pdas_output

        max_difference = float(
            np.max(
                np.abs(
                    U_lbfgsb
                    - U_pdas
                )
            )
        )

        if max_difference > 1e-6:
            raise RuntimeError(
                "Solver disagreement too large "
                f"for N={n_elements}: "
                f"{max_difference:.3e}"
            )

        lbfgsb_kkt = kkt_diagnostics(
            U_lbfgsb,
            psi_lbfgsb,
            A,
            F,
        )

        pdas_kkt = kkt_diagnostics(
            U_pdas,
            psi_pdas,
            A,
            F,
        )

        if pdas_kkt["min_gap"] < -1e-10:
            raise RuntimeError(
                "PDAS violates the obstacle "
                f"for N={n_elements}."
            )

        if (
            pdas_kkt["min_multiplier"]
            < -1e-10
        ):
            raise RuntimeError(
                "PDAS has a negative multiplier "
                f"for N={n_elements}."
            )

        if (
            pdas_kkt[
                "max_complementarity"
            ]
            > 1e-10
        ):
            raise RuntimeError(
                "PDAS violates complementarity "
                f"for N={n_elements}."
            )

        rows.append(
            {
                "n_elements": int(
                    n_elements
                ),
                "n_interior": int(
                    n_elements - 1
                ),
                "lbfgsb_time_ms": (
                    1000.0
                    * lbfgsb_time
                ),
                "pdas_time_ms": (
                    1000.0
                    * pdas_time
                ),
                "lbfgsb_iterations": int(
                    result_lbfgsb.nit
                ),
                "pdas_iterations": int(
                    result_pdas.nit
                ),
                "max_solution_difference": (
                    max_difference
                ),
                "lbfgsb_min_gap": (
                    lbfgsb_kkt[
                        "min_gap"
                    ]
                ),
                "lbfgsb_min_multiplier": (
                    lbfgsb_kkt[
                        "min_multiplier"
                    ]
                ),
                "lbfgsb_max_complementarity": (
                    lbfgsb_kkt[
                        "max_complementarity"
                    ]
                ),
                "pdas_min_gap": (
                    pdas_kkt[
                        "min_gap"
                    ]
                ),
                "pdas_min_multiplier": (
                    pdas_kkt[
                        "min_multiplier"
                    ]
                ),
                "pdas_max_complementarity": (
                    pdas_kkt[
                        "max_complementarity"
                    ]
                ),
                "lbfgsb_stop_reason": str(
                    result_lbfgsb.message
                ),
                "pdas_stop_reason": str(
                    result_pdas.message
                ),
            }
        )

        if (
            n_elements
            == COMPARISON_ELEMENTS
        ):
            comparison_data = (
                nodes,
                U_lbfgsb,
                U_pdas,
            )

    csv_path = (
        results_dir
        / "p1_alpha_0p2_solver_benchmark.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "N     L-BFGS-B ms   PDAS ms      "
        "L-BFGS-B it   PDAS it   "
        "max |U_LBFGSB-U_PDAS|"
    )

    print("-" * 86)

    for row in rows:
        print(
            f"{row['n_elements']:<5d} "
            f"{row['lbfgsb_time_ms']:<13.6f} "
            f"{row['pdas_time_ms']:<12.6f} "
            f"{row['lbfgsb_iterations']:<13d} "
            f"{row['pdas_iterations']:<9d} "
            f"{row['max_solution_difference']:.3e}"
        )

    print(
        "\nL-BFGS-B stopping reasons:"
    )

    for row in rows:
        print(
            f"  N={row['n_elements']}: "
            f"{row['lbfgsb_stop_reason']}"
        )

    print(
        "\nPDAS stopping reasons:"
    )

    for row in rows:
        print(
            f"  N={row['n_elements']}: "
            f"{row['pdas_stop_reason']}"
        )

    element_counts = np.array(
        [
            row["n_elements"]
            for row in rows
        ]
    )

    lbfgsb_times = np.array(
        [
            row["lbfgsb_time_ms"]
            for row in rows
        ]
    )

    pdas_times = np.array(
        [
            row["pdas_time_ms"]
            for row in rows
        ]
    )

    lbfgsb_iterations = np.array(
        [
            row["lbfgsb_iterations"]
            for row in rows
        ]
    )

    pdas_iterations = np.array(
        [
            row["pdas_iterations"]
            for row in rows
        ]
    )

    max_differences = np.array(
        [
            row[
                "max_solution_difference"
            ]
            for row in rows
        ]
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.5),
    )

    # Plot 1: runtime
    axes[0].semilogy(
        element_counts,
        lbfgsb_times,
        "o-",
        label="L-BFGS-B",
    )

    axes[0].semilogy(
        element_counts,
        pdas_times,
        "s-",
        label="PDAS",
    )

    axes[0].set_xlabel(
        "Number of elements"
    )

    axes[0].set_ylabel(
        "Median solve time [ms]"
    )

    axes[0].set_title(
        "Solver runtime"
    )

    axes[0].set_xticks(
        element_counts
    )

    axes[0].grid(
        True,
        which="both",
    )

    axes[0].legend()

    # Plot 2: iteration count
    axes[1].plot(
        element_counts,
        lbfgsb_iterations,
        "o-",
        label="L-BFGS-B",
    )

    axes[1].plot(
        element_counts,
        pdas_iterations,
        "s-",
        label="PDAS",
    )

    axes[1].set_xlabel(
        "Number of elements"
    )

    axes[1].set_ylabel(
        "Iterations"
    )

    axes[1].set_title(
        "Iteration count"
    )

    axes[1].set_xticks(
        element_counts
    )

    axes[1].grid(True)
    axes[1].legend()

    # Plot 3: agreement
    axes[2].semilogy(
        element_counts,
        max_differences,
        "o-",
    )

    axes[2].set_xlabel(
        "Number of elements"
    )

    axes[2].set_ylabel(
        r"$\|U_{\mathrm{LBFGS}}"
        r"-U_{\mathrm{PDAS}}\|_\infty$"
    )

    axes[2].set_title(
        "Agreement of discrete solutions"
    )

    axes[2].set_xticks(
        element_counts
    )

    axes[2].grid(
        True,
        which="both",
    )

    fig.suptitle(
        r"P1 obstacle solver benchmark, "
        r"$\alpha=0.2$"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    benchmark_figure_path = (
        figures_dir
        / "p1_alpha_0p2_solver_benchmark.png"
    )

    fig.savefig(
        benchmark_figure_path,
        dpi=200,
    )

    plt.close(fig)

    if comparison_data is None:
        raise RuntimeError(
            "Comparison mesh was not generated."
        )

    (
        nodes,
        U_lbfgsb,
        U_pdas,
    ) = comparison_data

    x_plot = np.linspace(
        0.0,
        1.0,
        1000,
    )

    u_exact = exact_solution(
        x_plot,
        ALPHA,
    )

    u_lbfgsb = interpolate_fem_solution(
        nodes,
        U_lbfgsb,
        x_plot,
    )

    u_pdas = interpolate_fem_solution(
        nodes,
        U_pdas,
        x_plot,
    )

    psi_plot = obstacle(
        x_plot,
        ALPHA,
    )

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x_plot,
        u_exact,
        label="Exact solution",
    )

    plt.plot(
        x_plot,
        u_lbfgsb,
        "--",
        label="L-BFGS-B FEM",
    )

    plt.plot(
        x_plot,
        u_pdas,
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
        "P1 solver comparison: "
        f"alpha={ALPHA}, "
        f"elements={COMPARISON_ELEMENTS}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    comparison_figure_path = (
        figures_dir
        / "p1_alpha_0p2_solver_comparison.png"
    )

    fig.savefig(
        comparison_figure_path,
        dpi=200,
    )

    plt.close(fig)

    print(
        "\nSaved benchmark data:",
        csv_path,
    )

    print(
        "Saved benchmark figure:",
        benchmark_figure_path,
    )

    print(
        "Saved solution comparison:",
        comparison_figure_path,
    )


if __name__ == "__main__":
    main()
