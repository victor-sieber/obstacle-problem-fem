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
    load,
    obstacle,
)

from src.solver import (
    solve_obstacle_problem_pdas,
    solve_obstacle_problem_pdas_nested,
)


ALPHA = 1.0 / 3.0

COARSEST_ELEMENTS = 10

ELEMENT_COUNTS = [
    20,
    40,
    80,
    160,
    320,
]

N_TIMING_REPEATS = 30


def obstacle_function(x):
    return obstacle(
        x,
        ALPHA,
    )


def assemble_problem(nodes):
    """
    Assemble the standard smooth P1 obstacle problem:

        f = -1,

        psi(x)
        =
        x(1-x) - 3/2 alpha^2,

        alpha = 1/3.
    """

    A = assemble_stiffness_matrix(
        nodes
    )

    F = assemble_load_vector(
        nodes,
        load,
    )

    return A, F


def solve_initial_coarse_problem():
    nodes = create_mesh(
        COARSEST_ELEMENTS
    )

    A, F = assemble_problem(
        nodes
    )

    U, _, result = (
        solve_obstacle_problem_pdas(
            nodes,
            A,
            F,
            obstacle_function,
        )
    )

    return nodes, U, result


def median_cpu_time_old(
    nodes,
    A,
    F,
):
    timings = []

    for _ in range(N_TIMING_REPEATS):
        start = time.process_time_ns()

        solve_obstacle_problem_pdas(
            nodes,
            A,
            F,
            obstacle_function,
        )

        end = time.process_time_ns()

        timings.append(
            (end - start) * 1e-9
        )

    return float(
        np.median(timings)
    )


def median_cpu_time_nested(
    coarse_nodes,
    coarse_U,
    nodes,
    A,
    F,
):
    timings = []

    for _ in range(N_TIMING_REPEATS):
        start = time.process_time_ns()

        initial_solution = (
            interpolate_fem_solution(
                coarse_nodes,
                coarse_U,
                nodes[1:-1],
            )
        )

        solve_obstacle_problem_pdas_nested(
            nodes,
            A,
            F,
            obstacle_function,
            initial_solution=(
                initial_solution
            ),
        )

        end = time.process_time_ns()

        timings.append(
            (end - start) * 1e-9
        )

    return float(
        np.median(timings)
    )


def run_comparison():
    coarse_nodes, coarse_U, _ = (
        solve_initial_coarse_problem()
    )

    rows = []

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(
            n_elements
        )

        A, F = assemble_problem(
            nodes
        )

        U_old, _, result_old = (
            solve_obstacle_problem_pdas(
                nodes,
                A,
                F,
                obstacle_function,
            )
        )

        initial_solution = (
            interpolate_fem_solution(
                coarse_nodes,
                coarse_U,
                nodes[1:-1],
            )
        )

        U_nested, _, result_nested = (
            solve_obstacle_problem_pdas_nested(
                nodes,
                A,
                F,
                obstacle_function,
                initial_solution=(
                    initial_solution
                ),
            )
        )

        old_cpu = median_cpu_time_old(
            nodes,
            A,
            F,
        )

        nested_cpu = median_cpu_time_nested(
            coarse_nodes,
            coarse_U,
            nodes,
            A,
            F,
        )

        difference = float(
            np.max(
                np.abs(
                    U_old
                    - U_nested
                )
            )
        )

        rows.append({
            "elements": n_elements,
            "dofs": n_elements - 1,
            "old_iterations": (
                result_old.nit
            ),
            "nested_iterations": (
                result_nested.nit
            ),
            "old_cpu": old_cpu,
            "nested_cpu": nested_cpu,
            "max_difference": difference,
        })

        coarse_nodes = nodes
        coarse_U = U_nested

    return rows


def print_results(rows):
    print()
    print(
        "Smooth nonaligned problem, "
        r"alpha = 1/3"
    )

    print(
        f"Median CPU time over "
        f"{N_TIMING_REPEATS} runs"
    )

    print()

    print(
        "elements  DOFs   "
        "old it   nested it   "
        "old CPU [ms]   nested CPU [ms]   "
        "max difference"
    )

    print("-" * 93)

    for row in rows:
        print(
            f"{row['elements']:<9d} "
            f"{row['dofs']:<6d} "
            f"{row['old_iterations']:<8d} "
            f"{row['nested_iterations']:<11d} "
            f"{1000.0 * row['old_cpu']:<14.4f} "
            f"{1000.0 * row['nested_cpu']:<17.4f} "
            f"{row['max_difference']:.3e}"
        )


def save_csv(rows, results_dir):
    path = (
        results_dir
        / "pdas_nested_initialization_smooth.csv"
    )

    fieldnames = [
        "elements",
        "dofs",
        "old_iterations",
        "nested_iterations",
        "old_cpu",
        "nested_cpu",
        "max_difference",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, figures_dir):
    elements = np.array([
        row["elements"]
        for row in rows
    ])

    old_iterations = np.array([
        row["old_iterations"]
        for row in rows
    ])

    nested_iterations = np.array([
        row["nested_iterations"]
        for row in rows
    ])

    old_cpu_ms = 1000.0 * np.array([
        row["old_cpu"]
        for row in rows
    ])

    nested_cpu_ms = 1000.0 * np.array([
        row["nested_cpu"]
        for row in rows
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    axes[0].plot(
        elements,
        old_iterations,
        "o-",
        label="existing initialization",
    )

    axes[0].plot(
        elements,
        nested_iterations,
        "s--",
        label=r"nested $V_H$ initialization",
    )

    axes[0].set_xscale(
        "log",
        base=2,
    )

    axes[0].set_xlabel(
        "number of elements"
    )

    axes[0].set_ylabel(
        "PDAS iterations"
    )

    axes[0].set_title(
        "PDAS iteration count"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[0].legend()

    axes[1].plot(
        elements,
        old_cpu_ms,
        "o-",
        label="existing initialization",
    )

    axes[1].plot(
        elements,
        nested_cpu_ms,
        "s--",
        label=r"nested $V_H$ initialization",
    )

    axes[1].set_xscale(
        "log",
        base=2,
    )

    axes[1].set_xlabel(
        "number of elements"
    )

    axes[1].set_ylabel(
        "median CPU time [ms]"
    )

    axes[1].set_title(
        "Median solver CPU time"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    axes[1].legend()

    fig.suptitle(
        r"Smooth problem, $\alpha=1/3$: "
        r"PDAS initialization comparison"
    )

    fig.savefig(
        figures_dir
        / "pdas_nested_initialization_smooth.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    figures_dir = ROOT / "figures"
    results_dir = ROOT / "results"

    figures_dir.mkdir(
        exist_ok=True
    )

    results_dir.mkdir(
        exist_ok=True
    )

    rows = run_comparison()

    print_results(
        rows
    )

    save_csv(
        rows,
        results_dir,
    )

    plot_results(
        rows,
        figures_dir,
    )


if __name__ == "__main__":
    main()