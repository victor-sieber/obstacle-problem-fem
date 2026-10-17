from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from experiments.dirac_adaptive_refinement import (
    run_adaptive_refinement,
)
from experiments.dirac_example import (
    assemble_dirac_load,
    compute_errors,
    obstacle_dirac,
)
from src.fem import (
    assemble_stiffness_matrix,
    create_mesh,
)
from src.solver import solve_obstacle_problem_pdas


UNIFORM_ELEMENT_COUNTS = [
    20,
    40,
    80,
    160,
    320,
]


def solve_on_mesh(nodes):
    A = assemble_stiffness_matrix(nodes)
    F = assemble_dirac_load(nodes)

    U, _, result_pdas = solve_obstacle_problem_pdas(
        nodes,
        A,
        F,
        obstacle_dirac,
    )

    l2_error, h1_error = compute_errors(
        nodes,
        U,
    )

    return {
        "nodes": nodes,
        "U": U,
        "l2_error": l2_error,
        "h1_error": h1_error,
        "pdas_iterations": result_pdas.nit,
    }


def collect_adaptive_data():
    history = run_adaptive_refinement()

    rows = []

    for snapshot in history:
        nodes = snapshot["nodes"]
        U = snapshot["U"]

        l2_error, h1_error = compute_errors(
            nodes,
            U,
        )

        n_elements = len(nodes) - 1

        rows.append({
            "strategy": "adaptive",
            "level": snapshot["level"],
            "elements": n_elements,
            "dofs": n_elements - 1,
            "h1_error": h1_error,
            "l2_error": l2_error,
        })

    return rows


def collect_uniform_data():
    rows = []

    for n_elements in UNIFORM_ELEMENT_COUNTS:
        nodes = create_mesh(n_elements)

        result = solve_on_mesh(nodes)

        rows.append({
            "strategy": "uniform",
            "level": "",
            "elements": n_elements,
            "dofs": n_elements - 1,
            "h1_error": result["h1_error"],
            "l2_error": result["l2_error"],
        })

    return rows


def print_data(title, rows):
    print()
    print(title)
    print(
        "elements  DOFs   "
        "H1 error       L2 error"
    )
    print("-" * 51)

    for row in rows:
        print(
            f"{row['elements']:<9d} "
            f"{row['dofs']:<6d} "
            f"{row['h1_error']:<14.6e} "
            f"{row['l2_error']:.6e}"
        )


def save_csv(
    adaptive_rows,
    uniform_rows,
    results_dir,
):
    path = (
        results_dir
        / "p1_dirac_adaptive_vs_uniform.csv"
    )

    fieldnames = [
        "strategy",
        "level",
        "elements",
        "dofs",
        "h1_error",
        "l2_error",
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

        for row in adaptive_rows + uniform_rows:
            writer.writerow(row)


def plot_comparison(
    adaptive_rows,
    uniform_rows,
    figures_dir,
):
    adaptive_dofs = np.array([
        row["dofs"]
        for row in adaptive_rows
    ])

    uniform_dofs = np.array([
        row["dofs"]
        for row in uniform_rows
    ])

    adaptive_h1 = np.array([
        row["h1_error"]
        for row in adaptive_rows
    ])

    uniform_h1 = np.array([
        row["h1_error"]
        for row in uniform_rows
    ])

    adaptive_l2 = np.array([
        row["l2_error"]
        for row in adaptive_rows
    ])

    uniform_l2 = np.array([
        row["l2_error"]
        for row in uniform_rows
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    axes[0].loglog(
        adaptive_dofs,
        adaptive_h1,
        "o-",
        label="adaptive",
    )

    axes[0].loglog(
        uniform_dofs,
        uniform_h1,
        "s--",
        label="uniform",
    )

    axes[0].set_xlabel(
        "interior P1 degrees of freedom"
    )

    axes[0].set_ylabel(
        r"$|u-u_h|_{H^1}$"
    )

    axes[0].set_title(
        r"$H^1$ seminorm error"
    )

    axes[0].grid(
        True,
        which="both",
        alpha=0.25,
    )

    axes[0].legend()

    axes[1].loglog(
        adaptive_dofs,
        adaptive_l2,
        "o-",
        label="adaptive",
    )

    axes[1].loglog(
        uniform_dofs,
        uniform_l2,
        "s--",
        label="uniform",
    )

    axes[1].set_xlabel(
        "interior P1 degrees of freedom"
    )

    axes[1].set_ylabel(
        r"$\|u-u_h\|_{L^2}$"
    )

    axes[1].set_title(
        r"$L^2$ error"
    )

    axes[1].grid(
        True,
        which="both",
        alpha=0.25,
    )

    axes[1].legend()

    fig.suptitle(
        "P1 Dirac problem: adaptive vs uniform refinement"
    )

    fig.savefig(
        figures_dir
        / "p1_dirac_adaptive_vs_uniform.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    figures_dir = ROOT / "figures"
    results_dir = ROOT / "results"

    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    adaptive_rows = collect_adaptive_data()
    uniform_rows = collect_uniform_data()

    print_data(
        "Adaptive refinement",
        adaptive_rows,
    )

    print_data(
        "Uniform refinement",
        uniform_rows,
    )

    save_csv(
        adaptive_rows,
        uniform_rows,
        results_dir,
    )

    plot_comparison(
        adaptive_rows,
        uniform_rows,
        figures_dir,
    )

    print()
    print("Saved:")
    print(
        "  figures/"
        "p1_dirac_adaptive_vs_uniform.png"
    )
    print(
        "  results/"
        "p1_dirac_adaptive_vs_uniform.csv"
    )


if __name__ == "__main__":
    main()