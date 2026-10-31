
from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.dirac_p2_residual_adaptive_refinement import (
    run_adaptive_refinement,
)

from experiments.p1_p2_dirac_comparison import (
    assemble_dirac_load_p2,
    compute_errors,
    obstacle_dirac,
)

from src.fem import create_mesh

from src.fem_p2 import (
    assemble_stiffness_matrix,
    create_p2_nodes,
    evaluate_fem_derivative,
    evaluate_fem_solution,
)

from src.solver import solve_obstacle_problem_pdas


# ============================================================
# Configuration
# ============================================================

UNIFORM_ELEMENT_COUNTS = [20, 40, 80, 160, 320]

# Adaptive tolerance is configured separately in:
# experiments/dirac_p2_residual_adaptive_refinement.py
#
# Recommended value: 1e-4


# ============================================================
# Uniform refinement
# ============================================================

def solve_uniform_mesh(n_elements):
    nodes = create_mesh(n_elements)
    p2_nodes = create_p2_nodes(nodes)

    A = assemble_stiffness_matrix(nodes)
    F = assemble_dirac_load_p2(nodes)

    U, _, result_pdas = solve_obstacle_problem_pdas(
        p2_nodes,
        A,
        F,
        obstacle_dirac,
        maxiter=2000,
    )

    l2_error, h1_error = compute_errors(
        nodes,
        U,
        evaluate_fem_solution,
        evaluate_fem_derivative,
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
        n_elements = len(snapshot["nodes"]) - 1

        rows.append({
            "strategy": "adaptive",
            "level": snapshot["level"],
            "elements": n_elements,
            "dofs": 2 * n_elements - 1,
            "eta_h": snapshot["eta_global"],
            "h1_error": snapshot["h1_error"],
            "l2_error": snapshot["l2_error"],
        })

    return rows


def collect_uniform_data():
    rows = []

    for n_elements in UNIFORM_ELEMENT_COUNTS:
        print(f"Uniform P2 solve: {n_elements} elements")

        result = solve_uniform_mesh(n_elements)

        rows.append({
            "strategy": "uniform",
            "level": "",
            "elements": n_elements,
            "dofs": 2 * n_elements - 1,
            "eta_h": "",
            "h1_error": result["h1_error"],
            "l2_error": result["l2_error"],
        })

    return rows


# ============================================================
# Output
# ============================================================

def print_data(title, rows):
    print()
    print(title)
    print("elements  DOFs   H1 error       L2 error")
    print("-" * 51)

    for row in rows:
        print(
            f"{row['elements']:<9d} "
            f"{row['dofs']:<6d} "
            f"{row['h1_error']:<14.6e} "
            f"{row['l2_error']:.6e}"
        )


def save_csv(adaptive_rows, uniform_rows, results_dir):
    path = (
        results_dir
        / "p2_dirac_residual_adaptive_vs_uniform.csv"
    )

    fieldnames = [
        "strategy",
        "level",
        "elements",
        "dofs",
        "eta_h",
        "h1_error",
        "l2_error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(adaptive_rows + uniform_rows)


# ============================================================
# Reference lines
# ============================================================

def reference_line(x_values, rate, x_anchor, y_anchor):
    x_values = np.asarray(x_values, dtype=float)

    return y_anchor * (x_values / x_anchor) ** (-rate)


def add_reference_segment(
    ax,
    x_start,
    x_end,
    rate,
    x_anchor,
    y_anchor,
    label,
    offset=1.4,
):
    """
    Draw a theoretical convergence-rate reference line.

    The line has slope -rate in a log-log plot.
    It is anchored close to the uniform-refinement data.
    """

    x_reference = np.array([x_start, x_end], dtype=float)

    y_reference = reference_line(
        x_reference,
        rate=rate,
        x_anchor=x_anchor,
        y_anchor=y_anchor * offset,
    )

    ax.loglog(
        x_reference,
        y_reference,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label=f"reference {label}",
    )


# ============================================================
# Observed convergence rates
# ============================================================

def compute_observed_rates(uniform_rows):
    dofs = np.array(
        [row["dofs"] for row in uniform_rows],
        dtype=float,
    )

    h1 = np.array(
        [row["h1_error"] for row in uniform_rows],
        dtype=float,
    )

    l2 = np.array(
        [row["l2_error"] for row in uniform_rows],
        dtype=float,
    )

    # Fit log(error) = log(C) - p * log(DOFs).
    #
    # The fitted exponent p is the observed convergence rate.

    p_h1 = -np.polyfit(np.log(dofs), np.log(h1), 1)[0]
    p_l2 = -np.polyfit(np.log(dofs), np.log(l2), 1)[0]

    print()
    print("Observed uniform convergence rates")
    print("-" * 42)
    print(f"H1 seminorm: {p_h1:.4f} (reference: 0.5)")
    print(f"L2 norm:     {p_l2:.4f} (reference: 1.5)")

    return p_h1, p_l2


# ============================================================
# Plotting
# ============================================================

def plot_comparison(adaptive_rows, uniform_rows, figures_dir):
    adaptive_dofs = np.array(
        [row["dofs"] for row in adaptive_rows]
    )

    uniform_dofs = np.array(
        [row["dofs"] for row in uniform_rows]
    )

    adaptive_h1 = np.array(
        [row["h1_error"] for row in adaptive_rows]
    )

    uniform_h1 = np.array(
        [row["h1_error"] for row in uniform_rows]
    )

    adaptive_l2 = np.array(
        [row["l2_error"] for row in adaptive_rows]
    )

    uniform_l2 = np.array(
        [row["l2_error"] for row in uniform_rows]
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )

    # --------------------------------------------------------
    # H1 seminorm error
    # --------------------------------------------------------

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

    # Uniform Dirac-limited convergence: N^(-1/2)

    uniform_anchor = 2

    add_reference_segment(
        axes[0],
        x_start=110.0,
        x_end=600.0,
        rate=0.5,
        x_anchor=uniform_dofs[uniform_anchor],
        y_anchor=uniform_h1[uniform_anchor],
        label=r"$N^{-1/2}$",
        offset=1.4,
    )

    axes[0].set_xlabel(
        "interior P2 degrees of freedom"
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

    axes[0].legend(fontsize=9)

    # --------------------------------------------------------
    # L2 error
    # --------------------------------------------------------

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

    # Uniform Dirac-limited convergence: N^(-3/2)

    add_reference_segment(
        axes[1],
        x_start=110.0,
        x_end=600.0,
        rate=1.5,
        x_anchor=uniform_dofs[uniform_anchor],
        y_anchor=uniform_l2[uniform_anchor],
        label=r"$N^{-3/2}$",
        offset=1.4,
    )

    axes[1].set_xlabel(
        "interior P2 degrees of freedom"
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

    axes[1].legend(fontsize=9)

    # --------------------------------------------------------
    # X-axis formatting
    # --------------------------------------------------------

    # Use readable integer labels instead of scientific notation.
    # Preserve logarithmic scaling.

    x_ticks = [40, 100, 200, 400, 600]

    all_dofs = np.concatenate(
        [adaptive_dofs, uniform_dofs]
    )

    x_min = min(35.0, float(np.min(all_dofs)) * 0.85)
    x_max = max(700.0, float(np.max(all_dofs)) * 1.10)

    for ax in axes:
        ax.set_xscale("log")

        ax.xaxis.set_major_locator(
            FixedLocator(x_ticks)
        )

        ax.xaxis.set_major_formatter(
            ScalarFormatter()
        )

        ax.xaxis.set_minor_formatter(
            NullFormatter()
        )

        ax.set_xlim(x_min, x_max)

        ax.tick_params(
            axis="x",
            labelsize=9,
        )

    # --------------------------------------------------------
    # Figure title and save
    # --------------------------------------------------------

    fig.suptitle(
        "P2 Dirac problem: "
        "residual-based adaptive vs uniform refinement"
    )

    fig.savefig(
        figures_dir
        / "p2_dirac_residual_adaptive_vs_uniform.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    figures_dir = ROOT / "figures"
    results_dir = ROOT / "results"

    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    adaptive_rows = collect_adaptive_data()
    uniform_rows = collect_uniform_data()

    print_data(
        "Residual-based adaptive P2 refinement",
        adaptive_rows,
    )

    print_data(
        "Uniform P2 refinement",
        uniform_rows,
    )

    compute_observed_rates(uniform_rows)

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
        "p2_dirac_residual_adaptive_vs_uniform.png"
    )
    print(
        "  results/"
        "p2_dirac_residual_adaptive_vs_uniform.csv"
    )


if __name__ == "__main__":
    main()
