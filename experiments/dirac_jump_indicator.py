from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.error_indicators import compute_p1_jump_indicator
from src.fem import (
    assemble_stiffness_matrix,
    create_mesh,
)
from src.solver import solve_obstacle_problem_pdas

from experiments.dirac_example import (
    ALPHA,
    BETA,
    assemble_dirac_load,
    obstacle_dirac,
)


ELEMENT_COUNTS = [20, 40, 80, 160]

# Main presentation-style indicator plot.
PLOT_ELEMENT_COUNT = 80


def solve_and_compute_indicator(n_elements):
    nodes = create_mesh(n_elements)

    A = assemble_stiffness_matrix(nodes)

    F = assemble_dirac_load(nodes)

    U, _, result_pdas = solve_obstacle_problem_pdas(
        nodes,
        A,
        F,
        obstacle_dirac,
    )

    indicator = compute_p1_jump_indicator(
        nodes,
        U,
    )

    return nodes, U, result_pdas, indicator


def print_largest_indicators(
    nodes,
    indicator,
    n_show=8,
):
    eta = indicator["eta"]
    midpoints = indicator["midpoints"]

    order = np.argsort(eta)[::-1]

    print(
        "rank  element        interval"
        "                 midpoint      eta_K"
        "         dist. to nearest Dirac"
    )

    print("-" * 92)

    for rank, element in enumerate(
        order[:n_show],
        start=1,
    ):
        left = nodes[element]
        right = nodes[element + 1]
        midpoint = midpoints[element]

        distance = min(
            abs(midpoint - ALPHA),
            abs(midpoint - BETA),
        )

        print(
            f"{rank:<5d} "
            f"{element:<14d} "
            f"[{left:.6f}, {right:.6f}]   "
            f"{midpoint:.6f}      "
            f"{eta[element]:.6e}   "
            f"{distance:.6e}"
        )


def plot_single_indicator(
    nodes,
    indicator,
    n_elements,
    figures_dir,
):
    midpoints = indicator["midpoints"]
    eta = indicator["eta"]

    fig, ax = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    ax.plot(
        midpoints,
        eta,
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        label=r"local indicator $\eta_K$",
    )

    ax.axvline(
        ALPHA,
        linestyle="--",
        linewidth=1.2,
        label=r"$\alpha=1/3$",
    )

    ax.axvline(
        BETA,
        linestyle=":",
        linewidth=1.4,
        label=r"$\beta=2/3$",
    )

    ax.set_xlabel(r"element midpoint $x_K$")
    ax.set_ylabel(r"$\eta_K$")

    ax.set_title(
        rf"P1 jump indicator for Dirac problem, $M={n_elements}$"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.savefig(
        figures_dir
        / "p1_dirac_jump_indicator.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_indicator_mesh_comparison(
    results,
    figures_dir,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, (
        n_elements,
        (_, _, _, indicator),
    ) in zip(
        axes,
        results.items(),
    ):
        midpoints = indicator["midpoints"]
        eta = indicator["eta"]

        ax.plot(
            midpoints,
            eta,
            marker="o",
            markersize=2.5,
            linewidth=1.0,
        )

        ax.axvline(
            ALPHA,
            linestyle="--",
            linewidth=1.0,
        )

        ax.axvline(
            BETA,
            linestyle=":",
            linewidth=1.2,
        )

        ax.set_title(
            rf"$M={n_elements}$"
        )

        ax.set_ylabel(r"$\eta_K$")

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[-2].set_xlabel(r"element midpoint $x_K$")
    axes[-1].set_xlabel(r"element midpoint $x_K$")

    fig.suptitle(
        "Localization of the P1 jump indicator under uniform refinement"
    )

    fig.savefig(
        figures_dir
        / "p1_dirac_jump_indicator_refinement.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    results = {}

    for n_elements in ELEMENT_COUNTS:
        results[n_elements] = (
            solve_and_compute_indicator(
                n_elements
            )
        )

    (
        nodes,
        _,
        result_pdas,
        indicator,
    ) = results[PLOT_ELEMENT_COUNT]

    print()
    print(
        f"Dirac jump indicator, "
        f"M={PLOT_ELEMENT_COUNT}"
    )

    print(
        f"PDAS iterations: "
        f"{result_pdas.nit}"
    )

    print()

    print_largest_indicators(
        nodes,
        indicator,
    )

    plot_single_indicator(
        nodes,
        indicator,
        PLOT_ELEMENT_COUNT,
        figures_dir,
    )

    plot_indicator_mesh_comparison(
        results,
        figures_dir,
    )

    print()
    print("Saved:")
    print(
        "  figures/"
        "p1_dirac_jump_indicator.png"
    )
    print(
        "  figures/"
        "p1_dirac_jump_indicator_refinement.png"
    )


if __name__ == "__main__":
    main()