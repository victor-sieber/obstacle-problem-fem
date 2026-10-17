from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.adaptive import (
    bisect_marked_elements,
    mark_by_maximum,
)
from src.error_indicators import (
    compute_p1_jump_indicator,
)
from src.fem import (
    assemble_stiffness_matrix,
    create_mesh,
)
from src.solver import (
    solve_obstacle_problem_pdas,
)

from experiments.dirac_example import (
    ALPHA,
    BETA,
    assemble_dirac_load,
    obstacle_dirac,
)


INITIAL_ELEMENTS = 20
N_REFINEMENTS = 5
THETA = 0.5


def solve_on_mesh(nodes):
    """
    Solve the P1 Dirac obstacle problem on the given mesh
    and compute the local jump indicator.
    """

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

    return U, result_pdas, indicator


def run_adaptive_refinement():
    """
    Adaptive loop:

        solve
        -> compute eta_K
        -> mark large eta_K
        -> bisect marked elements
        -> repeat.
    """

    nodes = create_mesh(INITIAL_ELEMENTS)

    history = []

    for level in range(N_REFINEMENTS + 1):
        U, result_pdas, indicator = solve_on_mesh(
            nodes
        )

        eta = indicator["eta"]
        eta_squared = indicator["eta_squared"]

        snapshot = {
            "level": level,
            "nodes": nodes.copy(),
            "U": U.copy(),
            "indicator": indicator,
            "eta_global": np.sqrt(
                np.sum(eta_squared)
            ),
            "max_eta": np.max(eta),
            "pdas_iterations": result_pdas.nit,
            "marked": None,
        }

        history.append(snapshot)

        if level == N_REFINEMENTS:
            break

        marked = mark_by_maximum(
            eta,
            theta=THETA,
        )

        history[-1]["marked"] = marked.copy()

        nodes = bisect_marked_elements(
            nodes,
            marked,
        )

    return history


def print_history(history):
    print()
    print(
        "level  elements  marked  "
        "min(h)       max(h)       "
        "max(eta_K)   eta_h        PDAS it"
    )
    print("-" * 91)

    for snapshot in history:
        nodes = snapshot["nodes"]
        h = np.diff(nodes)

        marked = snapshot["marked"]

        if marked is None:
            n_marked = "-"
        else:
            n_marked = str(
                int(np.sum(marked))
            )

        print(
            f"{snapshot['level']:<6d} "
            f"{len(nodes) - 1:<9d} "
            f"{n_marked:<7s} "
            f"{np.min(h):<12.4e} "
            f"{np.max(h):<12.4e} "
            f"{snapshot['max_eta']:<12.4e} "
            f"{snapshot['eta_global']:<12.4e} "
            f"{snapshot['pdas_iterations']}"
        )


def add_dirac_lines(ax):
    ax.axvline(
        ALPHA,
        linestyle="--",
        linewidth=1.1,
        label=r"$\alpha=1/3$",
    )

    ax.axvline(
        BETA,
        linestyle=":",
        linewidth=1.3,
        label=r"$\beta=2/3$",
    )


def plot_adaptive_history(
    history,
    figures_dir,
):
    selected_levels = [
        0,
        len(history) // 2,
        len(history) - 1,
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.2),
        constrained_layout=True,
    )

    for ax, level in zip(
        axes,
        selected_levels,
    ):
        snapshot = history[level]

        indicator = snapshot["indicator"]
        midpoints = indicator["midpoints"]
        eta = indicator["eta"]

        ax.plot(
            midpoints,
            eta,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )

        marked = snapshot["marked"]

        if marked is not None:
            ax.scatter(
                midpoints[marked],
                eta[marked],
                marker="x",
                s=45,
                linewidths=1.5,
                label="marked",
            )

        add_dirac_lines(ax)

        ax.set_title(
            rf"level {level}, "
            rf"{len(snapshot['nodes']) - 1} elements"
        )

        ax.set_xlabel(
            r"element midpoint $x_K$"
        )

        ax.set_ylabel(
            r"$\eta_K$"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

    axes[-1].legend()

    fig.suptitle(
        rf"Adaptive P1 refinement, "
        rf"$\theta={THETA}$"
    )

    fig.savefig(
        figures_dir
        / "p1_dirac_adaptive_history.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_adaptive_mesh(
    history,
    figures_dir,
):
    final = history[-1]

    nodes = final["nodes"]
    indicator = final["indicator"]

    midpoints = indicator["midpoints"]
    eta = indicator["eta"]
    h = indicator["h"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(
        midpoints,
        eta,
        marker="o",
        markersize=3.5,
        linewidth=1.2,
    )

    add_dirac_lines(axes[0])

    axes[0].set_ylabel(
        r"$\eta_K$"
    )

    axes[0].set_title(
        "Final local jump indicators"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[0].legend()

    axes[1].step(
        midpoints,
        h,
        where="mid",
        linewidth=1.4,
    )

    add_dirac_lines(axes[1])

    axes[1].set_xlabel(
        r"element midpoint $x_K$"
    )

    axes[1].set_ylabel(
        r"element size $h_K$"
    )

    axes[1].set_title(
        "Final adaptive mesh sizes"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    fig.suptitle(
        rf"P1 adaptive Dirac problem after "
        rf"{N_REFINEMENTS} refinement steps"
    )

    fig.savefig(
        figures_dir
        / "p1_dirac_adaptive_mesh.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    history = run_adaptive_refinement()

    print_history(history)

    plot_adaptive_history(
        history,
        figures_dir,
    )

    plot_final_adaptive_mesh(
        history,
        figures_dir,
    )

    print()
    print("Saved:")
    print(
        "  figures/"
        "p1_dirac_adaptive_history.png"
    )
    print(
        "  figures/"
        "p1_dirac_adaptive_mesh.png"
    )


if __name__ == "__main__":
    main()