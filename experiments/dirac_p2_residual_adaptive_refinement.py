from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from experiments.p1_p2_dirac_comparison import (
    ALPHA,
    BETA,
    assemble_dirac_load_p2,
    compute_errors,
    obstacle_dirac,
    smooth_load,
)

from src.adaptive import (
    bisect_marked_elements,
    mark_by_maximum,
)

from src.error_indicators import (
    compute_p2_residual_indicator,
)

from src.fem import (
    create_mesh,
)

from src.fem_p2 import (
    assemble_stiffness_matrix,
    create_p2_nodes,
    evaluate_fem_derivative,
    evaluate_fem_solution,
)

from src.solver import (
    solve_obstacle_problem_pdas,
)


INITIAL_ELEMENTS = 20

THETA = 0.5

ETA_TOL = 1.0e-3

MAX_INTERIOR_DOFS = 5000
MAX_REFINEMENTS = 30


def solve_on_mesh(
    nodes,
):
    """
    Solve the P2 Dirac obstacle problem and compute

        eta_K^2
        =
        h_K^2 ||f_reg + lambda_h + u_h''||_L2(K)^2
        +
        h_K/2 (J_left^2 + J_right^2).
    """
    p2_nodes = create_p2_nodes(
        nodes
    )

    A = assemble_stiffness_matrix(
        nodes
    )

    F = assemble_dirac_load_p2(
        nodes
    )

    U, _, result_pdas = solve_obstacle_problem_pdas(
        p2_nodes,
        A,
        F,
        obstacle_dirac,
        maxiter=2000,
    )

    indicator = compute_p2_residual_indicator(
        nodes,
        U,
        A,
        F,
        smooth_load,
    )

    l2_error, h1_error = compute_errors(
        nodes,
        U,
        evaluate_fem_solution,
        evaluate_fem_derivative,
    )

    return {
        "A": A,
        "F": F,
        "U": U,
        "pdas": result_pdas,
        "indicator": indicator,
        "l2_error": l2_error,
        "h1_error": h1_error,
    }


def run_adaptive_refinement():
    nodes = create_mesh(
        INITIAL_ELEMENTS
    )

    history = []

    for level in range(
        MAX_REFINEMENTS + 1
    ):
        result = solve_on_mesh(
            nodes
        )

        indicator = result[
            "indicator"
        ]

        eta = indicator[
            "eta"
        ]

        eta_global = float(
            np.sqrt(
                np.sum(
                    indicator[
                        "eta_squared"
                    ]
                )
            )
        )

        n_elements = (
            len(nodes) - 1
        )

        n_dofs = (
            2 * n_elements - 1
        )

        snapshot = {
            "level": level,

            "nodes": nodes.copy(),

            "U": result[
                "U"
            ].copy(),

            "indicator": indicator,

            "eta_global": eta_global,

            "max_eta": float(
                np.max(
                    eta
                )
            ),

            "l2_error": result[
                "l2_error"
            ],

            "h1_error": result[
                "h1_error"
            ],

            "pdas_iterations": result[
                "pdas"
            ].nit,

            "marked": None,

            "stop_reason": None,
        }

        history.append(
            snapshot
        )

        if eta_global <= ETA_TOL:

            history[-1][
                "stop_reason"
            ] = "indicator tolerance"

            break

        if n_dofs >= MAX_INTERIOR_DOFS:

            history[-1][
                "stop_reason"
            ] = "DOF safety cap"

            break

        if level == MAX_REFINEMENTS:

            history[-1][
                "stop_reason"
            ] = "refinement safety cap"

            break

        marked = mark_by_maximum(
            eta,
            theta=THETA,
        )

        if not np.any(
            marked
        ):

            history[-1][
                "stop_reason"
            ] = "no marked elements"

            break

        history[-1][
            "marked"
        ] = marked.copy()

        nodes = bisect_marked_elements(
            nodes,
            marked,
        )

    return history


def print_history(
    history,
):
    print()

    print(
        "level  elements  P2 DOFs  marked  "
        "min(h)       max(h)       "
        "max(eta_K)   eta_h        "
        "H1 error     L2 error     PDAS"
    )

    print(
        "-" * 132
    )

    for snapshot in history:

        nodes = snapshot[
            "nodes"
        ]

        h = np.diff(
            nodes
        )

        n_elements = (
            len(nodes) - 1
        )

        n_dofs = (
            2 * n_elements - 1
        )

        marked = snapshot[
            "marked"
        ]

        if marked is None:
            n_marked = "-"
        else:
            n_marked = str(
                int(
                    np.sum(
                        marked
                    )
                )
            )

        print(
            f"{snapshot['level']:<6d} "
            f"{n_elements:<9d} "
            f"{n_dofs:<8d} "
            f"{n_marked:<7s} "
            f"{np.min(h):<12.4e} "
            f"{np.max(h):<12.4e} "
            f"{snapshot['max_eta']:<12.4e} "
            f"{snapshot['eta_global']:<12.4e} "
            f"{snapshot['h1_error']:<12.4e} "
            f"{snapshot['l2_error']:<12.4e} "
            f"{snapshot['pdas_iterations']}"
        )

    print()

    print(
        "Stop reason:",
        history[-1][
            "stop_reason"
        ],
    )


def add_dirac_lines(
    ax,
):
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

    selected_levels = list(
        dict.fromkeys(
            selected_levels
        )
    )

    fig, axes = plt.subplots(
        1,
        len(selected_levels),
        figsize=(
            4.7
            * len(selected_levels),
            4.2,
        ),
        constrained_layout=True,
    )

    if len(
        selected_levels
    ) == 1:
        axes = [
            axes
        ]

    for ax, level in zip(
        axes,
        selected_levels,
    ):
        snapshot = history[
            level
        ]

        indicator = snapshot[
            "indicator"
        ]

        midpoints = indicator[
            "midpoints"
        ]

        eta = indicator[
            "eta"
        ]

        ax.plot(
            midpoints,
            eta,
            marker="o",
            markersize=3,
            linewidth=1.2,
        )

        marked = snapshot[
            "marked"
        ]

        if marked is not None:

            ax.scatter(
                midpoints[
                    marked
                ],
                eta[
                    marked
                ],
                marker="x",
                s=45,
                linewidths=1.5,
                label="marked",
            )

        add_dirac_lines(
            ax
        )

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
        rf"P2 residual-based adaptive refinement, "
        rf"$\theta={THETA}$"
    )

    fig.savefig(
        figures_dir
        / "p2_dirac_residual_adaptive_history.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_final_adaptive_mesh(
    history,
    figures_dir,
):
    final = history[
        -1
    ]

    indicator = final[
        "indicator"
    ]

    midpoints = indicator[
        "midpoints"
    ]

    eta = indicator[
        "eta"
    ]

    h = indicator[
        "h"
    ]

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

    add_dirac_lines(
        axes[0]
    )

    axes[0].set_ylabel(
        r"$\eta_K$"
    )

    axes[0].set_title(
        "Final local residual indicators"
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

    add_dirac_lines(
        axes[1]
    )

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

    final_level = final[
        "level"
    ]

    final_eta = final[
        "eta_global"
    ]

    fig.suptitle(
        rf"P2 adaptive Dirac problem: "
        rf"level {final_level}, "
        rf"$\eta_h={final_eta:.2e}$"
    )

    fig.savefig(
        figures_dir
        / "p2_dirac_residual_adaptive_mesh.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def main():
    figures_dir = (
        ROOT / "figures"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    print(
        "Indicator tolerance:",
        ETA_TOL,
    )

    history = (
        run_adaptive_refinement()
    )

    print_history(
        history
    )

    plot_adaptive_history(
        history,
        figures_dir,
    )

    plot_final_adaptive_mesh(
        history,
        figures_dir,
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  figures/"
        "p2_dirac_residual_adaptive_history.png"
    )

    print(
        "  figures/"
        "p2_dirac_residual_adaptive_mesh.png"
    )


if __name__ == "__main__":
    main()  