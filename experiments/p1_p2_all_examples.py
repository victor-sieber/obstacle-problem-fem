from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    FixedFormatter,
    FixedLocator,
    NullLocator,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from experiments.p1_p2_comparison import (
    ALPHA_ALIGNED,
    ALPHA_NONALIGNED,
    compute_errors as compute_errors_smooth,
    solve_p1 as solve_p1_smooth,
    solve_p2 as solve_p2_smooth,
)

from experiments.p1_p2_dirac_comparison import (
    compute_errors as compute_errors_dirac,
    solve_p1 as solve_p1_dirac,
    solve_p2 as solve_p2_dirac,
)

from src.fem import (
    create_mesh,
    evaluate_fem_derivative as evaluate_fem_derivative_p1,
    evaluate_fem_solution as evaluate_fem_solution_p1,
)

from src.fem_p2 import (
    evaluate_fem_derivative as evaluate_fem_derivative_p2,
    evaluate_fem_solution as evaluate_fem_solution_p2,
)


# Use the same refinement sequence in the consolidated comparison
# so that all three examples are visually comparable.
ALIGNED_COUNTS = np.array([
    20,
    40,
    80,
    160,
    320,
])

NONALIGNED_COUNTS = np.array([
    20,
    40,
    80,
    160,
    320,
])

DIRAC_COUNTS = np.array([
    20,
    40,
    80,
    160,
    320,
])


def smooth_errors(
    alpha,
    counts,
):
    mesh_sizes = 1.0 / counts

    p1_l2 = []
    p1_h1 = []

    p2_l2 = []
    p2_h1 = []

    for n_elements in counts:
        nodes = create_mesh(
            int(n_elements)
        )

        U_p1, _, _ = (
            solve_p1_smooth(
                nodes,
                alpha,
            )
        )

        U_p2, _, _, _, _, _ = (
            solve_p2_smooth(
                nodes,
                alpha,
            )
        )

        l2, h1 = compute_errors_smooth(
            nodes,
            U_p1,
            alpha,
            evaluate_fem_solution_p1,
            evaluate_fem_derivative_p1,
        )

        p1_l2.append(l2)
        p1_h1.append(h1)

        l2, h1 = compute_errors_smooth(
            nodes,
            U_p2,
            alpha,
            evaluate_fem_solution_p2,
            evaluate_fem_derivative_p2,
        )

        p2_l2.append(l2)
        p2_h1.append(h1)

    return (
        mesh_sizes,
        np.array(p1_l2),
        np.array(p1_h1),
        np.array(p2_l2),
        np.array(p2_h1),
    )


def dirac_errors(
    counts,
):
    mesh_sizes = 1.0 / counts

    p1_l2 = []
    p1_h1 = []

    p2_l2 = []
    p2_h1 = []

    for n_elements in counts:
        nodes = create_mesh(
            int(n_elements)
        )

        U_p1, _, _, _, _ = (
            solve_p1_dirac(
                nodes
            )
        )

        U_p2, _, _, _, _, _ = (
            solve_p2_dirac(
                nodes
            )
        )

        l2, h1 = compute_errors_dirac(
            nodes,
            U_p1,
            evaluate_fem_solution_p1,
            evaluate_fem_derivative_p1,
        )

        p1_l2.append(l2)
        p1_h1.append(h1)

        l2, h1 = compute_errors_dirac(
            nodes,
            U_p2,
            evaluate_fem_solution_p2,
            evaluate_fem_derivative_p2,
        )

        p2_l2.append(l2)
        p2_h1.append(h1)

    return (
        mesh_sizes,
        np.array(p1_l2),
        np.array(p1_h1),
        np.array(p2_l2),
        np.array(p2_h1),
    )


def format_mesh_axis(
    ax,
    counts,
):
    """
    Show exactly the mesh sizes used:
        1/20, 1/40, 1/80, 1/160, 1/320.

    Remove minor log ticks so the axis remains readable.
    """

    tick_values = (
        1.0
        / np.asarray(
            counts,
            dtype=float,
        )
    )

    tick_labels = [
        rf"$1/{int(n)}$"
        for n in counts
    ]

    ax.xaxis.set_major_locator(
        FixedLocator(
            tick_values
        )
    )

    ax.xaxis.set_major_formatter(
        FixedFormatter(
            tick_labels
        )
    )

    ax.xaxis.set_minor_locator(
        NullLocator()
    )

    ax.tick_params(
        axis="x",
        labelsize=9,
    )


def add_reference(
    ax,
    mesh_sizes,
    base_values,
    order,
    label,
    linestyle="--",
):
    reference = (
        base_values[0]
        * (
            mesh_sizes
            / mesh_sizes[0]
        )
        ** order
    )

    ax.loglog(
        mesh_sizes,
        reference,
        linestyle,
        label=label,
    )


def add_panel(
    ax,
    mesh_sizes,
    p1_values,
    p2_values,
    counts,
    title,
    ylabel,
):
    ax.loglog(
        mesh_sizes,
        p1_values,
        "o-",
        label="P1",
    )

    ax.loglog(
        mesh_sizes,
        p2_values,
        "s-",
        label="P2",
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        r"Mesh size $h$"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        True,
        which="major",
        alpha=0.6,
    )

    # Coarse mesh on the left,
    # fine mesh on the right.
    ax.invert_xaxis()

    format_mesh_axis(
        ax,
        counts,
    )


def main():
    aligned = smooth_errors(
        ALPHA_ALIGNED,
        ALIGNED_COUNTS,
    )

    nonaligned = smooth_errors(
        ALPHA_NONALIGNED,
        NONALIGNED_COUNTS,
    )

    dirac = dirac_errors(
        DIRAC_COUNTS,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 8.5),
    )

    # =================================================
    # 1. ALIGNED SMOOTH PROBLEM
    # alpha = 0.2
    # =================================================

    add_panel(
        axes[0, 0],
        aligned[0],
        aligned[1],
        aligned[3],
        ALIGNED_COUNTS,
        (
            r"Aligned smooth, "
            r"$\alpha=0.2$: $L^2$"
        ),
        r"$L^2$ error",
    )

    add_reference(
        axes[0, 0],
        aligned[0],
        aligned[1],
        order=2.0,
        label=r"$O(h^2)$ reference",
    )

    format_mesh_axis(
        axes[0, 0],
        ALIGNED_COUNTS,
    )

    axes[0, 0].legend(
        fontsize=9
    )

    add_panel(
        axes[1, 0],
        aligned[0],
        aligned[2],
        aligned[4],
        ALIGNED_COUNTS,
        (
            r"Aligned smooth, "
            r"$\alpha=0.2$: $H^1$"
        ),
        r"$H^1$ seminorm error",
    )

    add_reference(
        axes[1, 0],
        aligned[0],
        aligned[2],
        order=1.0,
        label=r"$O(h)$ reference",
    )

    format_mesh_axis(
        axes[1, 0],
        ALIGNED_COUNTS,
    )

    axes[1, 0].legend(
        fontsize=9
    )

    # =================================================
    # 2. NONALIGNED SMOOTH PROBLEM
    # alpha = 1/3
    #
    # Here the free boundary never aligns with the
    # dyadic mesh. Under factor-two refinement its
    # relative position inside the cut element alternates.
    # =================================================

    add_panel(
        axes[0, 1],
        nonaligned[0],
        nonaligned[1],
        nonaligned[3],
        NONALIGNED_COUNTS,
        (
            r"Nonaligned smooth, "
            r"$\alpha=1/3$: $L^2$"
        ),
        r"$L^2$ error",
    )

    add_reference(
        axes[0, 1],
        nonaligned[0],
        nonaligned[1],
        order=2.0,
        label=r"$O(h^2)$ reference",
    )

    format_mesh_axis(
        axes[0, 1],
        NONALIGNED_COUNTS,
    )

    axes[0, 1].legend(
        fontsize=9
    )

    add_panel(
        axes[1, 1],
        nonaligned[0],
        nonaligned[2],
        nonaligned[4],
        NONALIGNED_COUNTS,
        (
            r"Nonaligned smooth, "
            r"$\alpha=1/3$: $H^1$"
        ),
        r"$H^1$ seminorm error",
    )

    add_reference(
        axes[1, 1],
        nonaligned[0],
        nonaligned[2],
        order=1.0,
        label=r"$O(h)$ reference",
        linestyle="--",
    )

    add_reference(
        axes[1, 1],
        nonaligned[0],
        nonaligned[4],
        order=1.5,
        label=r"$O(h^{3/2})$ reference",
        linestyle=":",
    )

    format_mesh_axis(
        axes[1, 1],
        NONALIGNED_COUNTS,
    )

    axes[1, 1].legend(
        fontsize=9
    )

    # =================================================
    # 3. DIRAC PROBLEM
    # =================================================

    add_panel(
        axes[0, 2],
        dirac[0],
        dirac[1],
        dirac[3],
        DIRAC_COUNTS,
        r"Dirac load: $L^2$",
        r"$L^2$ error",
    )

    add_reference(
        axes[0, 2],
        dirac[0],
        dirac[1],
        order=1.5,
        label=r"$O(h^{3/2})$ reference",
    )

    format_mesh_axis(
        axes[0, 2],
        DIRAC_COUNTS,
    )

    axes[0, 2].legend(
        fontsize=9
    )

    add_panel(
        axes[1, 2],
        dirac[0],
        dirac[2],
        dirac[4],
        DIRAC_COUNTS,
        r"Dirac load: $H^1$",
        r"$H^1$ seminorm error",
    )

    add_reference(
        axes[1, 2],
        dirac[0],
        dirac[2],
        order=0.5,
        label=r"$O(h^{1/2})$ reference",
    )

    format_mesh_axis(
        axes[1, 2],
        DIRAC_COUNTS,
    )

    axes[1, 2].legend(
        fontsize=9
    )

    # =================================================
    # Final layout
    # =================================================

    fig.suptitle(
        (
            "P1 vs P2 across all "
            "obstacle benchmarks"
        ),
        fontsize=18,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.09,
        top=0.90,
        wspace=0.30,
        hspace=0.32,
    )

    figures_dir = (
        ROOT / "figures"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    output_path = (
        figures_dir
        / "p1_p2_all_examples_convergence.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(
        "  figures/"
        "p1_p2_all_examples_convergence.png"
    )


if __name__ == "__main__":
    main()