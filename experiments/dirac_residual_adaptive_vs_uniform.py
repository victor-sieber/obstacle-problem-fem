from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from experiments.dirac_residual_adaptive_refinement import (
    INITIAL_ELEMENTS,
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

from src.solver import (
    solve_obstacle_problem_pdas,
)


# ============================================================
# Uniform P1 solve
# ============================================================

def solve_uniform_mesh(
    n_elements,
):
    """
    Solve the Dirac obstacle problem on a uniform P1 mesh.
    """
    nodes = create_mesh(
        n_elements
    )

    A = assemble_stiffness_matrix(
        nodes
    )

    F = assemble_dirac_load(
        nodes
    )

    U, _, result_pdas = solve_obstacle_problem_pdas(
        nodes,
        A,
        F,
        obstacle_dirac,
        maxiter=1000,
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


# ============================================================
# Collect adaptive data
# ============================================================

def collect_adaptive_data():
    """
    Run the residual-based adaptive algorithm and extract
    the quantities required for the comparison plot.
    """
    history = run_adaptive_refinement()

    rows = []

    for snapshot in history:

        nodes = snapshot[
            "nodes"
        ]

        n_elements = (
            len(nodes) - 1
        )

        rows.append({
            "strategy": "adaptive",

            "level": snapshot[
                "level"
            ],

            "elements": n_elements,

            "dofs": (
                n_elements - 1
            ),

            "eta_h": snapshot[
                "eta_global"
            ],

            "h1_error": snapshot[
                "h1_error"
            ],

            "l2_error": snapshot[
                "l2_error"
            ],
        })

    return rows


# ============================================================
# Uniform mesh sizes
# ============================================================

def choose_uniform_element_counts(
    target_elements,
):
    """
    Use dyadically refined uniform meshes

        20, 40, 80, 160, ...

    until the uniform mesh reaches approximately the
    size of the final adaptive mesh.
    """
    counts = []

    n_elements = INITIAL_ELEMENTS

    while n_elements < target_elements:

        counts.append(
            n_elements
        )

        n_elements *= 2

    counts.append(
        n_elements
    )

    # Include one additional level only when it remains
    # reasonably small for the current dense implementation.
    if n_elements < 640:

        counts.append(
            2 * n_elements
        )

    return counts


# ============================================================
# Collect uniform data
# ============================================================

def collect_uniform_data(
    element_counts,
):
    rows = []

    for n_elements in element_counts:

        print(
            f"Uniform solve: "
            f"{n_elements} elements"
        )

        result = solve_uniform_mesh(
            n_elements
        )

        rows.append({
            "strategy": "uniform",

            "level": "",

            "elements": n_elements,

            "dofs": (
                n_elements - 1
            ),

            "eta_h": "",

            "h1_error": result[
                "h1_error"
            ],

            "l2_error": result[
                "l2_error"
            ],
        })

    return rows


# ============================================================
# Console output
# ============================================================

def print_data(
    title,
    rows,
):
    print()

    print(
        title
    )

    print(
        "elements  DOFs   "
        "H1 error       L2 error"
    )

    print(
        "-" * 51
    )

    for row in rows:

        print(
            f"{row['elements']:<9d} "
            f"{row['dofs']:<6d} "
            f"{row['h1_error']:<14.6e} "
            f"{row['l2_error']:.6e}"
        )


# ============================================================
# Save numerical results
# ============================================================

def save_csv(
    adaptive_rows,
    uniform_rows,
    results_dir,
):
    path = (
        results_dir
        / "p1_dirac_residual_adaptive_vs_uniform.csv"
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

        for row in (
            adaptive_rows
            + uniform_rows
        ):
            writer.writerow(
                row
            )


# ============================================================
# Theoretical reference slopes
# ============================================================

def reference_line(
    x_values,
    rate,
    x_anchor,
    y_anchor,
):
    r"""
    Construct a theoretical power-law reference curve

        E(N) = C N^{-rate}.

    We choose C so that the reference line passes through

        (x_anchor, y_anchor).

    Hence

        E(N)
        =
        y_anchor
        * (N / x_anchor)^{-rate}.

    The vertical position is not the theoretical statement.
    Only the slope -rate matters on the log-log plot.
    """
    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    return (
        y_anchor
        * (
            x_values / x_anchor
        ) ** (-rate)
    )


def add_reference_segment(
    ax,
    x_start,
    x_end,
    rate,
    x_anchor,
    y_anchor,
    label,
    linestyle,
    text_factor=1.20,
):
    """
    Add a short theoretical slope reference to a log-log plot.

    The segment is deliberately short so that it acts only as
    a slope guide rather than looking like another data curve.
    """
    x_reference = np.array([
        x_start,
        x_end,
    ])

    y_reference = reference_line(
        x_reference,
        rate=rate,
        x_anchor=x_anchor,
        y_anchor=y_anchor,
    )

    ax.loglog(
        x_reference,
        y_reference,
        color="black",
        linestyle=linestyle,
        linewidth=1.4,
    )

    # Geometric midpoint is the natural midpoint
    # on logarithmic axes.
    x_text = np.sqrt(
        x_start * x_end
    )

    y_text = (
        np.sqrt(
            y_reference[0]
            * y_reference[1]
        )
        * text_factor
    )

    ax.text(
        x_text,
        y_text,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
    )


# ============================================================
# Plot adaptive vs uniform convergence
# ============================================================

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

    # --------------------------------------------------------
    # Choose actual numerical data points as anchors.
    #
    # IMPORTANT:
    # This is NOT a fitted convergence rate.
    #
    # The theoretical slopes are fixed beforehand:
    #
    # H1 uniform:   N^{-1/2}
    # H1 adaptive:  N^{-1}
    #
    # L2 uniform:   N^{-3/2}
    # L2 adaptive:  N^{-2}
    #
    # The actual data point only determines the vertical
    # position of the reference segment.
    # --------------------------------------------------------

    uniform_anchor_index = int(
        np.argmin(
            np.abs(
                uniform_dofs - 319
            )
        )
    )

    adaptive_anchor_index = int(
        np.argmin(
            np.abs(
                adaptive_dofs - 284
            )
        )
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 4.5),
        constrained_layout=True,
    )

    # ========================================================
    # H1 seminorm error
    # ========================================================

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

    # --------------------------------------------------------
    # Uniform Dirac-limited reference:
    #
    #     |u-u_h|_H1 ~ N^{-1/2}
    # --------------------------------------------------------

    add_reference_segment(
        axes[0],

        x_start=200.0,
        x_end=800.0,

        rate=0.5,

        x_anchor=uniform_dofs[
            uniform_anchor_index
        ],

        y_anchor=uniform_h1[
            uniform_anchor_index
        ],

        label=r"$N^{-1/2}$",

        linestyle=":",
        text_factor=1.18,
    )

    # --------------------------------------------------------
    # Adaptive P1 reference:
    #
    #     |u-u_h|_H1 ~ N^{-1}
    # --------------------------------------------------------

    add_reference_segment(
        axes[0],

        x_start=220.0,
        x_end=800.0,

        rate=1.0,

        x_anchor=adaptive_dofs[
            adaptive_anchor_index
        ],

        y_anchor=adaptive_h1[
            adaptive_anchor_index
        ],

        label=r"$N^{-1}$",

        linestyle="-.",
        text_factor=1.16,
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

    # Only actual numerical methods belong
    # in the legend.
    axes[0].legend()

    # ========================================================
    # L2 error
    # ========================================================

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

    # --------------------------------------------------------
    # Uniform Dirac-limited reference:
    #
    #     ||u-u_h||_L2 ~ N^{-3/2}
    # --------------------------------------------------------

    add_reference_segment(
        axes[1],

        x_start=200.0,
        x_end=800.0,

        rate=1.5,

        x_anchor=uniform_dofs[
            uniform_anchor_index
        ],

        y_anchor=uniform_l2[
            uniform_anchor_index
        ],

        label=r"$N^{-3/2}$",

        linestyle=":",
        text_factor=1.35,
    )

    # --------------------------------------------------------
    # Adaptive P1 reference:
    #
    #     ||u-u_h||_L2 ~ N^{-2}
    # --------------------------------------------------------

    add_reference_segment(
        axes[1],

        x_start=220.0,
        x_end=800.0,

        rate=2.0,

        x_anchor=adaptive_dofs[
            adaptive_anchor_index
        ],

        y_anchor=adaptive_l2[
            adaptive_anchor_index
        ],

        label=r"$N^{-2}$",

        linestyle="-.",
        text_factor=1.35,
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
        "P1 Dirac problem: "
        "residual-based adaptive vs uniform refinement"
    )

    fig.savefig(
        figures_dir
        / "p1_dirac_residual_adaptive_vs_uniform.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main():
    figures_dir = (
        ROOT / "figures"
    )

    results_dir = (
        ROOT / "results"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    results_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Adaptive computation
    # --------------------------------------------------------

    adaptive_rows = (
        collect_adaptive_data()
    )

    final_adaptive_elements = (
        adaptive_rows[-1][
            "elements"
        ]
    )

    # --------------------------------------------------------
    # Uniform comparison meshes
    # --------------------------------------------------------

    uniform_counts = (
        choose_uniform_element_counts(
            final_adaptive_elements
        )
    )

    print()

    print(
        "Uniform element counts:",
        uniform_counts,
    )

    uniform_rows = (
        collect_uniform_data(
            uniform_counts
        )
    )

    # --------------------------------------------------------
    # Numerical output
    # --------------------------------------------------------

    print_data(
        "Residual-based adaptive refinement",
        adaptive_rows,
    )

    print_data(
        "Uniform refinement",
        uniform_rows,
    )

    # --------------------------------------------------------
    # Save data and figure
    # --------------------------------------------------------

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

    print(
        "Saved:"
    )

    print(
        "  figures/"
        "p1_dirac_residual_adaptive_vs_uniform.png"
    )

    print(
        "  results/"
        "p1_dirac_residual_adaptive_vs_uniform.csv"
    )


if __name__ == "__main__":
    main()