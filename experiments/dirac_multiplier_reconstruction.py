from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from experiments.dirac_example import (
    ALPHA,
    GAMMA,
    assemble_dirac_load,
    obstacle_dirac,
    smooth_load,
)

from src.error_indicators import (
    compute_p1_residual_indicator,
    evaluate_p1_multiplier,
    reconstruct_p1_multiplier_l2,
)

from src.fem import (
    assemble_stiffness_matrix,
    create_mesh,
)

from src.solver import (
    solve_obstacle_problem_pdas,
)


N_ELEMENTS = 80
THETA = 0.5


def exact_contact_multiplier(x):
    """
    Regular exact obstacle multiplier for
    the current Dirac benchmark.

    On the contact region [ALPHA, GAMMA]:

        u = -1,
        u'' = 0,
        f_reg = -6,

    hence

        lambda = -u'' - f_reg = 6.

    Outside the contact region lambda = 0.

    The Dirac distributions themselves are part
    of the load f and are not part of this regular
    multiplier function.
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    return np.where(
        (
            (x >= ALPHA)
            & (x <= GAMMA)
        ),
        6.0,
        0.0,
    )


def main():
    figures_dir = (
        ROOT / "figures"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    # =================================================
    # Solve obstacle problem once
    # =================================================

    nodes = create_mesh(
        N_ELEMENTS
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
    )

    # =================================================
    # Two reconstructions of the SAME KKT multiplier
    # =================================================

    consistent = reconstruct_p1_multiplier_l2(
        nodes,
        U,
        A,
        F,
        method="consistent",
    )

    lumped = reconstruct_p1_multiplier_l2(
        nodes,
        U,
        A,
        F,
        method="lumped",
    )

    # =================================================
    # Corresponding residual indicators
    # =================================================

    indicator_consistent = compute_p1_residual_indicator(
        nodes,
        U,
        A,
        F,
        smooth_load,
        multiplier_method="consistent",
    )

    indicator_lumped = compute_p1_residual_indicator(
        nodes,
        U,
        A,
        F,
        smooth_load,
        multiplier_method="lumped",
    )

    # Maximum marking for comparison only.
    marked_consistent = (
        indicator_consistent["eta"]
        >=
        THETA
        * np.max(
            indicator_consistent["eta"]
        )
    )

    marked_lumped = (
        indicator_lumped["eta"]
        >=
        THETA
        * np.max(
            indicator_lumped["eta"]
        )
    )

    # =================================================
    # Console diagnostics
    # =================================================

    print()
    print(
        f"Elements: {N_ELEMENTS}"
    )

    print(
        f"PDAS iterations: "
        f"{result_pdas.nit}"
    )

    print()
    print(
        "Discrete KKT multiplier "
        "Lambda = A U - F"
    )

    print(
        "  min Lambda: "
        f"{np.min(consistent['Lambda']):.6e}"
    )

    print(
        "  max Lambda: "
        f"{np.max(consistent['Lambda']):.6e}"
    )

    print()
    print(
        "Consistent L2-Riesz coefficients mu"
    )

    print(
        "  min mu: "
        f"{np.min(consistent['mu']):.6e}"
    )

    print(
        "  max mu: "
        f"{np.max(consistent['mu']):.6e}"
    )

    print()
    print(
        "Lumped coefficients mu"
    )

    print(
        "  min mu: "
        f"{np.min(lumped['mu']):.6e}"
    )

    print(
        "  max mu: "
        f"{np.max(lumped['mu']):.6e}"
    )

    eta_consistent = np.sqrt(
        np.sum(
            indicator_consistent[
                "eta_squared"
            ]
        )
    )

    eta_lumped = np.sqrt(
        np.sum(
            indicator_lumped[
                "eta_squared"
            ]
        )
    )

    print()
    print(
        "Indicator comparison"
    )

    print(
        "  eta_h consistent: "
        f"{eta_consistent:.6e}"
    )

    print(
        "  eta_h lumped:     "
        f"{eta_lumped:.6e}"
    )

    print(
        "  identical marking: "
        f"{np.array_equal(
            marked_consistent,
            marked_lumped
        )}"
    )

    print(
        "  marked consistent: "
        f"{np.flatnonzero(
            marked_consistent
        ).tolist()}"
    )

    print(
        "  marked lumped:     "
        f"{np.flatnonzero(
            marked_lumped
        ).tolist()}"
    )

    # =================================================
    # Multiplier plot
    # =================================================

    x_plot = np.linspace(
        0.0,
        1.0,
        4000,
    )

    lambda_consistent = evaluate_p1_multiplier(
        nodes,
        consistent["mu"],
        x_plot,
    )

    lambda_lumped = evaluate_p1_multiplier(
        nodes,
        lumped["mu"],
        x_plot,
    )

    lambda_exact = exact_contact_multiplier(
        x_plot
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        constrained_layout=True,
    )

    axes[0].plot(
        x_plot,
        lambda_exact,
        label="exact regular multiplier",
    )

    axes[0].plot(
        x_plot,
        lambda_consistent,
        "--",
        label="consistent L2-Riesz",
    )

    axes[0].plot(
        x_plot,
        lambda_lumped,
        ":",
        label="mass lumped",
    )

    axes[0].set_ylabel(
        r"$\lambda_h(x)$"
    )

    axes[0].set_title(
        "Multiplier reconstruction"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[0].legend()

    # =================================================
    # Local indicator comparison
    # =================================================

    midpoints = (
        indicator_consistent[
            "midpoints"
        ]
    )

    axes[1].plot(
        midpoints,
        indicator_consistent["eta"],
        "o-",
        markersize=3,
        label="consistent L2-Riesz",
    )

    axes[1].plot(
        midpoints,
        indicator_lumped["eta"],
        "s--",
        markersize=3,
        label="mass lumped",
    )

    axes[1].scatter(
        midpoints[marked_consistent],
        indicator_consistent[
            "eta"
        ][marked_consistent],
        marker="x",
        s=45,
        label="marked (consistent)",
    )

    axes[1].set_xlabel(
        r"element midpoint $x_K$"
    )

    axes[1].set_ylabel(
        r"$\eta_K$"
    )

    axes[1].set_title(
        rf"Residual indicator, "
        rf"maximum marking $\theta={THETA}$"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    axes[1].legend()

    fig.savefig(
        figures_dir
        / "p1_dirac_multiplier_reconstruction.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(
        "  figures/"
        "p1_dirac_multiplier_reconstruction.png"
    )


if __name__ == "__main__":
    main()