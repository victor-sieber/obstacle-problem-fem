from pathlib import Path
import sys

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
from src.solver import (
    solve_obstacle_problem,
    solve_obstacle_problem_pdas,
)


ALPHA = 1.0 / 3.0
BETA = 2.0 / 3.0
GAMMA = 1.0 - np.sqrt(2.0) / 3.0

ELEMENT_COUNTS = [
    20,
    40,
    80,
    160,
    320,
]


def obstacle_dirac(x):
    x = np.asarray(x)

    return -np.ones_like(
        x,
        dtype=float,
    )


def smooth_load(x):
    x = np.asarray(x)

    return -6.0 * np.ones_like(
        x,
        dtype=float,
    )


def add_point_load(
    F,
    nodes,
    point,
    weight,
):
    n_nodes = len(nodes)

    element = (
        np.searchsorted(
            nodes,
            point,
            side="right",
        )
        - 1
    )

    element = min(
        max(element, 0),
        n_nodes - 2,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    phi_left = (
        x_right - point
    ) / h

    phi_right = (
        point - x_left
    ) / h

    left_node = element
    right_node = element + 1

    if 1 <= left_node <= n_nodes - 2:
        F[left_node - 1] += (
            weight * phi_left
        )

    if 1 <= right_node <= n_nodes - 2:
        F[right_node - 1] += (
            weight * phi_right
        )


def assemble_dirac_load(nodes):
    F = assemble_load_vector(
        nodes,
        smooth_load,
    )

    add_point_load(
        F,
        nodes,
        ALPHA,
        -2.0,
    )

    add_point_load(
        F,
        nodes,
        BETA,
        -1.0,
    )

    return F


def exact_solution_dirac(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    u = np.empty_like(x)

    region_1 = x <= ALPHA

    region_2 = (
        (x > ALPHA)
        & (x <= GAMMA)
    )

    region_3 = (
        (x > GAMMA)
        & (x <= BETA)
    )

    region_4 = x > BETA

    u[region_1] = (
        3.0 * x[region_1] ** 2
        - 4.0 * x[region_1]
    )

    u[region_2] = -1.0

    u[region_3] = (
        -1.0
        + 3.0
        * (
            x[region_3]
            - GAMMA
        )
        ** 2
    )

    u[region_4] = (
        3.0 * x[region_4] ** 2
        + (
            2.0 * np.sqrt(2.0)
            - 5.0
        )
        * x[region_4]
        + 2.0
        - 2.0 * np.sqrt(2.0)
    )

    return u


def exact_derivative_dirac(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    du = np.empty_like(x)

    region_1 = x < ALPHA

    region_2 = (
        (x >= ALPHA)
        & (x < GAMMA)
    )

    region_3 = (
        (x >= GAMMA)
        & (x < BETA)
    )

    region_4 = x >= BETA

    du[region_1] = (
        6.0 * x[region_1]
        - 4.0
    )

    du[region_2] = 0.0

    du[region_3] = (
        6.0
        * (
            x[region_3]
            - GAMMA
        )
    )

    du[region_4] = (
        6.0 * x[region_4]
        + 2.0 * np.sqrt(2.0)
        - 5.0
    )

    return du


def compute_errors(
    nodes,
    U,
):
    x = np.linspace(
        0.0,
        1.0,
        20001,
    )

    uh = interpolate_fem_solution(
        nodes,
        U,
        x,
    )

    u = exact_solution_dirac(x)

    l2_error = np.sqrt(
        np.trapezoid(
            (uh - u) ** 2,
            x,
        )
    )

    h1_squared = 0.0

    for element in range(
        len(nodes) - 1
    ):
        a = nodes[element]
        b = nodes[element + 1]

        mask = (
            (x >= a)
            & (x <= b)
        )

        x_local = x[mask]

        if len(x_local) < 2:
            continue

        if element == 0:
            u_left = 0.0
        else:
            u_left = U[
                element - 1
            ]

        if (
            element
            == len(nodes) - 2
        ):
            u_right = 0.0
        else:
            u_right = U[element]

        uh_derivative = (
            u_right - u_left
        ) / (b - a)

        exact_derivative = (
            exact_derivative_dirac(
                x_local
            )
        )

        h1_squared += np.trapezoid(
            (
                uh_derivative
                - exact_derivative
            )
            ** 2,
            x_local,
        )

    return (
        float(l2_error),
        float(
            np.sqrt(h1_squared)
        ),
    )


def main():
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    l2_errors = []
    h1_errors = []
    max_differences = []

    comparison_data = None

    print(
        "M     L2 error       "
        "H1-semi error   "
        "PDAS it   "
        "max solver difference"
    )

    print("-" * 76)

    for n_elements in ELEMENT_COUNTS:
        nodes = create_mesh(
            n_elements
        )

        A = assemble_stiffness_matrix(
            nodes
        )

        F = assemble_dirac_load(
            nodes
        )

        (
            U_lbfgsb,
            psi,
            result_lbfgsb,
        ) = solve_obstacle_problem(
            nodes,
            A,
            F,
            obstacle_dirac,
        )

        (
            U_pdas,
            _,
            result_pdas,
        ) = solve_obstacle_problem_pdas(
            nodes,
            A,
            F,
            obstacle_dirac,
        )

        difference = float(
            np.max(
                np.abs(
                    U_lbfgsb
                    - U_pdas
                )
            )
        )

        gap = U_pdas - psi
        multiplier = A @ U_pdas - F

        if np.min(gap) < -1e-10:
            raise RuntimeError(
                f"Obstacle violation for M={n_elements}"
            )

        if np.min(multiplier) < -1e-10:
            raise RuntimeError(
                f"Negative multiplier for M={n_elements}"
            )

        if (
            np.max(
                np.abs(
                    gap * multiplier
                )
            )
            > 1e-10
        ):
            raise RuntimeError(
                f"Complementarity failure for M={n_elements}"
            )

        l2_error, h1_error = (
            compute_errors(
                nodes,
                U_pdas,
            )
        )

        l2_errors.append(
            l2_error
        )

        h1_errors.append(
            h1_error
        )

        max_differences.append(
            difference
        )

        print(
            f"{n_elements:<5d} "
            f"{l2_error:<14.6e} "
            f"{h1_error:<15.6e} "
            f"{result_pdas.nit:<9d} "
            f"{difference:.3e}"
        )

        if n_elements == 20:
            comparison_data = (
                nodes,
                U_pdas,
            )

    if comparison_data is None:
        raise RuntimeError(
            "Comparison data missing."
        )

    # -------------------------------------------------
    # Solution plot
    # -------------------------------------------------

    nodes, U_pdas = comparison_data

    x_plot = np.linspace(
        0.0,
        1.0,
        2000,
    )

    exact = exact_solution_dirac(
        x_plot
    )

    discrete = (
        interpolate_fem_solution(
            nodes,
            U_pdas,
            x_plot,
        )
    )

    obstacle_plot = (
        obstacle_dirac(
            x_plot
        )
    )

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        x_plot,
        exact,
        label="Exact solution",
    )

    plt.plot(
        x_plot,
        discrete,
        "--",
        label="P1 PDAS solution",
    )

    plt.plot(
        x_plot,
        obstacle_plot,
        label="Obstacle",
    )

    plt.axvline(
        ALPHA,
        linestyle=":",
        label=r"$\alpha=1/3$",
    )

    plt.axvline(
        BETA,
        linestyle=":",
        label=r"$\beta=2/3$",
    )

    plt.axvline(
        GAMMA,
        linestyle="--",
        label=r"$\gamma=1-\sqrt{2}/3$",
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")

    plt.title(
        "P1 obstacle problem with Dirac loads"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    fig.savefig(
        figures_dir
        / "p1_dirac_solution.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -------------------------------------------------
    # Convergence plot
    # -------------------------------------------------

    element_counts = np.array(
        ELEMENT_COUNTS,
        dtype=float,
    )

    h = 1.0 / element_counts

    l2_errors = np.array(
        l2_errors
    )

    h1_errors = np.array(
        h1_errors
    )

    # Expected reduced convergence rates:
    # L2  ~ O(h^(3/2))
    # H1  ~ O(h^(1/2))
    #
    # Scale each reference curve so that it passes
    # through the corresponding coarsest-mesh error.
    l2_reference = (
        l2_errors[0]
        / h[0] ** 1.5
    ) * h ** 1.5

    h1_reference = (
        h1_errors[0]
        / h[0] ** 0.5
    ) * h ** 0.5

    fig = plt.figure(
        figsize=(7, 5)
    )

    plt.loglog(
        h,
        l2_errors,
        "o-",
        label=r"$L^2$ error",
    )

    plt.loglog(
        h,
        h1_errors,
        "s-",
        label=r"$H^1$ seminorm error",
    )

    plt.loglog(
        h,
        l2_reference,
        "--",
        label=r"$O(h^{3/2})$ reference",
    )

    plt.loglog(
        h,
        h1_reference,
        "--",
        label=r"$O(h^{1/2})$ reference",
    )

    plt.xlabel(
        "Mesh size h"
    )

    plt.ylabel(
        "Error"
    )

    plt.title(
        "P1 convergence with Dirac loads"
    )

    plt.legend()

    plt.grid(
        True,
        which="both",
    )

    plt.tight_layout()

    fig.savefig(
        figures_dir
        / "p1_dirac_convergence.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    main()