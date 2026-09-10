from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.fem import (
    create_mesh,
    assemble_stiffness_matrix,
    assemble_load_vector,
)
from src.problem import (
    load,
    obstacle,
    exact_solution,
    exact_derivative,
)
from src.solver import solve_obstacle_problem


def compute_errors(
    nodes,
    interior_values,
    alpha,
    quadrature_order=12,
):
    """
    Compute

        ||u - u_h||_L2

    and the H1 seminorm error

        ||u' - u_h'||_L2.
    """

    full_values = np.concatenate(
        ([0.0], interior_values, [0.0])
    )

    xi, weights = leggauss(quadrature_order)

    l2_error_squared = 0.0
    h1_error_squared = 0.0

    for element in range(len(nodes) - 1):
        x_left = nodes[element]
        x_right = nodes[element + 1]

        h = x_right - x_left

        x_quad = (
            0.5 * (x_left + x_right)
            + 0.5 * h * xi
        )

        w_quad = 0.5 * h * weights

        phi_left = (x_right - x_quad) / h
        phi_right = (x_quad - x_left) / h

        u_h = (
            full_values[element] * phi_left
            + full_values[element + 1] * phi_right
        )

        u_h_derivative = (
            full_values[element + 1]
            - full_values[element]
        ) / h

        u_exact = exact_solution(
            x_quad,
            alpha,
        )

        du_exact = exact_derivative(
            x_quad,
            alpha,
        )

        l2_error_squared += np.sum(
            w_quad * (u_exact - u_h) ** 2
        )

        h1_error_squared += np.sum(
            w_quad
            * (du_exact - u_h_derivative) ** 2
        )

    return (
        np.sqrt(l2_error_squared),
        np.sqrt(h1_error_squared),
    )


def main():
    alpha = 0.2

    element_counts = np.array(
        [10, 20, 40, 80, 160]
    )

    mesh_sizes = []
    l2_errors = []
    h1_errors = []

    for n_elements in element_counts:
        nodes = create_mesh(n_elements)

        A = assemble_stiffness_matrix(nodes)

        F = assemble_load_vector(
            nodes,
            load,
        )

        obstacle_function = (
            lambda x: obstacle(x, alpha)
        )

        U, _, _ = solve_obstacle_problem(
            nodes,
            A,
            F,
            obstacle_function,
        )

        l2_error, h1_error = compute_errors(
            nodes,
            U,
            alpha,
        )

        h = 1.0 / n_elements

        mesh_sizes.append(h)
        l2_errors.append(l2_error)
        h1_errors.append(h1_error)

    mesh_sizes = np.array(mesh_sizes)
    l2_errors = np.array(l2_errors)
    h1_errors = np.array(h1_errors)

    print(
        "N       h          L2 error       rate       "
        "H1-semi error    rate"
    )

    print("-" * 72)

    for i, n_elements in enumerate(element_counts):
        if i == 0:
            l2_rate = np.nan
            h1_rate = np.nan
        else:
            l2_rate = np.log(
                l2_errors[i - 1] / l2_errors[i]
            ) / np.log(2.0)

            h1_rate = np.log(
                h1_errors[i - 1] / h1_errors[i]
            ) / np.log(2.0)

        print(
            f"{n_elements:<6d} "
            f"{mesh_sizes[i]:<10.5f} "
            f"{l2_errors[i]:<14.6e} "
            f"{l2_rate:<10.3f} "
            f"{h1_errors[i]:<14.6e} "
            f"{h1_rate:<10.3f}"
        )

    plt.figure(figsize=(7, 5))

    plt.loglog(
        mesh_sizes,
        l2_errors,
        "o-",
        label=r"$L^2$ error",
    )

    plt.loglog(
        mesh_sizes,
        h1_errors,
        "s-",
        label=r"$H^1$ seminorm error",
    )

    plt.loglog(
        mesh_sizes,
        mesh_sizes,
        "--",
        label=r"$O(h)$ reference",
    )

    plt.loglog(
        mesh_sizes,
        mesh_sizes**2,
        "--",
        label=r"$O(h^2)$ reference",
    )

    plt.xlabel("Mesh size h")
    plt.ylabel("Error")

    plt.title(
        "Convergence of the P1 FEM obstacle solution"
    )

    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()

    figure_path = (
        ROOT / "figures" / "convergence.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()