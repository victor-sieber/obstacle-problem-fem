from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.fem import (
    create_mesh,
    assemble_stiffness_matrix,
    assemble_load_vector,
    interpolate_fem_solution,
)
from src.problem import load, obstacle, exact_solution
from src.solver import solve_obstacle_problem


def main():
    alpha = 0.2
    n_elements = 20

    nodes = create_mesh(n_elements)

    A = assemble_stiffness_matrix(nodes)
    F = assemble_load_vector(nodes, load)

    obstacle_function = lambda x: obstacle(x, alpha)

    U, psi_nodes, result = solve_obstacle_problem(
        nodes,
        A,
        F,
        obstacle_function,
    )

    x_plot = np.linspace(0.0, 1.0, 1000)

    u_exact = exact_solution(x_plot, alpha)

    u_fem = interpolate_fem_solution(
        nodes,
        U,
        x_plot,
    )

    psi_plot = obstacle(x_plot, alpha)

    full_U = np.concatenate(([0.0], U, [0.0]))

    plt.figure(figsize=(8, 5))

    plt.plot(
        x_plot,
        u_exact,
        label="Exact solution",
    )

    plt.plot(
        x_plot,
        u_fem,
        "--",
        label="P1 FEM solution",
    )

    plt.plot(
        x_plot,
        psi_plot,
        label="Obstacle",
    )

    plt.scatter(
        nodes,
        full_U,
        s=20,
        label="FEM nodes",
    )

    plt.axvline(
        alpha,
        linestyle=":",
        label="Exact free boundary",
    )

    plt.axvline(
        1.0 - alpha,
        linestyle=":",
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        f"1D obstacle problem: N = {n_elements}"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    figure_path = ROOT / "figures" / "solution.png"
    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.show()

    residual = A @ U - F

    print("Optimization success:")
    print(result.success)

    print("\nMinimum obstacle gap:")
    print(np.min(U - psi_nodes))

    print("\nMinimum KKT residual:")
    print(np.min(residual))

    print("\nMaximum complementarity product:")
    print(
        np.max(
            np.abs(
                (U - psi_nodes) * residual
            )
        )
    )


if __name__ == "__main__":
    main()