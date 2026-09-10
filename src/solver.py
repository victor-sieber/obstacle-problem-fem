import numpy as np
from scipy.optimize import minimize


def solve_obstacle_problem(nodes, A, F, obstacle_function):
    """
    Solve

        min_U  1/2 U^T A U - F^T U

    subject to

        U_i >= Psi_i,

    where Psi_i is the obstacle evaluated at the interior nodes.
    """

    interior_nodes = nodes[1:-1]
    psi = obstacle_function(interior_nodes)

    # Unconstrained FEM solution: A U = F
    unconstrained_solution = np.linalg.solve(A, F)

    # Construct a feasible initial guess
    initial_guess = np.maximum(unconstrained_solution, psi)

    def energy(U):
        return 0.5 * U @ A @ U - F @ U

    def gradient(U):
        return A @ U - F

    bounds = [(psi_i, None) for psi_i in psi]

    result = minimize(
        energy,
        initial_guess,
        jac=gradient,
        bounds=bounds,
        method="L-BFGS-B",
        options={
            "ftol": 1e-14,
            "gtol": 1e-10,
            "maxiter": 10000,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Optimization failed: {result.message}"
        )

    return result.x, psi, result