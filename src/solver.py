import numpy as np
from scipy.optimize import OptimizeResult, minimize


def solve_obstacle_problem(nodes, A, F, obstacle_function):
    """
    Solve the discrete obstacle problem with L-BFGS-B:

        min_U  1/2 U^T A U - F^T U

    subject to

        U_i >= Psi_i.
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


def solve_obstacle_problem_pdas(
    nodes,
    A,
    F,
    obstacle_function,
    c=1.0,
    maxiter=100,
):
    """
    Solve the discrete obstacle problem with a primal-dual active-set method:

        min_V  1/2 V^T A V - F^T V

    subject to

        V >= Z.

    The multiplier is

        Lambda = A V - F.

    The active/contact set is updated by

        C^{j+1}
        =
        {i : Lambda_i^{j+1} - c (V_i^{j+1} - Z_i) > 0}.
    """
    if c <= 0.0:
        raise ValueError("PDAS parameter c must be positive.")

    interior_nodes = nodes[1:-1]
    psi = obstacle_function(interior_nodes)
    n = len(psi)

    # Practical initial guess for C^0:
    # nodes where the unconstrained solution lies below the obstacle.
    unconstrained_solution = np.linalg.solve(A, F)
    active = unconstrained_solution <= psi

    active_set_history = [
        np.flatnonzero(active).copy()
    ]

    for iteration in range(maxiter):
        free = ~active

        V = np.empty(n, dtype=float)

        # Contact condition: V_C = Z_C
        V[active] = psi[active]

        # Free condition: Lambda_F = 0, hence
        #
        # A_FF V_F = F_F - A_FC Z_C.
        if np.any(free):
            A_ff = A[np.ix_(free, free)]
            rhs = F[free]

            if np.any(active):
                rhs = (
                    rhs
                    - A[np.ix_(free, active)]
                    @ psi[active]
                )

            V[free] = np.linalg.solve(
                A_ff,
                rhs,
            )

        # Full discrete multiplier / contact force
        multiplier = A @ V - F

        # PDAS active-set update
        new_active = (
            multiplier
            - c * (V - psi)
            > 0.0
        )

        active_set_history.append(
            np.flatnonzero(new_active).copy()
        )

        # Stable active set => KKT conditions hold
        if np.array_equal(
            new_active,
            active,
        ):
            gap = V - psi
            complementarity = (
                gap * multiplier
            )

            result = OptimizeResult(
                x=V,
                success=True,
                status=0,
                message="PDAS active set converged.",
                nit=iteration + 1,
                contact_set=np.flatnonzero(
                    active
                ),
                multiplier=multiplier,
                min_gap=float(
                    np.min(gap)
                ),
                min_multiplier=float(
                    np.min(multiplier)
                ),
                max_complementarity=float(
                    np.max(
                        np.abs(
                            complementarity
                        )
                    )
                ),
                active_set_history=(
                    active_set_history
                ),
            )

            return V, psi, result

        active = new_active

    raise RuntimeError(
        f"PDAS failed to converge within "
        f"{maxiter} iterations."
    )