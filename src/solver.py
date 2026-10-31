
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import spsolve


def solve_obstacle_problem(nodes, A, F, obstacle_function):
    """
    Original L-BFGS-B solver, retained for comparisons.
    """
    interior_nodes = nodes[1:-1]
    psi = obstacle_function(interior_nodes)

    unconstrained_solution = np.linalg.solve(A, F)
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
    Standard primal-dual active-set method.

    Minimize:
        1/2 V^T A V - F^T V

    Subject to:
        V >= psi

    Multiplier:
        Lambda = A V - F

    Contact set:
        C = {i : Lambda_i - c(V_i - psi_i) > 0}
    """
    if c <= 0.0:
        raise ValueError("PDAS parameter c must be positive.")

    A_sparse = A.tocsc() if issparse(A) else csc_matrix(A)

    psi = obstacle_function(nodes[1:-1])
    n = len(psi)

    # Original cold-start initialization:
    # C^0 = {i : (A^-1 F)_i <= psi_i}.
    unconstrained_solution = spsolve(A_sparse, F)
    active = unconstrained_solution <= psi

    active_set_history = [
        np.flatnonzero(active).copy()
    ]

    for iteration in range(maxiter):
        free = ~active

        V = np.empty(n, dtype=float)

        # Contact condition.
        V[active] = psi[active]

        # Free condition:
        # A_FF V_F = F_F - A_FC psi_C.
        if np.any(free):
            A_ff = A_sparse[free][:, free].tocsc()
            rhs = F[free].copy()

            if np.any(active):
                rhs -= (
                    A_sparse[free][:, active]
                    @ psi[active]
                )

            V[free] = spsolve(A_ff, rhs)

        # Multiplier.
        multiplier = A_sparse @ V - F

        # Active-set update.
        new_active = (
            multiplier - c * (V - psi) > 0.0
        )

        active_set_history.append(
            np.flatnonzero(new_active).copy()
        )

        if np.array_equal(new_active, active):
            gap = V - psi

            result = OptimizeResult(
                x=V,
                success=True,
                status=0,
                message="PDAS active set converged.",
                nit=iteration + 1,
                contact_set=np.flatnonzero(active),
                multiplier=multiplier,
                min_gap=float(np.min(gap)),
                min_multiplier=float(np.min(multiplier)),
                max_complementarity=float(
                    np.max(np.abs(gap * multiplier))
                ),
                active_set_history=active_set_history,
            )

            return V, psi, result

        active = new_active

    raise RuntimeError(
        f"PDAS failed to converge within {maxiter} iterations."
    )


def solve_obstacle_problem_pdas_nested(
    nodes,
    A,
    F,
    obstacle_function,
    initial_solution,
    c=1.0,
    maxiter=100,
):
    """
    Original nested PDAS initialization, with sparse solves.

    Given the interpolated coarse-grid solution V0:

        Lambda0 = A V0 - F

        C0 = {i : Lambda0_i - c(V0_i - psi_i) > 0}

    Afterwards, the standard PDAS equations are used.
    """
    if c <= 0.0:
        raise ValueError("PDAS parameter c must be positive.")

    A_sparse = A.tocsc() if issparse(A) else csc_matrix(A)

    psi = obstacle_function(nodes[1:-1])
    n = len(psi)

    initial_solution = np.asarray(
        initial_solution,
        dtype=float,
    )

    if initial_solution.shape != (n,):
        raise ValueError(
            "initial_solution must contain one value "
            "for each interior node."
        )

    # --------------------------------------------------
    # Nested primal-dual initialization
    # --------------------------------------------------

    initial_multiplier = (
        A_sparse @ initial_solution - F
    )

    active = (
        initial_multiplier
        - c * (initial_solution - psi)
        > 0.0
    )

    active_set_history = [
        np.flatnonzero(active).copy()
    ]

    # --------------------------------------------------
    # Standard PDAS iteration
    # --------------------------------------------------

    for iteration in range(maxiter):
        free = ~active

        V = np.empty(n, dtype=float)

        # Contact nodes.
        V[active] = psi[active]

        # Free nodes.
        if np.any(free):
            A_ff = A_sparse[free][:, free].tocsc()
            rhs = F[free].copy()

            if np.any(active):
                rhs -= (
                    A_sparse[free][:, active]
                    @ psi[active]
                )

            V[free] = spsolve(A_ff, rhs)

        # Multiplier.
        multiplier = A_sparse @ V - F

        # Same active-set update as ordinary PDAS.
        new_active = (
            multiplier - c * (V - psi) > 0.0
        )

        active_set_history.append(
            np.flatnonzero(new_active).copy()
        )

        # Same convergence criterion.
        if np.array_equal(new_active, active):
            gap = V - psi

            result = OptimizeResult(
                x=V,
                success=True,
                status=0,
                message="Nested PDAS active set converged.",
                nit=iteration + 1,
                contact_set=np.flatnonzero(active),
                multiplier=multiplier,
                min_gap=float(np.min(gap)),
                min_multiplier=float(np.min(multiplier)),
                max_complementarity=float(
                    np.max(np.abs(gap * multiplier))
                ),
                active_set_history=active_set_history,
            )

            return V, psi, result

        active = new_active

    raise RuntimeError(
        f"Nested PDAS failed to converge within "
        f"{maxiter} iterations."
    )
