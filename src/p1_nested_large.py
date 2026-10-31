
"""Large-scale 1D P1 FEM with explicit tridiagonal PDAS."""

import numpy as np


ALPHA = 1.0 / 3.0
BETA = 2.0 / 3.0


def assemble_p1_dirac(nodes):
    """
    Assemble -u'' = -6 - 2 delta_alpha - delta_beta + lambda.

    Store only the three diagonals of the P1 stiffness matrix.
    """
    h = np.diff(nodes)

    if len(h) < 2 or np.any(h <= 0):
        raise ValueError("Invalid mesh")

    n = len(nodes) - 2

    # Stiffness matrix:
    # A_ii = 1/h_(i-1) + 1/h_i
    # A_(i,i+1) = -1/h_i

    diagonal = 1.0 / h[:-1] + 1.0 / h[1:]

    lower = np.zeros(n)
    upper = np.zeros(n)

    lower[1:] = -1.0 / h[1:-1]
    upper[:-1] = -1.0 / h[1:-1]

    # Regular load f_reg = -6.
    F = -3.0 * (h[:-1] + h[1:])

    # Point loads enter F separately.
    for z, weight in ((ALPHA, -2.0), (BETA, -1.0)):
        k = int(
            np.searchsorted(nodes, z, side="right") - 1
        )

        phi_right = (z - nodes[k]) / h[k]

        if k > 0:
            F[k - 1] += weight * (1.0 - phi_right)

        if k < n:
            F[k] += weight * phi_right

    return lower, diagonal, upper, F


def matvec(lower, diagonal, upper, values):
    """Compute A @ values without constructing the full matrix."""
    result = diagonal * values

    result[1:] += lower[1:] * values[:-1]
    result[:-1] += upper[:-1] * values[1:]

    return result


def thomas(lower, diagonal, upper, rhs):
    """Explicit Thomas algorithm for tridiagonal systems."""
    n = len(diagonal)

    if n == 0:
        return np.empty(0, dtype=float)

    c = np.empty(n)
    d = np.empty(n)

    pivot = diagonal[0]

    if pivot <= 0 or not np.isfinite(pivot):
        raise ArithmeticError("Invalid pivot")

    c[0] = upper[0] / pivot
    d[0] = rhs[0] / pivot

    # Forward elimination.
    for i in range(1, n):
        pivot = diagonal[i] - lower[i] * c[i - 1]

        if pivot <= 0 or not np.isfinite(pivot):
            raise ArithmeticError(
                f"Invalid pivot at index {i}"
            )

        c[i] = upper[i] / pivot

        d[i] = (
            rhs[i] - lower[i] * d[i - 1]
        ) / pivot

    # Backward substitution.
    result = np.empty(n)
    result[-1] = d[-1]

    for i in range(n - 2, -1, -1):
        result[i] = d[i] - c[i] * result[i + 1]

    return result


def prolongate_p1(coarse_nodes, coarse_values, fine_nodes):
    """Interpolate coarse nodal values onto fine interior nodes."""
    full_values = np.concatenate(
        ([0.0], coarse_values, [0.0])
    )

    return np.interp(
        fine_nodes[1:-1],
        coarse_nodes,
        full_values,
    )


def solve_pdas_p1(
    nodes,
    lower,
    diagonal,
    upper,
    F,
    initial_solution=None,
    initial_multiplier=None,
    c=1.0,
    maxiter=50,
    verbose=False,
):
    """PDAS with tridiagonal free-set solves.

    Cold start:
        C0 = {A^-1 F <= psi}

    Nested start:
        C0 = {Lambda0 - c(V0 - psi) > 0}

    The PDAS update and stopping criterion are unchanged.
    """
    if c <= 0 or maxiter < 1:
        raise ValueError("Invalid PDAS parameters")

    n = len(diagonal)
    psi = -np.ones(n)

    # --------------------------------------------------
    # Initial active set
    # --------------------------------------------------

    if initial_solution is None:
        unconstrained = thomas(
            lower, diagonal, upper, F
        )

        active = unconstrained <= psi

    else:
        V0 = np.asarray(initial_solution, dtype=float)

        if V0.shape != (n,):
            raise ValueError("Incorrect initial solution")

        if initial_multiplier is None:
            # Original nested initialization.
            Lambda0 = (
                matvec(lower, diagonal, upper, V0) - F
            )

        else:
            # Transfer coarse multiplier to the fine mesh.
            Lambda0 = np.asarray(
                initial_multiplier, dtype=float
            )

            if Lambda0.shape != (n,):
                raise ValueError("Incorrect initial multiplier")

        active = (
            Lambda0 - c * (V0 - psi) > 0.0
        )

    # --------------------------------------------------
    # Standard PDAS iteration
    # --------------------------------------------------

    for iteration in range(1, maxiter + 1):
        free = ~active
        V = psi.copy()

        # Consecutive free nodes form tridiagonal blocks.
        starts = np.flatnonzero(
            free & ~np.r_[False, free[:-1]]
        )

        ends = np.flatnonzero(
            free & ~np.r_[free[1:], False]
        ) + 1

        for first, last in zip(starts, ends):
            rhs = F[first:last].copy()

            # Contact values on neighboring nodes.
            if first > 0:
                rhs[0] -= lower[first] * psi[first - 1]

            if last < n:
                rhs[-1] -= upper[last - 1] * psi[last]

            # Free condition A_FF V_F = F_F - A_FC psi_C.
            V[first:last] = thomas(
                lower[first:last],
                diagonal[first:last],
                upper[first:last],
                rhs,
            )

        # Lambda = A V - F.
        multiplier = (
            matvec(lower, diagonal, upper, V) - F
        )

        # Original active-set update.
        new_active = (
            multiplier - c * (V - psi) > 0.0
        )

        changed = int(
            np.count_nonzero(new_active != active)
        )

        if verbose:
            print(
                f"PDAS {iteration}: "
                f"changed={changed}, "
                f"active={np.count_nonzero(new_active)}",
                flush=True,
            )

        # Original convergence condition C_new = C.
        if changed == 0:
            gap = V - psi

            return V, {
                "iterations": iteration,
                "active_dofs": int(np.count_nonzero(active)),
                "min_gap": float(np.min(gap)),
                "min_multiplier": float(np.min(multiplier)),
                "max_complementarity": float(
                    np.max(np.abs(gap * multiplier))
                ),
            }

        active = new_active

    raise RuntimeError(
        f"PDAS failed to stabilize in {maxiter} iterations "
        f"(DOFs={n}, last changed={changed})"
    )
