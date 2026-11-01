
"""Large-scale P2 Dirac FEM with static condensation and PDAS."""

import numpy as np

from src.p1_nested_large import thomas


ALPHA = 1.0 / 3.0
BETA = 2.0 / 3.0


def assemble_p2_dirac(nodes):
    """
    Assemble the P2 load vector for

        -u'' = -6 - 2 delta_alpha - delta_beta + lambda.

    The stiffness matrix is represented by its known
    local coefficients rather than assembled densely.

    Fv: load entries at mesh vertices, including boundaries.
    Fm: load entries at element midpoints.
    """
    h = np.diff(nodes)

    if h.size < 2 or np.any(h <= 0):
        raise ValueError("Invalid mesh")

    M = h.size

    Fv = np.zeros(M + 1)

    # Integral of the midpoint basis = 2h/3.
    Fm = -4.0 * h

    # Integral of either endpoint basis = h/6.
    Fv[:-1] -= h
    Fv[1:] -= h

    # Dirac contributions:
    # integral delta_z * phi_i = phi_i(z).
    for z, weight in (
        (ALPHA, -2.0),
        (BETA, -1.0),
    ):
        k = int(
            np.clip(
                np.searchsorted(
                    nodes, z, side="right"
                ) - 1,
                0,
                M - 1,
            )
        )

        xi = (z - nodes[k]) / h[k]

        phi_left = 2 * (xi - 0.5) * (xi - 1.0)
        phi_mid = 4 * xi * (1.0 - xi)
        phi_right = 2 * xi * (xi - 0.5)

        Fv[k] += weight * phi_left
        Fm[k] += weight * phi_mid
        Fv[k + 1] += weight * phi_right

    return Fv, Fm


def split_p2(U, M):
    """
    Convert interleaved interior P2 coefficients into
    vertex and midpoint coefficient arrays.

    Ordering:
        midpoint_0, vertex_1, midpoint_1, ...
    """
    U = np.asarray(U, dtype=float)

    if U.shape != (2 * M - 1,):
        raise ValueError("Wrong P2 interior shape")

    Vv = np.zeros(M + 1)
    Vv[1:-1] = U[1::2]

    Vm = U[0::2].copy()

    return Vv, Vm


def join_p2(Vv, Vm):
    """Return interior coefficients in standard P2 ordering."""
    M = len(Vm)

    U = np.empty(2 * M - 1)

    U[0::2] = Vm
    U[1::2] = Vv[1:-1]

    return U


def p2_matvec(nodes, U):
    """
    Compute A @ U from the local P2 stiffness matrix.

    No full stiffness matrix is stored.
    """
    M = len(nodes) - 1

    Vv, Vm = split_p2(U, M)

    inv3h = 1.0 / (3.0 * np.diff(nodes))

    Av = np.zeros(M + 1)

    Av[:-1] += inv3h * (
        7 * Vv[:-1] - 8 * Vm + Vv[1:]
    )

    Av[1:] += inv3h * (
        Vv[:-1] - 8 * Vm + 7 * Vv[1:]
    )

    Am = inv3h * (
        -8 * Vv[:-1]
        + 16 * Vm
        - 8 * Vv[1:]
    )

    return join_p2(Av, Am)


def solve_on_active_set(nodes, Fv, Fm, active):
    """
    Solve the P2 contact/free equations.

    Free midpoints are eliminated locally.

    The remaining vertex equations form a tridiagonal
    system, solved using our explicit Thomas algorithm.
    """
    M = len(nodes) - 1

    if active.shape != (2 * M - 1,):
        raise ValueError("Incorrect active-set shape")

    active_m = active[0::2]

    active_v = np.zeros(M + 1, dtype=bool)
    active_v[1:-1] = active[1::2]

    free_m = ~active_m

    h = np.diff(nodes)

    # Local P2 stiffness coefficients.
    a = 7.0 / (3.0 * h)
    b = -8.0 / (3.0 * h)
    c = 1.0 / (3.0 * h)
    d = 16.0 / (3.0 * h)

    # --------------------------------------------------
    # 1. Eliminate free midpoint DOFs.
    # --------------------------------------------------

    correction = b * b / d

    dv_local = a - free_m * correction
    off = c - free_m * correction

    diagonal_vertices = np.zeros(M + 1)

    diagonal_vertices[:-1] += dv_local
    diagonal_vertices[1:] += dv_local

    # Free midpoint:
    #   RHS shift = -(b/d) * Fm.
    #
    # Active midpoint:
    #   u_m = -1, RHS shift = -b*(-1) = b.
    shift = np.where(
        free_m,
        -b / d * Fm,
        b,
    )

    rhs_vertices = Fv.copy()

    rhs_vertices[:-1] += shift
    rhs_vertices[1:] += shift

    # --------------------------------------------------
    # 2. Enforce active vertex constraints.
    # --------------------------------------------------

    fixed_v = np.zeros(M + 1)
    fixed_v[active_v] = -1.0

    rhs = rhs_vertices[1:-1].copy()

    rhs -= (
        off[:-1] * fixed_v[:-2]
        + off[1:] * fixed_v[2:]
    )

    free_v = ~active_v[1:-1]

    main = diagonal_vertices[1:-1].copy()

    main[~free_v] = 1.0
    rhs[~free_v] = -1.0

    # Couplings between two free interior vertices.
    coupling = (
        off[1:-1]
        * (free_v[:-1] & free_v[1:])
    )

    lower = np.zeros(M - 1)
    upper = np.zeros(M - 1)

    lower[1:] = coupling
    upper[:-1] = coupling

    # --------------------------------------------------
    # 3. Solve reduced tridiagonal vertex system.
    # --------------------------------------------------

    vertices = fixed_v

    vertices[1:-1] = thomas(
        lower,
        main,
        upper,
        rhs,
    )

    # --------------------------------------------------
    # 4. Recover all free midpoint values.
    # --------------------------------------------------

    midpoints = np.full(M, -1.0)

    midpoints[free_m] = (
        Fm[free_m]
        - b[free_m]
        * (
            vertices[:-1][free_m]
            + vertices[1:][free_m]
        )
    ) / d[free_m]

    return join_p2(vertices, midpoints)


def prolongate_p2(coarse_nodes, coarse_U, fine_nodes):
    """
    Evaluate the coarse quadratic FEM function at
    all new interior P2 interpolation points.

    Uses the P2 Lagrange basis explicitly.
    """
    M = len(coarse_nodes) - 1

    old_v, old_m = split_p2(coarse_U, M)

    n = len(fine_nodes) - 1

    x = np.empty(2 * n - 1)

    x[0::2] = (
        fine_nodes[:-1] + fine_nodes[1:]
    ) / 2

    x[1::2] = fine_nodes[1:-1]

    k = np.clip(
        np.searchsorted(
            coarse_nodes,
            x,
            side="right",
        ) - 1,
        0,
        M - 1,
    )

    xi = (
        (x - coarse_nodes[k])
        / (coarse_nodes[k + 1] - coarse_nodes[k])
    )

    return (
        2 * (xi - 0.5) * (xi - 1.0) * old_v[k]
        + 4 * xi * (1.0 - xi) * old_m[k]
        + 2 * xi * (xi - 0.5) * old_v[k + 1]
    )


def solve_pdas_p2(
    nodes,
    Fv,
    Fm,
    initial_solution=None,
    initial_multiplier=None,
    c=1.0,
    maxiter=100,
    verbose=False,
):
    """
    PDAS for P2, with original active-set update.

    Cold:
        C0 = {unconstrained solution <= obstacle}.

    Nested:
        C0 = {Lambda0 - c(V0 - obstacle) > 0}.

    If initial_multiplier is omitted:
        Lambda0 = A V0 - F.

    If supplied:
        Lambda0 is the transferred coarse multiplier.
    """
    M = len(nodes) - 1
    n = 2 * M - 1

    if c <= 0 or maxiter < 1:
        raise ValueError("Invalid PDAS parameters")

    # --------------------------------------------------
    # Initial active set.
    # --------------------------------------------------

    if initial_solution is None:
        initial = solve_on_active_set(
            nodes,
            Fv,
            Fm,
            np.zeros(n, dtype=bool),
        )

        active = initial <= -1.0

    else:
        initial = np.asarray(
            initial_solution,
            dtype=float,
        )

        if initial.shape != (n,):
            raise ValueError("Incorrect initial solution")

        F = join_p2(Fv, Fm)

        if initial_multiplier is None:
            # Original nested initialization.
            lam0 = p2_matvec(nodes, initial) - F

        else:
            # Optional transferred-dual initialization.
            lam0 = np.asarray(
                initial_multiplier,
                dtype=float,
            )

            if lam0.shape != (n,):
                raise ValueError("Incorrect initial multiplier")

        active = (
            lam0 - c * (initial + 1.0) > 0.0
        )

    # --------------------------------------------------
    # Standard PDAS iteration.
    # --------------------------------------------------

    F = join_p2(Fv, Fm)

    for iteration in range(1, maxiter + 1):
        U = solve_on_active_set(
            nodes,
            Fv,
            Fm,
            active,
        )

        lam = p2_matvec(nodes, U) - F

        new_active = (
            lam - c * (U + 1.0) > 0.0
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

        # Original stopping criterion: C_new = C.
        if changed == 0:
            gap = U + 1.0

            return U, {
                "iterations": iteration,
                "active_dofs": int(
                    np.count_nonzero(active)
                ),
                "min_gap": float(np.min(gap)),
                "min_multiplier": float(np.min(lam)),
                "max_complementarity": float(
                    np.max(np.abs(gap * lam))
                ),
            }

        active = new_active

    raise RuntimeError(
        f"P2 PDAS failed to stabilize in "
        f"{maxiter} iterations "
        f"(DOFs={n}, last changed={changed})"
    )
