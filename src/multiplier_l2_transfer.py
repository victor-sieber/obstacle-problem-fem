
"""L2-Riesz multiplier reconstruction and nested transfer.

Given the algebraic multiplier

    Lambda_H = A_H U_H - F_H,

reconstruct the finite-element function with coefficients mu_H:

    M_H mu_H = Lambda_H.

Transfer the reconstructed function:

    mu_h^0 = I_h mu_H,

and convert it back into a fine-grid algebraic multiplier:

    Lambda_h^0 = M_h mu_h^0.

Supports P1 and P2 on 1D meshes.
No external linear-system solver is used.
"""

import numpy as np

from src.p1_nested_large import thomas

from src.p2_nested_large import (
    join_p2,
    prolongate_p2,
    split_p2,
)


def _tridiagonal(off, diagonal, rhs):
    """Solve a symmetric tridiagonal system with Thomas."""

    n = len(diagonal)

    lower = np.zeros(n, dtype=float)
    upper = np.zeros(n, dtype=float)

    lower[1:] = off
    upper[:-1] = off

    return thomas(
        lower,
        diagonal,
        upper,
        rhs,
    )


# ============================================================
# P1 mass matrix and L2 reconstruction
# ============================================================

def p1_mass_action(nodes, coefficients):
    """Compute M_h @ coefficients without a dense matrix.

    For interior P1 hat functions:

        M_ii = (h_left + h_right) / 3

        M_i,i+1 = h_i / 6
    """

    h = np.diff(nodes)

    result = (
        (h[:-1] + h[1:]) / 3.0
        * coefficients
    )

    off = h[1:-1] / 6.0

    result[:-1] += off * coefficients[1:]
    result[1:] += off * coefficients[:-1]

    return result


def p1_riesz_coefficients(nodes, dual):
    """Solve the consistent P1 mass system M_h mu = dual."""

    h = np.diff(nodes)

    diagonal = (
        h[:-1] + h[1:]
    ) / 3.0

    off = h[1:-1] / 6.0

    return _tridiagonal(
        off,
        diagonal,
        dual,
    )


# ============================================================
# P2 mass matrix and L2 reconstruction
# ============================================================

def p2_mass_action(nodes, coefficients):
    """Compute M_h @ coefficients using local P2 mass matrices.

    In local ordering (left vertex, midpoint, right vertex):

               [ 4   2  -1 ]
        M_K = h/30 * [ 2  16   2 ]
               [ -1  2   4 ]

    The full mass matrix is never assembled densely.
    """

    m = len(nodes) - 1

    vertices, midpoints = split_p2(
        coefficients,
        m,
    )

    a = np.diff(nodes) / 30.0

    mv = np.zeros(m + 1, dtype=float)

    mv[:-1] += a * (
        4 * vertices[:-1]
        + 2 * midpoints
        - vertices[1:]
    )

    mv[1:] += a * (
        -vertices[:-1]
        + 2 * midpoints
        + 4 * vertices[1:]
    )

    mm = a * (
        2 * vertices[:-1]
        + 16 * midpoints
        + 2 * vertices[1:]
    )

    return join_p2(mv, mm)


def p2_riesz_coefficients(nodes, dual):
    """Solve M_h mu = dual using static condensation.

    The P2 midpoint unknowns are eliminated locally.

    The resulting vertex system is tridiagonal:

        local vertex diagonal = h / 8

        local vertex coupling = -h / 24

    Then recover the midpoint coefficients.
    """

    m = len(nodes) - 1

    dual_vertices, dual_midpoints = split_p2(
        dual,
        m,
    )

    h = np.diff(nodes)

    # --------------------------------------------------
    # 1. Reduced vertex system.
    # --------------------------------------------------

    diagonal = (
        h[:-1] + h[1:]
    ) / 8.0

    off = -h[1:-1] / 24.0

    rhs = (
        dual_vertices[1:-1]
        - (
            dual_midpoints[:-1]
            + dual_midpoints[1:]
        ) / 8.0
    )

    # Homogeneous boundary values.
    vertices = np.zeros(m + 1, dtype=float)

    vertices[1:-1] = _tridiagonal(
        off,
        diagonal,
        rhs,
    )

    # --------------------------------------------------
    # 2. Recover midpoint coefficients.
    # --------------------------------------------------

    midpoints = (
        15.0 * dual_midpoints / (8.0 * h)
        - (
            vertices[:-1]
            + vertices[1:]
        ) / 8.0
    )

    return join_p2(vertices, midpoints)


# ============================================================
# Transfer of L2-reconstructed multipliers
# ============================================================

def transfer_l2_multiplier_p1(
    coarse_nodes,
    coarse_dual,
    fine_nodes,
):
    """Lambda_h^0 = M_h I_h (M_H^{-1} Lambda_H), P1."""

    # Reconstruct the coarse L2 multiplier.
    mu_H = p1_riesz_coefficients(
        coarse_nodes,
        coarse_dual,
    )

    # Homogeneous boundary values for the FEM function.
    mu_full = np.r_[
        0.0,
        mu_H,
        0.0,
    ]

    # Interpolate the multiplier function onto fine nodes.
    mu_h = np.interp(
        fine_nodes[1:-1],
        coarse_nodes,
        mu_full,
    )

    # Convert back to the fine algebraic dual vector.
    return p1_mass_action(
        fine_nodes,
        mu_h,
    )


def transfer_l2_multiplier_p2(
    coarse_nodes,
    coarse_dual,
    fine_nodes,
):
    """Lambda_h^0 = M_h I_h (M_H^{-1} Lambda_H), P2."""

    # Reconstruct the coarse L2 multiplier.
    mu_H = p2_riesz_coefficients(
        coarse_nodes,
        coarse_dual,
    )

    # Evaluate the coarse P2 function at fine P2 nodes.
    mu_h = prolongate_p2(
        coarse_nodes,
        mu_H,
        fine_nodes,
    )

    # Convert back to the fine algebraic dual vector.
    return p2_mass_action(
        fine_nodes,
        mu_h,
    )
