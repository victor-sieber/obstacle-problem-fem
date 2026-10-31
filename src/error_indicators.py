import numpy as np
from numpy.polynomial.legendre import leggauss

from src.fem import (
    assemble_mass_matrix,
    p1_basis,
)


def compute_p1_jump_indicator(nodes, U):
    """
    Compute the original P1 slope-jump indicator

        eta_{K_i}^2
        =
        h_i / 2
        * (J_i^2 + J_{i+1}^2).

    This remains available as the original
    jump-only baseline.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    if nodes.ndim != 1 or U.ndim != 1:
        raise ValueError(
            "nodes and U must be "
            "one-dimensional arrays."
        )

    if len(U) != len(nodes) - 2:
        raise ValueError(
            "For homogeneous Dirichlet P1 FEM, "
            "len(U) must equal len(nodes) - 2."
        )

    h = np.diff(nodes)

    if np.any(h <= 0.0):
        raise ValueError(
            "Mesh nodes must be strictly increasing."
        )

    U_full = np.zeros(
        len(nodes),
        dtype=float,
    )

    U_full[1:-1] = U

    # P1 derivative on each element.
    slopes = (
        np.diff(U_full)
        / h
    )

    # Convention:
    #
    # J_i
    # =
    # u_h'(x_i^+)
    # -
    # u_h'(x_i^-).
    jumps = np.zeros(
        len(nodes),
        dtype=float,
    )

    jumps[1:-1] = (
        slopes[1:]
        - slopes[:-1]
    )

    eta_squared = (
        0.5
        * h
        * (
            jumps[:-1] ** 2
            + jumps[1:] ** 2
        )
    )

    midpoints = (
        0.5
        * (
            nodes[:-1]
            + nodes[1:]
        )
    )

    return {
        "midpoints": midpoints,
        "h": h,
        "slopes": slopes,
        "jumps": jumps,
        "eta_squared": eta_squared,
        "eta": np.sqrt(eta_squared),
    }


def reconstruct_p1_multiplier_l2(
    nodes,
    U,
    A,
    F,
    method="consistent",
):
    """
    Reconstruct an L2 finite-element representative
    lambda_h of the discrete KKT multiplier functional.

    The discrete multiplier vector is

        Lambda = A U - F.

    --------------------------------------------------
    Consistent L2-Riesz reconstruction
    --------------------------------------------------

    Define lambda_h in V_h such that

        (lambda_h, v_h)_L2
        =
        <Lambda_h, v_h>

    for every v_h in V_h.

    In coordinates this is

        M mu = Lambda.

    --------------------------------------------------
    Lumped reconstruction
    --------------------------------------------------

    Use nodal quadrature:

        mu_i
        =
        Lambda_i / integral(phi_i).

    This is included only so that we can compare
    positivity/localisation against the exact
    L2-Riesz reconstruction.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    A = np.asarray(
        A,
        dtype=float,
    )

    F = np.asarray(
        F,
        dtype=float,
    )

    n_interior = (
        len(nodes) - 2
    )

    if (
        len(U) != n_interior
        or len(F) != n_interior
    ):
        raise ValueError(
            "U and F must contain one value "
            "per interior P1 node."
        )

    if A.shape != (
        n_interior,
        n_interior,
    ):
        raise ValueError(
            "A has incompatible dimensions."
        )

    # ---------------------------------------------
    # Discrete multiplier functional
    #
    # Lambda = A U - F.
    # ---------------------------------------------
    Lambda = (
        A @ U
        - F
    )

    # ---------------------------------------------
    # L2 Riesz map on V_h.
    # ---------------------------------------------
    M = assemble_mass_matrix(
        nodes
    )

    if method == "consistent":

        # Solve
        #
        # M mu = Lambda.
        #
        # np.linalg.solve does NOT form M^{-1}.
        mu = np.linalg.solve(
            M,
            Lambda,
        )

    elif method == "lumped":

        h = np.diff(nodes)

        # For the interior P1 hat phi_i,
        #
        # integral phi_i
        #
        # =
        # (h_{i-1} + h_i) / 2.
        lumped_weights = (
            0.5
            * (
                h[:-1]
                + h[1:]
            )
        )

        mu = (
            Lambda
            / lumped_weights
        )

    else:
        raise ValueError(
            "method must be "
            "'consistent' or 'lumped'."
        )

    return {
        "Lambda": Lambda,
        "mu": mu,
        "method": method,
        "mass_matrix": M,
    }


def evaluate_p1_multiplier(
    nodes,
    mu,
    x,
):
    """
    Evaluate

        lambda_h
        =
        sum_i mu_i phi_i

    at points x.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    mu = np.asarray(
        mu,
        dtype=float,
    )

    x = np.asarray(
        x,
        dtype=float,
    )

    if len(mu) != len(nodes) - 2:
        raise ValueError(
            "mu must contain one coefficient "
            "per interior P1 node."
        )

    if (
        np.any(x < nodes[0])
        or np.any(x > nodes[-1])
    ):
        raise ValueError(
            "Evaluation points must lie inside "
            "the domain."
        )

    mu_full = np.zeros(
        len(nodes),
        dtype=float,
    )

    mu_full[1:-1] = mu

    element = (
        np.searchsorted(
            nodes,
            x,
            side="right",
        )
        - 1
    )

    element = np.clip(
        element,
        0,
        len(nodes) - 2,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    xi = (
        x - x_left
    ) / h

    basis_values = p1_basis(xi)

    return (
        mu_full[element]
        * basis_values[0]
        +
        mu_full[element + 1]
        * basis_values[1]
    )


def compute_p1_residual_indicator(
    nodes,
    U,
    A,
    F,
    regular_load,
    multiplier_method="consistent",
    quadrature_order=4,
):
    r"""
    Compute the P1 residual indicator

        eta_K^2
        =
        h_K^2
        ||f_reg + lambda_h + u_h''||_{L2(K)}^2

        +

        h_K/2
        (J_left^2 + J_right^2).

    For P1,

        u_h''|_K = 0,

    therefore

        R_K
        =
        f_reg + lambda_h.

    For a Dirac load only the regular L2 part
    of f is supplied as regular_load.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    if len(U) != len(nodes) - 2:
        raise ValueError(
            "For homogeneous Dirichlet P1 FEM, "
            "len(U) must equal len(nodes) - 2."
        )

    h = np.diff(nodes)
    n_elements = len(h)

    if np.any(h <= 0.0):
        raise ValueError(
            "Mesh nodes must be strictly increasing."
        )

    # =================================================
    # 1. Reconstruct lambda_h
    # =================================================

    multiplier = reconstruct_p1_multiplier_l2(
        nodes,
        U,
        A,
        F,
        method=multiplier_method,
    )

    mu = multiplier["mu"]

    mu_full = np.zeros(
        len(nodes),
        dtype=float,
    )

    mu_full[1:-1] = mu

    # =================================================
    # 2. Jump contribution
    # =================================================

    U_full = np.zeros(
        len(nodes),
        dtype=float,
    )

    U_full[1:-1] = U

    # Since u_h is P1,
    # u_h' is constant on each element.
    slopes = (
        np.diff(U_full)
        / h
    )

    jumps = np.zeros(
        len(nodes),
        dtype=float,
    )

    jumps[1:-1] = (
        slopes[1:]
        - slopes[:-1]
    )

    jump_eta_squared = (
        0.5
        * h
        * (
            jumps[:-1] ** 2
            + jumps[1:] ** 2
        )
    )

    # =================================================
    # 3. Volume residual contribution
    # =================================================

    gauss_points, gauss_weights = leggauss(
        quadrature_order
    )

    residual_l2_squared = np.zeros(
        n_elements,
        dtype=float,
    )

    volume_eta_squared = np.zeros(
        n_elements,
        dtype=float,
    )

    for element in range(n_elements):

        x_left = nodes[element]
        x_right = nodes[element + 1]
        h_K = h[element]

        # Map Gauss points
        # [-1,1] -> K.
        x_quad = (
            0.5
            * (x_left + x_right)
            +
            0.5
            * h_K
            * gauss_points
        )

        w_quad = (
            0.5
            * h_K
            * gauss_weights
        )

        xi = (
            x_quad - x_left
        ) / h_K

        basis_values = p1_basis(xi)

        # lambda_h restricted to K.
        lambda_values = (
            mu_full[element]
            * basis_values[0]
            +
            mu_full[element + 1]
            * basis_values[1]
        )

        f_values = np.asarray(
            regular_load(x_quad),
            dtype=float,
        )

        # P1:
        #
        # u_h'' = 0
        #
        # inside every element.
        residual_values = (
            f_values
            + lambda_values
        )

        residual_l2_squared[element] = np.sum(
            w_quad
            * residual_values ** 2
        )

        volume_eta_squared[element] = (
            h_K ** 2
            * residual_l2_squared[element]
        )

    # =================================================
    # 4. Full residual indicator
    # =================================================

    eta_squared = (
        volume_eta_squared
        + jump_eta_squared
    )

    midpoints = (
        0.5
        * (
            nodes[:-1]
            + nodes[1:]
        )
    )

    return {
        "midpoints": midpoints,
        "h": h,
        "slopes": slopes,
        "jumps": jumps,

        "Lambda": multiplier["Lambda"],
        "mu": mu,
        "multiplier_method": multiplier_method,

        "residual_l2_squared": (
            residual_l2_squared
        ),

        "volume_eta_squared": (
            volume_eta_squared
        ),

        "jump_eta_squared": (
            jump_eta_squared
        ),

        "eta_squared": eta_squared,
        "eta": np.sqrt(eta_squared),
    }


from src.fem_p2 import (
    assemble_mass_matrix as assemble_mass_matrix_p2,
    p2_basis,
)


def reconstruct_p2_multiplier_l2(
    nodes,
    U,
    A,
    F,
):
    """
    Construct the P2 L2-Riesz representative of the
    discrete multiplier functional.

    Discrete KKT multiplier:

        Lambda = A U - F.

    L2-Riesz representation:

        M mu = Lambda,

    followed by

        lambda_h = sum_i mu_i phi_i.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    A = np.asarray(
        A,
        dtype=float,
    )

    F = np.asarray(
        F,
        dtype=float,
    )

    n_elements = len(nodes) - 1
    n_interior = 2 * n_elements - 1

    if len(U) != n_interior:
        raise ValueError(
            "U must contain one value "
            "for every interior P2 DOF."
        )

    if len(F) != n_interior:
        raise ValueError(
            "F must contain one value "
            "for every interior P2 DOF."
        )

    if A.shape != (
        n_interior,
        n_interior,
    ):
        raise ValueError(
            "A has incompatible dimensions."
        )

    # Discrete multiplier functional.
    Lambda = A @ U - F

    # L2 Riesz map.
    M = assemble_mass_matrix_p2(
        nodes
    )

    # Solve
    #
    #     M mu = Lambda.
    #
    # np.linalg.solve does not form M^{-1}.
    mu = np.linalg.solve(
        M,
        Lambda,
    )

    return {
        "Lambda": Lambda,
        "mu": mu,
        "mass_matrix": M,
    }


def compute_p2_residual_indicator(
    nodes,
    U,
    A,
    F,
    regular_load,
    quadrature_order=4,
):
    r"""
    Compute

        eta_K^2
        =
        h_K^2
        ||f_reg + lambda_h + u_h''||_L2(K)^2

        +

        h_K/2
        (J_left^2 + J_right^2).

    For P2,

        u_h''|_K

    is constant on every element.
    """
    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    U = np.asarray(
        U,
        dtype=float,
    )

    h = np.diff(nodes)

    n_elements = len(h)
    n_interior = 2 * n_elements - 1

    if len(U) != n_interior:
        raise ValueError(
            "For homogeneous Dirichlet P2 FEM, "
            "len(U) must equal 2*n_elements - 1."
        )

    if np.any(h <= 0.0):
        raise ValueError(
            "Mesh nodes must be strictly increasing."
        )

    # =================================================
    # 1. L2-Riesz multiplier
    # =================================================

    multiplier = reconstruct_p2_multiplier_l2(
        nodes,
        U,
        A,
        F,
    )

    mu = multiplier[
        "mu"
    ]

    # Full global P2 vectors:
    #
    # x_0, midpoint_0, x_1, midpoint_1, ..., x_M.
    #
    # Boundary values are zero.
    U_full = np.zeros(
        2 * n_elements + 1,
        dtype=float,
    )

    U_full[1:-1] = U

    mu_full = np.zeros(
        2 * n_elements + 1,
        dtype=float,
    )

    mu_full[1:-1] = mu

    # =================================================
    # 2. u_h'' and one-sided derivatives
    # =================================================

    second_derivatives = np.zeros(
        n_elements,
        dtype=float,
    )

    derivative_left = np.zeros(
        n_elements,
        dtype=float,
    )

    derivative_right = np.zeros(
        n_elements,
        dtype=float,
    )

    # P2 basis derivatives on the reference element:
    #
    # at xi = 0:
    #
    #     (-3, 4, -1)
    #
    # at xi = 1:
    #
    #     (1, -4, 3)
    dphi_left_reference = np.array(
        [-3.0, 4.0, -1.0]
    )

    dphi_right_reference = np.array(
        [1.0, -4.0, 3.0]
    )

    for element in range(
        n_elements
    ):
        h_K = h[element]

        local_values = U_full[
            2 * element:
            2 * element + 3
        ]

        # P2 basis second derivatives with respect
        # to xi are
        #
        #     4, -8, 4.
        #
        # Since
        #
        #     d^2/dx^2 = h_K^{-2} d^2/dxi^2,
        #
        # we get
        #
        # u_h''|_K
        #
        # =
        #
        # 4/h_K^2
        # (U_left - 2 U_mid + U_right).
        second_derivatives[element] = (
            4.0
            * (
                local_values[0]
                - 2.0 * local_values[1]
                + local_values[2]
            )
            / h_K ** 2
        )

        derivative_left[element] = (
            dphi_left_reference
            @ local_values
        ) / h_K

        derivative_right[element] = (
            dphi_right_reference
            @ local_values
        ) / h_K

    # =================================================
    # 3. Interface derivative jumps
    # =================================================

    # Convention:
    #
    # J_i
    #
    # =
    #
    # u_h'(x_i^+)
    # -
    # u_h'(x_i^-).
    #
    # IMPORTANT:
    # P2 midpoints are NOT interfaces.
    # There is no jump term there.
    jumps = np.zeros(
        len(nodes),
        dtype=float,
    )

    jumps[1:-1] = (
        derivative_left[1:]
        - derivative_right[:-1]
    )

    jump_eta_squared = (
        0.5
        * h
        * (
            jumps[:-1] ** 2
            + jumps[1:] ** 2
        )
    )

    # =================================================
    # 4. Volume residual
    # =================================================

    gauss_points, gauss_weights = leggauss(
        quadrature_order
    )

    residual_l2_squared = np.zeros(
        n_elements,
        dtype=float,
    )

    volume_eta_squared = np.zeros(
        n_elements,
        dtype=float,
    )

    for element in range(
        n_elements
    ):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h_K = h[element]

        # Gauss points:
        #
        # [-1,1] -> K.
        x_quad = (
            0.5 * (x_left + x_right)
            +
            0.5 * h_K * gauss_points
        )

        w_quad = (
            0.5
            * h_K
            * gauss_weights
        )

        xi = (
            x_quad - x_left
        ) / h_K

        basis_values = p2_basis(
            xi
        )

        local_mu = mu_full[
            2 * element:
            2 * element + 3
        ]

        # lambda_h restricted to K.
        lambda_values = (
            local_mu
            @ basis_values
        )

        f_values = np.asarray(
            regular_load(
                x_quad
            ),
            dtype=float,
        )

        # The full P2 strong residual.
        residual_values = (
            f_values
            + lambda_values
            + second_derivatives[element]
        )

        residual_l2_squared[element] = np.sum(
            w_quad
            * residual_values ** 2
        )

        volume_eta_squared[element] = (
            h_K ** 2
            * residual_l2_squared[element]
        )

    # =================================================
    # 5. Total local indicator
    # =================================================

    eta_squared = (
        volume_eta_squared
        + jump_eta_squared
    )

    midpoints = (
        0.5
        * (
            nodes[:-1]
            + nodes[1:]
        )
    )

    return {
        "midpoints": midpoints,
        "h": h,

        "jumps": jumps,

        "derivative_left": (
            derivative_left
        ),

        "derivative_right": (
            derivative_right
        ),

        "second_derivatives": (
            second_derivatives
        ),

        "Lambda": multiplier[
            "Lambda"
        ],

        "mu": mu,

        "residual_l2_squared": (
            residual_l2_squared
        ),

        "volume_eta_squared": (
            volume_eta_squared
        ),

        "jump_eta_squared": (
            jump_eta_squared
        ),

        "eta_squared": eta_squared,
        "eta": np.sqrt(
            eta_squared
        ),
    }