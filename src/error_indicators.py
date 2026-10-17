import numpy as np


def compute_p1_jump_indicator(nodes, U):
    """
    Compute the local P1 slope-jump indicator

        s_i = (U_{i+1} - U_i) / h_i,

        J_i = s_i - s_{i-1},

        eta_{K_i}^2
            = h_i / 2 * (J_i^2 + J_{i+1}^2).

    Parameters
    ----------
    nodes : array_like
        Mesh nodes x_0 < ... < x_M, including boundary nodes.

    U : array_like
        Interior P1 degrees of freedom.
        Homogeneous Dirichlet values at x_0 and x_M are inserted here.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "midpoints": element midpoints
        - "h": element lengths
        - "slopes": P1 derivative on each element
        - "jumps": slope jumps at mesh nodes
        - "eta_squared": squared local indicators
        - "eta": local indicators
    """

    nodes = np.asarray(nodes, dtype=float)
    U = np.asarray(U, dtype=float)

    if nodes.ndim != 1 or U.ndim != 1:
        raise ValueError("nodes and U must be one-dimensional arrays.")

    if len(U) != len(nodes) - 2:
        raise ValueError(
            "For homogeneous Dirichlet P1 FEM, "
            "len(U) must equal len(nodes) - 2."
        )

    h = np.diff(nodes)

    if np.any(h <= 0.0):
        raise ValueError("Mesh nodes must be strictly increasing.")

    # Full nodal vector:
    #
    # u_h(x_0) = 0,
    # u_h(x_M) = 0,
    #
    # because u_h belongs to H_0^1.
    U_full = np.zeros(len(nodes), dtype=float)
    U_full[1:-1] = U

    # P1 derivative on element K_i = [x_i, x_{i+1}]:
    #
    # s_i = u_h'|_{K_i}
    #     = (U_{i+1} - U_i) / h_i.
    slopes = np.diff(U_full) / h

    # J_i = s_i - s_{i-1}
    #
    # There is no inter-element jump at the physical boundary,
    # so we set J_0 = J_M = 0.
    jumps = np.zeros(len(nodes), dtype=float)
    jumps[1:-1] = slopes[1:] - slopes[:-1]

    # eta_{K_i}^2
    #     = h_i / 2 * (J_i^2 + J_{i+1}^2).
    eta_squared = 0.5 * h * (
        jumps[:-1] ** 2
        + jumps[1:] ** 2
    )

    eta = np.sqrt(eta_squared)

    midpoints = 0.5 * (
        nodes[:-1]
        + nodes[1:]
    )

    return {
        "midpoints": midpoints,
        "h": h,
        "slopes": slopes,
        "jumps": jumps,
        "eta_squared": eta_squared,
        "eta": eta,
    }