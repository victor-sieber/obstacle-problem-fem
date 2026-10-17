import numpy as np


def mark_by_maximum(eta, theta=0.5):
    """
    Mark all elements whose local indicator is at least
    theta times the largest indicator:

        eta_K >= theta * max_L eta_L.

    Parameters
    ----------
    eta : array_like
        Local error indicators, one value per element.

    theta : float
        Marking threshold with 0 < theta <= 1.

    Returns
    -------
    marked : ndarray of bool
        Boolean array with one entry per element.
    """

    eta = np.asarray(eta, dtype=float)

    if eta.ndim != 1:
        raise ValueError("eta must be one-dimensional.")

    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must satisfy 0 < theta <= 1.")

    if len(eta) == 0:
        return np.zeros(0, dtype=bool)

    eta_max = np.max(eta)

    if eta_max <= 0.0:
        return np.zeros(len(eta), dtype=bool)

    return eta >= theta * eta_max


def bisect_marked_elements(nodes, marked):
    """
    Bisect every marked element once.

    If

        K_i = [x_i, x_{i+1}]

    is marked, insert its midpoint

        m_i = (x_i + x_{i+1}) / 2.

    Unmarked elements remain unchanged.

    Parameters
    ----------
    nodes : array_like
        Strictly increasing mesh nodes.

    marked : array_like of bool
        One marking value per element.

    Returns
    -------
    new_nodes : ndarray
        Refined, strictly increasing mesh.
    """

    nodes = np.asarray(nodes, dtype=float)
    marked = np.asarray(marked, dtype=bool)

    n_elements = len(nodes) - 1

    if len(marked) != n_elements:
        raise ValueError(
            "marked must contain one value per element."
        )

    if np.any(np.diff(nodes) <= 0.0):
        raise ValueError(
            "Mesh nodes must be strictly increasing."
        )

    new_nodes = [nodes[0]]

    for element in range(n_elements):
        left = nodes[element]
        right = nodes[element + 1]

        if marked[element]:
            midpoint = 0.5 * (left + right)
            new_nodes.append(midpoint)

        new_nodes.append(right)

    return np.asarray(new_nodes, dtype=float)