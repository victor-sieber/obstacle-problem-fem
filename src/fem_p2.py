import numpy as np
from numpy.polynomial.legendre import leggauss


def p2_basis(xi):
    """
    Evaluate the three local P2 Lagrange basis functions on [0, 1].

    The interpolation points are

        xi_0 = 0,  xi_1 = 1/2,  xi_2 = 1.
    """
    xi = np.asarray(xi, dtype=float)

    return np.array([
        2.0 * (xi - 0.5) * (xi - 1.0),
        4.0 * xi * (1.0 - xi),
        2.0 * xi * (xi - 0.5),
    ])


def p2_basis_derivative(xi):
    """
    Evaluate derivatives of the local P2 basis with respect to xi.
    """
    xi = np.asarray(xi, dtype=float)

    return np.array([
        4.0 * xi - 3.0,
        4.0 - 8.0 * xi,
        4.0 * xi - 1.0,
    ])


def create_p2_nodes(nodes):
    """
    Create the global P2 interpolation-point array from the original mesh.

    If the original mesh is

        x_0, x_1, ..., x_M,

    the P2 coordinates are ordered as

        x_0, m_0, x_1, m_1, ..., m_{M-1}, x_M,

    where m_e is the midpoint of element e.
    """
    n_elements = len(nodes) - 1

    p2_nodes = np.empty(
        2 * n_elements + 1,
        dtype=float,
    )

    p2_nodes[0::2] = nodes
    p2_nodes[1::2] = 0.5 * (
        nodes[:-1] + nodes[1:]
    )

    return p2_nodes


def _reference_gauss_rule(quadrature_order):
    """
    Return Gauss-Legendre points and weights on [0, 1].
    """
    points, weights = leggauss(quadrature_order)

    xi = 0.5 * (points + 1.0)
    weights = 0.5 * weights

    return xi, weights


def assemble_stiffness_matrix(nodes, quadrature_order=2):
    """
    Assemble the P2 stiffness matrix

        A_ij = integral phi_i'(x) phi_j'(x) dx

    with homogeneous Dirichlet boundary conditions.

    As in the P1 implementation, the local matrix is built directly
    from the reference-basis derivatives using Gauss quadrature.
    """
    n_elements = len(nodes) - 1
    n_interior = 2 * n_elements - 1

    A = np.zeros(
        (n_interior, n_interior)
    )

    xi, weights = _reference_gauss_rule(
        quadrature_order
    )

    basis_derivatives = p2_basis_derivative(xi)

    for element in range(n_elements):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        # x = x_left + h*xi, so
        # d/dx = (1/h)d/dxi and dx = h dxi.
        A_local = (
            (basis_derivatives * weights)
            @ basis_derivatives.T
        ) / h

        # Local P2 DOFs: left vertex, midpoint, right vertex.
        global_dofs = [
            2 * element,
            2 * element + 1,
            2 * element + 2,
        ]

        for i_local, i_global in enumerate(global_dofs):
            if 1 <= i_global <= 2 * n_elements - 1:
                for j_local, j_global in enumerate(global_dofs):
                    if 1 <= j_global <= 2 * n_elements - 1:
                        A[
                            i_global - 1,
                            j_global - 1,
                        ] += A_local[
                            i_local,
                            j_local,
                        ]

    return A


def assemble_load_vector(nodes, f, quadrature_order=3):
    """
    Assemble the P2 load vector

        F_i = integral f(x) phi_i(x) dx

    by Gauss-Legendre quadrature on each element.
    """
    n_elements = len(nodes) - 1
    n_interior = 2 * n_elements - 1

    F = np.zeros(n_interior)

    xi, weights = _reference_gauss_rule(
        quadrature_order
    )

    basis_values = p2_basis(xi)

    for element in range(n_elements):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        x_quad = x_left + h * xi
        f_values = f(x_quad)

        # dx = h dxi.
        F_local = h * (
            basis_values
            @ (weights * f_values)
        )

        global_dofs = [
            2 * element,
            2 * element + 1,
            2 * element + 2,
        ]

        for i_local, i_global in enumerate(global_dofs):
            if 1 <= i_global <= 2 * n_elements - 1:
                F[i_global - 1] += F_local[i_local]

    return F


def _full_nodal_values(nodes, interior_values):
    """
    Insert homogeneous Dirichlet values at the two boundary vertices.
    """
    interior_values = np.asarray(
        interior_values,
        dtype=float,
    )

    expected = 2 * (len(nodes) - 1) - 1

    if len(interior_values) != expected:
        raise ValueError(
            "interior_values must contain one value for each "
            "interior P2 degree of freedom."
        )

    return np.concatenate(
        ([0.0], interior_values, [0.0])
    )


def _element_indices(nodes, x):
    """
    Find the original mesh element containing each evaluation point.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < nodes[0]) or np.any(x > nodes[-1]):
        raise ValueError(
            "Evaluation points must lie inside the mesh domain."
        )

    element = (
        np.searchsorted(
            nodes,
            x,
            side="right",
        )
        - 1
    )

    return np.clip(
        element,
        0,
        len(nodes) - 2,
    )


def evaluate_fem_solution(nodes, interior_values, x):
    """
    Evaluate the P2 FEM function explicitly from its local basis.

    On element [x_e, x_{e+1}],

        xi = (x - x_e) / h_e,

    and

        u_h(x)
        = U_left phi_0(xi)
        + U_mid phi_1(xi)
        + U_right phi_2(xi).
    """
    x = np.asarray(x, dtype=float)

    full_values = _full_nodal_values(
        nodes,
        interior_values,
    )

    element = _element_indices(
        nodes,
        x,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    xi = (x - x_left) / h

    basis_values = p2_basis(xi)

    left_values = full_values[2 * element]
    midpoint_values = full_values[2 * element + 1]
    right_values = full_values[2 * element + 2]

    return (
        left_values * basis_values[0]
        + midpoint_values * basis_values[1]
        + right_values * basis_values[2]
    )


def evaluate_fem_derivative(nodes, interior_values, x):
    """
    Evaluate the derivative of the P2 FEM function explicitly.
    """
    x = np.asarray(x, dtype=float)

    full_values = _full_nodal_values(
        nodes,
        interior_values,
    )

    element = _element_indices(
        nodes,
        x,
    )

    x_left = nodes[element]
    x_right = nodes[element + 1]
    h = x_right - x_left

    xi = (x - x_left) / h

    # d/dx = (1/h)d/dxi.
    basis_derivatives = (
        p2_basis_derivative(xi) / h
    )

    left_values = full_values[2 * element]
    midpoint_values = full_values[2 * element + 1]
    right_values = full_values[2 * element + 2]

    return (
        left_values * basis_derivatives[0]
        + midpoint_values * basis_derivatives[1]
        + right_values * basis_derivatives[2]
    )