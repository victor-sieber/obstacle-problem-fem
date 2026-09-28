import numpy as np
from numpy.polynomial.legendre import leggauss


def create_mesh(n_elements, left=0.0, right=1.0):
    """
    Create a uniform 1D mesh with n_elements finite elements.

    Parameters
    ----------
    n_elements : int
        Number of finite elements.
    left, right : float
        Domain endpoints.

    Returns
    -------
    nodes : ndarray
        Array of the n_elements + 1 mesh vertices.
    """
    return np.linspace(left, right, n_elements + 1)


def p1_basis(xi):
    """
    Evaluate the two local P1 Lagrange basis functions on [0, 1].

        phi_0(xi) = 1 - xi,
        phi_1(xi) = xi.

    Parameters
    ----------
    xi : float or ndarray
        Reference-element coordinate(s).

    Returns
    -------
    values : ndarray
        Basis values. The first axis indexes the two local basis functions.
    """
    xi = np.asarray(xi, dtype=float)
    return np.array([1.0 - xi, xi])


def p1_basis_derivative(xi):
    """
    Evaluate derivatives of the local P1 basis with respect to xi.

        phi_0'(xi) = -1,
        phi_1'(xi) =  1.
    """
    xi = np.asarray(xi, dtype=float)
    return np.array([
        -np.ones_like(xi),
        np.ones_like(xi),
    ])


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
    Assemble the P1 stiffness matrix

        A_ij = integral phi_i'(x) phi_j'(x) dx

    with homogeneous Dirichlet boundary conditions.

    The local matrix is computed from the reference P1 basis by
    Gauss-Legendre quadrature. For P1, the derivative products are
    constant, so the quadrature is exact.
    """
    n_elements = len(nodes) - 1
    n_interior = n_elements - 1

    A = np.zeros((n_interior, n_interior))

    xi, weights = _reference_gauss_rule(quadrature_order)
    basis_derivatives = p1_basis_derivative(xi)

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

        global_nodes = [element, element + 1]

        for i_local, i_global in enumerate(global_nodes):
            if 1 <= i_global <= n_elements - 1:
                for j_local, j_global in enumerate(global_nodes):
                    if 1 <= j_global <= n_elements - 1:
                        A[i_global - 1, j_global - 1] += A_local[
                            i_local, j_local
                        ]

    return A


def assemble_load_vector(nodes, f, quadrature_order=2):
    """
    Assemble the P1 load vector

        F_i = integral f(x) phi_i(x) dx

    by Gauss-Legendre quadrature on each element.
    """
    n_elements = len(nodes) - 1
    n_interior = n_elements - 1

    F = np.zeros(n_interior)

    gauss_points, gauss_weights = leggauss(quadrature_order)

    for element in range(n_elements):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        # Map Gauss points from [-1, 1] to the physical element.
        x_quad = (
            0.5 * (x_left + x_right)
            + 0.5 * h * gauss_points
        )
        w_quad = 0.5 * h * gauss_weights

        # Map physical quadrature points back to xi in [0, 1]
        # and evaluate the same P1 basis used everywhere else.
        xi = (x_quad - x_left) / h
        basis_values = p1_basis(xi)

        f_values = f(x_quad)

        F_local = basis_values @ (w_quad * f_values)

        global_nodes = [element, element + 1]

        for i_local, i_global in enumerate(global_nodes):
            if 1 <= i_global <= n_elements - 1:
                F[i_global - 1] += F_local[i_local]

    return F


def _full_nodal_values(nodes, interior_values):
    """
    Insert the homogeneous Dirichlet values at the boundary nodes.
    """
    interior_values = np.asarray(interior_values, dtype=float)

    if len(interior_values) != len(nodes) - 2:
        raise ValueError(
            "interior_values must contain one value "
            "for each interior mesh node."
        )

    return np.concatenate(([0.0], interior_values, [0.0]))


def _element_indices(nodes, x):
    """
    Find the element containing each evaluation point.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < nodes[0]) or np.any(x > nodes[-1]):
        raise ValueError(
            "Evaluation points must lie inside the mesh domain."
        )

    element = np.searchsorted(nodes, x, side="right") - 1

    return np.clip(
        element,
        0,
        len(nodes) - 2,
    )


def evaluate_fem_solution(nodes, interior_values, x):
    """
    Evaluate the P1 FEM function explicitly from its local basis.

    On the element [x_e, x_{e+1}],

        xi = (x - x_e) / h_e,

    and

        u_h(x)
        =
        U_e phi_0(xi)
        +
        U_{e+1} phi_1(xi).
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

    basis_values = p1_basis(xi)

    return (
        full_values[element] * basis_values[0]
        + full_values[element + 1] * basis_values[1]
    )


def evaluate_fem_derivative(nodes, interior_values, x):
    """
    Evaluate the derivative of the P1 FEM function.

    Since

        d/dx = (1/h_e) d/dxi,

    the reference-basis derivatives are divided by h_e.
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

    basis_derivatives = (
        p1_basis_derivative(xi) / h
    )

    return (
        full_values[element] * basis_derivatives[0]
        + full_values[element + 1] * basis_derivatives[1]
    )


def interpolate_fem_solution(nodes, interior_values, x):
    """
    Backward-compatible name for evaluate_fem_solution.

    Evaluation is performed explicitly using the P1 basis.
    np.interp is not used.
    """
    return evaluate_fem_solution(
        nodes,
        interior_values,
        x,
    )