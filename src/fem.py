import numpy as np
from numpy.polynomial.legendre import leggauss


def create_mesh(n_elements, left=0.0, right=1.0):
    """
    Create a uniform 1D mesh.

    Parameters
    ----------
    n_elements : int
        Number of finite elements.
    left, right : float
        Domain endpoints.

    Returns
    -------
    nodes : ndarray
        Array [x_0, ..., x_N].
    """
    return np.linspace(left, right, n_elements + 1)


def assemble_stiffness_matrix(nodes):
    """
    Assemble the stiffness matrix

        A_ij = integral lambda_i'(x) lambda_j'(x) dx

    for continuous piecewise-linear (P1) finite elements with
    homogeneous Dirichlet boundary conditions.
    """
    n_elements = len(nodes) - 1
    n_interior = n_elements - 1

    A = np.zeros((n_interior, n_interior))

    for element in range(n_elements):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        # Local P1 stiffness matrix
        A_local = (1.0 / h) * np.array(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )

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
    Assemble the load vector

        F_i = integral f(x) lambda_i(x) dx

    using Gauss-Legendre quadrature element by element.
    """
    n_elements = len(nodes) - 1
    n_interior = n_elements - 1

    F = np.zeros(n_interior)

    xi, weights = leggauss(quadrature_order)

    for element in range(n_elements):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        # Map Gauss points from [-1,1] to the current element
        x_quad = 0.5 * (x_left + x_right) + 0.5 * h * xi
        w_quad = 0.5 * h * weights

        # Local linear basis functions
        phi_left = (x_right - x_quad) / h
        phi_right = (x_quad - x_left) / h

        basis_values = np.vstack((phi_left, phi_right))

        f_values = f(x_quad)

        F_local = basis_values @ (w_quad * f_values)

        global_nodes = [element, element + 1]

        for i_local, i_global in enumerate(global_nodes):
            if 1 <= i_global <= n_elements - 1:
                F[i_global - 1] += F_local[i_local]

    return F


def interpolate_fem_solution(nodes, interior_values, x):
    """
    Evaluate the piecewise-linear FEM solution at arbitrary points x.
    """
    full_values = np.concatenate(
        ([0.0], interior_values, [0.0])
    )

    return np.interp(x, nodes, full_values)