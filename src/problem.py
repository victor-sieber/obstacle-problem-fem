import numpy as np


def load(x):
    """
    Load function f(x) = -1 on [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return -np.ones_like(x)


def obstacle(x, alpha=0.2):
    """
    Obstacle

        psi(x) = x(1-x) - (3/2) alpha^2,

    with 0 < alpha < 1/2.
    """
    x = np.asarray(x, dtype=float)
    return x * (1.0 - x) - 1.5 * alpha**2


def exact_solution(x, alpha=0.2):
    """
    Exact solution of the benchmark obstacle problem.
    """
    x = np.asarray(x, dtype=float)

    left = 0.5 * x**2 + (1.0 - 3.0 * alpha) * x

    middle = obstacle(x, alpha)

    y = 1.0 - x
    right = 0.5 * y**2 + (1.0 - 3.0 * alpha) * y

    return np.where(
        x < alpha,
        left,
        np.where(x <= 1.0 - alpha, middle, right),
    )


def exact_derivative(x, alpha=0.2):
    """
    Derivative of the exact solution.
    """
    x = np.asarray(x, dtype=float)

    left = x + (1.0 - 3.0 * alpha)

    middle = 1.0 - 2.0 * x

    right = -(1.0 - x) - (1.0 - 3.0 * alpha)

    return np.where(
        x < alpha,
        left,
        np.where(x <= 1.0 - alpha, middle, right),
    )