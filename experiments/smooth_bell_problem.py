
"""Exact manufactured smooth obstacle problem on (0,1)."""

import numpy as np

ALPHA = 1.0 / 3.0
BETA = 2.0 / 3.0
POWER = 6


def exact_solution(x):
    x = np.asarray(x, dtype=float)
    t = x - 0.5
    return -np.sin(np.pi * x) * np.exp(-6.0 * t**2)


def exact_derivative(x):
    x = np.asarray(x, dtype=float)
    t = x - 0.5

    return np.exp(-6.0 * t**2) * (
        12.0 * t * np.sin(np.pi * x)
        - np.pi * np.cos(np.pi * x)
    )


def exact_second_derivative(x):
    x = np.asarray(x, dtype=float)
    t = x - 0.5

    return np.exp(-6.0 * t**2) * (
        (12.0 + np.pi**2 - 144.0 * t**2)
        * np.sin(np.pi * x)
        + 24.0 * np.pi * t * np.cos(np.pi * x)
    )


def obstacle(x):
    """C^5 obstacle equal to u on [1/3, 2/3], c=1."""
    x = np.asarray(x, dtype=float)

    distance_scaled = np.maximum(
        0.0,
        np.maximum(1.0 - 3.0*x, 3.0*x - 2.0)
    )

    return exact_solution(x) - distance_scaled**POWER


def exact_multiplier(x):
    """C^5 polynomial bump with maximum 1."""
    x = np.asarray(x, dtype=float)

    bump = np.maximum(
        0.0,
        36.0 * (x - ALPHA) * (BETA - x)
    )

    return bump**POWER


def regular_load(x):
    """Manufactured forcing f = -u'' - lambda; no Dirac."""
    return -exact_second_derivative(x) - exact_multiplier(x)
