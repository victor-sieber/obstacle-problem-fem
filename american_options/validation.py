from __future__ import annotations

import numpy as np


def american_put_crr(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    n_steps: int = 5000,
) -> float:
    """
    Price an American put using a Cox-Ross-Rubinstein binomial tree.

    This provides an independent benchmark for validating the FEM solver.
    """

    if spot <= 0.0:
        raise ValueError("spot must be positive")

    if strike <= 0.0:
        raise ValueError("strike must be positive")

    if volatility <= 0.0:
        raise ValueError("volatility must be positive")

    if maturity <= 0.0:
        raise ValueError("maturity must be positive")

    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    dt = maturity / n_steps

    up = np.exp(
        volatility * np.sqrt(dt)
    )

    down = 1.0 / up

    growth = np.exp(
        rate * dt
    )

    probability_up = (
        growth - down
    ) / (
        up - down
    )

    if not 0.0 < probability_up < 1.0:
        raise ValueError(
            "CRR risk-neutral probability is outside (0, 1). "
            "Increase n_steps or check the model parameters."
        )

    discount = np.exp(
        -rate * dt
    )

    # ---------------------------------------------------------------
    # Terminal stock prices
    #
    # After N steps, a path with j upward moves has stock price
    #
    # S_N,j = S_0 * up^j * down^(N-j).
    # ---------------------------------------------------------------

    j = np.arange(
        n_steps + 1
    )

    stock = (
        spot
        * up**j
        * down**(
            n_steps - j
        )
    )

    # At maturity an American put is worth its payoff.
    values = np.maximum(
        strike - stock,
        0.0,
    )

    # ---------------------------------------------------------------
    # Backward induction
    #
    # At every node compare:
    #
    # 1. continuation value
    # 2. immediate exercise value
    #
    # American value = max(continuation, exercise).
    # ---------------------------------------------------------------

    for step in range(
        n_steps - 1,
        -1,
        -1,
    ):

        continuation = (
            discount
            * (
                probability_up
                * values[1:]
                +
                (
                    1.0
                    - probability_up
                )
                * values[:-1]
            )
        )

        j = np.arange(
            step + 1
        )

        stock = (
            spot
            * up**j
            * down**(
                step - j
            )
        )

        exercise = np.maximum(
            strike - stock,
            0.0,
        )

        values = np.maximum(
            continuation,
            exercise,
        )

    return float(
        values[0]
    )