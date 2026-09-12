from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def american_put_payoff(S: np.ndarray, strike: float) -> np.ndarray:
    """American put payoff max(K - S, 0)."""
    return np.maximum(strike - S, 0.0)


def european_put_black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
) -> float:
    """Closed-form European put price, used only as a lower-bound sanity check."""
    d1 = (
        np.log(spot / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))

    d2 = d1 - volatility * np.sqrt(maturity)

    return (
        strike * np.exp(-rate * maturity) * norm.cdf(-d2)
        - spot * norm.cdf(-d1)
    )


def assemble_p1_matrices(
    nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble full P1 mass and stiffness matrices on a 1D mesh.

    The boundary nodes are kept because the American-option problem uses
    non-zero Dirichlet data at the left boundary.
    """
    n_nodes = len(nodes)

    mass = np.zeros((n_nodes, n_nodes))
    stiffness = np.zeros((n_nodes, n_nodes))

    for element in range(n_nodes - 1):
        x_left = nodes[element]
        x_right = nodes[element + 1]
        h = x_right - x_left

        mass_local = (h / 6.0) * np.array(
            [
                [2.0, 1.0],
                [1.0, 2.0],
            ]
        )

        stiffness_local = (1.0 / h) * np.array(
            [
                [1.0, -1.0],
                [-1.0, 1.0],
            ]
        )

        global_nodes = [element, element + 1]

        for i_local, i_global in enumerate(global_nodes):
            for j_local, j_global in enumerate(global_nodes):
                mass[i_global, j_global] += mass_local[i_local, j_local]
                stiffness[i_global, j_global] += stiffness_local[i_local, j_local]

    return mass, stiffness


def price_american_put_fem(
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.20,
    maturity: float = 1.0,
    s_min_ratio: float = 0.10,
    s_max_ratio: float = 4.0,
    n_elements: int = 200,
    n_time_steps: int = 200,
) -> dict:
    """
    Price a one-dimensional American put in the Black-Scholes model.

    Method:
    1. Transform S to log-moneyness x = log(S / K).
    2. Remove the drift and reaction terms so the continuation PDE becomes
       a heat equation.
    3. Discretize space with P1 finite elements.
    4. Discretize time with implicit Euler.
    5. At every time step, solve the resulting lower-bound constrained
       convex quadratic problem with SciPy L-BFGS-B.
    """

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    if volatility <= 0 or maturity <= 0:
        raise ValueError("volatility and maturity must be positive")

    if not 0 < s_min_ratio < 1 < s_max_ratio:
        raise ValueError("require 0 < s_min_ratio < 1 < s_max_ratio")

    # ---------------------------------------------------------------
    # 1. Log-price coordinate
    #
    # x = log(S / K), hence S = K * exp(x).
    # ---------------------------------------------------------------

    x_min = np.log(s_min_ratio)
    x_max = np.log(s_max_ratio)

    x = np.linspace(
        x_min,
        x_max,
        n_elements + 1,
    )

    stock_grid = strike * np.exp(x)

    # ---------------------------------------------------------------
    # 2. Assemble P1 finite-element matrices
    # ---------------------------------------------------------------

    mass, stiffness = assemble_p1_matrices(x)

    # ---------------------------------------------------------------
    # 3. Transform the Black-Scholes PDE
    #
    # In log-price coordinates:
    #
    # v_tau =
    #     0.5*sigma^2*v_xx
    #     + mu*v_x
    #     - r*v
    #
    # where mu = r - 0.5*sigma^2.
    #
    # The exponential transformation below eliminates the
    # first-derivative and reaction terms.
    # ---------------------------------------------------------------

    mu = rate - 0.5 * volatility**2

    alpha = -mu / volatility**2

    beta = (
        0.5 * volatility**2 * alpha**2
        + mu * alpha
        - rate
    )

    diffusion = 0.5 * volatility**2

    # ---------------------------------------------------------------
    # 4. Time discretization
    # ---------------------------------------------------------------

    dt = maturity / n_time_steps

    # Weak form of the transformed heat equation:
    #
    # M u_tau + diffusion * A u = 0.
    #
    # Implicit Euler gives:
    #
    # (M/dt + diffusion*A) u^{n+1}
    #     = (M/dt) u^n.
    # ---------------------------------------------------------------

    mass_over_dt = mass / dt

    system = (
        mass_over_dt
        + diffusion * stiffness
    )

    # Interior and boundary node indices.
    interior = np.arange(1, n_elements)

    boundary = np.array(
        [
            0,
            n_elements,
        ]
    )

    system_interior = system[
        np.ix_(interior, interior)
    ]

    # ---------------------------------------------------------------
    # 5. Payoff / obstacle
    # ---------------------------------------------------------------

    payoff = american_put_payoff(
        stock_grid,
        strike,
    )

    def transformed_obstacle(tau: float) -> np.ndarray:
        """
        Transform the American put obstacle into the u-variable.
        """
        return (
            np.exp(
                -alpha * x
                - beta * tau
            )
            * payoff
        )

    def transformed_boundary_values(
        tau: float,
    ) -> np.ndarray:
        """
        Approximate Black-Scholes boundary conditions.

        At very small S:
            American put ~= K - S

        At very large S:
            American put ~= 0
        """

        value_left = (
            strike
            - stock_grid[0]
        )

        value_right = 0.0

        return np.array(
            [
                np.exp(
                    -alpha * x[0]
                    - beta * tau
                )
                * value_left,

                np.exp(
                    -alpha * x[-1]
                    - beta * tau
                )
                * value_right,
            ]
        )

    # ---------------------------------------------------------------
    # 6. Terminal condition
    #
    # At maturity:
    #
    # V(T,S) = max(K-S, 0).
    # ---------------------------------------------------------------

    u = transformed_obstacle(0.0)

    tau_values = [0.0]

    exercise_boundaries = [
        strike
    ]

    solver_iterations = []

    # ---------------------------------------------------------------
    # 7. March forward in time-to-maturity tau
    #
    # This corresponds to solving the option-pricing problem
    # backwards from maturity toward today.
    # ---------------------------------------------------------------

    for step in range(n_time_steps):

        tau_new = (
            step + 1
        ) * dt

        boundary_new = (
            transformed_boundary_values(
                tau_new
            )
        )

        # -----------------------------------------------------------
        # Interior equation
        #
        # B_II u_I^{n+1}
        #
        # =
        #
        # (M/dt)_{I,:} u^n
        # -
        # B_{I,B} u_B^{n+1}
        #
        # The second term accounts for the known boundary values.
        # -----------------------------------------------------------

        rhs = (
            mass_over_dt[
                interior,
                :
            ]
            @ u
            -
            system[
                np.ix_(
                    interior,
                    boundary,
                )
            ]
            @ boundary_new
        )

        # American option constraint:
        #
        # option value >= immediate exercise payoff.
        lower_bound = (
            transformed_obstacle(
                tau_new
            )[interior]
        )

        # Previous time step is already close to the new solution.
        #
        # max(...) guarantees feasibility.
        initial_guess = np.maximum(
            u[interior],
            lower_bound,
        )

        # -----------------------------------------------------------
        # Convex quadratic energy at the current time step
        #
        # J(U) = 1/2 U^T B U - rhs^T U
        # -----------------------------------------------------------

        def energy(
            interior_values: np.ndarray,
        ) -> float:

            return (
                0.5
                * interior_values
                @ system_interior
                @ interior_values
                -
                rhs
                @ interior_values
            )

        def gradient(
            interior_values: np.ndarray,
        ) -> np.ndarray:

            return (
                system_interior
                @ interior_values
                - rhs
            )

        # -----------------------------------------------------------
        # Solve:
        #
        # min J(U)
        #
        # subject to
        #
        # U >= obstacle.
        # -----------------------------------------------------------

        result = minimize(
            energy,
            initial_guess,
            jac=gradient,
            method="L-BFGS-B",
            bounds=[
                (
                    float(bound),
                    None,
                )
                for bound in lower_bound
            ],
            options={
                "gtol": 1e-8,
                "ftol": 1e-10,
                "maxiter": 2000,
                "maxls": 50,
            },
        )

        if not result.success:
            raise RuntimeError(
                "L-BFGS-B failed at "
                f"time step {step + 1}: "
                f"{result.message}"
            )

        # Reconstruct the complete FEM vector,
        # including boundary values.
        u_new = np.empty_like(u)

        u_new[0] = boundary_new[0]

        u_new[-1] = boundary_new[1]

        u_new[interior] = result.x

        u = u_new

        solver_iterations.append(
            result.nit
        )

        # -----------------------------------------------------------
        # Transform back to the financial option value V.
        # -----------------------------------------------------------

        option_value = (
            np.exp(
                alpha * x
                + beta * tau_new
            )
            * u
        )

        # -----------------------------------------------------------
        # Estimate the early-exercise boundary.
        #
        # Contact means:
        #
        # V ~= payoff.
        # -----------------------------------------------------------

        gap = (
            option_value
            - payoff
        )

        contact_tolerance = (
            1e-5 * strike
        )

        contact = (
            (stock_grid < strike)
            &
            (
                gap
                <= contact_tolerance
            )
        )

        if np.any(contact):
            exercise_boundary = np.max(
                stock_grid[contact]
            )
        else:
            exercise_boundary = np.nan

        tau_values.append(
            tau_new
        )

        exercise_boundaries.append(
            exercise_boundary
        )

    # ---------------------------------------------------------------
    # 8. Final value corresponds to today's option price
    # ---------------------------------------------------------------

    option_value = (
        np.exp(
            alpha * x
            + beta * maturity
        )
        * u
    )

    american_price = float(
        np.interp(
            spot,
            stock_grid,
            option_value,
        )
    )

    european_price = (
        european_put_black_scholes(
            spot,
            strike,
            rate,
            volatility,
            maturity,
        )
    )

    intrinsic_value = max(
        strike - spot,
        0.0,
    )

    return {
        "american_price": american_price,
        "european_price": european_price,
        "intrinsic_value": intrinsic_value,
        "stock_grid": stock_grid,
        "option_value": option_value,
        "payoff": payoff,
        "tau": np.asarray(
            tau_values
        ),
        "exercise_boundary": np.asarray(
            exercise_boundaries
        ),
        "min_obstacle_gap": float(
            np.min(
                option_value
                - payoff
            )
        ),
        "max_solver_iterations": int(
            max(
                solver_iterations
            )
        ),
        "alpha": alpha,
        "beta": beta,
    }