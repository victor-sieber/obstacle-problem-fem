from __future__ import annotations

from typing import Any

import numpy as np

from american_options.american_option import price_american_put_fem


MODEL_PARAMETERS = {
    "spot",
    "strike",
    "rate",
    "volatility",
    "maturity",
}


DEFAULT_PARAMETERS = {
    "spot": 100.0,
    "strike": 100.0,
    "rate": 0.05,
    "volatility": 0.20,
    "maturity": 1.0,
}


def _validate_parameters(
    parameters: dict[str, float],
) -> None:
    """
    Validate the financial parameters accepted by the analytics layer.
    """

    unknown = set(parameters) - MODEL_PARAMETERS

    if unknown:
        raise ValueError(
            f"Unknown parameter(s): {sorted(unknown)}"
        )

    if parameters["spot"] <= 0:
        raise ValueError(
            "spot must be positive"
        )

    if parameters["strike"] <= 0:
        raise ValueError(
            "strike must be positive"
        )

    if parameters["volatility"] <= 0:
        raise ValueError(
            "volatility must be positive"
        )

    if parameters["maturity"] <= 0:
        raise ValueError(
            "maturity must be positive"
        )


def _complete_parameters(
    parameters: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Fill missing parameters with the default benchmark values.
    """

    result = DEFAULT_PARAMETERS.copy()

    if parameters is not None:
        result.update(
            parameters
        )

    _validate_parameters(
        result
    )

    return result


def price_option(
    parameters: dict[str, float] | None = None,
    n_elements: int = 200,
    n_time_steps: int = 200,
) -> dict[str, Any]:
    """
    Price one American put and return structured numerical results.
    """

    params = _complete_parameters(
        parameters
    )

    result = price_american_put_fem(
        **params,
        n_elements=n_elements,
        n_time_steps=n_time_steps,
    )

    early_exercise_premium = (
        result["american_price"]
        - result["european_price"]
    )

    return {
        "parameters": params,
        "american_price": float(
            result["american_price"]
        ),
        "european_price": float(
            result["european_price"]
        ),
        "intrinsic_value": float(
            result["intrinsic_value"]
        ),
        "early_exercise_premium": float(
            early_exercise_premium
        ),
        "exercise_boundary_today": float(
            result["exercise_boundary"][-1]
        ),
        "min_obstacle_gap": float(
            result["min_obstacle_gap"]
        ),
        "numerical_method": {
            "spatial_method": "P1 finite elements",
            "time_method": "implicit Euler",
            "optimizer": "L-BFGS-B",
            "n_elements": n_elements,
            "n_time_steps": n_time_steps,
        },
    }


def compare_scenarios(
    base_parameters: dict[str, float],
    shocked_parameters: dict[str, float],
    n_elements: int = 200,
    n_time_steps: int = 200,
) -> dict[str, Any]:
    """
    Compare a base scenario with a shocked scenario.

    Example:
    base volatility = 20%
    shocked volatility = 30%
    """

    base = _complete_parameters(
        base_parameters
    )

    shocked = base.copy()

    shocked.update(
        shocked_parameters
    )

    _validate_parameters(
        shocked
    )

    base_result = price_option(
        base,
        n_elements=n_elements,
        n_time_steps=n_time_steps,
    )

    shocked_result = price_option(
        shocked,
        n_elements=n_elements,
        n_time_steps=n_time_steps,
    )

    price_change = (
        shocked_result["american_price"]
        - base_result["american_price"]
    )

    if base_result["american_price"] != 0:
        relative_change = (
            price_change
            / base_result["american_price"]
        )
    else:
        relative_change = np.nan

    boundary_change = (
        shocked_result["exercise_boundary_today"]
        - base_result["exercise_boundary_today"]
    )

    return {
        "base": base_result,
        "scenario": shocked_result,
        "price_change": float(
            price_change
        ),
        "relative_price_change": float(
            relative_change
        ),
        "exercise_boundary_change": float(
            boundary_change
        ),
    }


def sweep_parameter(
    parameter: str,
    values: list[float],
    base_parameters: dict[str, float] | None = None,
    n_elements: int = 200,
    n_time_steps: int = 200,
) -> dict[str, Any]:
    """
    Reprice the option over several values of one model parameter.
    """

    if parameter not in MODEL_PARAMETERS:
        raise ValueError(
            f"parameter must be one of {sorted(MODEL_PARAMETERS)}"
        )

    if len(values) == 0:
        raise ValueError(
            "values must contain at least one value"
        )

    base = _complete_parameters(
        base_parameters
    )

    rows = []

    for value in values:

        scenario = base.copy()

        scenario[parameter] = float(
            value
        )

        _validate_parameters(
            scenario
        )

        result = price_option(
            scenario,
            n_elements=n_elements,
            n_time_steps=n_time_steps,
        )

        rows.append(
            {
                parameter: float(
                    value
                ),
                "american_price": result[
                    "american_price"
                ],
                "european_price": result[
                    "european_price"
                ],
                "early_exercise_premium": result[
                    "early_exercise_premium"
                ],
                "exercise_boundary_today": result[
                    "exercise_boundary_today"
                ],
            }
        )

    return {
        "parameter": parameter,
        "base_parameters": base,
        "results": rows,
    }


def analyze_exercise_boundary(
    parameters: dict[str, float] | None = None,
    n_elements: int = 200,
    n_time_steps: int = 200,
) -> dict[str, Any]:
    """
    Return the complete estimated early-exercise boundary.
    """

    params = _complete_parameters(
        parameters
    )

    result = price_american_put_fem(
        **params,
        n_elements=n_elements,
        n_time_steps=n_time_steps,
    )

    tau = np.asarray(
        result["tau"]
    )

    boundary = np.asarray(
        result["exercise_boundary"]
    )

    finite = np.isfinite(
        boundary
    )

    return {
        "parameters": params,
        "time_to_maturity": tau.tolist(),
        "exercise_boundary": boundary.tolist(),
        "boundary_today": float(
            boundary[-1]
        ),
        "minimum_finite_boundary": float(
            np.min(
                boundary[finite]
            )
        ),
        "maximum_finite_boundary": float(
            np.max(
                boundary[finite]
            )
        ),
    }


def get_model_information() -> dict[str, Any]:
    """
    Return deterministic information about the numerical pricing engine.
    """

    return {
        "model": "Black-Scholes American put",
        "pricing_problem": (
            "Optimal stopping formulated as "
            "a time-dependent obstacle problem"
        ),
        "spatial_discretization": (
            "P1 finite elements"
        ),
        "time_discretization": (
            "implicit Euler"
        ),
        "optimizer": (
            "SciPy L-BFGS-B"
        ),
        "validation": (
            "Validated against a 5000-step "
            "Cox-Ross-Rubinstein American put benchmark"
        ),
        "quantitative_source_of_truth": (
            "All option prices and exercise-boundary "
            "values are produced by the deterministic "
            "numerical solver."
        ),
        "current_limitations": [
            "American puts only",
            "Black-Scholes assumptions",
            "constant volatility",
            "constant risk-free rate",
            "one underlying asset",
            "generic L-BFGS-B optimizer",
        ],
        "planned_extensions": [
            "primal-dual active-set solver",
            "P2 finite elements",
            "higher-dimensional obstacle problems",
        ],
    }