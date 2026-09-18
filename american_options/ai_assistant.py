from __future__ import annotations

import json
import math
from typing import Any

from ollama import chat

from american_options.analytics import (
    analyze_exercise_boundary,
    compare_scenarios,
    get_model_information,
    price_option,
    sweep_parameter,
)


MODEL = "qwen3:8b"


SYSTEM_PROMPT = """
You are an analytics assistant for an experimental American option
pricing project.

You have access to deterministic numerical tools built around a
finite-element American put pricing engine.

Rules:

1. For ALL quantitative option prices, scenario comparisons,
   parameter sweeps, or exercise-boundary values, use a tool.

2. Never invent or mentally estimate an option price if a tool can
   compute it.

3. Treat tool outputs as the quantitative source of truth.

4. Clearly distinguish:
   - numerical results produced by tools,
   - qualitative financial interpretation.

5. The current numerical engine supports American put options under
   Black-Scholes assumptions with:
   - one underlying asset,
   - constant volatility,
   - constant risk-free rate.

6. Do not claim support for unsupported option types or models.

7. If a tool returns an error because an input is invalid, explain
   the error instead of inventing a result.

8. Be concise and explain results clearly to a finance user who may
   not know finite element methods.

9. This is an educational analytical prototype, not investment advice.

10. Do not infer that a lower early-exercise boundary means early
    exercise is more likely. For an American put, a lower boundary
    means the underlying must fall further before immediate exercise
    becomes optimal.

11. When reporting intrinsic value, describe it as the intrinsic
    value at the current input spot. Do not make statements about
    the underlying price at expiration.

12. The numerical optimizer solves a bound-constrained convex
    quadratic optimization problem at each time step. Do not describe
    it as solving a nonlinear system.

13. Be conservative with qualitative financial explanations.
    If a conclusion is not directly supported by a tool result or
    the model information, report the numerical change without
    inventing a causal interpretation.

14. When explaining an exercise-boundary change, interpret it in
    terms of the trade-off between immediate exercise and continuation
    value. For an American put, a lower exercise boundary means the
    stock must fall further before immediate exercise becomes optimal.

15. Do not claim that higher volatility makes early exercise more
    likely. Higher volatility generally increases continuation value
    and can lower the American put exercise boundary.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "price_option",
            "description": (
                "Price one American put using the deterministic "
                "finite-element pricing engine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot": {"type": "number"},
                            "strike": {"type": "number"},
                            "rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "maturity": {"type": "number"},
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_scenarios",
            "description": (
                "Compare a base American put scenario with a shocked "
                "scenario."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "base_parameters": {
                        "type": "object",
                        "properties": {
                            "spot": {"type": "number"},
                            "strike": {"type": "number"},
                            "rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "maturity": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                    "shocked_parameters": {
                        "type": "object",
                        "properties": {
                            "spot": {"type": "number"},
                            "strike": {"type": "number"},
                            "rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "maturity": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "base_parameters",
                    "shocked_parameters",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sweep_parameter",
            "description": (
                "Calculate American put values for several values "
                "of one model parameter."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "enum": [
                            "spot",
                            "strike",
                            "rate",
                            "volatility",
                            "maturity",
                        ],
                    },
                    "values": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                    },
                    "base_parameters": {
                        "type": "object",
                        "properties": {
                            "spot": {"type": "number"},
                            "strike": {"type": "number"},
                            "rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "maturity": {"type": "number"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "parameter",
                    "values",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_exercise_boundary",
            "description": (
                "Calculate the estimated early-exercise boundary "
                "for an American put."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spot": {"type": "number"},
                            "strike": {"type": "number"},
                            "rate": {"type": "number"},
                            "volatility": {"type": "number"},
                            "maturity": {"type": "number"},
                        },
                        "additionalProperties": False,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_model_information",
            "description": (
                "Return authoritative information about the model, "
                "numerical method, validation and limitations."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


FUNCTIONS = {
    "price_option": price_option,
    "compare_scenarios": compare_scenarios,
    "sweep_parameter": sweep_parameter,
    "analyze_exercise_boundary": analyze_exercise_boundary,
    "get_model_information": get_model_information,
}


def _make_json_safe(value: Any) -> Any:
    """
    Convert numerical results into standard JSON-safe values.
    """

    if isinstance(value, dict):
        return {
            key: _make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _make_json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _make_json_safe(item)
            for item in value
        ]

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def execute_tool(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """
    Execute one deterministic analytics tool.
    """

    function = FUNCTIONS.get(name)

    if function is None:
        return {
            "status": "error",
            "error": f"Unknown tool: {name}",
        }

    try:
        result = function(
            **arguments
        )

        return {
            "status": "success",
            "result": _make_json_safe(
                result
            ),
        }

    except Exception as error:
        return {
            "status": "error",
            "error": str(error),
        }


def ask(
    question: str,
    verbose: bool = True,
) -> str:
    """
    Ask the local LLM a question.

    The LLM may choose deterministic tools. Tool outputs are then
    returned to the model until it can produce a final answer.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    while True:

        response = chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            think=False,
        )

        messages.append(
            response.message
        )

        tool_calls = (
            response.message.tool_calls
            or []
        )

        if not tool_calls:
            return response.message.content

        for tool_call in tool_calls:

            name = tool_call.function.name

            arguments = (
                tool_call.function.arguments
                or {}
            )

            if verbose:
                print(
                    f"\n[tool call] {name}"
                )
                print(
                    f"[arguments] {arguments}"
                )

            result = execute_tool(
                name,
                arguments,
            )

            if verbose:
                print(
                    f"[tool status] "
                    f"{result['status']}"
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(
                        result
                    ),
                }
            )