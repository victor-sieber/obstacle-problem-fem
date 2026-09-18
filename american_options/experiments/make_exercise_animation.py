from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from american_options.american_option import price_american_put_fem


# Benchmark parameters
SPOT = 100.0
STRIKE = 100.0
RATE = 0.05
VOLATILITY = 0.20

# This animation is explanatory rather than a convergence experiment.
# A moderately fine discretization keeps generation reasonably fast.
N_ELEMENTS = 100
N_TIME_STEPS = 100

# We move from very close to maturity backward toward today.
N_FRAMES = 30
MIN_MATURITY = 0.02
MAX_MATURITY = 1.00


def compute_frames():
    """
    Compute American put solutions for several times to maturity.

    Each frame is produced by the deterministic FEM pricing engine.
    """

    maturities = np.linspace(
        MIN_MATURITY,
        MAX_MATURITY,
        N_FRAMES,
    )

    frames = []

    print("Generating American put animation frames...")

    for index, maturity in enumerate(
        maturities,
        start=1,
    ):

        print(
            f"Frame {index:02d}/{N_FRAMES}: "
            f"T = {maturity:.3f}"
        )

        result = price_american_put_fem(
            spot=SPOT,
            strike=STRIKE,
            rate=RATE,
            volatility=VOLATILITY,
            maturity=float(maturity),
            n_elements=N_ELEMENTS,
            n_time_steps=N_TIME_STEPS,
        )

        frames.append(
            {
                "maturity": float(maturity),
                "stock_grid": np.asarray(
                    result["stock_grid"],
                    dtype=float,
                ),
                "option_value": np.asarray(
                    result["option_value"],
                    dtype=float,
                ),
                "payoff": np.asarray(
                    result["payoff"],
                    dtype=float,
                ),
                "exercise_boundary": float(
                    result["exercise_boundary"][-1]
                ),
                "american_price": float(
                    result["american_price"]
                ),
            }
        )

    return frames


def main():
    frames = compute_frames()

    stock_grid = frames[0]["stock_grid"]

    max_value = max(
        np.max(frame["option_value"])
        for frame in frames
    )

    max_payoff = max(
        np.max(frame["payoff"])
        for frame in frames
    )

    y_max = 1.08 * max(
        max_value,
        max_payoff,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    value_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        label="American put value",
    )

    payoff_line, = ax.plot(
        [],
        [],
        "--",
        linewidth=2.0,
        label="Immediate-exercise payoff",
    )

    boundary_point, = ax.plot(
        [],
        [],
        "o",
        markersize=8,
        label="Early-exercise boundary",
    )

    boundary_line = ax.axvline(
        x=0.0,
        linestyle=":",
        linewidth=1.5,
        visible=False,
    )

    info_text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.set_xlim(
        float(stock_grid[0]),
        float(stock_grid[-1]),
    )

    ax.set_ylim(
        -1.0,
        float(y_max),
    )

    ax.set_xlabel(
        "Underlying price S"
    )

    ax.set_ylabel(
        "Option value"
    )

    ax.set_title(
        "American put as a time-dependent obstacle problem"
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        alpha=0.25
    )

    def update(frame_index):
        frame = frames[frame_index]

        stock = frame["stock_grid"]
        value = frame["option_value"]
        payoff = frame["payoff"]
        boundary = frame["exercise_boundary"]

        value_line.set_data(
            stock,
            value,
        )

        payoff_line.set_data(
            stock,
            payoff,
        )

        if np.isfinite(boundary):

            boundary_payoff = max(
                STRIKE - boundary,
                0.0,
            )

            boundary_point.set_data(
                [boundary],
                [boundary_payoff],
            )

            boundary_line.set_xdata(
                [boundary, boundary]
            )

            boundary_line.set_visible(
                True
            )

            boundary_text = (
                f"{boundary:.2f}"
            )

        else:

            boundary_point.set_data(
                [],
                [],
            )

            boundary_line.set_visible(
                False
            )

            boundary_text = (
                "not detected"
            )

        info_text.set_text(
            f"Time to maturity: "
            f"{frame['maturity']:.2f} years\n"
            f"American value at S0=100: "
            f"{frame['american_price']:.2f}\n"
            f"Exercise boundary: "
            f"{boundary_text}"
        )

        return (
            value_line,
            payoff_line,
            boundary_point,
            boundary_line,
            info_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=140,
        blit=False,
    )

    output_path = (
        ROOT
        / "figures"
        / "american_put_evolution.gif"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    animation.save(
        output_path,
        writer=PillowWriter(
            fps=7
        ),
    )

    plt.close(
        fig
    )

    print()
    print(
        f"Saved animation to:"
    )
    print(
        output_path
    )


if __name__ == "__main__":
    main()