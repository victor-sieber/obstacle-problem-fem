from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from american_options.american_option import price_american_put_fem

def main():

    result = price_american_put_fem(
        spot=100.0,
        strike=100.0,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        n_elements=200,
        n_time_steps=200,
    )

    print(
        "American put price (FEM): "
        f"{result['american_price']:.6f}"
    )

    print(
        "European put price (Black-Scholes): "
        f"{result['european_price']:.6f}"
    )

    print(
        "Intrinsic value: "
        f"{result['intrinsic_value']:.6f}"
    )

    print(
        "Early-exercise premium over European put: "
        f"{result['american_price'] - result['european_price']:.6f}"
    )

    print(
        "Minimum obstacle gap: "
        f"{result['min_obstacle_gap']:.3e}"
    )

    print(
        "Maximum L-BFGS-B iterations in one time step: "
        f"{result['max_solver_iterations']}"
    )

    figures = ROOT / "figures"

    figures.mkdir(
        exist_ok=True
    )

    stock = result[
        "stock_grid"
    ]

    value = result[
        "option_value"
    ]

    payoff = result[
        "payoff"
    ]

    # ---------------------------------------------------------------
    # Option value figure
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        stock,
        value,
        label="American put value",
    )

    plt.plot(
        stock,
        payoff,
        "--",
        label="Immediate exercise payoff",
    )

    plt.axvline(
        100.0,
        linestyle=":",
        label="Current spot $S_0$",
    )

    plt.xlim(
        20.0,
        180.0,
    )

    plt.xlabel(
        "Stock price S"
    )

    plt.ylabel(
        "Option value"
    )

    plt.title(
        "American put: P1 FEM + bound-constrained optimization"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figures
        / "american_put_value.png",
        dpi=200,
    )

    plt.show()

    # ---------------------------------------------------------------
    # Exercise boundary figure
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        result["tau"],
        result["exercise_boundary"],
        marker="o",
        markersize=2,
    )

    plt.xlabel(
        "Time to maturity"
    )

    plt.ylabel(
        "Approximate exercise boundary"
    )

    plt.title(
        "American put early-exercise boundary"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figures
        / "american_put_boundary.png",
        dpi=200,
    )

    plt.show()


if __name__ == "__main__":
    main()