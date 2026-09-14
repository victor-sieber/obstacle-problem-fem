from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT),
)


from american_options.american_option import price_american_put_fem
from american_options.validation import american_put_crr


def main():

    # ---------------------------------------------------------------
    # Model parameters
    # ---------------------------------------------------------------

    spot = 100.0
    strike = 100.0
    rate = 0.05
    volatility = 0.20
    maturity = 1.0

    # ---------------------------------------------------------------
    # Independent CRR reference
    # ---------------------------------------------------------------

    crr_steps = 5000

    print(
        f"Computing CRR reference with {crr_steps} steps..."
    )

    reference_price = american_put_crr(
        spot=spot,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        n_steps=crr_steps,
    )

    print(
        f"CRR reference price: {reference_price:.8f}"
    )

    print()

    # ---------------------------------------------------------------
    # Joint spatial / temporal refinement
    #
    # We increase both:
    #
    #     number of P1 elements
    #
    # and
    #
    #     number of implicit-Euler time steps.
    #
    # This is a stabilization study rather than an attempt to isolate
    # the spatial and temporal convergence orders separately.
    # ---------------------------------------------------------------

    resolutions = [
        50,
        100,
        200,
        400,
    ]

    results = []

    for resolution in resolutions:

        print(
            f"Running FEM with "
            f"{resolution} elements and "
            f"{resolution} time steps..."
        )

        fem_result = price_american_put_fem(
            spot=spot,
            strike=strike,
            rate=rate,
            volatility=volatility,
            maturity=maturity,
            n_elements=resolution,
            n_time_steps=resolution,
        )

        fem_price = fem_result[
            "american_price"
        ]

        absolute_error = abs(
            fem_price
            - reference_price
        )

        results.append(
            {
                "n_elements": resolution,
                "n_time_steps": resolution,
                "fem_price": fem_price,
                "absolute_error": absolute_error,
                "min_obstacle_gap": fem_result[
                    "min_obstacle_gap"
                ],
                "max_solver_iterations": fem_result[
                    "max_solver_iterations"
                ],
            }
        )

    # ---------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------

    print()
    print(
        "FEM convergence against CRR reference"
    )

    print(
        "-" * 86
    )

    print(
        f"{'Elements':>10} "
        f"{'Time steps':>12} "
        f"{'FEM price':>14} "
        f"{'Abs. error':>14} "
        f"{'Min gap':>14} "
        f"{'Max iter':>10}"
    )

    print(
        "-" * 86
    )

    for row in results:

        print(
            f"{row['n_elements']:>10d} "
            f"{row['n_time_steps']:>12d} "
            f"{row['fem_price']:>14.8f} "
            f"{row['absolute_error']:>14.6e} "
            f"{row['min_obstacle_gap']:>14.6e} "
            f"{row['max_solver_iterations']:>10d}"
        )

    print(
        "-" * 86
    )

    print(
        f"CRR reference: {reference_price:.8f}"
    )

    # ---------------------------------------------------------------
    # Output directories
    # ---------------------------------------------------------------

    figures_dir = (
        ROOT
        / "figures"
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    results_dir = (
        ROOT
        / "american_options"
        / "results"
    )

    results_dir.mkdir(
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Save numerical results as CSV
    # ---------------------------------------------------------------

    csv_path = (
        results_dir
        / "convergence_validation.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "n_elements",
                "n_time_steps",
                "fem_price",
                "absolute_error",
                "min_obstacle_gap",
                "max_solver_iterations",
            ],
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ---------------------------------------------------------------
    # Convert results to NumPy arrays for plotting
    # ---------------------------------------------------------------

    n_values = np.array(
        [
            row["n_elements"]
            for row in results
        ]
    )

    fem_prices = np.array(
        [
            row["fem_price"]
            for row in results
        ]
    )

    errors = np.array(
        [
            row["absolute_error"]
            for row in results
        ]
    )

    # ---------------------------------------------------------------
    # Figure 1:
    # FEM price stabilization
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        n_values,
        fem_prices,
        marker="o",
        label="P1 FEM",
    )

    plt.axhline(
        reference_price,
        linestyle="--",
        label="CRR reference",
    )

    plt.xlabel(
        "Number of spatial elements"
    )

    plt.ylabel(
        "American put price"
    )

    plt.title(
        "American put price under joint mesh/time refinement"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "american_put_convergence.png",
        dpi=200,
    )

    plt.show()

    # ---------------------------------------------------------------
    # Figure 2:
    # Absolute difference from CRR reference
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.loglog(
        n_values,
        errors,
        marker="o",
    )

    plt.xlabel(
        "Number of spatial elements"
    )

    plt.ylabel(
        "Absolute price difference"
    )

    plt.title(
        "FEM difference from high-resolution CRR benchmark"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "american_put_validation_error.png",
        dpi=200,
    )

    plt.show()

    print()
    print(
        f"Saved results to: {csv_path}"
    )


if __name__ == "__main__":
    main()