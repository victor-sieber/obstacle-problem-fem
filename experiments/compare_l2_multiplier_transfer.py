
"""P1/P2 nested PDAS comparison using median CPU time.

Dirac obstacle problem:
    f = -6 - 2*delta_(1/3) - delta_(2/3)
    psi = -1

Uniform meshes:
    M = 10, 100, 1000, 10000

Compare:
    1. Cold PDAS
    2. Original nested PDAS
    3. L2-Riesz multiplier nested PDAS

Timing:
    Median CPU time over repeated solves.
    Includes initialization and PDAS.
    Excludes FEM assembly and previous coarse-grid solves.
"""

import argparse
import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    NullFormatter,
)

from src.multiplier_l2_transfer import (
    transfer_l2_multiplier_p1,
    transfer_l2_multiplier_p2,
)

from src.p1_nested_large import (
    assemble_p1_dirac,
    matvec,
    prolongate_p1,
    solve_pdas_p1,
)

from src.p2_nested_large import (
    assemble_p2_dirac,
    join_p2,
    p2_matvec,
    prolongate_p2,
    solve_pdas_p2,
)


ROOT = Path(__file__).resolve().parents[1]

LEVELS = (10, 100, 1000, 10000)

METHODS = ("cold", "original", "l2")

LABELS = {
    "cold": "cold PDAS",
    "original": "original nested PDAS",
    "l2": r"nested $L^2$ multiplier",
}

FIELDNAMES = [
    "degree",
    "elements",
    "dofs",
    "method",
    "repetitions",
    "pdas_iterations",
    "median_initialization_cpu_seconds",
    "median_solver_cpu_seconds",
    "median_total_cpu_seconds",
    "max_difference_to_cold",
    "min_gap",
    "min_multiplier",
    "max_complementarity",
]


# ============================================================
# 1. Prepare P1 or P2 FEM problem
# ============================================================

def prepare_problem(degree, nodes, maxiter):
    """
    Assemble the FEM problem once on the current mesh.

    Return:
        solve           - existing PDAS algorithm
        multiplier      - Lambda = A U - F
        prolongate      - primal FEM interpolation
        l2_transfer     - L2-Riesz multiplier transfer
    """

    if degree == 1:

        lower, diagonal, upper, F = assemble_p1_dirac(
            nodes
        )

        def solve(U0, Lambda0):
            return solve_pdas_p1(
                nodes,
                lower,
                diagonal,
                upper,
                F,
                initial_solution=U0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
            )

        def multiplier(U):
            return (
                matvec(
                    lower,
                    diagonal,
                    upper,
                    U,
                )
                - F
            )

        return (
            solve,
            multiplier,
            prolongate_p1,
            transfer_l2_multiplier_p1,
        )

    elif degree == 2:

        Fv, Fm = assemble_p2_dirac(nodes)
        F = join_p2(Fv, Fm)

        def solve(U0, Lambda0):
            return solve_pdas_p2(
                nodes,
                Fv,
                Fm,
                initial_solution=U0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
            )

        def multiplier(U):
            return p2_matvec(nodes, U) - F

        return (
            solve,
            multiplier,
            prolongate_p2,
            transfer_l2_multiplier_p2,
        )

    else:
        raise ValueError("degree must be 1 or 2")


# ============================================================
# 2. Benchmark one initialization strategy
# ============================================================

def benchmark(
    method,
    previous,
    nodes,
    solve,
    prolongate,
    l2_transfer,
    repetitions,
):
    """
    Repeat the same discrete solve and report median CPU time.

    The FEM matrices are already assembled.

    Initializations:
        cold:
            Solve unconstrained system inside PDAS.

        original:
            U0 = I_h U_H
            Lambda0 = A_h U0 - F_h (inside PDAS).

        l2:
            U0 = I_h U_H
            M_H mu_H = Lambda_H
            Lambda0 = M_h I_h mu_H
    """

    initialization_times = []
    solver_times = []
    total_times = []

    reference_U = None
    reference_iterations = None
    final_info = None

    for repeat in range(repetitions):

        # --------------------------------------------------
        # Initialization CPU timing
        # --------------------------------------------------

        start = process_time()

        if method == "cold":

            U0 = None
            Lambda0 = None

        else:

            if previous is None:
                raise ValueError(
                    "Nested PDAS requires a coarse-grid solution"
                )

            coarse_nodes, coarse_U, coarse_Lambda = previous

            # Primal transfer:
            #
            # U_h^0 = I_h U_H

            U0 = prolongate(
                coarse_nodes,
                coarse_U,
                nodes,
            )

            if method == "original":

                # Original nested initialization.
                #
                # Lambda_h^0 = A_h U_h^0 - F_h
                #
                # The existing PDAS solver computes this
                # internally when Lambda0 is None.

                Lambda0 = None

            elif method == "l2":

                # L2-Riesz multiplier transfer:
                #
                # M_H mu_H = Lambda_H
                #
                # mu_h = I_h mu_H
                #
                # Lambda_h^0 = M_h mu_h

                Lambda0 = l2_transfer(
                    coarse_nodes,
                    coarse_Lambda,
                    nodes,
                )

            else:
                raise ValueError(
                    f"Unknown initialization method: {method}"
                )

        after_initialization = process_time()

        # --------------------------------------------------
        # Solve using the unchanged PDAS equations
        # --------------------------------------------------

        U, info = solve(
            U0,
            Lambda0,
        )

        end = process_time()

        # --------------------------------------------------
        # Record CPU times
        # --------------------------------------------------

        initialization_times.append(
            after_initialization - start
        )

        solver_times.append(
            end - after_initialization
        )

        total_times.append(
            end - start
        )

        # --------------------------------------------------
        # Verify deterministic behavior
        # --------------------------------------------------

        if reference_U is None:

            reference_U = U.copy()
            reference_iterations = info["iterations"]

        else:

            same_iterations = (
                info["iterations"] == reference_iterations
            )

            same_solution = np.allclose(
                U,
                reference_U,
                rtol=0.0,
                atol=1e-10,
            )

            if not (same_iterations and same_solution):
                raise RuntimeError(
                    "Repeated PDAS solves produced "
                    "inconsistent results."
                )

        final_info = info

        print(
            f"    run {repeat + 1}/{repetitions}: "
            f"CPU={end - start:.5f}s, "
            f"PDAS={info['iterations']}",
            flush=True,
        )

    # ------------------------------------------------------
    # Median CPU times across all repetitions
    # ------------------------------------------------------

    return {
        "U": reference_U,
        "info": final_info,

        "initialization_cpu": float(
            np.median(initialization_times)
        ),

        "solver_cpu": float(
            np.median(solver_times)
        ),

        "total_cpu": float(
            np.median(total_times)
        ),
    }


# ============================================================
# 3. Save numerical results
# ============================================================

def save_csv(rows, path):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. Plot P1 and P2 comparisons
# ============================================================

def plot_results(rows, path, repetitions):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    axes[0, 0].set_title(
        "PDAS iteration count"
    )

    axes[0, 1].set_title(
        "Median CPU time"
    )

    for i, degree in enumerate((1, 2)):

        for method in METHODS:

            subset = sorted(
                [
                    row for row in rows
                    if (
                        row["degree"] == degree
                        and row["method"] == method
                    )
                ],
                key=lambda row: row["elements"],
            )

            if not subset:
                continue

            elements = [
                row["elements"]
                for row in subset
            ]

            iterations = [
                row["pdas_iterations"]
                for row in subset
            ]

            cpu_times = [
                row["median_total_cpu_seconds"]
                for row in subset
            ]

            # ----------------------------------------------
            # Left column: PDAS iterations
            # ----------------------------------------------

            axes[i, 0].loglog(
                elements,
                iterations,
                "o-",
                label=LABELS[method],
            )

            # ----------------------------------------------
            # Right column: median CPU time
            # ----------------------------------------------

            axes[i, 1].loglog(
                elements,
                cpu_times,
                "o-",
                label=LABELS[method],
            )

        axes[i, 0].set_ylabel(
            f"P{degree}: PDAS iterations"
        )

        axes[i, 1].set_ylabel(
            f"P{degree}: CPU time [s]"
        )

        # ----------------------------------------------
        # Common plot formatting
        # ----------------------------------------------

        for ax in axes[i]:

            ax.set_xlabel(
                "number of elements M"
            )

            ax.set_xscale("log")

            ax.xaxis.set_major_locator(
                FixedLocator(LEVELS)
            )

            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, pos:
                    f"{int(x):,}" if x in LEVELS else ""
                )
            )

            ax.xaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.grid(
                True,
                which="both",
                alpha=0.23,
            )

        axes[i, 0].legend(
            fontsize=9,
            loc="upper left",
        )

    fig.suptitle(
        "Dirac obstacle problem: "
        "PDAS initialization comparison "
        f"(median CPU time, {repetitions} runs)"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 5. Main experiment
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-elements",
        type=int,
        choices=(1000, 10000),
        default=10000,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=6000,
    )

    args = parser.parse_args()

    if args.repeats < 1 or args.maxiter < 1:
        parser.error(
            "--repeats and --maxiter must be positive"
        )

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"

    results_dir.mkdir(
        exist_ok=True
    )

    figures_dir.mkdir(
        exist_ok=True
    )

    csv_path = (
        results_dir
        / "p1_p2_l2_multiplier_transfer.csv"
    )

    figure_path = (
        figures_dir
        / "p1_p2_l2_multiplier_transfer.png"
    )

    rows = []

    # ------------------------------------------------------
    # Run P1 and P2
    # ------------------------------------------------------

    for degree in (1, 2):

        print(
            f"\nP{degree}: CPU benchmark",
            flush=True,
        )

        # Each nested method has its own previous
        # solution and multiplier.

        previous = {
            "original": None,
            "l2": None,
        }

        for M in LEVELS:

            if M > args.max_elements:
                continue

            nodes = np.linspace(
                0.0,
                1.0,
                M + 1,
            )

            # ------------------------------------------------
            # Assemble FEM problem once.
            #
            # Assembly is excluded from CPU measurements.
            # ------------------------------------------------

            solve, multiplier, prolongate, l2_transfer = (
                prepare_problem(
                    degree,
                    nodes,
                    args.maxiter,
                )
            )

            cold_U = None

            # ------------------------------------------------
            # Compare three initialization methods
            # ------------------------------------------------

            for method in METHODS:

                # Initial mesh has no previous solution.
                # Therefore only cold PDAS is available.

                if M == LEVELS[0] and method != "cold":
                    continue

                print(
                    f"P{degree}, M={M:,}, "
                    f"{LABELS[method]}",
                    flush=True,
                )

                result = benchmark(
                    method=method,
                    previous=previous.get(method),
                    nodes=nodes,
                    solve=solve,
                    prolongate=prolongate,
                    l2_transfer=l2_transfer,
                    repetitions=args.repeats,
                )

                U = result["U"]
                info = result["info"]

                # --------------------------------------------
                # Store cold solution for validation
                # --------------------------------------------

                if method == "cold":
                    cold_U = U

                # --------------------------------------------
                # Check identical discrete solutions
                # --------------------------------------------

                difference = float(
                    np.max(
                        np.abs(U - cold_U)
                    )
                )

                if (
                    not np.isfinite(difference)
                    or difference > 1e-8
                ):
                    raise RuntimeError(
                        f"P{degree}, M={M}, {method}: "
                        f"solution differs from cold "
                        f"by {difference:.3e}"
                    )

                # --------------------------------------------
                # Store solution and algebraic multiplier
                # for nested initialization on the next mesh
                # --------------------------------------------

                state = (
                    nodes.copy(),
                    U.copy(),
                    multiplier(U),
                )

                if M == LEVELS[0]:

                    previous["original"] = state
                    previous["l2"] = state

                elif method in previous:

                    previous[method] = state

                # --------------------------------------------
                # Store results
                # --------------------------------------------

                row = {
                    "degree": degree,
                    "elements": M,
                    "dofs": len(U),
                    "method": method,
                    "repetitions": args.repeats,
                    "pdas_iterations": info["iterations"],

                    "median_initialization_cpu_seconds": (
                        result["initialization_cpu"]
                    ),

                    "median_solver_cpu_seconds": (
                        result["solver_cpu"]
                    ),

                    "median_total_cpu_seconds": (
                        result["total_cpu"]
                    ),

                    "max_difference_to_cold": difference,
                    "min_gap": info["min_gap"],
                    "min_multiplier": info["min_multiplier"],
                    "max_complementarity": (
                        info["max_complementarity"]
                    ),
                }

                rows.append(row)

                # Save successful results immediately.
                save_csv(
                    rows,
                    csv_path,
                )

                print(
                    f"  Median total CPU="
                    f"{result['total_cpu']:.5f}s, "
                    f"PDAS={info['iterations']}, "
                    f"max difference={difference:.2e}",
                    flush=True,
                )

    # ------------------------------------------------------
    # Generate final figure
    # ------------------------------------------------------

    plot_results(
        rows,
        figure_path,
        args.repeats,
    )

    print()
    print(
        f"Saved: {csv_path.relative_to(ROOT)}"
    )

    print(
        f"Saved: {figure_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
