
"""Compare cold and nested PDAS for P1 and P2.

Uniform Dirac obstacle problem.

Initializations:
    cold          : unconstrained solution
    recompute     : prolongate U_H, recompute Lambda_h
    transfer-dual : prolongate U_H and Lambda_H

Main comparison: M = 10, 100, 1000, 10000.
"""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, ScalarFormatter

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

METHODS = (
    "cold",
    "recompute",
    "transfer-dual",
)

LABELS = {
    "cold": "cold start",
    "recompute": "nested: recomputed multiplier",
    "transfer-dual": "nested: transferred multiplier",
}


def solve_one(
    degree,
    nodes,
    method,
    previous,
    repeats,
    maxiter,
    verbose,
):
    """Solve one discrete obstacle problem.

    Assembly and interpolation are excluded from
    measured PDAS solver time.
    """

    if degree == 1:
        lower, diagonal, upper, F = assemble_p1_dirac(
            nodes
        )

        V0 = None
        Lambda0 = None

        if method != "cold":
            old_nodes, old_U, old_Lambda = previous

            V0 = prolongate_p1(
                old_nodes, old_U, nodes
            )

            if method == "transfer-dual":
                Lambda0 = prolongate_p1(
                    old_nodes, old_Lambda, nodes
                )

        def solve():
            return solve_pdas_p1(
                nodes,
                lower,
                diagonal,
                upper,
                F,
                initial_solution=V0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
                verbose=verbose,
            )

        def get_multiplier(U):
            return (
                matvec(lower, diagonal, upper, U)
                - F
            )

    else:
        Fv, Fm = assemble_p2_dirac(nodes)
        F = join_p2(Fv, Fm)

        V0 = None
        Lambda0 = None

        if method != "cold":
            old_nodes, old_U, old_Lambda = previous

            V0 = prolongate_p2(
                old_nodes, old_U, nodes
            )

            if method == "transfer-dual":
                Lambda0 = prolongate_p2(
                    old_nodes, old_Lambda, nodes
                )

        def solve():
            return solve_pdas_p2(
                nodes,
                Fv,
                Fm,
                initial_solution=V0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
                verbose=verbose,
            )

        def get_multiplier(U):
            return p2_matvec(nodes, U) - F

    times = []

    for _ in range(repeats):
        start = perf_counter()
        U, info = solve()
        times.append(perf_counter() - start)

    state = (
        nodes.copy(),
        U.copy(),
        get_multiplier(U),
    )

    return U, info, float(np.median(times)), state


def collect_results(
    rows,
    max_elements,
    repeats,
    maxiter,
    verbose_large,
):
    """Compare initializations on identical uniform meshes."""

    for degree in (1, 2):
        print(
            f"\nP{degree}: cold versus nested PDAS",
            flush=True,
        )

        previous = {
            "recompute": None,
            "transfer-dual": None,
        }

        for M in LEVELS:
            if M > max_elements:
                continue

            nodes = np.linspace(0.0, 1.0, M + 1)
            computed = {}

            for method in METHODS:

                # The first mesh has no coarse-grid solution.
                if method != "cold" and (
                    M == LEVELS[0]
                    or previous[method] is None
                ):
                    continue

                print(
                    f"START P{degree}, M={M:,}, "
                    f"{LABELS[method]}",
                    flush=True,
                )

                try:
                    result = solve_one(
                        degree,
                        nodes,
                        method,
                        previous.get(method),
                        repeats,
                        maxiter,
                        verbose=(
                            verbose_large and M >= 10000
                        ),
                    )

                except (RuntimeError, ArithmeticError) as exc:
                    print(
                        f"  DID NOT CONVERGE: {exc}",
                        flush=True,
                    )

                    rows.append({
                        "degree": degree,
                        "elements": M,
                        "dofs": (
                            M - 1
                            if degree == 1
                            else 2 * M - 1
                        ),
                        "initialization": method,
                        "status": "not converged",
                        "pdas_iterations": "",
                        "solver_seconds": "",
                        "max_difference_to_cold": "",
                        "min_gap": "",
                        "min_multiplier": "",
                        "max_complementarity": "",
                    })

                    continue

                U, info, seconds, state = result
                computed[method] = result

                # Store the solution for the next mesh.
                if method in previous:
                    previous[method] = state

                if M == LEVELS[0] and method == "cold":
                    previous["recompute"] = state
                    previous["transfer-dual"] = state

                # Compare solutions on the same mesh.
                baseline = computed.get("cold")

                difference = (
                    float(
                        np.max(np.abs(U - baseline[0]))
                    )
                    if baseline is not None
                    else None
                )

                if difference is not None:
                    print(
                        f"  DONE: iterations={info['iterations']}, "
                        f"solve={seconds:.5f}s, "
                        f"max|U-U_cold|={difference:.3e}",
                        flush=True,
                    )
                else:
                    print(
                        f"  DONE: iterations={info['iterations']}, "
                        f"solve={seconds:.5f}s, "
                        "cold reference unavailable",
                        flush=True,
                    )

                rows.append({
                    "degree": degree,
                    "elements": M,
                    "dofs": (
                        M - 1
                        if degree == 1
                        else 2 * M - 1
                    ),
                    "initialization": method,
                    "status": "converged",
                    "pdas_iterations": info["iterations"],
                    "solver_seconds": seconds,
                    "max_difference_to_cold": (
                        difference
                        if difference is not None
                        else ""
                    ),
                    "min_gap": info["min_gap"],
                    "min_multiplier": info["min_multiplier"],
                    "max_complementarity": (
                        info["max_complementarity"]
                    ),
                })


def save_csv(rows, path):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows, path, max_elements):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    axes[0, 0].set_title("PDAS iterations")
    axes[0, 1].set_title("PDAS solver wall time")

    for i, degree in enumerate((1, 2)):
        for method in METHODS:

            subset = [
                row for row in rows
                if row["degree"] == degree
                and row["initialization"] == method
                and row["status"] == "converged"
            ]

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

            seconds = [
                row["solver_seconds"]
                for row in subset
            ]

            axes[i, 0].semilogx(
                elements,
                iterations,
                "o-",
                label=LABELS[method],
            )

            axes[i, 1].loglog(
                elements,
                seconds,
                "o-",
                label=LABELS[method],
            )

        axes[i, 0].set_ylabel(
            f"P{degree}: iterations"
        )

        axes[i, 1].set_ylabel(
            f"P{degree}: time [s]"
        )

        for ax in axes[i]:
            ax.set_xlabel(
                "number of elements M"
            )

            ax.set_xscale("log")

            ticks = [
                M for M in LEVELS
                if M <= max_elements
            ]

            ax.xaxis.set_major_locator(
                FixedLocator(ticks)
            )

            ax.xaxis.set_major_formatter(
                ScalarFormatter()
            )

            ax.grid(
                True,
                which="both",
                alpha=0.25,
            )

        axes[i, 0].legend(fontsize=8)

    fig.suptitle(
        "Dirac obstacle problem: "
        f"cold vs nested PDAS, M up to {max_elements:,}"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--verbose-large",
        action="store_true",
    )

    args = parser.parse_args()

    if args.maxiter < 1 or args.repeats < 1:
        parser.error(
            "--maxiter and --repeats must be positive"
        )

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    rows = []

    try:
        collect_results(
            rows,
            args.max_elements,
            args.repeats,
            args.maxiter,
            args.verbose_large,
        )

    finally:
        # Preserve completed measurements even if the
        # experiment is interrupted.
        if rows:
            csv_path = (
                results_dir
                / f"p1_p2_nested_initializations_M{args.max_elements}.csv"
            )

            figure_path = (
                figures_dir
                / f"p1_p2_nested_initializations_M{args.max_elements}.png"
            )

            save_csv(rows, csv_path)

            plot_comparison(
                rows,
                figure_path,
                args.max_elements,
            )

            print(f"\nSaved: {csv_path}")
            print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
