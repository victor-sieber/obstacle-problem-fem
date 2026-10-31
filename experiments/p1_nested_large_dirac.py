
"""Large-scale P1 nested PDAS benchmark."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from src.p1_nested_large import (
    assemble_p1_dirac,
    matvec,
    prolongate_p1,
    solve_pdas_p1,
)


ROOT = Path(__file__).resolve().parents[1]


def run(target, maxiter, initialization):
    levels = [10, 100, 1000, 10000, 100000]
    levels = [m for m in levels if m < target] + [target]

    coarse_nodes = None
    coarse_solution = None
    coarse_multiplier = None

    rows = []

    for m in levels:
        nodes = np.linspace(0.0, 1.0, m + 1)

        # --------------------------------------------
        # Assembly
        # --------------------------------------------
        start = perf_counter()

        lower, diagonal, upper, F = assemble_p1_dirac(
            nodes
        )

        assembly_time = perf_counter() - start

        # --------------------------------------------
        # Nested initialization
        # --------------------------------------------
        start = perf_counter()

        V0 = None
        Lambda0 = None

        if coarse_nodes is not None:
            V0 = prolongate_p1(
                coarse_nodes,
                coarse_solution,
                nodes,
            )

            if initialization == "transfer-dual":
                Lambda0 = prolongate_p1(
                    coarse_nodes,
                    coarse_multiplier,
                    nodes,
                )

        interpolation_time = perf_counter() - start

        # --------------------------------------------
        # Solve
        # --------------------------------------------
        start = perf_counter()

        try:
            U, info = solve_pdas_p1(
                nodes,
                lower,
                diagonal,
                upper,
                F,
                initial_solution=V0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
                verbose=(m >= 1_000_000),
            )

        except (RuntimeError, ArithmeticError) as exc:
            print(
                f"FAILED at M={m:,}: {exc}",
                flush=True,
            )
            break

        solve_time = perf_counter() - start

        row = {
            "elements": m,
            "dofs": m - 1,
            "pdas_iterations": info["iterations"],
            "assembly_seconds": assembly_time,
            "interpolation_seconds": interpolation_time,
            "pdas_seconds": solve_time,
            "active_dofs": info["active_dofs"],
            "min_gap": info["min_gap"],
            "min_multiplier": info["min_multiplier"],
            "max_complementarity": info["max_complementarity"],
        }

        rows.append(row)

        print(
            f"M={m:>8,d} "
            f"DOFs={m-1:>8,d} "
            f"PDAS={info['iterations']:>3d} "
            f"assembly={assembly_time:.3f}s "
            f"interpolate={interpolation_time:.3f}s "
            f"solve={solve_time:.3f}s "
            f"min_gap={info['min_gap']:.2e} "
            f"min_lambda={info['min_multiplier']:.2e} "
            f"max_compl={info['max_complementarity']:.2e}",
            flush=True,
        )

        # Store coarse solution and its discrete multiplier.
        coarse_nodes = nodes
        coarse_solution = U

        coarse_multiplier = (
            matvec(lower, diagonal, upper, U) - F
        )

    # --------------------------------------------
    # Save successful levels
    # --------------------------------------------
    path = (
        ROOT
        / "results"
        / f"p1_nested_dirac_M{target}.csv"
    )

    path.parent.mkdir(exist_ok=True)

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

    print(f"Saved converged levels: {path}")

    return bool(
        rows and rows[-1]["elements"] == target
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--elements",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--initialization",
        choices=("transfer-dual", "recompute"),
        default="transfer-dual",
    )

    args = parser.parse_args()

    if args.elements < 10 or args.maxiter < 1:
        parser.error("Invalid benchmark parameters")

    success = run(
        args.elements,
        args.maxiter,
        args.initialization,
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
