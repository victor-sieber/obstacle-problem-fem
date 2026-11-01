
"""Large-scale uniform P2 nested PDAS benchmark."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from src.p2_nested_large import (
    assemble_p2_dirac,
    join_p2,
    p2_matvec,
    prolongate_p2,
    solve_pdas_p2,
)


ROOT = Path(__file__).resolve().parents[1]


def run(target, maxiter, initialization):
    levels = [10, 100, 1000, 10000, 100000]

    levels = [
        m for m in levels if m < target
    ] + [target]

    coarse_nodes = None
    coarse_U = None
    coarse_lambda = None

    rows = []

    for m in levels:
        # --------------------------------------------
        # 1. Assemble FEM problem.
        # --------------------------------------------

        start = perf_counter()

        nodes = np.linspace(0.0, 1.0, m + 1)

        Fv, Fm = assemble_p2_dirac(nodes)

        assembly_time = perf_counter() - start

        # --------------------------------------------
        # 2. Nested initialization.
        # --------------------------------------------

        start = perf_counter()

        V0 = None
        Lambda0 = None

        if coarse_nodes is not None:
            V0 = prolongate_p2(
                coarse_nodes,
                coarse_U,
                nodes,
            )

            if initialization == "transfer-dual":
                Lambda0 = prolongate_p2(
                    coarse_nodes,
                    coarse_lambda,
                    nodes,
                )

        interpolation_time = perf_counter() - start

        # --------------------------------------------
        # 3. PDAS solve.
        # --------------------------------------------

        start = perf_counter()

        try:
            U, info = solve_pdas_p2(
                nodes,
                Fv,
                Fm,
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

        # --------------------------------------------
        # 4. Store results.
        # --------------------------------------------

        row = {
            "elements": m,
            "dofs": 2 * m - 1,
            "pdas_iterations": info["iterations"],
            "assembly_seconds": assembly_time,
            "interpolation_seconds": interpolation_time,
            "pdas_seconds": solve_time,
            "active_dofs": info["active_dofs"],
            "min_gap": info["min_gap"],
            "min_multiplier": info["min_multiplier"],
            "max_complementarity": info[
                "max_complementarity"
            ],
        }

        rows.append(row)

        print(
            f"M={m:>9,d} "
            f"DOFs={2*m-1:>9,d} "
            f"PDAS={info['iterations']:>3d} "
            f"assembly={assembly_time:.3f}s "
            f"interpolate={interpolation_time:.3f}s "
            f"solve={solve_time:.3f}s "
            f"min_gap={info['min_gap']:.2e} "
            f"min_lambda={info['min_multiplier']:.2e} "
            f"max_compl={info['max_complementarity']:.2e}",
            flush=True,
        )

        # Store solution and multiplier for next level.
        coarse_nodes = nodes
        coarse_U = U

        coarse_lambda = (
            p2_matvec(nodes, U) - join_p2(Fv, Fm)
        )

    if not rows:
        return False

    # Use the last converged size in the filename.
    last = rows[-1]["elements"]

    path = (
        ROOT
        / "results"
        / f"p2_nested_dirac_M{last}.csv"
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

    print(
        f"Saved converged levels: {path.relative_to(ROOT)}"
    )

    return last == target


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--elements",
        type=int,
        default=100000,
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
