
"""Compare cold, original nested, and L2-Riesz nested PDAS.

The previously computed cold/original nested results are
read from CSV. Only the new L2-initialized solves are rerun.

No raw multiplier-vector transfer is performed here.
"""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    FixedLocator,
    NullFormatter,
    ScalarFormatter,
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

LABELS = {
    "cold": "cold start",
    "recompute": "original nested",
    "l2-riesz": r"nested: $L^2$ multiplier",
}


# ============================================================
# L2-based nested PDAS
# ============================================================

def run_l2_nested(degree, maxiter=1000):
    """Solve uniform nested FEM problems with L2 multiplier transfer."""

    previous = None
    rows = []

    for M in LEVELS:

        nodes = np.linspace(0.0, 1.0, M + 1)

        # --------------------------------------------------
        # 1. Assemble FEM matrices and load.
        # --------------------------------------------------

        if degree == 1:

            lower, diagonal, upper, F = (
                assemble_p1_dirac(nodes)
            )

            def solve(u0, lambda0):
                return solve_pdas_p1(
                    nodes,
                    lower,
                    diagonal,
                    upper,
                    F,
                    initial_solution=u0,
                    initial_multiplier=lambda0,
                    maxiter=maxiter,
                )

            def multiplier(U):
                return (
                    matvec(
                        lower, diagonal, upper, U
                    )
                    - F
                )

            prolongate = prolongate_p1
            transfer_multiplier = transfer_l2_multiplier_p1

        else:

            Fv, Fm = assemble_p2_dirac(nodes)
            F = join_p2(Fv, Fm)

            def solve(u0, lambda0):
                return solve_pdas_p2(
                    nodes,
                    Fv,
                    Fm,
                    initial_solution=u0,
                    initial_multiplier=lambda0,
                    maxiter=maxiter,
                )

            def multiplier(U):
                return (
                    p2_matvec(nodes, U) - F
                )

            prolongate = prolongate_p2
            transfer_multiplier = transfer_l2_multiplier_p2

        # --------------------------------------------------
        # 2. Initial mesh: ordinary cold PDAS.
        # --------------------------------------------------

        if previous is None:

            U, info = solve(None, None)

            print(
                f"P{degree}, M={M:>5,}: "
                f"initial cold solve "
                f"({info['iterations']} iterations)",
                flush=True,
            )

        else:

            coarse_nodes, coarse_U, coarse_Lambda = previous

            # ----------------------------------------------
            # 3. Transfer the coarse solution.
            # ----------------------------------------------

            t0 = perf_counter()

            U0 = prolongate(
                coarse_nodes,
                coarse_U,
                nodes,
            )

            # ----------------------------------------------
            # 4. L2-Riesz multiplier transfer.
            #
            # M_H mu_H = Lambda_H
            #
            # mu_h = I_h mu_H
            #
            # Lambda_h^0 = M_h mu_h
            # ----------------------------------------------

            Lambda0 = transfer_multiplier(
                coarse_nodes,
                coarse_Lambda,
                nodes,
            )

            transfer_seconds = perf_counter() - t0

            # ----------------------------------------------
            # 5. Solve the fine obstacle problem with PDAS.
            # ----------------------------------------------

            t0 = perf_counter()

            U, info = solve(
                U0,
                Lambda0,
            )

            solver_seconds = perf_counter() - t0

            # ----------------------------------------------
            # 6. Store numerical results and KKT diagnostics.
            # ----------------------------------------------

            rows.append({
                "degree": degree,
                "elements": M,
                "dofs": len(U),
                "initialization": "l2-riesz",
                "status": "converged",
                "pdas_iterations": info["iterations"],
                "solver_seconds": solver_seconds,
                "transfer_seconds": transfer_seconds,
                "min_gap": info["min_gap"],
                "min_multiplier": info["min_multiplier"],
                "max_complementarity": (
                    info["max_complementarity"]
                ),
            })

            print(
                f"P{degree}, M={M:>5,}: "
                f"PDAS={info['iterations']:>3d}, "
                f"solve={solver_seconds:.4f}s, "
                f"transfer={transfer_seconds:.4f}s, "
                f"min_gap={info['min_gap']:.2e}, "
                f"min_lambda={info['min_multiplier']:.2e}, "
                f"max_compl={info['max_complementarity']:.2e}",
                flush=True,
            )

        # --------------------------------------------------
        # 7. Save solution and multiplier for next mesh.
        # --------------------------------------------------

        previous = (
            nodes,
            U,
            multiplier(U),
        )

    return rows


# ============================================================
# Plot three initialization strategies
# ============================================================

def plot_comparison(
    baseline_rows,
    l2_rows,
    path,
):
    """Plot cold, original nested and L2-nested PDAS."""

    rows = baseline_rows + l2_rows

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )

    for i, degree in enumerate((1, 2)):

        for method, label in LABELS.items():

            subset = sorted(
                (
                    row for row in rows
                    if int(row["degree"]) == degree
                    and row["initialization"] == method
                    and row["status"] == "converged"
                ),
                key=lambda row: int(row["elements"]),
            )

            if not subset:
                continue

            elements = [
                int(row["elements"])
                for row in subset
            ]

            iterations = [
                int(row["pdas_iterations"])
                for row in subset
            ]

            seconds = [
                float(row["solver_seconds"])
                for row in subset
            ]

            # PDAS iterations: logarithmic axes.
            axes[i, 0].loglog(
                elements,
                iterations,
                "o-",
                label=label,
            )

            # Solver runtime: logarithmic axes.
            axes[i, 1].loglog(
                elements,
                seconds,
                "o-",
                label=label,
            )

        axes[i, 0].set_ylabel(
            f"P{degree}: PDAS iterations"
        )

        axes[i, 1].set_ylabel(
            f"P{degree}: solver time [s]"
        )

        for ax in axes[i]:

            ax.set_xlabel(
                "number of elements M"
            )

            ax.xaxis.set_major_locator(
                FixedLocator(LEVELS)
            )

            ax.xaxis.set_major_formatter(
                ScalarFormatter()
            )

            ax.xaxis.set_minor_formatter(
                NullFormatter()
            )

            ax.grid(
                True,
                which="both",
                alpha=0.22,
            )

        axes[i, 0].legend(
            fontsize=9,
            loc="upper left",
        )

    axes[0, 0].set_title(
        "PDAS iterations"
    )

    axes[0, 1].set_title(
        "Solver time (excluding transfer)"
    )

    fig.suptitle(
        "Dirac obstacle problem: nested PDAS initialization"
    )

    fig.savefig(
        path,
        dpi=210,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():

    baseline_path = (
        ROOT
        / "results"
        / "p1_p2_nested_initializations_M10000.csv"
    )

    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Missing {baseline_path}. "
            "Run the M=10000 cold/nested comparison first."
        )

    # ------------------------------------------------------
    # 1. Read only cold and original nested measurements.
    # ------------------------------------------------------

    with baseline_path.open(
        newline="",
        encoding="utf-8",
    ) as file:

        baseline_rows = [
            row
            for row in csv.DictReader(file)
            if row["initialization"] in (
                "cold",
                "recompute",
            )
        ]

    # ------------------------------------------------------
    # 2. Run only the L2-based nested experiments.
    # ------------------------------------------------------

    l2_rows = []

    for degree in (1, 2):
        l2_rows.extend(
            run_l2_nested(degree)
        )

    # ------------------------------------------------------
    # 3. Save results.
    # ------------------------------------------------------

    results_path = (
        ROOT
        / "results"
        / "p1_p2_l2_multiplier_transfer.csv"
    )

    figure_path = (
        ROOT
        / "figures"
        / "p1_p2_l2_multiplier_transfer.png"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(l2_rows[0]),
        )

        writer.writeheader()
        writer.writerows(l2_rows)

    plot_comparison(
        baseline_rows,
        l2_rows,
        figure_path,
    )

    print(
        f"Saved: {results_path.relative_to(ROOT)}"
    )

    print(
        f"Saved: {figure_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
