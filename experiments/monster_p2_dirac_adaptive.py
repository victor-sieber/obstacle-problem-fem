
"""P2 Dirac monster: cold PDAS versus L2-nested PDAS.

Both methods solve on identical adaptively refined meshes.

The existing P2 residual indicator, maximum marking,
and local mesh bisection remain unchanged.

Timing includes PDAS initialization and solving,
but excludes FEM assembly, indicator evaluation,
and mesh refinement.
"""

import argparse
import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np

from experiments import (
    dirac_p2_residual_adaptive_refinement as original,
)

from experiments.p1_p2_dirac_comparison import (
    assemble_dirac_load_p2,
    compute_errors,
    smooth_load,
)

from src.adaptive import (
    bisect_marked_elements,
    mark_by_maximum,
)

from src.error_indicators import (
    compute_p2_residual_indicator,
)

from src.fem import create_mesh

from src.fem_p2 import (
    assemble_stiffness_matrix,
    evaluate_fem_derivative,
    evaluate_fem_solution,
)

from src.multiplier_l2_transfer import (
    transfer_l2_multiplier_p2,
)

from src.p2_nested_large import (
    assemble_p2_dirac,
    join_p2,
    p2_matvec,
    prolongate_p2,
    solve_pdas_p2,
)


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "level",
    "elements",
    "dofs",
    "marked",
    "min_h",
    "max_h",
    "eta_h",
    "h1_error",
    "l2_error",
    "cold_iterations",
    "nested_iterations",
    "cold_cpu_seconds",
    "nested_solver_cpu_seconds",
    "nested_transfer_cpu_seconds",
    "nested_total_cpu_seconds",
    "max_solution_difference",
    "min_gap",
    "min_multiplier",
    "max_complementarity",
    "stop_reason",
]


# ============================================================
# 1. CPU time measurement
# ============================================================

def timed_solve(solve, repeats, prepare=None):
    """
    Repeat PDAS on one fixed mesh.

    Report median CPU time and verify that repeated
    solves return the same solution and iteration count.
    """

    preparation_times = []
    solver_times = []
    total_times = []

    reference_U = None
    reference_info = None

    for _ in range(repeats):

        t0 = process_time()

        if prepare is None:
            U0 = None
            Lambda0 = None
        else:
            U0, Lambda0 = prepare()

        t1 = process_time()

        U, info = solve(U0, Lambda0)

        t2 = process_time()

        if reference_U is None:

            reference_U = U.copy()
            reference_info = info

        else:

            same_iterations = (
                info["iterations"]
                == reference_info["iterations"]
            )

            same_solution = np.allclose(
                U,
                reference_U,
                rtol=0.0,
                atol=1e-10,
            )

            if not (same_iterations and same_solution):
                raise RuntimeError(
                    "Repeated PDAS solutions differ."
                )

        preparation_times.append(t1 - t0)
        solver_times.append(t2 - t1)
        total_times.append(t2 - t0)

    times = {
        "prepare": float(
            np.median(preparation_times)
        ),
        "solve": float(
            np.median(solver_times)
        ),
        "total": float(
            np.median(total_times)
        ),
    }

    return reference_U, reference_info, times


# ============================================================
# 2. Save CSV
# ============================================================

def save_csv(rows, path):

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. Plot comparison
# ============================================================

def plot_results(rows, path, repeats):

    if len(rows) < 2:
        return

    after_first = rows[1:]

    x = np.array([
        row["dofs"]
        for row in after_first
    ])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.4),
        constrained_layout=True,
    )

    # --------------------------------------------------
    # PDAS iteration counts
    # --------------------------------------------------

    axes[0].loglog(
        x,
        [
            r["cold_iterations"]
            for r in after_first
        ],
        "o-",
        label="cold PDAS",
    )

    axes[0].loglog(
        x,
        [
            r["nested_iterations"]
            for r in after_first
        ],
        "s-",
        label=r"nested $L^2$ PDAS",
    )

    axes[0].set_title("PDAS iterations")
    axes[0].set_ylabel("PDAS iterations")

    # --------------------------------------------------
    # Median CPU times
    # --------------------------------------------------

    axes[1].loglog(
        x,
        [
            r["cold_cpu_seconds"]
            for r in after_first
        ],
        "o-",
        label="cold PDAS",
    )

    axes[1].loglog(
        x,
        [
            r["nested_total_cpu_seconds"]
            for r in after_first
        ],
        "s-",
        label=r"nested $L^2$ PDAS",
    )

    axes[1].set_title("Median CPU time")
    axes[1].set_ylabel("CPU time [s]")

    # --------------------------------------------------
    # Adaptive error reduction
    # --------------------------------------------------

    dofs = [
        r["dofs"]
        for r in rows
    ]

    for key, style, label in (
        (
            "eta_h",
            "o-",
            r"estimator $\eta_h$",
        ),
        (
            "h1_error",
            "s-",
            r"$H^1$ seminorm error",
        ),
        (
            "l2_error",
            "^-",
            r"$L^2$ error",
        ),
    ):

        axes[2].loglog(
            dofs,
            [
                r[key]
                for r in rows
            ],
            style,
            label=label,
        )

    axes[2].set_title("Adaptive error reduction")
    axes[2].set_ylabel("estimator / error")

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    for ax in axes:

        ax.set_xlabel(
            "interior P2 degrees of freedom"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.23,
        )

        ax.legend(
            fontsize=8
        )

    fig.suptitle(
        "P2 Dirac: adaptive cold vs "
        + r"$L^2$-nested PDAS "
        + f"(median CPU time, {repeats} runs)"
    )

    fig.savefig(
        path,
        dpi=210,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 4. Main adaptive + nested algorithm
# ============================================================

def run(tol, max_levels, maxiter, repeats):

    nodes = create_mesh(
        original.INITIAL_ELEMENTS
    )

    previous = None
    rows = []

    # Short tests receive distinct output filenames.
    suffix = (
        ""
        if max_levels == original.MAX_REFINEMENTS
        else f"_L{max_levels}"
    )

    result_dir = ROOT / "results"
    figure_dir = ROOT / "figures"

    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    csv_path = (
        result_dir
        / f"p2_monster_dirac{suffix}.csv"
    )

    figure_path = (
        figure_dir
        / f"p2_monster_dirac{suffix}.png"
    )

    cumulative_cold = 0.0
    cumulative_nested = 0.0

    print(
        f"P2 monster: "
        f"theta={original.THETA:g}, "
        f"tol={tol:g}, "
        f"max_levels={max_levels}, "
        f"repetitions={repeats}",
        flush=True,
    )

    # ====================================================
    # Adaptive refinement loop
    # ====================================================

    for level in range(max_levels + 1):

        # ------------------------------------------------
        # A. Assemble structured P2 FEM load
        #
        # f = -6 - 2 delta_alpha - delta_beta
        # ------------------------------------------------

        Fv, Fm = assemble_p2_dirac(
            nodes
        )

        F = join_p2(
            Fv,
            Fm,
        )

        # Check agreement with our original P2
        # Dirac load assembly.
        F_original = assemble_dirac_load_p2(
            nodes
        )

        if not np.allclose(
            F,
            F_original,
            rtol=0.0,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"P2 load assembly differs "
                f"at level {level}."
            )

        # ------------------------------------------------
        # B. Existing P2 PDAS solver
        #
        # Uses static condensation + Thomas algorithm.
        # ------------------------------------------------

        def solve(U0, Lambda0):

            return solve_pdas_p2(
                nodes,
                Fv,
                Fm,
                initial_solution=U0,
                initial_multiplier=Lambda0,
                maxiter=maxiter,
            )

        # ------------------------------------------------
        # C. Cold PDAS
        # ------------------------------------------------

        cold_U, cold_info, cold_time = timed_solve(
            solve,
            repeats,
        )

        # ------------------------------------------------
        # D. L2-nested PDAS
        # ------------------------------------------------

        if previous is None:

            # Initial mesh has no previous solution.
            nested_U = cold_U
            nested_info = cold_info
            nested_time = cold_time

        else:

            coarse_nodes, coarse_U, coarse_Lambda = (
                previous
            )

            def prepare():

                # Primal interpolation:
                #
                # U_h^0 = I_h U_H

                U0 = prolongate_p2(
                    coarse_nodes,
                    coarse_U,
                    nodes,
                )

                # L2 multiplier transfer:
                #
                # M_H mu_H = Lambda_H
                #
                # Lambda_h^0 = M_h I_h mu_H

                Lambda0 = transfer_l2_multiplier_p2(
                    coarse_nodes,
                    coarse_Lambda,
                    nodes,
                )

                return U0, Lambda0

            nested_U, nested_info, nested_time = (
                timed_solve(
                    solve,
                    repeats,
                    prepare=prepare,
                )
            )

        # ------------------------------------------------
        # E. Compare both FEM solutions
        # ------------------------------------------------

        difference = float(
            np.max(
                np.abs(
                    cold_U - nested_U
                )
            )
        )

        if (
            not np.isfinite(difference)
            or difference > 1e-8
        ):
            raise RuntimeError(
                f"Level {level}: cold and nested "
                f"solutions differ by {difference:.3e}"
            )

        # ------------------------------------------------
        # F. Original P2 residual indicator
        #
        # This is the Grässle-approved indicator.
        #
        # f_reg = -6 enters the volume residual.
        # Dirac point loads remain in F.
        #
        # No change to the indicator formula.
        # ------------------------------------------------

        A_indicator = assemble_stiffness_matrix(
            nodes
        )

        indicator = compute_p2_residual_indicator(
            nodes,
            nested_U,
            A_indicator,
            F,
            smooth_load,
        )

        eta = indicator[
            "eta"
        ]

        eta_global = float(
            np.sqrt(
                np.sum(
                    indicator["eta_squared"]
                )
            )
        )

        # ------------------------------------------------
        # G. Actual H1 and L2 errors
        # ------------------------------------------------

        l2_error, h1_error = compute_errors(
            nodes,
            nested_U,
            evaluate_fem_solution,
            evaluate_fem_derivative,
        )

        M = len(nodes) - 1
        dofs = 2 * M - 1

        h = np.diff(nodes)

        # ------------------------------------------------
        # H. Accumulate solver-stage CPU times
        # ------------------------------------------------

        cumulative_cold += cold_time["total"]
        cumulative_nested += nested_time["total"]

        # ------------------------------------------------
        # I. Store level results
        # ------------------------------------------------

        row = {
            "level": level,
            "elements": M,
            "dofs": dofs,
            "marked": 0,
            "min_h": float(np.min(h)),
            "max_h": float(np.max(h)),
            "eta_h": eta_global,
            "h1_error": h1_error,
            "l2_error": l2_error,

            "cold_iterations": cold_info[
                "iterations"
            ],

            "nested_iterations": (
                ""
                if level == 0
                else nested_info["iterations"]
            ),

            "cold_cpu_seconds": cold_time["total"],

            "nested_solver_cpu_seconds": (
                ""
                if level == 0
                else nested_time["solve"]
            ),

            "nested_transfer_cpu_seconds": (
                ""
                if level == 0
                else nested_time["prepare"]
            ),

            "nested_total_cpu_seconds": (
                ""
                if level == 0
                else nested_time["total"]
            ),

            "max_solution_difference": difference,

            "min_gap": nested_info["min_gap"],

            "min_multiplier": nested_info[
                "min_multiplier"
            ],

            "max_complementarity": nested_info[
                "max_complementarity"
            ],

            "stop_reason": "",
        }

        rows.append(row)

        # ------------------------------------------------
        # J. Original adaptive stopping conditions
        # ------------------------------------------------

        if eta_global <= tol:

            row["stop_reason"] = (
                "indicator tolerance"
            )

        elif dofs >= original.MAX_INTERIOR_DOFS:

            row["stop_reason"] = (
                "DOF safety cap"
            )

        elif level == max_levels:

            row["stop_reason"] = (
                "refinement limit"
            )

        else:

            # --------------------------------------------
            # K. Maximum marking, theta = 0.5
            # --------------------------------------------

            marked = mark_by_maximum(
                eta,
                theta=original.THETA,
            )

            row["marked"] = int(
                np.count_nonzero(marked)
            )

            if not np.any(marked):

                row["stop_reason"] = (
                    "no marked elements"
                )

            else:

                # ----------------------------------------
                # L. Local bisection
                # ----------------------------------------

                next_nodes = bisect_marked_elements(
                    nodes,
                    marked,
                )

        # ------------------------------------------------
        # M. Print results
        # ------------------------------------------------

        nested_it = (
            "-"
            if level == 0
            else str(nested_info["iterations"])
        )

        nested_cpu = (
            "-"
            if level == 0
            else f"{nested_time['total']:.4f}s"
        )

        print(
            f"level={level:2d} "
            f"M={M:5d} "
            f"DOFs={dofs:5d} "
            f"eta={eta_global:.3e} "
            f"H1={h1_error:.3e} "
            f"L2={l2_error:.3e} "
            f"cold={cold_info['iterations']:4d} "
            f"L2-nested={nested_it} "
            f"coldCPU={cold_time['total']:.4f}s "
            f"nestedCPU={nested_cpu} "
            f"diff={difference:.1e} "
            f"marked={row['marked']}",
            flush=True,
        )

        # Preserve all successful levels.
        save_csv(
            rows,
            csv_path,
        )

        if row["stop_reason"]:
            break

        # ------------------------------------------------
        # N. Store coarse solution and multiplier
        #
        # Lambda_H = A_H U_H - F_H
        # ------------------------------------------------

        previous = (
            nodes.copy(),
            nested_U.copy(),
            p2_matvec(
                nodes,
                nested_U,
            ) - F,
        )

        nodes = next_nodes

    # ====================================================
    # Final outputs
    # ====================================================

    plot_results(
        rows,
        figure_path,
        repeats,
    )

    print()
    print(
        "Stop:",
        rows[-1]["stop_reason"],
    )

    print(
        "Cumulative cold PDAS CPU:",
        f"{cumulative_cold:.4f}s",
    )

    print(
        "Cumulative L2-nested PDAS CPU "
        "including transfer:",
        f"{cumulative_nested:.4f}s",
    )

    print(
        f"Saved: {csv_path.relative_to(ROOT)}"
    )

    if len(rows) > 1:
        print(
            f"Saved: {figure_path.relative_to(ROOT)}"
        )

    return rows


# ============================================================
# 5. Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tol",
        type=float,
        default=original.ETA_TOL,
    )

    parser.add_argument(
        "--max-levels",
        type=int,
        default=original.MAX_REFINEMENTS,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    if (
        args.tol <= 0
        or args.max_levels < 0
        or args.maxiter < 1
        or args.repeats < 1
    ):
        parser.error(
            "Require tol > 0, max-levels >= 0, "
            "maxiter >= 1, repeats >= 1."
        )

    run(
        args.tol,
        args.max_levels,
        args.maxiter,
        args.repeats,
    )


if __name__ == "__main__":
    main()
