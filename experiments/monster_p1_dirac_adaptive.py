
"""P1 Dirac: residual adaptivity with L2-nested PDAS.

Compare cold and L2-nested PDAS on identical adaptive meshes.

The residual indicator, marking, and local refinement
are taken unchanged from the original P1 experiment.

Run from project root:

python -u -m experiments.monster_p1_dirac_adaptive \
    --max-levels 3 --repeats 2

python -u -m experiments.monster_p1_dirac_adaptive \
    --repeats 5
"""

import argparse
import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np

from experiments import (
    dirac_residual_adaptive_refinement as original,
)

from experiments.dirac_example import (
    assemble_dirac_load,
    compute_errors,
    smooth_load,
)

from src.adaptive import (
    bisect_marked_elements,
    mark_by_maximum,
)

from src.error_indicators import (
    compute_p1_residual_indicator,
)

from src.fem import (
    assemble_stiffness_matrix,
    create_mesh,
)

from src.multiplier_l2_transfer import (
    transfer_l2_multiplier_p1,
)

from src.p1_nested_large import (
    assemble_p1_dirac,
    matvec,
    prolongate_p1,
    solve_pdas_p1,
)


ROOT = Path(__file__).resolve().parents[1]

FIELDNAMES = [
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
# 1. Measure PDAS CPU time on a fixed mesh
# ============================================================

def timed_solve(solve, repeats, prepare=None):
    """
    Repeat a solve on one fixed mesh.

    Returns:
        FEM solution,
        PDAS diagnostics,
        median CPU times.

    For nested PDAS, prepare() computes:
        U_h^0 and Lambda_h^0.
    """

    total_times = []
    preparation_times = []
    solver_times = []

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

            if not (
                same_iterations and same_solution
            ):
                raise RuntimeError(
                    "Repeated PDAS solves disagree."
                )

        preparation_times.append(t1 - t0)
        solver_times.append(t2 - t1)
        total_times.append(t2 - t0)

    times = {
        "total": float(np.median(total_times)),
        "prepare": float(
            np.median(preparation_times)
        ),
        "solve": float(np.median(solver_times)),
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
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. Plot convergence and efficiency
# ============================================================

def plot_results(rows, path, repeats):

    # At level 0 there is no nested initialization.
    measured = rows[1:]

    if not measured:
        return

    x = np.array([
        row["dofs"]
        for row in measured
    ])

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.4),
        constrained_layout=True,
    )

    # --------------------------------------------------
    # Plot 1: PDAS iteration counts
    # --------------------------------------------------

    axes[0].loglog(
        x,
        [
            row["cold_iterations"]
            for row in measured
        ],
        "o-",
        label="cold PDAS",
    )

    axes[0].loglog(
        x,
        [
            row["nested_iterations"]
            for row in measured
        ],
        "s-",
        label=r"nested $L^2$ PDAS",
    )

    axes[0].set_title("PDAS iterations")
    axes[0].set_ylabel("PDAS iterations")

    # --------------------------------------------------
    # Plot 2: Median CPU time
    # --------------------------------------------------

    axes[1].loglog(
        x,
        [
            row["cold_cpu_seconds"]
            for row in measured
        ],
        "o-",
        label="cold PDAS",
    )

    axes[1].loglog(
        x,
        [
            row["nested_total_cpu_seconds"]
            for row in measured
        ],
        "s-",
        label=r"nested $L^2$ PDAS",
    )

    axes[1].set_title("Median CPU time")
    axes[1].set_ylabel("CPU time [s]")

    # --------------------------------------------------
    # Plot 3: Error estimator and actual errors
    # --------------------------------------------------

    all_dofs = np.array([
        row["dofs"]
        for row in rows
    ])

    axes[2].loglog(
        all_dofs,
        [
            row["eta_h"]
            for row in rows
        ],
        "o-",
        label=r"estimator $\eta_h$",
    )

    axes[2].loglog(
        all_dofs,
        [
            row["h1_error"]
            for row in rows
        ],
        "s-",
        label=r"$H^1$ error",
    )

    axes[2].loglog(
        all_dofs,
        [
            row["l2_error"]
            for row in rows
        ],
        "^-",
        label=r"$L^2$ error",
    )

    axes[2].set_title("Adaptive error reduction")
    axes[2].set_ylabel("estimator / error")

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    for ax in axes:

        ax.set_xlabel(
            "interior P1 degrees of freedom"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.23,
        )

        ax.legend(fontsize=9)

    fig.suptitle(
        "P1 Dirac: adaptive cold vs "
        r"$L^2$-nested PDAS "
        f"(median CPU time, {repeats} runs)"
    )

    fig.savefig(
        path,
        dpi=210,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 4. Monster algorithm
# ============================================================

def run(
    tolerance,
    max_levels,
    maxiter,
    repeats,
):

    # Same initial mesh as the original adaptive code.
    nodes = create_mesh(
        original.INITIAL_ELEMENTS
    )

    previous = None
    rows = []

    # Short diagnostic runs get separate output files.
    suffix = (
        ""
        if max_levels == original.MAX_REFINEMENTS
        else f"_L{max_levels}"
    )

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    csv_path = (
        results_dir
        / f"p1_monster_dirac{suffix}.csv"
    )

    figure_path = (
        figures_dir
        / f"p1_monster_dirac{suffix}.png"
    )

    cumulative_cold = 0.0
    cumulative_nested = 0.0

    print(
        f"P1 monster: "
        f"theta={original.THETA}, "
        f"tol={tolerance:g}, "
        f"max_levels={max_levels}, "
        f"repetitions={repeats}",
        flush=True,
    )

    # ====================================================
    # Adaptive loop
    # ====================================================

    for level in range(max_levels + 1):

        # ------------------------------------------------
        # A. Assemble structured P1 FEM problem
        # ------------------------------------------------

        lower, diagonal, upper, F = (
            assemble_p1_dirac(nodes)
        )

        # Verify initial Dirac RHS against original code.
        if level == 0:

            original_F = assemble_dirac_load(
                nodes
            )

            if not np.allclose(
                F,
                original_F,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "Dirac load assembly disagrees "
                    "with the original implementation."
                )

        # ------------------------------------------------
        # B. Define our existing structured PDAS solver
        # ------------------------------------------------

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

        # ------------------------------------------------
        # C. Cold PDAS on the current mesh
        # ------------------------------------------------

        cold_U, cold_info, cold_time = (
            timed_solve(
                solve,
                repeats,
            )
        )

        # ------------------------------------------------
        # D. L2-nested PDAS on the SAME mesh
        # ------------------------------------------------

        if previous is None:

            # Level zero has no previous mesh.
            nested_U = cold_U
            nested_info = cold_info

            nested_time = {
                "total": cold_time["total"],
                "prepare": 0.0,
                "solve": cold_time["solve"],
            }

        else:

            coarse_nodes, coarse_U, coarse_Lambda = (
                previous
            )

            def prepare():

                # ----------------------------------------
                # Primal transfer
                #
                # U_h^0 = I_h U_H
                # ----------------------------------------

                U0 = prolongate_p1(
                    coarse_nodes,
                    coarse_U,
                    nodes,
                )

                # ----------------------------------------
                # Dual L2-Riesz transfer
                #
                # M_H mu_H = Lambda_H
                #
                # Lambda_h^0 = M_h I_h mu_H
                # ----------------------------------------

                Lambda0 = transfer_l2_multiplier_p1(
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
        # E. Check both solutions agree
        # ------------------------------------------------

        difference = float(
            np.max(
                np.abs(cold_U - nested_U)
            )
        )

        if (
            not np.isfinite(difference)
            or difference > 1e-8
        ):
            raise RuntimeError(
                f"Level {level}: cold and nested "
                f"solutions disagree: {difference:.3e}"
            )

        # ------------------------------------------------
        # F. Original Grässle residual estimator
        #
        # IMPORTANT:
        # This is the unchanged existing indicator.
        #
        # Only f_reg = -6 enters its volume term.
        # Dirac loads remain in the FEM load vector F.
        # ------------------------------------------------

        A_indicator = assemble_stiffness_matrix(
            nodes
        )

        indicator = compute_p1_residual_indicator(
            nodes,
            nested_U,
            A_indicator,
            F,
            smooth_load,
            multiplier_method=original.MULTIPLIER_METHOD,
        )

        eta = indicator["eta"]

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
        )

        M = len(nodes) - 1
        dofs = M - 1

        h = np.diff(nodes)

        # ------------------------------------------------
        # H. Cumulative solver CPU time
        # ------------------------------------------------

        cumulative_cold += cold_time["total"]

        if level == 0:
            cumulative_nested += cold_time["total"]
        else:
            cumulative_nested += nested_time["total"]

        # ------------------------------------------------
        # I. Store results
        # ------------------------------------------------

        row = {
            "level": level,
            "elements": M,
            "dofs": dofs,
            "marked": 0,
            "min_h": float(np.min(h)),
            "max_h": float(np.max(h)),
            "eta_h": eta_global,
            "h1_error": float(h1_error),
            "l2_error": float(l2_error),

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
        # J. Original adaptive stopping criteria
        # ------------------------------------------------

        if eta_global <= tolerance:

            row["stop_reason"] = "indicator tolerance"

        elif dofs >= original.MAX_INTERIOR_DOFS:

            row["stop_reason"] = "DOF safety cap"

        elif level == max_levels:

            row["stop_reason"] = "refinement limit"

        else:

            # --------------------------------------------
            # K. Original maximum marking
            # --------------------------------------------

            marked = mark_by_maximum(
                eta,
                theta=original.THETA,
            )

            row["marked"] = int(
                np.count_nonzero(marked)
            )

            if not np.any(marked):

                row["stop_reason"] = "no marked elements"

            else:

                # ----------------------------------------
                # L. Original local bisection
                # ----------------------------------------

                next_nodes = bisect_marked_elements(
                    nodes,
                    marked,
                )

        # ------------------------------------------------
        # M. Print results
        # ------------------------------------------------

        nested_iterations_text = (
            "-"
            if level == 0
            else str(nested_info["iterations"])
        )

        nested_cpu_text = (
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
            f"L2-nested={nested_iterations_text} "
            f"coldCPU={cold_time['total']:.4f}s "
            f"nestedCPU={nested_cpu_text} "
            f"diff={difference:.1e} "
            f"marked={row['marked']}",
            flush=True,
        )

        # Save successful levels immediately.
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
            matvec(
                lower,
                diagonal,
                upper,
                nested_U,
            ) - F,
        )

        # Move to the refined mesh.
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
        "Cumulative nested PDAS CPU "
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
