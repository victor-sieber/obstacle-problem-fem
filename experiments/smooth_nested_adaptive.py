
"""Smooth obstacle: cold versus L2-Riesz-nested PDAS on identical adaptive meshes.

Reuses the existing smooth manufactured solution and UNCHANGED residual indicator.
Uses nodal P1/P2 sparse FEM, 10-point load quadrature, and 12-point error
quadrature (split at 1/3 and 2/3). Timings cover PDAS and transfer but exclude
assembly, residual indicator, marking, refinement, and error evaluation.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
     python -m experiments.smooth_nested_adaptive --repeats 5
"""

import argparse
import csv
from pathlib import Path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from experiments.smooth_bell_problem import (
    ALPHA,
    BETA,
    exact_solution,
    exact_derivative,
    obstacle,
    regular_load,
)

from src.adaptive import (
    mark_by_maximum,
    bisect_marked_elements,
)

from src.error_indicators import (
    compute_p1_residual_indicator,
    compute_p2_residual_indicator,
)

from src.fem import (
    create_mesh,
    evaluate_fem_solution as eval_p1,
)

from src.fem_p2 import (
    evaluate_fem_solution as eval_p2,
)

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 1. P1 / P2 degrees of freedom and local basis functions
# ============================================================

def dof_coordinates(nodes, degree):
    if degree == 1:
        return nodes[1:-1]

    full = np.empty(2 * (len(nodes) - 1) + 1)
    full[0::2] = nodes
    full[1::2] = (nodes[:-1] + nodes[1:]) / 2

    return full[1:-1]


def local_basis(xi, degree):
    if degree == 1:
        return np.stack(
            (1 - xi, xi),
            axis=0,
        )

    return np.stack(
        (
            2 * (xi - 0.5) * (xi - 1),
            4 * xi * (1 - xi),
            2 * xi * (xi - 0.5),
        ),
        axis=0,
    )


# ============================================================
# 2. Sparse stiffness matrix, mass matrix and load vector
# ============================================================

def assemble_sparse(nodes, degree):
    """Build P1/P2 stiffness A, mass M and load F (10-point Gauss)."""

    ne = len(nodes) - 1
    ndof = degree * ne - 1

    q, w = leggauss(10)
    xi, w = (q + 1) / 2, w / 2
    phi = local_basis(xi, degree)

    if degree == 1:
        stiff0 = np.array([
            [1., -1.],
            [-1., 1.],
        ])

        mass0 = np.array([
            [2., 1.],
            [1., 2.],
        ]) / 6

    else:
        stiff0 = np.array([
            [7., -8., 1.],
            [-8., 16., -8.],
            [1., -8., 7.],
        ]) / 3

        mass0 = np.array([
            [4., 2., -1.],
            [2., 16., 2.],
            [-1., 2., 4.],
        ]) / 30

    ii, jj, av, mv = [], [], [], []
    F = np.zeros(ndof)

    for el, (a, b) in enumerate(
        zip(nodes[:-1], nodes[1:])
    ):
        h = b - a

        loc = (
            np.arange(
                degree * el,
                degree * el + degree + 1,
            ) - 1
        )

        # Interior DOF indices; boundary vertices are excluded.
        f_loc = h * (
            phi @ (
                w * regular_load(a + h * xi)
            )
        )

        for j, r in enumerate(loc):
            if 0 <= r < ndof:
                F[r] += f_loc[j]

                for k, c in enumerate(loc):
                    if 0 <= c < ndof:
                        ii.append(r)
                        jj.append(c)

                        av.append(
                            stiff0[j, k] / h
                        )

                        mv.append(
                            h * mass0[j, k]
                        )

    shape = (ndof, ndof)

    A = coo_matrix(
        (av, (ii, jj)),
        shape=shape,
    ).tocsc()

    M = coo_matrix(
        (mv, (ii, jj)),
        shape=shape,
    ).tocsc()

    return A, M, F


# ============================================================
# 3. Primal-dual active-set solver
# ============================================================

def pdas(
    A,
    F,
    psi,
    initial_active=None,
    maxiter=2000,
    c=1.0,
    active_tol=1e-10,
    kkt_tol=1e-8,
):
    """Same PDAS equations for cold and L2-nested initialization."""

    n = len(F)

    if initial_active is None:
        active = (
            spsolve(A, F) - psi <= active_tol
        )
    else:
        active = np.array(
            initial_active,
            dtype=bool,
            copy=True,
        )

        if active.shape != (n,):
            raise ValueError(
                "Initial active set has wrong shape"
            )

    for it in range(1, maxiter + 1):
        free = ~active

        U = np.empty(n)
        U[active] = psi[active]

        if free.any():
            rhs = F[free].copy()

            if active.any():
                rhs -= (
                    A[free][:, active]
                    @ psi[active]
                )

            U[free] = spsolve(
                A[free][:, free].tocsc(),
                rhs,
            )

        Lam = A @ U - F

        updated = (
            Lam - c * (U - psi)
        ) > active_tol

        if np.array_equal(updated, active):
            gap = U - psi

            violation = max(
                float(
                    max(0., -np.min(gap))
                ),
                float(
                    max(0., -np.min(Lam))
                ),
                float(
                    np.max(np.abs(gap * Lam))
                ),
            )

            if violation > kkt_tol:
                raise RuntimeError(
                    "PDAS stable but KKT violation="
                    f"{violation:.2e}"
                )

            return U, np.asarray(Lam), it

        active = updated

    raise RuntimeError(
        f"PDAS failed after {maxiter} iterations"
    )


# ============================================================
# 4. L2-Riesz nested initialization
# ============================================================

def nested_initialization(
    coarse,
    nodes,
    M,
    psi,
    degree,
    c=1.0,
):
    """mu_H = M_H^{-1} Lambda_H, then Lambda_h^0 = M_h I_h mu_H."""

    old_nodes, old_U, old_Lam, old_M = coarse

    evaluate = (
        eval_p1 if degree == 1
        else eval_p2
    )

    coords = dof_coordinates(
        nodes,
        degree,
    )

    # Interpolate previous primal solution.
    U0 = np.asarray(
        evaluate(
            old_nodes,
            old_U,
            coords,
        ),
        dtype=float,
    )

    # Convert dual multiplier to L2 representation.
    muH = spsolve(
        old_M,
        old_Lam,
    )

    # Transfer L2 multiplier to refined mesh.
    mu0 = np.asarray(
        evaluate(
            old_nodes,
            muH,
            coords,
        ),
        dtype=float,
    )

    # Convert back to fine-grid dual representation.
    Lam0 = M @ mu0

    # Initial active set.
    return (
        np.asarray(Lam0)
        - c * (U0 - psi)
    ) > 1e-10


# ============================================================
# 5. CPU timing
# ============================================================

def timed(call, repeats):
    """Median process CPU time; verify repeated solves agree."""

    times = []
    first = None

    for _ in range(repeats):
        t0 = process_time()

        answer = call()

        times.append(
            process_time() - t0
        )

        if first is None:
            first = answer

        elif not np.allclose(
            answer[0],
            first[0],
            atol=1e-10,
            rtol=0,
        ):
            raise RuntimeError(
                "Repeated solver runs disagree"
            )

    return first, float(
        np.median(times)
    )


# ============================================================
# 6. Exact H1 and L2 errors
# ============================================================

def errors(nodes, U, degree):
    """12-point elementwise Gauss H1-semi and L2 errors."""

    q, w = leggauss(12)

    full = np.concatenate((
        [0.],
        U,
        [0.],
    ))

    l2sq = 0.
    h1sq = 0.

    for el, (a, b) in enumerate(
        zip(nodes[:-1], nodes[1:])
    ):
        splits = (
            [a]
            + [
                t for t in (ALPHA, BETA)
                if a < t < b
            ]
            + [b]
        )

        loc = full[
            degree * el:
            degree * el + degree + 1
        ]

        h = b - a

        for left, right in zip(
            splits[:-1],
            splits[1:],
        ):
            x = (
                (left + right) / 2
                + (right - left) / 2 * q
            )

            wt = (
                (right - left) / 2 * w
            )

            xi = (x - a) / h

            if degree == 1:
                uh = (
                    loc[0] * (1 - xi)
                    + loc[1] * xi
                )

                duh = np.full_like(
                    x,
                    (loc[1] - loc[0]) / h,
                )

            else:
                basis = local_basis(xi, 2)

                uh = loc @ basis

                dbasis = np.stack(
                    (
                        4 * xi - 3,
                        4 - 8 * xi,
                        4 * xi - 1,
                    ),
                    axis=0,
                )

                duh = (
                    loc @ dbasis
                ) / h

            l2sq += float(
                np.sum(
                    wt * (
                        exact_solution(x) - uh
                    ) ** 2
                )
            )

            h1sq += float(
                np.sum(
                    wt * (
                        exact_derivative(x) - duh
                    ) ** 2
                )
            )

    return (
        np.sqrt(l2sq),
        np.sqrt(h1sq),
    )


# ============================================================
# 7. Adaptive experiment: cold vs nested on same mesh
# ============================================================

def run_degree(
    degree,
    tol,
    repeats,
    max_levels,
    max_dofs,
):
    nodes = create_mesh(20)

    previous = None
    history = []

    print(
        f"\nSmooth P{degree}: "
        f"tol={tol}, repeats={repeats}, theta=0.5"
    )

    print(
        "lvl  M   DOFs      eta       H1       L2"
        "     cold/nested its   cold/nested CPU"
    )

    for level in range(max_levels + 1):
        A, M, F = assemble_sparse(
            nodes,
            degree,
        )

        psi = np.asarray(
            obstacle(
                dof_coordinates(nodes, degree)
            ),
            dtype=float,
        )

        # Cold solve, repeated for timing.
        (U, Lam, k_cold), cold_cpu = timed(
            lambda: pdas(A, F, psi),
            repeats,
        )

        if previous is None:
            # Level 0: both methods perform same cold solve.
            nested_U = U
            nested_Lam = Lam
            k_nested = k_cold
            nested_cpu = cold_cpu
            diff = 0.

        else:
            def nested_solve():
                active0 = nested_initialization(
                    previous,
                    nodes,
                    M,
                    psi,
                    degree,
                )

                return pdas(
                    A,
                    F,
                    psi,
                    initial_active=active0,
                )

            (
                nested_U,
                nested_Lam,
                k_nested,
            ), nested_cpu = timed(
                nested_solve,
                repeats,
            )

            diff = float(
                np.max(
                    np.abs(U - nested_U)
                )
            )

            if diff > 1e-8:
                raise RuntimeError(
                    f"P{degree}, level {level}: "
                    f"solution mismatch {diff}"
                )

        # Original Grässle-approved indicator: unchanged.
        if degree == 1:
            indicator = compute_p1_residual_indicator(
                nodes,
                U,
                A.toarray(),
                F,
                regular_load,
            )
        else:
            indicator = compute_p2_residual_indicator(
                nodes,
                U,
                A.toarray(),
                F,
                regular_load,
            )

        loc_eta = np.asarray(
            indicator["eta"],
            dtype=float,
        )

        eta = float(
            np.linalg.norm(loc_eta)
        )

        l2, h1 = errors(
            nodes,
            U,
            degree,
        )

        rec = dict(
            degree=degree,
            level=level,
            elements=len(nodes) - 1,
            dofs=len(U),
            eta_h=eta,
            h1_error=h1,
            l2_error=l2,
            cold_iterations=k_cold,
            nested_iterations=k_nested,
            cold_cpu_seconds=cold_cpu,
            nested_total_cpu_seconds=nested_cpu,
            max_solution_difference=diff,
            repetitions=repeats,
        )

        history.append(rec)

        print(
            f"{level:3d} "
            f"{len(nodes)-1:4d} "
            f"{len(U):5d} "
            f"{eta:9.3e} "
            f"{h1:9.3e} "
            f"{l2:9.3e}  "
            f"{k_cold:3d}/{k_nested:<3d}  "
            f"{cold_cpu:8.4f}/{nested_cpu:.4f}s",
            flush=True,
        )

        if eta <= tol:
            print(
                "Stop: estimator tolerance"
            )
            break

        if level == max_levels:
            raise RuntimeError(
                "Reached max levels without meeting tolerance"
            )

        marked = mark_by_maximum(
            loc_eta,
            theta=0.5,
        )

        if not np.any(marked):
            raise RuntimeError(
                "No elements marked"
            )

        refined = bisect_marked_elements(
            nodes,
            marked,
        )

        if (
            degree * (len(refined) - 1) - 1
            > max_dofs
        ):
            raise RuntimeError(
                "Refinement exceeds --max-dofs; "
                "increase it deliberately"
            )

        previous = (
            nodes.copy(),
            nested_U.copy(),
            nested_Lam.copy(),
            M,
        )

        nodes = refined

    return history


# ============================================================
# 8. Reproducible six-panel figure
# ============================================================

def make_figure(rows, dest, repeats):
    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 9.4),
    )

    for idx, degree in enumerate((1, 2)):
        r = [
            row for row in rows
            if row["degree"] == degree
        ]

        N = np.array(
            [row["dofs"] for row in r],
            dtype=float,
        )

        # Iterations and CPU time.
        for col, title, yc, yn in (
            (
                0,
                "PDAS iterations",
                "cold_iterations",
                "nested_iterations",
            ),
            (
                1,
                "Median CPU time [s]",
                "cold_cpu_seconds",
                "nested_total_cpu_seconds",
            ),
        ):
            ax = axes[idx, col]

            ax.loglog(
                N,
                [rr[yc] for rr in r],
                "o-",
                label="Cold adaptive PDAS",
            )

            ax.loglog(
                N,
                [rr[yn] for rr in r],
                "s-",
                label=r"$L^2$-nested adaptive PDAS",
            )

            ax.set_title(
                f"P{degree}: {title}"
            )

            ax.legend(fontsize=8)

            if col == 0:
                ax.set_ylabel(
                    "Iterations per mesh"
                )
            else:
                ax.set_ylabel(
                    f"Median of {repeats} CPU runs [s]"
                )

                ratio = (
                    sum(rr[yc] for rr in r)
                    / sum(rr[yn] for rr in r)
                )

                ax.text(
                    0.96,
                    0.06,
                    f"Sum-of-medians: {ratio:.1f}x",
                    transform=ax.transAxes,
                    ha="right",
                    fontsize=9,
                    bbox=dict(
                        facecolor="white",
                        alpha=0.85,
                    ),
                )

        # Accuracy panel.
        ax = axes[idx, 2]

        for key, label, marker in (
            ("eta_h", r"Estimator $\eta_h$", "o-"),
            ("h1_error", r"$H^1$ error", "s-"),
            ("l2_error", r"$L^2$ error", "^-"),
        ):
            ax.loglog(
                N,
                [rr[key] for rr in r],
                marker,
                label=label,
                markersize=4,
            )

        x = np.geomspace(
            N[0],
            N[-1],
            80,
        )

        # Smooth-problem FEM reference guides.
        for key, p, color in (
            ("h1_error", degree, "0.35"),
            ("l2_error", degree + 1, "0.65"),
        ):
            anchor = r[0][key]

            ax.loglog(
                x,
                anchor * (x / N[0]) ** (-p),
                "--",
                color=color,
                label=rf"Smooth uniform guide $N^{{-{p}}}$",
            )

        # Observed finite-range adaptive slopes.
        for key, color, label in (
            ("h1_error", "tab:orange", "H1"),
            ("l2_error", "tab:green", "L2"),
        ):
            yy = np.array(
                [rr[key] for rr in r]
            )

            slope, intercept = np.polyfit(
                np.log(N),
                np.log(yy),
                1,
            )

            ax.loglog(
                x,
                np.exp(
                    intercept + slope * np.log(x)
                ),
                ":",
                color=color,
                label=rf"Observed {label} $N^{{{-slope:.2f}}}$",
            )

            print(
                f"P{degree} adaptive {label}: "
                f"fitted exponent {-slope:.3f} "
                "(finite-range)"
            )

        ax.set_title(
            f"P{degree}: adaptive accuracy"
        )

        ax.set_ylabel(
            "Estimator / error"
        )

        ax.legend(
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )

        for a in axes[idx]:
            a.set_xlabel(
                "Interior degrees of freedom $N$"
            )

            a.grid(
                True,
                which="both",
                alpha=0.22,
            )

    fig.suptitle(
        "Smooth obstacle: adaptive P1/P2 FEM, "
        "cold vs $L^2$-nested PDAS\n"
        f"Identical meshes · {repeats} CPU "
        "repetitions/method/level · "
        "cold solve charged to both on level 0",
        fontsize=13,
    )

    fig.text(
        0.5,
        0.025,
        "CPU excludes assembly, indicator, marking, "
        "refinement and error quadrature. "
        "Dashed smooth-solution guides are uniform FEM "
        "reference rates; dotted slopes are "
        "empirical adaptive fits.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(0, 0.065, 0.88, 0.92)
    )

    fig.savefig(
        dest,
        dpi=185,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--p1-tol",
        type=float,
        default=2e-2,
    )

    parser.add_argument(
        "--p2-tol",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--max-levels",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-dofs",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    if (
        args.repeats < 1
        or args.p1_tol <= 0
        or args.p2_tol <= 0
    ):
        parser.error(
            "repeats >= 1, tolerances > 0 required"
        )

    rows = []

    for degree, tol in (
        (1, args.p1_tol),
        (2, args.p2_tol),
    ):
        rows.extend(
            run_degree(
                degree,
                tol,
                args.repeats,
                args.max_levels,
                args.max_dofs,
            )
        )

    results = ROOT / "results"
    figures = ROOT / "figures"

    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    csvpath = (
        results / "smooth_nested_adaptive_p1_p2.csv"
    )

    with csvpath.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    figpath = (
        figures / "smooth_nested_adaptive_p1_p2.png"
    )

    make_figure(
        rows,
        figpath,
        args.repeats,
    )

    print(
        "\nSummary (sum of per-mesh medians; "
        "NOT end-to-end CPU):"
    )

    for degree in (1, 2):
        r = [
            row for row in rows
            if row["degree"] == degree
        ]

        cold = sum(
            row["cold_cpu_seconds"]
            for row in r
        )

        nested = sum(
            row["nested_total_cpu_seconds"]
            for row in r
        )

        print(
            f"P{degree}: {len(r)} mesh levels, "
            f"final {r[-1]['dofs']} DOFs; "
            "last iterations cold/nested="
            f"{r[-1]['cold_iterations']}/"
            f"{r[-1]['nested_iterations']}; "
            f"cold={cold:.5f}s "
            f"nested={nested:.5f}s "
            f"speedup={cold/nested:.2f}x"
        )

    print(
        "Saved:",
        csvpath.relative_to(ROOT),
        figpath.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
