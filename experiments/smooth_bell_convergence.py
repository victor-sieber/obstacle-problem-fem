
"""Manufactured C^5 bell obstacle: P1/P2 uniform and adaptive FEM.

Uses existing FEM assembly, residual indicators and maximum marking.
The local PDAS supports a nonconstant obstacle.

Run:
    python -m experiments.smooth_bell_convergence --quick
    python -m experiments.smooth_bell_convergence
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from experiments.smooth_bell_problem import (
    ALPHA, BETA, exact_derivative, exact_multiplier,
    exact_second_derivative, exact_solution,
    obstacle, regular_load,
)

from src.adaptive import (
    bisect_marked_elements,
    mark_by_maximum,
)

from src.error_indicators import (
    compute_p1_residual_indicator,
    compute_p2_residual_indicator,
)

from src.fem import (
    assemble_load_vector as load_p1,
    assemble_stiffness_matrix as stiffness_p1,
    create_mesh,
    evaluate_fem_derivative as derivative_p1,
    evaluate_fem_solution as evaluate_p1,
)

from src.fem_p2 import (
    assemble_load_vector as load_p2,
    assemble_stiffness_matrix as stiffness_p2,
    create_p2_nodes,
    evaluate_fem_derivative as derivative_p2,
    evaluate_fem_solution as evaluate_p2,
)


ROOT = Path(__file__).resolve().parents[1]
THETA = 0.5

FIELDS = [
    "degree", "strategy", "level", "elements", "dofs",
    "eta_h", "h1_error", "l2_error", "pdas_iterations",
    "min_gap", "min_multiplier", "complementarity",
    "marked", "stop_reason",
]


# ============================================================
# 1. Verify the exact manufactured problem
# ============================================================

def verify_exact_problem():

    x = np.linspace(0.0, 1.0, 2001)

    u = exact_solution(x)
    psi = obstacle(x)
    lam = exact_multiplier(x)

    assert abs(u[0]) < 1e-13
    assert abs(u[-1]) < 1e-13

    assert np.min(u - psi) >= -1e-12
    assert np.min(lam) >= 0.0

    assert np.max(
        np.abs(lam * (u - psi))
    ) < 1e-12

    assert np.max(
        np.abs(
            -exact_second_derivative(x)
            - regular_load(x)
            - lam
        )
    ) < 1e-12

    assert np.allclose(
        obstacle([0.0, 1.0]),
        [-1.0, -1.0],
    )

    assert np.isclose(
        exact_multiplier(0.5),
        1.0,
    )

    print(
        "Analytical checks passed: boundary, contact, "
        "KKT, f=-u''-lambda."
    )


# ============================================================
# 2. PDAS for the nonconstant obstacle
# ============================================================

def pdas(A, F, psi, tol=1e-10, maxiter=2000):
    """Cold PDAS with a small numerical tie threshold.

    Lambda = A@U - F.
    Free DOFs: Lambda = 0.
    Contact DOFs: U = psi.
    """

    A = csc_matrix(A)

    unconstrained = spsolve(A, F)
    active = unconstrained <= psi

    for k in range(1, maxiter + 1):

        free = ~active

        U = np.empty_like(psi)
        U[active] = psi[active]

        if np.any(free):

            rhs = F[free].copy()

            if np.any(active):
                rhs -= (
                    A[free][:, active]
                    @ psi[active]
                )

            U[free] = spsolve(
                A[free][:, free],
                rhs,
            )

        Lambda = np.asarray(A @ U - F)
        gap = U - psi

        new_active = Lambda - gap > tol

        if np.array_equal(new_active, active):

            diag = {
                "pdas_iterations": k,
                "min_gap": float(np.min(gap)),
                "min_multiplier": float(np.min(Lambda)),
                "complementarity": float(
                    np.max(np.abs(gap * Lambda))
                ),
            }

            # Do not accept a stable active set unless
            # the discrete KKT residuals are small.
            if (
                min(
                    diag["min_gap"],
                    diag["min_multiplier"],
                ) < -1e-8
                or diag["complementarity"] > 1e-8
            ):
                raise RuntimeError(
                    "PDAS stabilized without KKT convergence: "
                    f"{diag}"
                )

            return U, diag

        active = new_active

    raise RuntimeError(
        f"PDAS did not stabilize after {maxiter} iterations"
    )


# ============================================================
# 3. Actual L2 and H1 errors
# ============================================================

def exact_errors(nodes, U, degree):
    """Elementwise 12-point Gauss quadrature.

    Split integration at the exact contact boundaries
    1/3 and 2/3.
    """

    if degree == 1:
        ev = evaluate_p1
        deriv = derivative_p1
    else:
        ev = evaluate_p2
        deriv = derivative_p2

    points, weights = leggauss(12)

    L2sq = 0.0
    H1sq = 0.0

    for left, right in zip(
        nodes[:-1],
        nodes[1:],
    ):

        cuts = (
            [left]
            + [
                z for z in (ALPHA, BETA)
                if left < z < right
            ]
            + [right]
        )

        for a, b in zip(
            cuts[:-1],
            cuts[1:],
        ):

            xq = (
                (a + b) / 2
                + (b - a) * points / 2
            )

            wq = (
                (b - a) * weights / 2
            )

            L2sq += float(
                np.dot(
                    wq,
                    (
                        ev(nodes, U, xq)
                        - exact_solution(xq)
                    )**2,
                )
            )

            H1sq += float(
                np.dot(
                    wq,
                    (
                        deriv(nodes, U, xq)
                        - exact_derivative(xq)
                    )**2,
                )
            )

    return np.sqrt(L2sq), np.sqrt(H1sq)


# ============================================================
# 4. Assemble and solve one mesh
# ============================================================

def solve_mesh(nodes, degree):

    if degree == 1:

        A = stiffness_p1(nodes)

        F = load_p1(
            nodes,
            regular_load,
            quadrature_order=10,
        )

        points = nodes
        indicator_fn = compute_p1_residual_indicator

    else:

        A = stiffness_p2(nodes)

        F = load_p2(
            nodes,
            regular_load,
            quadrature_order=10,
        )

        points = create_p2_nodes(nodes)
        indicator_fn = compute_p2_residual_indicator

    psi = obstacle(points[1:-1])

    U, diag = pdas(A, F, psi)

    # Original Grässle-approved indicator.
    indicator = indicator_fn(
        nodes,
        U,
        A,
        F,
        regular_load,
    )

    L2, H1 = exact_errors(
        nodes,
        U,
        degree,
    )

    return U, diag, indicator, L2, H1


def record(degree, strategy, level, nodes, data):

    U, diag, indicator, L2, H1 = data

    return {
        "degree": degree,
        "strategy": strategy,
        "level": level,
        "elements": len(nodes) - 1,
        "dofs": len(U),
        "eta_h": float(
            np.sqrt(np.sum(indicator["eta_squared"]))
        ),
        "h1_error": float(H1),
        "l2_error": float(L2),
        **diag,
        "marked": 0,
        "stop_reason": "",
    }


# ============================================================
# 5. Uniform refinement
# ============================================================

def run_uniform(degree, counts):

    rows = []

    for level, M in enumerate(counts):

        nodes = create_mesh(M)
        data = solve_mesh(nodes, degree)

        r = record(
            degree, "uniform", level, nodes, data
        )

        rows.append(r)

        print(
            f"P{degree} uniform "
            f"M={M:4d} N={r['dofs']:4d} "
            f"eta={r['eta_h']:.3e} "
            f"H1={r['h1_error']:.3e} "
            f"L2={r['l2_error']:.3e} "
            f"PDAS={r['pdas_iterations']}",
            flush=True,
        )

    return rows


# ============================================================
# 6. Adaptive refinement
# ============================================================

def run_adaptive(
    degree,
    max_levels,
    max_dofs,
    tolerance,
):

    nodes = create_mesh(20)

    rows = []
    last = None

    for level in range(max_levels + 1):

        data = solve_mesh(nodes, degree)

        r = record(
            degree, "adaptive", level, nodes, data
        )

        rows.append(r)

        # mu contains coefficients of the L2-Riesz
        # reconstruction returned by the indicator.
        last = (
            nodes.copy(),
            data[0].copy(),
            data[2]["mu"].copy(),
        )

        if r["eta_h"] <= tolerance:

            r["stop_reason"] = "indicator tolerance"

        elif r["dofs"] >= max_dofs:

            r["stop_reason"] = "DOF cap"

        elif level == max_levels:

            r["stop_reason"] = "level cap"

        else:

            marked = mark_by_maximum(
                data[2]["eta"],
                theta=THETA,
            )

            r["marked"] = int(
                np.count_nonzero(marked)
            )

            if not np.any(marked):

                r["stop_reason"] = "no marked elements"

            else:

                next_nodes = bisect_marked_elements(
                    nodes,
                    marked,
                )

        print(
            f"P{degree} adaptive "
            f"level={level:2d} "
            f"M={r['elements']:4d} "
            f"N={r['dofs']:4d} "
            f"eta={r['eta_h']:.3e} "
            f"H1={r['h1_error']:.3e} "
            f"L2={r['l2_error']:.3e} "
            f"PDAS={r['pdas_iterations']:3d} "
            f"marked={r['marked']:3d} "
            f"{r['stop_reason']}",
            flush=True,
        )

        if r["stop_reason"]:
            break

        nodes = next_nodes

    return rows, last


# ============================================================
# 7. Convergence plot with theoretical reference slopes
# ============================================================

def plot_convergence(rows, path):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 4.8),
        constrained_layout=True,
    )

    metrics = ("h1_error", "l2_error")

    labels = (
        "H1 seminorm error",
        "L2 error",
    )

    for ax, metric, ylabel in zip(
        axes, metrics, labels
    ):

        for degree in (1, 2):

            for method in ("uniform", "adaptive"):

                items = [
                    r for r in rows
                    if r["degree"] == degree
                    and r["strategy"] == method
                ]

                x = np.array(
                    [r["dofs"] for r in items],
                    dtype=float,
                )

                y = np.array(
                    [r[metric] for r in items],
                    dtype=float,
                )

                ax.loglog(
                    x,
                    y,
                    marker=(
                        "o" if method == "uniform" else "s"
                    ),
                    linestyle=(
                        "-" if method == "uniform" else "-."
                    ),
                    label=f"P{degree} {method}",
                )

            # Reference slopes are anchored to the
            # final uniform refinement result.
            uniform = [
                r for r in rows
                if r["degree"] == degree
                and r["strategy"] == "uniform"
            ]

            if len(uniform) > 1:

                x = np.array(
                    [r["dofs"] for r in uniform],
                    dtype=float,
                )

                y = np.array(
                    [r[metric] for r in uniform],
                    dtype=float,
                )

                power = (
                    degree
                    if metric == "h1_error"
                    else degree + 1
                )

                reference = (
                    y[-1]
                    * (x / x[-1])**(-power)
                )

                ax.loglog(
                    x,
                    reference,
                    "--",
                    alpha=0.7,
                    label=(
                        rf"P{degree} reference "
                        rf"$N^{{-{power}}}$"
                    ),
                )

        ax.set_xlabel(
            "interior degrees of freedom N"
        )

        ax.set_ylabel(ylabel)

        ax.grid(
            True,
            which="both",
            alpha=0.2,
        )

        ax.legend(fontsize=8)

    fig.suptitle(
        "Smooth manufactured bell obstacle: "
        "P1/P2 uniform vs adaptive"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 8. Plot exact solution and multiplier
# ============================================================

def plot_solution(final, path):

    x = np.linspace(
        0.0,
        1.0,
        1001,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4.7),
        constrained_layout=True,
    )

    axes[0].plot(
        x,
        exact_solution(x),
        "k-",
        linewidth=2,
        label="exact solution",
    )

    axes[0].plot(
        x,
        obstacle(x),
        "--",
        label="obstacle",
    )

    axes[1].plot(
        x,
        exact_multiplier(x),
        "k--",
        linewidth=2,
        label="exact multiplier",
    )

    for degree, (nodes, U, mu) in final.items():

        ev = (
            evaluate_p1
            if degree == 1
            else evaluate_p2
        )

        axes[0].plot(
            x,
            ev(nodes, U, x),
            label=f"P{degree} adaptive",
            alpha=0.8,
        )

        axes[1].plot(
            x,
            ev(nodes, mu, x),
            label=rf"P{degree} $L^2$-Riesz",
            alpha=0.8,
        )

    for ax in axes:

        ax.axvline(
            ALPHA,
            color="gray",
            linestyle=":",
            alpha=0.7,
        )

        ax.axvline(
            BETA,
            color="gray",
            linestyle=":",
            alpha=0.7,
        )

        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x")
        ax.legend(fontsize=9)

    axes[0].set_title(
        "Exact and numerical solutions / obstacle"
    )

    axes[1].set_title(
        "Exact and reconstructed multipliers"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 9. Run all experiments
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--quick",
        action="store_true",
    )

    ap.add_argument(
        "--p1-tol",
        type=float,
        default=2e-2,
    )

    ap.add_argument(
        "--p2-tol",
        type=float,
        default=1e-3,
    )

    ap.add_argument(
        "--max-levels",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--max-dofs",
        type=int,
        default=1500,
    )

    ap.add_argument(
        "--max-elements",
        type=int,
        default=160,
    )

    args = ap.parse_args()

    if (
        args.p1_tol <= 0
        or args.p2_tol <= 0
        or args.max_levels < 0
        or args.max_dofs < 40
        or args.max_elements < 20
    ):
        ap.error(
            "Positive tolerances, max-levels >= 0, "
            "max-dofs >= 40, max-elements >= 20 required."
        )

    verify_exact_problem()

    counts = []
    M = 10

    while M <= (
        40 if args.quick else args.max_elements
    ):
        counts.append(M)
        M *= 2

    rows = []
    final = {}

    for degree in (1, 2):

        rows.extend(
            run_uniform(degree, counts)
        )

        adaptive, last = run_adaptive(
            degree,
            3 if args.quick else args.max_levels,
            args.max_dofs,
            (
                args.p1_tol
                if degree == 1
                else args.p2_tol
            ),
        )

        rows.extend(adaptive)
        final[degree] = last

    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    suffix = "_quick" if args.quick else ""

    csv_path = (
        ROOT / "results"
        / f"smooth_bell_convergence{suffix}.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    convergence_path = (
        ROOT / "figures"
        / f"smooth_bell_convergence{suffix}.png"
    )

    solution_path = (
        ROOT / "figures"
        / f"smooth_bell_solution{suffix}.png"
    )

    plot_convergence(
        rows,
        convergence_path,
    )

    plot_solution(
        final,
        solution_path,
    )

    print(
        "Saved:",
        csv_path.relative_to(ROOT),
        convergence_path.relative_to(ROOT),
        solution_path.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
