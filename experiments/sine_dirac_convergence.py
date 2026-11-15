"""Uniform/adaptive P1/P2 convergence, using the same nested L2-PDAS engine.

Sine benchmark: epsilon=1, original Dirac point loads and exact solution.
Adaptive: maximum marking theta=0.5; stop eta_l/eta_0 <= rel_tol.
Uniform: bisect all elements; DOF cap limits both study arms.
"""
import argparse
import csv
from pathlib import Path
from time import perf_counter
import numpy as np

from src.adaptive import bisect_marked_elements, mark_by_maximum
from src.p1_nested_large import prolongate_p1
from src.p2_nested_large import prolongate_p2
from src.multiplier_l2_transfer import (transfer_l2_multiplier_p1,
                                        transfer_l2_multiplier_p2)
from experiments.sine_perturbed_dirac import (solve_on_mesh, compute_errors)

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("benchmark", "degree", "method", "level", "elements", "dofs",
           "min_h", "eta", "eta_ratio", "l2_error", "h1_error",
           "iterations", "wall_seconds", "delta_l2_previous", "delta_h1_previous", "min_gap", "min_multiplier",
           "max_complementarity", "free_equation_residual",
           "min_active_multiplier", "marked", "stop_reason")


def run_series(degree, method, solver, error_function=None, benchmark="sine_dirac",
               max_levels=8, max_dofs=5000, rel_tol=1e-3, maxiter=2000,
               output_dir=None):
    """Run identical mesh/refinement bookkeeping for a benchmark-specific solver.

    solver(nodes, degree, maxiter, initial_solution, initial_multiplier)
    returns a dict containing nodes, W, Lambda, U, indicator, info.
    """
    if degree not in (1, 2) or method not in ("uniform", "adaptive"):
        raise ValueError("Select P1/P2 and uniform/adaptive")
    if not (0 < rel_tol < 1) or max_levels < 0 or max_dofs < 39:
        raise ValueError("Invalid study limits")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")

    nodes = np.linspace(0, 1, 21)  # original 20-element initial mesh
    previous = None
    initial_eta = None
    rows = []
    folder = Path(output_dir) if output_dir is not None else ROOT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{benchmark}_p{degree}_{method}.csv"
    for level in range(max_levels + 1):
        n_dofs = (degree * (len(nodes) - 1) - 1)
        if n_dofs > max_dofs:
            raise RuntimeError("DOF cap was exceeded before solving")
        if previous is None:
            primal0 = dual0 = None
        else:
            coarse_nodes = previous["nodes"]
            primal0 = (prolongate_p1 if degree == 1 else prolongate_p2)(
                coarse_nodes, previous["W"], nodes)
            dual0 = (transfer_l2_multiplier_p1 if degree == 1
                     else transfer_l2_multiplier_p2)(
                coarse_nodes, previous["Lambda"], nodes)

        t0 = perf_counter()
        result = solver(nodes, degree, maxiter, primal0, dual0)
        elapsed_wall = perf_counter() - t0  # includes assembly, PDAS, estimator
        eta_local = np.asarray(result["indicator"]["eta"], dtype=float)
        eta = float(np.linalg.norm(eta_local))
        if not np.isfinite(eta):
            raise RuntimeError("Nonfinite estimator")
        if initial_eta is None:
            initial_eta = eta
        ratio = eta / initial_eta if initial_eta > 0 else 0.0
        l2, h1 = (error_function(result) if error_function is not None
                  else (None, None))
        info = result["info"]
        new_row = dict(benchmark=benchmark, degree=degree, method=method,
                       level=level, elements=len(nodes)-1, dofs=n_dofs,
                       min_h=float(np.min(np.diff(nodes))), eta=eta,
                       eta_ratio=ratio, l2_error=l2, h1_error=h1,
                       iterations=int(info["iterations"]),
                       wall_seconds=elapsed_wall,
                       delta_l2_previous=info.get("delta_l2_previous", ""),
                       delta_h1_previous=info.get("delta_h1_previous", ""),
                       min_gap=float(info["min_gap_physical"]),
                       min_multiplier=float(info["min_multiplier_physical"]),
                       max_complementarity=float(info["max_complementarity_physical"]),
                       free_equation_residual=info.get("max_free_stationarity_residual", ""),
                       min_active_multiplier=info.get("min_active_multiplier", ""),
                       marked=0, stop_reason="")
        rows.append(new_row)
        reason = None
        if method == "adaptive" and ratio <= rel_tol:
            reason = "relative_estimator"
        elif level == max_levels:
            reason = "max_levels"
        else:
            if method == "adaptive":
                marked = mark_by_maximum(eta_local, theta=0.5)
            else:
                marked = np.ones(len(nodes)-1, dtype=bool)
            count = int(np.count_nonzero(marked))
            if count == 0:
                reason = "nothing_marked"
            else:
                candidate = bisect_marked_elements(nodes, marked)
                candidate_dofs = degree * (len(candidate)-1) - 1
                if candidate_dofs > max_dofs:
                    reason = "dof_cap"
                else:
                    new_row["marked"] = count
                    previous = result
                    nodes = candidate
        if reason is not None:
            new_row["stop_reason"] = reason
        print(f"{benchmark} P{degree} {method:8s} level={level:2d} "
              f"DOFs={n_dofs:5d} eta/eta0={ratio:.3e} "
              f"H1={h1 if h1 is not None else 'n/a'} "
              f"PDAS={info['iterations']:3d} "
              f"deltaH1={info.get('delta_h1_previous', float('nan')):.3e} "
              f"free_res={info.get('max_free_stationarity_residual', float('nan')):.2e} "
              f"stop={reason or '-'}", flush=True)
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if benchmark == "no_exact" and method == "adaptive":
            np.savez(folder / f"{benchmark}_p{degree}_{method}_final.npz",
                     nodes=result["nodes"], U=result["U"], Lambda=result["Lambda"])
        if reason is not None:
            break
    print("Saved:", file)
    return rows


def _sine_solver(nodes, degree, maxiter, initial_solution, initial_multiplier):
    return solve_on_mesh(nodes, degree, epsilon=1.0, maxiter=maxiter,
                         initial_solution=initial_solution,
                         initial_multiplier=initial_multiplier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, choices=(1, 2), required=True)
    parser.add_argument("--method", choices=("uniform", "adaptive"), required=True)
    parser.add_argument("--max-levels", type=int, default=8)
    parser.add_argument("--max-dofs", type=int, default=5000)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=2000)
    args = parser.parse_args()
    run_series(degree=args.degree, method=args.method, solver=_sine_solver,
               error_function=compute_errors,
               max_levels=args.max_levels, max_dofs=args.max_dofs,
               rel_tol=args.rel_tol, maxiter=args.maxiter)


if __name__ == "__main__":
    main()
