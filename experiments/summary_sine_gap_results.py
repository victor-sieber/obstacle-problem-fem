"""Review CSV diagnostics without converting roundoff residuals into false KKT passes."""
import csv
from pathlib import Path
import math

root = Path(__file__).resolve().parents[1] / "results"
threshold = 1e-8
for degree in (1, 2):
    for method in ("uniform", "adaptive"):
        path = root / f"sine_dirac_p{degree}_{method}.csv"
        if not path.exists():
            print(f"P{degree} {method}: missing {path.name}")
            continue
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        required = ("min_gap", "min_multiplier", "max_complementarity")
        finite = all(math.isfinite(float(r[k])) for r in rows for k in required)
        min_gap = min(float(r["min_gap"]) for r in rows)
        min_dual = min(float(r["min_multiplier"]) for r in rows)
        max_comp = max(float(r["max_complementarity"]) for r in rows)
        available = all(r.get("free_equation_residual", "") != "" and
                        r.get("min_active_multiplier", "") != "" for r in rows)
        print(f"\nP{degree} {method} | DOFs {last['dofs']} | levels {len(rows)} | stop {last['stop_reason']}")
        print(f"  eta/eta0={float(last['eta_ratio']):.6e}  H1={float(last['h1_error']):.6e}  L2={float(last['l2_error']):.6e}")
        print(f"  min_gap={min_gap:+.3e}, raw min_dual={min_dual:+.3e}, max_complementarity={max_comp:.3e}")
        if available:
            min_active = min(float(r["min_active_multiplier"]) for r in rows)
            free_max = max(float(r["free_equation_residual"]) for r in rows)
            print(f"  min_active_dual={min_active:+.3e}; max free-equation residual={free_max:.3e}")
            print(f"  active feasibility: {'PASS' if min_active >= -threshold and min_gap >= -threshold else 'INVESTIGATE'}")
            print(f"  absolute free-equation test (1e-8): {'PASS' if free_max <= threshold else 'EXCEEDS (roundoff to investigate)'}")
        else:
            print('  No separated active/free diagnostics; rerun with integrated driver.')
        print(f"  finite data: {'PASS' if finite else 'FAIL'}")
        if method == "adaptive":
            reached = float(last["eta_ratio"]) <= 1e-3
            print(f"  relative estimator target: {'REACHED' if reached else 'NOT REACHED'}")
            if reached and last["stop_reason"] != "relative_estimator":
                print('  WARNING: stopping label does not match target')
