
"""Six-panel, CSV-based P1/P2 adaptive Dirac PDAS comparison.

Recreates plots from the saved per-mesh benchmark medians; no FEM solves.

Each CPU point is the median of REPEATS independent timing repetitions
on that mesh. Verify REPEATS against the benchmark invocation/source;
the summary CSV contains medians, not the five raw samples.

Run from repository root:
    python -m experiments.plot_dirac_nested_pdas_summary
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INPUTS = {
    1: RESULTS / "p1_monster_dirac.csv",
    2: RESULTS / "p2_monster_dirac.csv",
}
REPEATS = 5  # Benchmark setting, NOT inferred from aggregated CSV.
MAX_SOLUTION_DIFFERENCE = 1e-8


def load_history(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark CSV: {path}")

    required = (
        "level", "dofs", "eta_h", "h1_error", "l2_error",
        "cold_iterations", "nested_iterations", "cold_cpu_seconds",
        "nested_total_cpu_seconds", "max_solution_difference",
    )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        rows = sorted(reader, key=lambda r: int(r["level"]))

    if len(rows) < 3:
        raise ValueError(f"Need at least 3 mesh levels in {path.name}")

    def column(name, allow_first_blank=False):
        values = []
        for i, row in enumerate(rows):
            cell = row[name].strip()
            if i == 0 and not cell and allow_first_blank:
                values.append(np.nan)
            elif not cell:
                raise ValueError(f"Missing {name} in {path.name}, row {i}")
            else:
                values.append(float(cell))
        return np.asarray(values, dtype=float)

    levels = column("level").astype(int)
    if not np.array_equal(levels, np.arange(len(rows))):
        raise ValueError(f"Nonconsecutive levels in {path.name}")

    data = {
        "dofs": column("dofs"),
        "eta": column("eta_h"),
        "h1": column("h1_error"),
        "l2": column("l2_error"),
        "cold_it": column("cold_iterations"),
        "nested_it": column("nested_iterations", allow_first_blank=True),
        "cold_cpu": column("cold_cpu_seconds"),
        "nested_cpu": column("nested_total_cpu_seconds", allow_first_blank=True),
        "difference": column("max_solution_difference"),
        "mesh_levels": len(rows),
    }

    if not np.all(np.diff(data["dofs"]) > 0):
        raise ValueError(f"DOFs do not strictly increase in {path.name}")
    for key in ("eta", "h1", "l2", "cold_it", "cold_cpu"):
        a = data[key]
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError(f"Invalid {key} values in {path.name}")
    for key in ("nested_it", "nested_cpu"):
        a = data[key][1:]
        if not np.all(np.isfinite(a) & (a > 0)):
            raise ValueError(f"Invalid {key} values in {path.name}")
    diff = data["difference"]
    if not np.all(np.isfinite(diff)) or np.max(diff) > MAX_SOLUTION_DIFFERENCE:
        raise ValueError(f"Cold and nested PDAS disagree in {path.name}")

    # Both strategies must pay for level 0: no coarse solution exists yet.
    data["nested_cpu"][0] = data["cold_cpu"][0]
    data["nested_it"][0] = data["cold_it"][0]

    data["cold_total"] = float(np.sum(data["cold_cpu"]))
    data["nested_total"] = float(np.sum(data["nested_cpu"]))
    data["speedup"] = data["cold_total"] / data["nested_total"]
    data["max_difference"] = float(np.max(diff))
    return data




def draw_accuracy(ax, d, degree):
    """Measured accuracy, reference guides and observed log-log fits."""

    n = np.asarray(d["dofs"], dtype=float)

    ax.loglog(
        n, d["eta"], "o-",
        markersize=4,
        label=r"Estimator $\eta_h$",
    )
    ax.loglog(
        n, d["h1"], "s-",
        markersize=4,
        label=r"$H^1$ error",
    )
    ax.loglog(
        n, d["l2"], "^-",
        markersize=4,
        label=r"$L^2$ error",
    )

    # Reference guides already used in our previous plot.
    if degree == 1:
        xref = np.geomspace(
            max(n[0], n[-1] / 7), n[-1], 70
        )
        reference = (
            ("h1", 1.0, "H1"),
            ("l2", 2.0, "L2"),
        )
    else:
        xref = np.geomspace(n[0], n[-1], 70)
        reference = (
            ("h1", 0.5, "H1"),
            ("l2", 1.5, "L2"),
        )

    for key, exponent, label in reference:
        if degree == 1:
            guide = (
                1.8 * d[key][-1]
                * (xref / n[-1]) ** (-exponent)
            )
            description = "P1 adaptive guide"
        else:
            guide = (
                d[key][0]
                * (xref / n[0]) ** (-exponent)
            )
            description = "Uniform Dirac guide"

        ax.loglog(
            xref,
            guide,
            "--",
            color=("0.25" if key == "h1" else "0.65"),
            linewidth=1.6,
            label=rf"{description} $N^{{-{exponent:g}}}$ ({label})",
        )

    # Empirical least-squares fit across ALL adaptive levels.
    xfit = np.geomspace(n[0], n[-1], 100)

    for key, label, color in (
        ("h1", "H1", "tab:orange"),
        ("l2", "L2", "tab:green"),
    ):
        y = np.asarray(d[key], dtype=float)

        slope, intercept = np.polyfit(
            np.log(n),
            np.log(y),
            1,
        )

        exponent = -float(slope)

        fitted = np.exp(
            intercept + slope * np.log(xfit)
        )

        # R-squared in log-log coordinates.
        log_y = np.log(y)
        predicted = intercept + slope * np.log(n)

        r2 = 1.0 - (
            np.sum((log_y - predicted) ** 2)
            / np.sum((log_y - np.mean(log_y)) ** 2)
        )

        ax.loglog(
            xfit,
            fitted,
            ":",
            color=color,
            linewidth=2.25,
            alpha=0.9,
            label=rf"Observed {label} fit $N^{{-{exponent:.2f}}}$",
        )

        print(
            f"P{degree} adaptive {label}: "
            f"fitted exponent p={exponent:.3f}, "
            f"R^2={r2:.3f} "
            f"({len(n)} mesh levels)"
        )

    ax.set_title(f"P{degree} · adaptive accuracy")
    ax.set_ylabel("Estimator / error")
    ax.legend(
    fontsize=7.1,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
)


def draw_figure(histories, path):
    fig, axes = plt.subplots(2, 3, figsize=(20, 9.4))

    for i, degree in enumerate((1, 2)):
        d = histories[degree]
        n = d["dofs"]
        label_cold = "Cold-start adaptive PDAS"
        label_nested = r"$L^2$-nested adaptive PDAS"

        ax = axes[i, 0]
        ax.loglog(n, d["cold_it"], "o-", label=label_cold, markersize=4)
        ax.loglog(n, d["nested_it"], "s-", label=label_nested, markersize=4)
        ax.set_title(f"P{degree} · PDAS iterations")
        ax.set_ylabel("Iterations on this mesh")
        ax.legend(fontsize=8)

        ax = axes[i, 1]
        ax.loglog(n, d["cold_cpu"], "o-", label=label_cold, markersize=4)
        ax.loglog(n, d["nested_cpu"], "s-", label=label_nested, markersize=4)
        ax.set_title(f"P{degree} · median solver CPU time")
        ax.set_ylabel(f"Median of {REPEATS} timed runs [s]")
        ax.text(
            0.96, 0.06,
            f"Sum-of-medians speedup: {d['speedup']:.1f}x",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.88, edgecolor="0.8"),
        )
        ax.legend(fontsize=8)

        draw_accuracy(axes[i, 2], d, degree)

        for ax in axes[i]:
            ax.set_xlabel("Interior degrees of freedom $N$")
            ax.grid(True, which="both", alpha=0.23)

    mesh_info = " · ".join(
        f"P{p}: {histories[p]['mesh_levels']} mesh levels / "
        f"{histories[p]['mesh_levels'] - 1} refinements" for p in (1, 2)
    )
    fig.suptitle(
        "Dirac obstacle: adaptive P1/P2 FEM, cold versus $L^2$-nested PDAS\n"
        f"Same adaptive meshes · {REPEATS} timing repetitions per method per level · "
        f"{mesh_info}",
        fontsize=13,
    )
    fig.text(
        0.5, 0.045,
        f"Each CPU point = median of {REPEATS} repetitions on that mesh. "
        "Initial cold solve included for both methods. "
        "Nested CPU includes $L^2$ transfer.\n"
        "Timing excludes assembly, estimator computation, marking and refinement. "
        "Dashed: P1 illustrative adaptive / P2 UNIFORM Dirac guides. "
        "Dotted: fitted log-log slopes across all adaptive levels.",
        ha="center", va="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    data = {p: load_history(file) for p, file in INPUTS.items()}
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    output_rows = []
    print("\nCold versus nested adaptive PDAS on IDENTICAL meshes")
    print(f"Timing methodology: {REPEATS} CPU repetitions per method and mesh;")
    print("each plotted point = median of those repetitions (not median across levels).")
    print("Aggregate CPU = sum of mesh-wise medians, including level 0 for both.\n")

    for degree in (1, 2):
        d = data[degree]
        nlevels = d["mesh_levels"]
        final_dofs = int(d["dofs"][-1])
        speedup = d["speedup"]
        print(
            f"P{degree}: {nlevels} mesh levels ({nlevels-1} refinements), "
            f"final N={final_dofs}; "
            f"last iterations {int(d['cold_it'][-1])} vs "
            f"{int(d['nested_it'][-1])}; "
            f"total CPU {d['cold_total']:.6f}s vs "
            f"{d['nested_total']:.6f}s; "
            f"{speedup:.1f}x solver-stage speedup; "
            f"max difference {d['max_difference']:.2e}"
        )
        output_rows.append({
            "degree": degree,
            "adaptive_refinements": nlevels - 1,
            "mesh_levels": nlevels,
            "timing_repetitions_per_mesh_and_method": REPEATS,
            "final_dofs": final_dofs,
            "final_cold_iterations": int(d["cold_it"][-1]),
            "final_nested_iterations": int(d["nested_it"][-1]),
            "cold_sum_of_medians_seconds": d["cold_total"],
            "nested_sum_of_medians_seconds": d["nested_total"],
            "solver_stage_speedup": speedup,
            "max_solution_difference": d["max_difference"],
        })

    csv_path = RESULTS / "dirac_nested_pdas_p1_p2_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    png_path = FIGURES / "dirac_nested_pdas_p1_p2_summary.png"
    draw_figure(data, png_path)
    print(f"\nSaved: {png_path.relative_to(ROOT)}")
    print(f"Saved: {csv_path.relative_to(ROOT)}")
    print("Caveat: CSV stores median timings, not five individual samples.")


if __name__ == "__main__":
    main()
