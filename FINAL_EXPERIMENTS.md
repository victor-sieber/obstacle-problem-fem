# Final experiment protocol — 18 November 2026

The English handout and its eight core histories are the submission baseline.
The three remaining P1 accuracy reruns were completed during this review.
No further large run is required to read, validate, or plot the delivered data.

## Fixed protocol

| Setting | Final value |
| --- | --- |
| Cases | `sine_dirac`, `no_exact` |
| FEM / mesh | P1 and P2; uniform and adaptive |
| Initial mesh | 20 equal elements |
| Initialization after refinement | Riesz dual transfer and interpolated shifted primal variable |
| Marking | Maximum rule, theta = 0.5; all elements for uniform refinement |
| Inner PDAS ratio | eta(k)/eta(0) on each fixed mesh, tolerance 0.001 |
| Inner relative-exit safeguard | Absolute nodal KKT tolerance 1e-8 |
| Stable-set acceptance | Wrapper checks gap >= -1e-9, active dual >= -1e-8, scaled free residual and complementarity <= 1e-12 |
| Outer estimator stopping | Disabled (`mesh_rel_tol=None`) |
| Interior DOF ceiling | 300000 |
| Maximum refinement level | 60 (initial level is zero) |
| Prospective minimum width | 1e-12; **2e-9 for no_exact P2 adaptive** |
| Precision | P1: Decimal 45 digits if max(h)/min(h)>1e6 or elements>=15000; P2: Decimal 40 digits if ratio>1e3 or elements>=15000 |

Boundary gaps are 1 and contact gaps are 0. Loads and returned arrays remain
binary64. Uniform bisection may stop well below the DOF ceiling because the next
complete refinement would exceed it. Per-series sidecars record mesh limits.
`configuration.json` is the latest targeted invocation, not combined provenance.
`summary.csv` is synchronized with `verified_summary.csv` in this delivery;
`run_progress.csv` remains only the most recent run's progress record.

## Routine checks and plots

```bash
python -m experiments.validation.solvers.final_checks
python -m experiments.validation.estimators.verify_gap_sine_integration
python -m experiments.key.final_delivery check
python -m experiments.validation.solvers.submission_checks
python -m experiments.key.final_delivery plots
```

The strengthened CSV check validates identities, finite errors, mesh limits,
full inner histories and ratios, stop reasons and recorded nodal/scaled KKT
quantities. It does not establish forward accuracy from a small residual.
`submission_checks` independently reproduces the finest warm-start counts once;
it does not replace three-repetition timing measurements.

Canonical outputs:

- `results/key/inner_study/budget_300000/{sine_dirac,no_exact}/*.csv`
- `results/key/inner_study/budget_300000/verified_summary.csv`
- `results/key/inner_study/budget_300000/warm_starts_inner.csv`
- `figures/key/inner_study/budget_300000/final/`

All 174 final mesh solves stabilized; no inner relative-indicator exit occurred.
No speedup is attributed to the tolerance.

## Reproduce the eight histories only if required

These commands overwrite the selected canonical histories. Preserve a copy or
Git commit first. They can take several minutes and create final `.npz`
snapshots; snapshots are not required for the report or final CSV plotter.
The following decomposition applies the P2 adaptive exception explicitly:

```bash
python -m experiments.key.inner_study run --cases sine_dirac --degrees 1 2 --methods uniform adaptive --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 1e-12
python -m experiments.key.inner_study run --cases no_exact --degrees 1 --methods uniform adaptive --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 1e-12
python -m experiments.key.inner_study run --cases no_exact --degrees 2 --methods uniform --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 1e-12
python -m experiments.key.inner_study run --cases no_exact --degrees 2 --methods adaptive --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 2e-9
python -m experiments.key.final_delivery check
python -m experiments.key.final_delivery plots
```

Five histories were retained from the uploaded archive, three P1 histories were
rerun. The new P1 trigger does not affect the retained sine–Dirac adaptive
history: its grading trigger already selected high precision before it reached
15000 elements. The P2 solver is unchanged. The source/data manifest records
review-time hashes; it does not retroactively certify generation revisions for
retained data.

## Controlled warm-start experiment

```bash
python -m experiments.key.final_delivery warm
```

This replaces the CSV and initialization figure. On identical uniform meshes
with 80, 320 and 1280 elements it compares cold, primal-only and Riesz starts.
Each uses the same coarse solution, fine problem and stopping settings.
The reported median is over three trials, excluding the common coarse solve
but including transfer, assembly, PDAS, indicators and diagnostic checks.
It excludes true-error evaluation and is not total adaptive runtime.

By contrast, convergence-history `wall_seconds` starts **after** transfer and
ends **before** true-error quadrature, marking, refinement and output. Do not
compare these columns as if they measured the same work.

## Precision diagnostics and provenance

```bash
python -m experiments.validation.solvers.precision_review
```

This writes a small P2 controlled quadratic check under
`results/validation/submission/` and a separate figure under
`figures/presentation/precision_controlled.png`. It never writes final curves.
It compares the retained binary64 physical solver against the current gap/
precision solver; it is **not** a recovered historical obstacle experiment.
The uploaded archive did not contain the earlier failing regular-load P2
history. See `docs/PRECISION_DIAGNOSTIC.md` for the limitation and safe replay.

Original P1 histories are preserved under
`results/validation/submission/pre_precision_fix/`. The recorded
`p1_tail_precision.csv` predates the broadened trigger and compares the same
meshes and fixed active sets. Repeating `precision_review --p1-tail` now uses
the corrected solver and should produce agreement with the high-precision
check, rather than recreate the earlier error.
