# Final numerical experiments — reproducibility checklist

## Scope and rules

The final *core* study consists of eight mesh histories: `sine_dirac` and
`no_exact`, each with P1/P2 finite elements and uniform/adaptive meshes.
Use the existing maximum marking rule (`theta=0.5`) and L2-Riesz nested PDAS
initialization on successive meshes. The P1/P2 free-set solves use extended
precision when needed for strongly graded meshes; P2 also uses extended precision
on very large uniform meshes. The FEM method and marking are unchanged.

- Inner PDAS indicator tolerance: `1e-3`, safeguarded by the discrete KKT conditions.
- Outer mesh estimator stopping tolerance: **disabled** (`mesh_rel_tol=None`).
- Maximum DOFs: `300000` **as a ceiling, not a target**.
- Minimum post-bisection element size: normally `1e-12` on `(0,1)`.
  **Exception:** the very accurate `no_exact` P2 adaptive series uses
  `2e-9` to stop at the onset of the verified floating-point accuracy floor.
  This is a numerical-safeguard parameter, not an outer estimator tolerance.
- Maximum adaptive levels: `60`.
- Accept terminal reasons `dof_cap`, `mesh_resolution`, `max_levels`, or
  `nothing_marked`. Do not combine histories from incompatible protocols.

## Steps (execute from the repository root)

### 0. Check disk space and repository changes

```bash
df -h .
git status --short
```

Free at least several gigabytes before large runs. The source changes from
the performance and protocol patches should remain visible until committed.

### 1. Run validation (short)

```bash
python -m experiments.validation.solvers.final_checks
```

### 2. Regenerate the regular-load P2 adaptive curve after the precision fix

This **one** series must be rerun with its stronger mesh-resolution safeguard.
The old CSV was calculated using the less accurate binary64 P2 solve:

```bash
python -m experiments.key.inner_study run \
  --cases no_exact --degrees 2 --methods adaptive \
  --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 2e-9
```

The seven other core histories were completed in the November 16 run. Do not
rerun them unless verification finds a real problem. Historical `split20`,
`split40`, `split80` experiments remain archived/optional, not required for
this final eight-series core.

### 2b. Recompute the two L2-accuracy regression histories

The following commands must be run **once after** the P1/P2 precision fix.
Previous histories were generated with binary64 factorization and can show
false L2 error increases at fine resolutions. Both files are overwritten
only after each completed solve; keep the old CSVs in Git history if needed.

```bash
python -m experiments.key.inner_study run \
  --cases sine_dirac --degrees 1 --methods adaptive \
  --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 1e-12

python -m experiments.key.inner_study run \
  --cases no_exact --degrees 2 --methods uniform \
  --max-dofs 300000 --max-levels 60 --inner-tol 1e-3 --min-h 1e-12
```

### 3. Verify all eight final core histories

```bash
python -m experiments.key.final_delivery check
```

This calls the existing strict protocol verifier. It checks the CSV contents,
inner tolerance, absence of an outer stop, mesh resolution, DOF limits and
terminal reasons. Success writes `verified_summary.csv`; it does **not**
independently prove numerical accuracy or validate unexpected convergence
behaviour. Review the actual error curves too. The P1/P2 accuracy fixes have
independent graded-mesh polynomial regressions in `final_checks.py`.

### 4. Build the final convergence plots (no PDE solves)

```bash
python -m experiments.key.final_delivery plots
```

Output: `figures/key/inner_study/budget_300000/final/`

- `h1_convergence.png`: true H1-seminorm errors for both benchmarks.
- `l2_convergence.png`: true L2 errors for both benchmarks.
- `estimator_convergence.png`: residual estimator versus DOFs.

Although named `no_exact`, that case has an independently derived analytical
reference in `final_study.reference`, used for **error evaluation only**, not
to generate the FE solutions or the adaptive indicator.

### 5. Compare cold/primal/Riesz starts under the same inner stopping rule

```bash
python -m experiments.key.final_delivery warm
```

This independent experiment compares the three initializations on identical
uniform fine meshes (80, 320, 1280 elements), with P1/P2 and both benchmark
loads. It checks agreement with the cold-start solution to `1e-8`. Results:

- `results/key/inner_study/budget_300000/warm_starts_inner.csv`
- `figures/key/inner_study/budget_300000/final/initialization_comparison.png`

The common coarse solve is excluded from timing; transfer and fine-mesh solve
are included. These are **solver-stage**, not end-to-end adaptive timings.

### 6. Review the protocol and results

```bash
python -m experiments.key.final_delivery check
git status --short
```

Check `verified_summary.csv` for each series' last DOF count, stop reason,
minimum element size, and PDAS stop counts. The inner indicator threshold may
never trigger because active-set stability or the KKT safeguard takes priority.
This is a result to report rather than a reason to adjust the tolerance.

### 7. Update the English handout and the German talk

Use only verified figures for final core convergence claims. Historical smooth
and Dirac solver/indicator validation figures may remain as *clearly labelled*
validation figures, but do not describe them as final inner-protocol curves.
Check every numerical sentence and notation in `beyond_the_obstacle.tex`.
Fill its Abstract, Conclusion, Future work, and AI-use sections; rebuild
`beyond_the_obstacle.pdf` with your LaTeX setup and visually inspect it.

Keep the talk to ~8 slides, ~12–15 minutes of prepared explanation, leaving
space for technical questions (35-minute session total).

### 8. Commit only reviewed deliverables

```bash
git status --short
git diff --check
# Add files intentionally; do NOT use git add . while large results remain untracked.
git add experiments/key/final_delivery.py FINAL_EXPERIMENTS.md \
  experiments/key/inner_study.py experiments/key/final_study.py \
  experiments/key/sine_dirac/convergence.py src/error_indicators.py \
  README.md beyond_the_obstacle.tex
# Add selected final figures and verified CSVs separately after reviewing their sizes.
git diff --cached --stat
git commit -m "Finalize inner-tolerance experiments and reproducible figures"
```

Do not assume the above staged-source list covers every local modification;
review `git status` and adjust it. Commit final figures/CSVs only according to
the repository's storage and Git policy. Never include temporary CSVs from
interrupted runs, massive `.npz` files, or machine-specific environments.
