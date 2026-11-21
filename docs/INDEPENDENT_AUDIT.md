# Independent submission review

18 November 2026 · English report due Friday 20 November; oral presentation
Tuesday 24 November. Reviewed the supplied ZIP, the newer attached English
LaTeX text, the attached 16-page PDF, and the actual final CSV histories.
The newer attached text superseded the repository's incomplete LaTeX draft.

## Outcome and scope

The final English report, README, protocol and canonical figures now agree with
the reviewed implementation. One remaining numerical precision defect was fixed,
three affected histories were rerun, and the finest warm-start iteration counts
were reproduced independently. The mathematical method and marking rule were
preserved. This review does not certify every optional or historical script.

### Critical — fixed before submission

1. **Large P1 meshes bypassed the precision safeguard.** The original activation
   depended only on mesh grading. Replaying the regular adaptive history and
   solving the same final active set with the existing 45-digit routine reduced
   L2 error from 4.41377e-10 to 8.60802e-12 at 276191 DOFs. The scaled residual
   was already tiny, so it did not expose the forward-error problem. Activation
   now also covers meshes with at least 15000 elements, mirroring the existing
   P2 policy. The regular P1 uniform/adaptive and sine–Dirac P1 uniform histories
   were rerun; original CSVs remain under `pre_precision_fix/`.
   **Location:** `src/p1_nested_large.py:386`; evidence:
   `results/validation/submission/p1_tail_precision.csv`.

2. **The report described a different primal transfer from the implementation.**
   The code prolongates `W=U-S` for sine–Dirac, then adds the new sine shift.
   Thus `U_h^0=P(U_H-S_H)+S_h`, not simply `P U_H`.
   The dual transfer is still `M_h P M_H^{-1} Lambda_H`.
   **Locations:** `experiments/key/inner_study.py:38`,
   `experiments/key/sine_dirac/convergence.py:67`,
   `experiments/key/sine_dirac/problem.py:1`. Corrected in the report and talk.

3. **Stale solver/KKT wording.** The final structured code has strict sign tests
   and no stated 1e-10 tie tolerance. Stable sets return from the inner solver;
   the experiment wrapper checks nodal feasibility, active multipliers and
   scaled free residual/complementarity. Relative-indicator exits additionally
   require absolute 1e-8 nodal KKT checks. The report now describes both layers
   and does not equate a small scaled residual with forward accuracy.
   **Locations:** `src/p1_nested_large.py:404`, `src/p2_nested_large.py:643`,
   `src/pdas_indicator.py:30`, `experiments/key/final_study.py:63`.

4. **Submission files were inconsistent.** The ZIP's report still contained
   unfinished sections and old experimental discussion; its protocol ordered
   already-completed reruns and used an obsolete talk length. Replaced the
   report with the newer attached foundation, corrected it, rebuilt its figures
   and PDF, and rewrote README/FINAL_EXPERIMENTS around the verified deliverables.

### Important — fixed or explicitly qualified

- **Nonemptiness:** continuous obstacles with nonpositive endpoint values do not
  alone ensure an H1 majorant with zero trace. For these benchmarks `psi<=0`,
  so `0 in K` is the simple valid argument. Existence uses closed convex K,
  coercivity and weak lower semicontinuity; strict convexity gives uniqueness.
- **P2 admissibility:** nodal constraints do not make `K_h` a subset of `K`.
  The recorded coarse-mesh minimum gap is about -2.12e-3 for sine–Dirac and
  -3.83e-4 for the regular case, relative to the interpolated obstacle.
  This limitation is now visible rather than hidden behind a generic caveat.
- **Riesz positivity:** a nonnegative dual vector need not produce a pointwise
  nonnegative consistent-mass reconstruction. The function represents a
  discrete functional; it is not a positive physical reaction approximation
  by construction.
- **Indicator derivation:** signs of jumps and volume residual are correct.
  The h weights follow local scaling, with geometry-dependent constants for
  the stated averaging argument. The half-weight convention yields interface
  weight `(h_left+h_right)/2`. No reliability theorem is asserted: multiplier
  reconstruction error is uncontrolled, P2 is nodally constrained and explicit
  Dirac residual terms are absent from the implemented indicator.
- **Regularity:** the regular-load solution is C1 and H2, with jumps of its
  second derivative at contact endpoints; it is not globally H3. The reference
  is an analytic construction with a numerically solved scalar free boundary,
  not an exact floating-point oracle.
- **Historical precision claim:** the failing regular-load P2 history and
  archived pre-fix solver revision are absent from the upload. Removed claims
  that this particular old curve had been independently recovered. Supplied
  a small controlled test of the retained binary64 vs current solver instead.
- **Verification holes:** the old verifier accepted nonfinite errors or
  inconsistent inner histories and checked the global minimum length rather
  than the stricter per-series limit. Strengthened `inner_study.verify`, including
  identity, history, stop and recorded KKT checks. Defaults now select the two
  core cases and minimum length 1e-12, rather than optional split-contact cases.
- **Timing boundaries:** convergence `wall_seconds` excludes transfer and true
  errors. Warm-start timings include transfer but exclude the common coarse
  solve. Neither is total adaptive runtime. Corrected documentation.
- **Dependencies:** Ollama moved to optional requirements. No service or API key
  is needed for the numerical core.

## Components independently verified

| Component | Evidence and result |
| --- | --- |
| Continuous formulation | Checked energy derivative, VI sign, distributional reaction and Dirac point evaluation in 1D |
| FEM bases/assembly | P1 hats, P2 basis on [-1,1] vs code on [0,1], derivatives, Jacobian factors, boundary elimination and SPD agree |
| KKT/PDAS | Maximum identity and fixed-set implication checked; small dense systems independently agree with structured solves |
| P2 condensation | Arbitrary active sets checked against independent dense free systems; free-midpoint Schur block is (1/h)[[1,-1],[-1,1]] |
| Mass/Riesz transfer | Small dense consistent mass solves; nested identities `A_H=P.T A_h P`, `M_H=P.T M_h P`; reconstructed transfer independently checked |
| Sine benchmark | Piecewise solution, jumps +2/+1, unchanged reaction, stable shift action and original-vs-gap regression |
| Regular benchmark | Contact endpoints 0.421398631728655 and 0.578601368271345; values, smooth fit, feasibility and contact reaction checked |
| Final protocol | Eight complete histories / 174 mesh solves; outer tolerance disabled; all stable-set exits; minimum inner ratio about 0.212 |
| Warm starts | All twelve counts at 1280 elements reproduced once; all three final solutions agree within 1e-8 (recorded differences zero) |
| Precision | Existing graded P1/P2 polynomial checks pass; new large-P1 public-path check passes; controlled P2 quadratic comparison recorded |

Commands and outputs are recorded in `results/validation/submission/`.
The source/data SHA-256 manifest is a review-time record. The original ZIP had
no Git history and lacked generation hashes per series. Therefore retained CSVs
cannot be retroactively assigned a verified historical commit. Five retained
histories were checked but not fully rerun; the change does not affect their
solver path. The sine–Dirac adaptive P1 grading trigger was already active before
its mesh reached 15000 elements (h_min about 2.33e-11 at 19002 elements).

## Remaining limitations and safely deferred work

- The regular P2 uniform L2 error increases from 4.69e-11 at 40959 DOFs to
  6.48e-11 at 81919 DOFs while H1 decreases. Independent solves of these same
  fine problems with the current 40-digit path reproduce the values. This is
  compatible with changing free-boundary alignment and the absence of an L2
  monotonicity guarantee; the cause is not uniquely proved. It is not labelled
  as a roundoff defect merely because the curve is nonmonotone. Final errors
  near 1e-13 also require caution about reference/finite-precision accuracy.
  Evidence: `p2_uniform_recheck.csv`; the analytic reference was used only
  as an initial guess for this diagnostic, never for final-study solves.
- Nodal P2 penetration and the absent indicator-reliability proof are method
  limitations, not issues cured by tighter linear solves. Changing the obstacle
  discretization or indicator would be a new study and is deferred.
- The 40/45-digit solves improve critical operations; assembled loads, mesh
  coordinates and returned arrays remain binary64. This is not arbitrary-
  precision computation of the entire FEM method.
- No benchmark has demonstrated savings from the inner 1e-3 indicator threshold.
  A genuinely inexact PDAS criterion requires additional mathematical work.
- Historical plotters may need omitted `.npz` files. Final report plots use CSVs
  and are reproducible without them. The optional finance/LLM extension was
  not part of this numerical-practicum audit.

## Bibliography verification

Checked publisher/author records for the three references and SciPy's official
L-BFGS-B documentation. DOIs are included in the report:

- Braess, third edition (2007): https://doi.org/10.1017/CBO9780511618635
- Brenner–Scott, third edition (2008): https://doi.org/10.1007/978-0-387-75934-0
- Hintermüller–Ito–Kunisch, SIAM J. Optim. 13(3), 865–888 (2002): https://doi.org/10.1137/S1052623401383558
- SciPy: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

Publisher catalogues use variant subtitles for Braess; the supplied book's
frontmatter identifies “Applications in Elasticity Theory”, retained here.
