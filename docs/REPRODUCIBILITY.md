# Reproducibility guide

One command reproduces everything that is backed by validated, deterministic code:

```bash
scripts/reproduce_all.sh        # ~45 s; no random state outside seeded NumPy generators
```

It runs the regression tests, the closure stress scan, the AMR convergence study and the
Schwarzschild prescription comparison, and writes to `outputs/repro/` and
`outputs/constraint_tests/closure_stress_test.json`.

Requirements: Python 3, numpy, scipy, sympy, torch (autograd only), matplotlib, pytest.

## What is validated

| Result | Code | Validation |
|---|---|---|
| Constraint closure of the spherically symmetric lattice constraints | `scripts/closure_stress_test.py` | Classical limit closes with error ∝ a² (7e-2 → 1e-3, n = 32 → 256); a deliberately broken constraint is rejected by > 100×. Tests: `tests/test_closure_stress.py`. |
| AMR integrals and convergence | `adaptive_mesh_refinement.py`, `scripts/amr_convergence_study.py` | Known-answer field (mass `0.2π`, energy `0.625π`); uniform grids converge at order 1.99; patches tile the domain exactly. Tests: `tests/test_amr.py`. |
| Schwarzschild holonomy-prescription coefficients | `alternative_polymer_prescriptions.py` (default run) | Deterministic (identical output across runs apart from timings). Not independently validated against the literature. |

### Findings that came out of the validation

* **Closure.** The naive polymer constraint (`K_φ → sin(δK_φ)/δ`) has a real anomaly ∝ (δK)²
  that survives lattice refinement; `γ` and `μ` enter only as `δ = γμ`, so no preferred `γ`
  exists. The form `K_φ² → sin²(δK_φ)/δ²`, `2K_φK_x → sin(2δK_φ)K_x/δ` closes with the
  deformed structure function `cos(2δK_φ)·E^x/(E^φ)²` to within 1.5× the classical lattice
  floor over the whole scan. Radial inhomogeneity only; no angular perturbations.
* **AMR.** Refined patches must be re-evaluated from the field (`update_function`); if they are
  only interpolated from the coarse data there is no accuracy gain (energy error 4.5e-1
  against 2e-2). With re-evaluation, AMR reaches the uniform-grid error with ≈1.5–1.9× fewer
  degrees of freedom on the test field. This is a modest gain, not orders of magnitude.

## What is NOT valid (do not cite)

* `constraint_closure_tests.py`, `constraint_anomaly_scanner.py`, `unified_qg/constraint_closure.py`
  and the stored `outputs/constraint_tests/constraint_closure_results.json`: these cannot detect
  an anomaly (they evaluate `{H,H}` or commute multiples of one matrix). Each module now says so.
* `unified_qg/amr.py`: `refine_or_coarsen` is a placeholder and never refines. Anything built on
  it (`demo_unified_qg.py`, `discoveries_runner.py`, `enhanced_qg_simplified.py`) computes error
  maps on a single coarse patch; no adaptive refinement takes place. Use
  `adaptive_mesh_refinement.AdaptiveMeshRefinement` instead.
* **Kerr part of `alternative_polymer_prescriptions.py` (`--kerr`).** It used to crash before
  producing anything; it now completes, but:
  * the `compare_kerr_prescriptions` stage fails to evaluate its coefficients ("Cannot convert
    expression to float"), so the summary lines "Kerr prescriptions analyzed: 0" and "most
    stable prescription" are meaningless;
  * horizon shifts evaluated at `r = r₊` are exactly 0 for Thiemann and NaN for AQEL, which
    suggests the μ-expansion is singular at the horizon; the shift formula
    `δr₊ = α M²/r₊` is carried over from the original code and has not been validated;
  * Bojowald and Improved coefficients come out identical for every spin, which may be by
    construction of those prescriptions but has not been checked.
  Hard-coded "fallback" constants used when a computation fails are now flagged (a `*` in the
  console table, a `fallback_coefficients` CSV column, and a warning line in the summary), and
  failed horizon-shift evaluations are recorded as errors, not zeros.
* Five figures cited in the manuscript have no generating code anywhere in the repository:
  `evolution_results.png`, `geometry_catalysis_plot.png`, `amr_error_plot.png`,
  `enhanced_amr_convergence.png`, `constraint_heatmap.png` (the last one was removed from the text).

## Defects fixed along the way

* `adaptive_mesh_refinement.py`: inconsistent node/cell spacing; child patches not aligned with
  parent nodes; parents re-refined each cycle (duplicate overlapping children); sibling children
  overlapping each other; conservation totals summed over overlapping patches (mass doubled in
  refined regions); "Richardson extrapolation" across levels covering different regions, with
  NaN convergence rates.
* `alternative_polymer_prescriptions.py`: `__main__` block in the middle of the file (Kerr path
  raised `NameError`); a truncated duplicate of the Kerr section at the end of the file that
  overrode the complete `compare_kerr_prescriptions`; `display_prescription_coefficient_table`
  and `create_cross_prescription_summary_table` written for a different data layout; coefficient
  CSV all `N/A`; stray `temp_config.json` written to the working directory.
