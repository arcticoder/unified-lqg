`main.pdf` now builds at 175 pages instead of 3. The only errors left are five missing figure files, and those don't stop the build. I haven't committed anything.

**Why you only got 3 pages:** every file that `main.tex` pulls in with `\input` is a complete standalone paper, with its own `\documentclass`, preamble and `\end{document}`. LaTeX stopped at the first `\end{document}`, which is in `00-intro.tex`.

**What I changed**
- **[main.tex](unified-lqg/papers/main.tex):**
  - I added `\usepackage{standalone}`, which makes `\input` skip each chapter's preamble and `\end{document}`. Your chapter files still compile on their own.
  - I added `subcaption` and a `\graphicspath` for two figures that live elsewhere in the workspace.
  - I added definitions the chapters needed: `\sinc`, `\ket`, `underscore`, and about 20 Unicode characters (Greek letters, ≈, ≥ and similar).
- **[warp_bubble_proof.tex](unified-lqg/papers/warp_bubble_proof.tex):** a block of text was pasted into the middle of the `\usepackage{amsmath,amssymb}` line. I restored the preamble. The pasted block was a fuller version of the "Enhancement Strategies" list further down, so I replaced the shorter list with it.
- **[recent_discoveries.tex](unified-lqg/papers/recent_discoveries.tex):** I removed a stray `\end{document}` at line 832 that hid about 545 lines (Discoveries 127–131). I also wrapped an unbracketed `10^{-15}` in `$…$`.
- **[results_performance.tex](unified-lqg/papers/results_performance.tex):** I fixed a table with 7 column specs but 8 columns, and a `**` typo (`\textbf{Memory Efficiency**:`).
- **[alternative_prescriptions.tex](unified-lqg/papers/alternative_prescriptions.tex):** the bibliography ended with `</thebibliography>` instead of `\end{thebibliography}`.
- **[warp_feasibility_complete.tex](unified-lqg/papers/warp_feasibility_complete.tex):** another `10^{-15}` outside math mode.

**Still open**
1. **Missing figures.** Five PNGs don't exist anywhere in the workspace, so the PDF shows placeholder boxes: `evolution_results`, `geometry_catalysis_plot`, `amr_error_plot`, `enhanced_amr_convergence`, `constraint_heatmap`. They need to be regenerated. In an interactive terminal run, `pdflatex` will also pause at each one, so use `-interaction=nonstopmode`.
2. **[amr_quantum_gravity.tex](unified-lqg/papers/amr_quantum_gravity.tex)** is two drafts glued together. Everything after its first `\end{document}` (line 118) is ignored, including a second "Conclusions" and a `\bibliography{references}` that has no `.bib` file. I left it alone because choosing which draft to keep is your call.
3. **Unused file.** `constraint_closure_analysis_new.tex` is not included in `main.tex`, though it was probably meant to replace `constraint_closure_analysis.tex`.

The clean-up is largely source-level only. I haven't read the 175 pages for layout problems like overfull boxes.
<!-- ------ -->
---
Item 1 is done, and the result contradicts the paper. Items 2–4 are not started.

**Why the paper's closure claim isn't supported**

The "anomaly-free closure at γ ≈ 0.2377, μ ∈ [0.1, 0.5]" claim can't be traced to a real test. None of the existing closure checks can fail:
- [constraint_closure_tests.py](unified-lqg/constraint_closure_tests.py) computes `{H,H}`, which is zero by antisymmetry.
- [constraint_anomaly_scanner.py](unified-lqg/constraint_anomaly_scanner.py) and [unified_qg/constraint_closure.py](unified-lqg/unified_qg/constraint_closure.py) build `H[N]` and `H[M]` from the same matrix, so they always commute.
- The stored [closure_results.json](unified-lqg/outputs/constraint_tests/constraint_closure_results.json) reports exactly `0.0` everywhere. Its one non-vacuous test, `mixed_closure`, fails with a max violation of 16.8.
- 0.2377 is the Immirzi parameter (0.2375) that the code uses as an input. Nothing in the repo selects it.

**What I built:** [scripts/closure_stress_test.py](unified-lqg/scripts/closure_stress_test.py) and [tests/test_closure_stress.py](unified-lqg/tests/test_closure_stress.py). It computes the lattice Poisson bracket of the spherically symmetric Hamiltonian-constraint densities and compares it with the hypersurface-deformation algebra, with no fitted structure function. I validated it first:
- The classical constraint closes, with the error falling as `a²` (7e-2 → 1e-3 from n=32 to n=256).
- A deliberately broken constraint is off by 10–80×.
- A free-`W` fit I tried first was vacuous: it "closed" the broken constraint to 1e-15. I removed it.
- The four tests pass.

**Result:**
- **Anomaly grows with δK:** the naive polymer constraint (`K_φ → sin(δK_φ)/δ`) has a real anomaly that survives lattice refinement. It grows roughly as `(δK)²`, with defects of 4e-3, 2e-2 and 9e-2 at `δK = γμK` ≈ 0.24, 0.59 and 1.19.
- **No closure at large δK:** neither the undeformed nor the `cos(2δK_φ)` structure function closes the algebra there. The literature fixes this with counterterms, which I haven't added.
- **No special γ:** γ and μ enter only through the product `δ = γμ`, so a γ ≈ 0.2377 optimum can't exist in this model. At small δK the defect is just the lattice floor, which is trivial classical agreement.

**Limits**
- The perturbations are inhomogeneous in the radial direction only. "Beyond midisuperspace" (angular, non-symmetric) is not tested, and this model can't test it.
- I assumed `δ = γμ`, the holonomy of `A = γK`. That is my choice, not something from your code.

I haven't edited the paper's claim in `constraint_closure_analysis.tex`.

**Before items 2–3, I need your input**
- **Item 2:** no code for `H_int = λ∫√f R φ²` exists in the repo. What is `f`, and which background do you want it in (static spherical, FRW, or the warp ansatz)? "Particle creation = annihilation" needs a definite mode-equation setup.
- **Item 3:** `drude_model.tex` has presets but no code. For the warp ansatz, would you accept Casimir negative energy between Drude-Lorentz plates (a Lifshitz calculation) as the link from permittivity to the energy requirement? That is the only well-defined route I see.
- **Item 4:** I can start now. First I'd look at which scripts produce the five missing figures.

Do you want me to continue with item 4 while you decide on 2 and 3?
<!-- ------ -->
---
The manuscript fix, the counterterm search and the item 4 cleanup are done, and item 2 stays on hold as you agreed. Nothing is committed. `scripts/reproduce_all.sh` runs everything in about 45 s. The Kerr half of `alternative_polymer_prescriptions.py` is still not valid (details below).

**Manuscript text:**
- **Closure claim:** I withdrew the γ ≈ 0.2377 / μ ∈ [0.1, 0.5] claim in [constraint_closure_analysis.tex](papers/constraint_closure_analysis.tex) and [constraint_closure.tex](papers/constraint_closure.tex). The new text states what was measured, with a new table.
- **Unsupported numbers:** I also withdrew three things that had no backing in the repo:
  - the "representative result 3.2×10⁻¹⁰";
  - the quoted JSON excerpt, whose values appear in no stored output;
  - the `<10⁻⁶` matter-coupling commutator bound in §7.
- **Heatmap:** I removed the heatmap figure, which was missing anyway.
- **Pipeline summary:** the one-line "anomaly-free parameter regions identified" in `extended_pipeline_summary.tex` is corrected.
- **Build:** the PDF still builds, now at 176 pages.

**Counterterm search succeeded**
- **Form:** writing the kinetic term as `sin²(δK_φ)/δ²` and the cross term as `sin(2δK_φ)K_x/δ` gives a constraint that closes with the deformed structure function `cos(2δK_φ)·E^x/(E^φ)²`.
- **Result:** the worst defect across the full scan (μ 0.1–0.5, E-amplitude up to 0.9, K up to 10, n=128) is 8×10⁻³, against a classical lattice floor of 5×10⁻³. The undeformed structure function fails by up to 8.6.
- **Caveats:** the algebra is deformed, and `cos(2δK)` changes sign for `δK > π/4`, which is a signature change. It's also still radial-only and `γ` still drops out. I found the form by testing candidates, not from a derivation. I cited Bojowald–Brahma–Reyes (PRD 92, 045043) in the text from memory, so please verify that reference.
- **Status:** it's in the paper as §5.2 and covered by a test.

**Item 4: what I found and changed**
- **AMR module:** [adaptive_mesh_refinement.py](adaptive_mesh_refinement.py) had five real bugs:
  - the "conserved" mass doubled in refined regions;
  - parents were re-refined every cycle;
  - sibling patches overlapped;
  - the grid spacing convention was inconsistent;
  - the "Richardson extrapolation" compared different regions and gave NaN.
- **AMR validation:** with these fixed, a known-answer study ([scripts/amr_convergence_study.py](scripts/amr_convergence_study.py)) shows uniform grids converging at order 1.99. AMR reaches the same error with only about 1.5–1.9× fewer degrees of freedom, not the large gains the paper implies.
- **Interpolated refinement:** AMR patches that are only interpolated, as the old demo ran, give no accuracy gain at all.
- **Placeholder AMR class:** `unified_qg/amr.py` never refines (its refinement step is a placeholder), so three demo scripts that use it never refine either.
- **Prescription module:** `alternative_polymer_prescriptions.py` is deterministic for Schwarzschild. Its Kerr path used to crash, and I fixed several bugs. Fallback constants and failed evaluations are now flagged instead of passing as data.
- **Kerr results still invalid:** the Kerr comparison stage fails to evaluate its coefficients. Horizon shifts at `r₊` are exactly 0 for Thiemann and NaN for AQEL. Don't cite the Kerr output.
- **Docs:** [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) lists what is validated and what isn't.

**Still open**
- **Missing figures:** five figures cited in the paper have no generating code anywhere (`evolution_results`, `geometry_catalysis_plot`, `amr_error_plot`, `enhanced_amr_convergence`, and the removed `constraint_heatmap`).
- **Other claims not checked:** I didn't verify the "stable across 6 orders of magnitude" claim or the "O(10⁵) error reduction" claim in the pipeline summary. Given what I found, I'd treat them as unverified.
- **Broken test file:** `tests/test_advanced_refinements.py` has an `IndentationError` that predates my changes.
- **`docs/history.md`:** I didn't touch it.

**Item 2:** it stays on hold. It also needs your answers on what `f` is and which background to use. A natural next step is the Kerr horizon-shift evaluation, or extending the closure test beyond radial symmetry. Which would you like?
<!-- ------ -->
---