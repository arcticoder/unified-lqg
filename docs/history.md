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