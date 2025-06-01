#!/usr/bin/env python3
"""
Compute negative energy integrals for refined wormhole metrics using real T^{00}(r).

This script:
1. Extracts T^{00}(r,t) from exotic_matter_density.tex
2. Converts the LaTeX expression to a numeric function via SymPy
3. Numerically integrates ∫|T^{00}(r)| dV over the throat region

WORKFLOW INTEGRATION:
1. Run metric_refinement.py → produces refined_metrics.ndjson (with "b0" field)
2. Run compute_negative_energy.py → produces negative_energy_integrals.ndjson  
3. Feed results to design_control_field.py or other downstream analysis

USAGE:
    python compute_negative_energy.py \\
        --refined metric_engineering/outputs/refined_metrics.ndjson \\
        --tex metric_engineering/exotic_matter_density.tex \\
        --out metric_engineering/outputs/negative_energy_integrals.ndjson \\
        --factor 10.0

DEPENDENCIES:
    pip install sympy scipy numpy python-ndjson
"""

import argparse
import ndjson
import os
import re

import numpy as np
from scipy.integrate import quad
from sympy import Symbol, lambdify, Abs, pi
from sympy.parsing.latex import parse_latex


def read_ndjson(path):
    """Load a list of dicts from an NDJSON file."""
    with open(path) as f:
        return ndjson.load(f)


def write_ndjson(obj_list, path):
    """Write a list of dicts to an NDJSON file."""
    with open(path, 'w') as f:
        writer = ndjson.writer(f)
        for obj in obj_list:
            writer.writerow(obj)


def extract_T00_latex(tex_path):
    """
    Read the .tex file and extract the first occurrence of T^{00}(r, ...) =
    ... enclosed in \\[ \\]. Returns the raw LaTeX string of the right‐hand side.
    """
    with open(tex_path, 'r') as f:
        full_tex = f.read()

    # Find all display‐math blocks \\[ ... \\]
    display_math_matches = re.findall(r"\\\\\\[(.*?)\\\\\\]", full_tex, flags=re.DOTALL)
    if not display_math_matches:
        raise ValueError(f"No display‐math (\\\\[ ... \\\\]) found in {tex_path}.")

    # Look for the one that contains T^{00}(r
    for block in display_math_matches:
        if "T^{00}" in block:
            # Example block: "  T^{00}(r,t) = <some LaTeX expression>  "
            # Split at the '=' sign and take the RHS
            parts = block.split("=", 1)
            if len(parts) != 2:
                continue
            rhs = parts[1].strip()
            return rhs

    raise ValueError(f"Could not locate a T^{{00}}(...) = ... equation in {tex_path}.")


def build_numeric_T00(latex_rhs, assume_t_constant=True):
    """
    Convert the LaTeX RHS of T^{00}(r,t) into a SymPy expression, then lambdify it.

    If `assume_t_constant` is True, we substitute t -> 0, so T00 becomes a function of r only.
    """
    # Create symbols
    r, t = Symbol('r', positive=True), Symbol('t')

    # Some LaTeX‐to‐SymPy preprocessing:
    #
    # 1. Replace \\left( ... \\right) with ( ... ) 
    # 2. Remove "..." truncation if it appears (user may need to fix the TeX manually).
    #
    # Note: parse_latex is sensitive to pure‐LaTeX syntax. You may need to adapt or simplify the input.
    cleaned = latex_rhs
    cleaned = cleaned.replace(r"\\left(", "(").replace(r"\\right)", ")")
    # If you see "..." inside, you must manually fill in the missing TeX before parsing.
    if "..." in cleaned:
        raise ValueError(
            "The extracted T^{00} expression contains '...'.\n"
            "Please open exotic_matter_density.tex and ensure the full T^{00}(r) "
            "expression is there (no ellipses) so SymPy can parse it."
        )

    # Now parse into a SymPy expression
    try:
        T00_sym = parse_latex(cleaned)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LaTeX into SymPy: {e}")

    # If time‐dependence is present and we want T00(r) alone, substitute t -> 0
    if assume_t_constant:
        T00_r_sym = T00_sym.subs(t, 0)
    else:
        T00_r_sym = T00_sym

    # Lambdify: numeric function of r
    T00_num = lambdify(r, T00_r_sym, 'numpy')
    return T00_num


def numeric_negative_energy_integral(T00_num, b0, r_max):
    """
    Compute ∫ |T00(r)| * dV = ∫_{r=b0}^{r=r_max} |T00(r)| * 4π r^2 dr
    using SciPy's quad integrator.
    """
    def integrand(r_val):
        return abs(T00_num(r_val)) * (4.0 * np.pi * r_val**2)

    # Use a simple adaptive quadrature. You can tweak epsabs/epsrel as needed.
    result, error_estimate = quad(integrand, b0, r_max, epsabs=1e-9, epsrel=1e-9)
    return result


def compute_negative_energy(
    refined_metrics_path,
    tex_T00_path,
    output_path,
    outer_radius_factor=10.0
):
    """
    For each entry in refined_metrics.ndjson, read its b0 and shape‐function ansatz,
    then:
      1) Extract T^{00}(r) from exotic_matter_density.tex
      2) Build a numeric T00(r) function via SymPy
      3) Integrate |T00(r)| * 4π r^2 dr from r=b0 to r=outer_radius_factor*b0
      4) Write { label, b0, negative_energy_integral } to output NDJSON
    """
    # 1) Load refined metrics (each record MUST have field "b0")
    metrics = read_ndjson(refined_metrics_path)

    # 2) Extract the raw LaTeX for T^{00}(r,t)
    latex_rhs = extract_T00_latex(tex_T00_path)

    # 3) Build numeric T00(r) function (drop t by substituting t=0)
    T00_numeric = build_numeric_T00(latex_rhs, assume_t_constant=True)

    results = []
    for entry in metrics:
        label = entry.get("label", "unknown_refined")
        if "b0" not in entry:
            raise KeyError(f"Entry '{label}' is missing key 'b0' in {refined_metrics_path}.")

        b0 = float(entry["b0"])
        r_max = outer_radius_factor * b0

        # 4) Numerically integrate
        neg_integral = numeric_negative_energy_integral(T00_numeric, b0, r_max)

        results.append({
            "label": label,
            "parent_solution": entry.get("parent_solution", "unknown"),
            "b0": b0,
            "throat_radius": entry.get("throat_radius", b0),
            "integration_range": {"r_min": b0, "r_max": r_max},
            "negative_energy_integral": neg_integral,
            "computation_method": "numerical_quadrature_sympy"
        })

    # 5) Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # 6) Write the results
    write_ndjson(results, output_path)
    print(f"Wrote {len(results)} entries to {output_path}")
    print(f"Used T^{00}(r) from {tex_T00_path} with numerical integration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute negative-energy integrals |T^{00}| dV for each refined metric."
    )
    parser.add_argument(
        '--refined', required=True,
        help="Path to refined_metrics.ndjson (each record must include 'b0')."
    )
    parser.add_argument(
        '--tex', required=True,
        help="Path to exotic_matter_density.tex (must contain a full T^{00}(r,t) = ...)."
    )
    parser.add_argument(
        '--out', required=True,
        help="Output NDJSON file for negative-energy integrals."
    )
    parser.add_argument(
        '--factor', type=float, default=10.0,
        help="If b0 is the throat radius, integrate from r=b0 to r=factor*b0."
    )

    args = parser.parse_args()
    compute_negative_energy(
        refined_metrics_path=args.refined,
        tex_T00_path=args.tex,
        output_path=args.out,
        outer_radius_factor=args.factor
    )
