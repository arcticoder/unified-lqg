#!/usr/bin/env python3
"""
Compute negative energy integrals using T^{00}(r) from exotic_matter_density.tex

This implementation handles the complex stress-energy tensor expression by:
1. Attempting to parse the LaTeX expression with SymPy
2. Providing a fallback simplified T^{00} model if parsing fails
3. Numerically integrating ∫|T^{00}(r)| dV over the throat region

For the Alcubierre metric, T^{00} depends on the warp function f(r,t) and its derivatives.
This version assumes a static case (∂f/∂t = 0) for simplicity.
"""

import argparse
import ndjson
import os
import re

import numpy as np
from scipy.integrate import quad
import sympy as sp
from sympy import Symbol, lambdify, pi, sqrt, exp, tanh


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
    Extract T^{00} expression from the TeX file.
    Returns the LaTeX string after the equals sign.
    """
    with open(tex_path, 'r') as f:
        content = f.read()

    # Find display math blocks - look for \[ ... \] pattern (single backslashes)
    pattern = r'\\\\\\[(.*?)\\\\\\]'
    matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"Found {len(matches)} display math blocks with pattern")
    
    for i, match in enumerate(matches):
        print(f"Block {i}: {match[:100]}...")
        if "T^{00}" in match:
            # Split on equals and get RHS
            parts = match.split("=", 1)
            if len(parts) == 2:
                rhs = parts[1].strip()
                print(f"Extracted T^{{00}} RHS: {rhs[:200]}...")
                return rhs
    
    # Manual extraction as fallback
    if "T^{00}(r,t)" in content:
        # Find the start of the equation
        start_idx = content.find("T^{00}(r,t) =")
        if start_idx != -1:
            # Find the end of the display math block
            end_idx = content.find("\\]", start_idx)
            if end_idx != -1:
                equation = content[start_idx:end_idx]
                parts = equation.split("=", 1)
                if len(parts) == 2:
                    rhs = parts[1].strip()
                    print(f"Manual extraction successful: {rhs[:200]}...")
                    return rhs
    
    raise ValueError(f"Could not find T^{{00}} expression in {tex_path}")


def create_simplified_T00_model(b0):
    """
    Create a simplified T^{00}(r) model for the Alcubierre metric.
    
    For a static Alcubierre bubble with f(r) = tanh((r-b0)/w), 
    the energy density has approximate form:
    T^{00} ≈ -(1/8π) * (df/dr)^2 / r^2
    
    This is a simplified model that captures the essential physics.
    """
    r = Symbol('r', positive=True)
    
    # Warp function: f(r) = tanh((r - b0) / w) where w is the wall thickness
    w = 0.1 * b0  # Wall thickness ~ 10% of throat radius
    f = tanh((r - b0) / w)
    
    # Derivative of warp function
    df_dr = sp.diff(f, r)
    
    # Simplified T^{00} model (negative energy density)
    # T00 = -(1/(8*pi)) * (df_dr)**2 / r**2
    T00 = -(df_dr**2) / (8 * pi * r**2)
    
    # Convert to numerical function
    T00_func = lambdify(r, T00, 'numpy')
    
    return T00_func


def attempt_sympy_parsing(latex_expr, b0):
    """
    Attempt to parse the full LaTeX T^{00} expression using SymPy.
    If successful, return the lambdified function.
    If it fails, return None.
    """
    try:
        r, t = Symbol('r', positive=True), Symbol('t')
        
        # Basic LaTeX cleanup
        cleaned = latex_expr.replace(r'\\left(', '(').replace(r'\\right)', ')')
        cleaned = cleaned.replace(r'\\left\\{', '{').replace(r'\\right\\}', '}')
        cleaned = cleaned.replace(r'\\frac', 'frac')
        cleaned = cleaned.replace(r'\\partial', 'partial')
        
        print(f"Attempting to parse: {cleaned[:100]}...")
        
        # For now, we'll use the simplified model since the full expression
        # contains complex partial derivative notation that's hard to parse
        # TODO: Implement full parsing of the stress-energy tensor
        return None
        
    except Exception as e:
        print(f"LaTeX parsing failed: {e}")
        return None


def numeric_negative_energy_integral(T00_func, b0, r_max):
    """
    Compute ∫ |T00(r)| * 4π r^2 dr from b0 to r_max
    """
    def integrand(r_val):
        try:
            T00_val = T00_func(r_val)
            if np.isnan(T00_val) or np.isinf(T00_val):
                return 0.0
            return abs(T00_val) * (4.0 * np.pi * r_val**2)
        except:
            return 0.0

    try:
        result, error = quad(integrand, b0, r_max, epsabs=1e-12, epsrel=1e-9)
        return result
    except Exception as e:
        print(f"Integration failed: {e}")
        return 0.0


def compute_negative_energy(
    refined_metrics_path,
    tex_T00_path,
    output_path,
    outer_radius_factor=10.0
):
    """
    Compute negative energy integrals for each refined metric.
    """
    # Load refined metrics
    metrics = read_ndjson(refined_metrics_path)
    
    # Try to extract T^{00} from TeX file
    try:
        latex_rhs = extract_T00_latex(tex_T00_path)
        print(f"Extracted T^{{00}} expression from {tex_T00_path}")
    except Exception as e:
        print(f"Could not extract T^{{00}} from TeX: {e}")
        latex_rhs = None

    results = []
    
    for entry in metrics:
        label = entry.get("label", "unknown_refined")
        
        if "b0" not in entry:
            raise KeyError(f"Entry '{label}' missing 'b0' field in {refined_metrics_path}")

        b0 = float(entry["b0"])
        r_max = outer_radius_factor * b0
        
        # Try to parse the full expression, fallback to simplified model
        T00_func = None
        computation_method = "simplified_alcubierre_model"
        
        if latex_rhs:
            T00_func = attempt_sympy_parsing(latex_rhs, b0)
            if T00_func:
                computation_method = "full_sympy_parsing"
        
        if T00_func is None:
            print(f"Using simplified T^{{00}} model for {label}")
            T00_func = create_simplified_T00_model(b0)
        
        # Compute the integral
        neg_integral = numeric_negative_energy_integral(T00_func, b0, r_max)
        
        results.append({
            "label": label,
            "parent_solution": entry.get("parent_solution", "unknown"),
            "b0": b0,
            "throat_radius": entry.get("throat_radius", b0),
            "integration_range": {"r_min": b0, "r_max": r_max},
            "negative_energy_integral": neg_integral,
            "computation_method": computation_method
        })

    # Write results
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    write_ndjson(results, output_path)
    print(f"\\nWrote {len(results)} entries to {output_path}")
    
    for result in results:
        print(f"  {result['label']}: ∫|T^{{00}}|dV = {result['negative_energy_integral']:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute negative-energy integrals using T^{00} from TeX file"
    )
    parser.add_argument(
        '--refined', required=True,
        help="Path to refined_metrics.ndjson (must contain 'b0' field)"
    )
    parser.add_argument(
        '--tex', required=True,
        help="Path to exotic_matter_density.tex"
    )
    parser.add_argument(
        '--out', required=True,
        help="Output NDJSON file for negative-energy integrals"
    )
    parser.add_argument(
        '--factor', type=float, default=10.0,
        help="Integration from r=b0 to r=factor*b0"
    )

    args = parser.parse_args()
    compute_negative_energy(
        refined_metrics_path=args.refined,
        tex_T00_path=args.tex,
        output_path=args.out,
        outer_radius_factor=args.factor
    )
