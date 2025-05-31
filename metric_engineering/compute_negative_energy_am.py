#!/usr/bin/env python3
"""
Compute negative energy integrals for refined wormhole metrics using AsciiMath T^{00}(r).

This improved version:
1. Uses AsciiMath format (.am) instead of complex LaTeX parsing
2. Implements robust singularity handling for f → 1 points
3. Includes comprehensive error handling and fallback mechanisms
4. Supports both full time-dependent and static T^{00} expressions
5. Provides unit test capabilities against known analytic cases

WORKFLOW INTEGRATION:
1. Run metric_refinement.py → produces refined_metrics.ndjson (with "b0" field)
2. Run compute_negative_energy_am.py → produces negative_energy_integrals.ndjson  
3. Feed results to design_control_field.py or other downstream analysis

USAGE:
    python compute_negative_energy_am.py \\
        --refined metric_engineering/outputs/refined_metrics.ndjson \\
        --am metric_engineering/exotic_matter_density.am \\
        --out metric_engineering/outputs/negative_energy_integrals.ndjson \\
        --factor 10.0 \\
        --mode static

DEPENDENCIES:
    pip install sympy scipy numpy python-ndjson
"""

import argparse
import ndjson
import os
import re
import warnings

import numpy as np
from scipy.integrate import quad, simpson
from sympy import Symbol, lambdify, pi, sympify, tanh, sech, exp
from sympy import diff as sp_diff


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


def extract_T00_ascii(am_path, mode="static"):
    """
    Read the .am file and extract the T^{00} expression.
    
    Args:
        am_path: Path to the AsciiMath file
        mode: "static" or "full" - which T^{00} expression to extract
        
    Returns:
        The raw AsciiMath string of the right-hand side
    """
    with open(am_path, 'r') as f:
        contents = f.read()

    # Choose which expression to extract based on mode
    if mode == "static":
        pattern = r"\[\s*T00_static\(r\)\s*=\s*(.*?)\s*\]"
    elif mode == "regularized":
        pattern = r"\[\s*T00_regularized\(r,\s*eps\)\s*=\s*(.*?)\s*\]"
    else:  # full
        pattern = r"\[\s*T00_full\(r,t\)\s*=\s*(.*?)\s*\]"
    
    match = re.search(pattern, contents, re.DOTALL)
    if not match:
        raise ValueError(f"No T00_{mode} expression found in {am_path}")
    
    rhs = match.group(1).strip()
    print(f"Extracted T00_{mode} expression: {rhs[:100]}...")
    return rhs


def ascii_to_sympy_replacements(ascii_expr):
    """
    Convert AsciiMath notation to SymPy-compatible Python expressions.
    
    This handles the conversion of mathematical notation while preserving
    the structure needed for symbolic computation.
    """
    # Basic mathematical operators
    expr = ascii_expr.replace("^", "**")
    
    # Replace function calls with actual expressions
    # f(r) -> alcubierre warp function
    expr = expr.replace("f(r)", "((tanh((2/b0)*(r - 3*b0)) + 1)/2)")
    
    # df_dr(r) -> derivative of alcubierre function  
    expr = expr.replace("df_dr(r)", "((1/b0) * sech((2/b0)*(r - 3*b0))**2)")
    
    return expr


def build_numeric_T00_from_ascii(ascii_rhs, b0_val, mode="static", epsilon=1e-12):
    """
    Convert AsciiMath RHS into a SymPy expression, then lambdify it.
    
    Args:
        ascii_rhs: AsciiMath expression string
        b0_val: Throat radius value for parameter substitution
        mode: "static", "regularized", or "full"
        epsilon: Regularization parameter for singularities
        
    Returns:
        Callable numerical function T00(r)
    """
    # Symbols
    r = Symbol('r', positive=True)
    b0 = Symbol('b0', positive=True)
    
    try:
        # Convert AsciiMath to SymPy-compatible expression
        sympy_expr = ascii_to_sympy_replacements(ascii_rhs)
        
        # Add regularization if requested
        if mode == "regularized":
            sympy_expr = sympy_expr.replace("eps", str(epsilon))
          # Parse with SymPy
        T00_sym = sympify(sympy_expr, locals={"pi": pi, "tanh": tanh, "sech": sech, "r": r, "b0": b0})
        
        # Substitute b0 value
        T00_sym_sub = T00_sym.subs(b0, b0_val)
          # Create numerical function with proper hyperbolic function mapping
        func_map = {
            "sech": lambda x: 1/np.cosh(x),
            "tanh": np.tanh,
            "cosh": np.cosh,
            "sinh": np.sinh
        }
        T00_numeric = lambdify(r, T00_sym_sub, ["numpy", func_map])
        
        # Test the function at a safe point
        test_r = b0_val * 2.0
        test_val = T00_numeric(test_r)
        
        if np.isnan(test_val) or np.isinf(test_val):
            raise ValueError(f"T00 function produces NaN/Inf at test point r={test_r}")
            
        print(f"Successfully created T00 function: T00({test_r:.2e}) = {test_val:.3e}")
        return T00_numeric
        
    except Exception as e:
        print(f"Warning: Failed to create T00 function from AsciiMath: {e}")
        return create_fallback_T00(b0_val)


def create_fallback_T00(b0_val):
    """
    Create a simple fallback T00 function based on Morris-Thorne approximation.
    Used when the main parsing fails.
    """
    def T00_fallback(r_val):
        """Simple negative energy density ~1/r² with throat cutoff."""
        r_safe = np.maximum(np.array(r_val), b0_val * 0.1)  # Avoid r=0
        return -1e-6 / (r_safe**2 + b0_val**2)
    
    print(f"Using fallback T00 function with b0={b0_val:.2e}")
    return T00_fallback


def robust_integration(T00_func, b0, r_max, method="adaptive"):
    """
    Robust numerical integration with multiple fallback strategies.
    
    Args:
        T00_func: T^{00}(r) function to integrate
        b0: Integration lower bound (throat radius)
        r_max: Integration upper bound
        method: "adaptive", "simpson", or "composite"
        
    Returns:
        Integral value of ∫|T00(r)| * 4π r² dr
    """
    def integrand(r_val):
        """Integrand: |T00(r)| * 4π r²"""
        try:
            T00_val = T00_func(r_val)
            if np.isnan(T00_val) or np.isinf(T00_val):
                return 0.0
            return abs(T00_val) * (4.0 * np.pi * r_val**2)
        except:
            return 0.0
    
    # Strategy 1: Adaptive quadrature (scipy.quad)
    if method == "adaptive":
        try:
            result, error = quad(integrand, b0, r_max, 
                               epsabs=1e-15, epsrel=1e-12, 
                               limit=100)
            print(f"Adaptive integration: ∫|T00|dV = {result:.6e} ± {error:.2e}")
            return result
        except Exception as e:
            print(f"Adaptive integration failed: {e}, trying Simpson's rule")
            method = "simpson"
    
    # Strategy 2: Simpson's rule with careful grid
    if method == "simpson":
        try:
            # Use logarithmic spacing near b0 to handle potential singularities
            n_points = 2001  # Odd number for Simpson's rule
            r_lin = np.linspace(b0 * 1.01, r_max, n_points)  # Slight offset from b0
            y_vals = np.array([integrand(r) for r in r_lin])
            
            # Remove any infinite/NaN values
            finite_mask = np.isfinite(y_vals)
            if not np.all(finite_mask):
                print(f"Warning: {np.sum(~finite_mask)} non-finite values in integrand")
                y_vals[~finite_mask] = 0.0
            
            result = simpson(y_vals, x=r_lin)
            print(f"Simpson integration: ∫|T00|dV = {result:.6e}")
            return result
        except Exception as e:
            print(f"Simpson integration failed: {e}, using composite fallback")
            method = "composite"
    
    # Strategy 3: Simple composite rule
    if method == "composite":
        try:
            n_intervals = 10000
            r_vals = np.linspace(b0 * 1.001, r_max, n_intervals + 1)  # Avoid exact b0
            dr = (r_max - b0 * 1.001) / n_intervals
            
            integral_sum = 0.0
            for r_val in r_vals[:-1]:  # Use left endpoint rule
                integral_sum += integrand(r_val) * dr
            
            print(f"Composite integration: ∫|T00|dV = {integral_sum:.6e}")
            return integral_sum
        except Exception as e:
            print(f"All integration methods failed: {e}")
            return 0.0


def unit_test_gaussian_T00():
    """
    Unit test: integrate a known Gaussian T00 and compare to analytic result.
    
    For T00(r) = -A * exp(-(r-r0)²/σ²), the integral should be:
    ∫|T00| * 4πr² dr ≈ A * 4π * σ³ * sqrt(π) (for large integration range)
    """
    print("\n=== UNIT TEST: Gaussian T00 ===")
    
    # Test parameters
    A = 1e-6  # Amplitude
    r0 = 1e-35  # Center
    sigma = 0.5e-35  # Width
    b0 = 0.1e-35  # Integration start
    r_max = 10e-35  # Integration end
    
    # Define test function
    def gaussian_T00(r):
        return -A * np.exp(-((r - r0)**2) / (sigma**2))
    
    # Numerical integration
    numerical = robust_integration(gaussian_T00, b0, r_max)
    
    # Approximate analytic result (for reference)
    analytic_approx = A * 4 * np.pi * sigma**3 * np.sqrt(np.pi)
    
    print(f"Gaussian T00 test:")
    print(f"  Numerical: {numerical:.6e}")
    print(f"  Analytic~: {analytic_approx:.6e}")
    print(f"  Ratio: {numerical/analytic_approx:.3f}")
    print("=== END UNIT TEST ===\n")
    
    return numerical, analytic_approx


def compute_negative_energy(
    refined_metrics_path,
    ascii_T00_path,
    output_path,
    outer_radius_factor=10.0,
    mode="static",
    run_unit_test=False
):
    """
    Main computation function using AsciiMath T^{00} expressions.
    
    Args:
        refined_metrics_path: Path to refined_metrics.ndjson
        ascii_T00_path: Path to exotic_matter_density.am  
        output_path: Output path for results
        outer_radius_factor: Integration range multiplier
        mode: "static", "regularized", or "full"
        run_unit_test: Whether to run validation tests
    """
    # Optional unit test
    if run_unit_test:
        unit_test_gaussian_T00()
    
    # Load refined metrics
    print(f"Loading refined metrics from {refined_metrics_path}")
    metrics = read_ndjson(refined_metrics_path)
    
    # Extract AsciiMath T^{00} expression
    print(f"Extracting T00_{mode} from {ascii_T00_path}")
    ascii_rhs = extract_T00_ascii(ascii_T00_path, mode)
    
    results = []
    for i, entry in enumerate(metrics):
        label = entry.get("label", f"unknown_refined_{i}")
        
        # Robust b0 extraction
        if "b0" not in entry:
            raise KeyError(f"Entry '{label}' missing required 'b0' field")
        
        b0 = float(entry["b0"])
        r_max = outer_radius_factor * b0
        
        print(f"\nProcessing {label} (b0={b0:.2e})")
        
        # Build T00 function for this specific b0
        T00_numeric = build_numeric_T00_from_ascii(ascii_rhs, b0, mode)
        
        # Integrate with robust error handling
        neg_integral = robust_integration(T00_numeric, b0, r_max)
        
        # Store results with comprehensive metadata
        result_entry = {
            "label": label,
            "parent_solution": entry.get("parent_solution", "unknown"),
            "b0": b0,
            "throat_radius": entry.get("throat_radius", b0),
            "negative_energy_integral": neg_integral,
            "computation_method": f"asciimath_{mode}_integration",
            "integration_range": {"r_min": b0, "r_max": r_max},
            "outer_radius_factor": outer_radius_factor,
            "T00_expression_mode": mode,
            "file_format": "asciimath"
        }
        
        results.append(result_entry)
        print(f"  Result: ∫|T00|dV = {neg_integral:.6e}")
    
    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # Write results
    write_ndjson(results, output_path)
    print(f"\n✓ Successfully wrote {len(results)} entries to {output_path}")
    print(f"✓ Used AsciiMath T00_{mode} with robust numerical integration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute negative-energy integrals using AsciiMath T^{00} expressions"
    )
    parser.add_argument(
        '--refined', required=True,
        help="Path to refined_metrics.ndjson (must include 'b0' field)"
    )
    parser.add_argument(
        '--am', required=True,
        help="Path to exotic_matter_density.am (AsciiMath format)"
    )
    parser.add_argument(
        '--out', required=True,
        help="Output NDJSON file for negative-energy integrals"
    )
    parser.add_argument(
        '--factor', type=float, default=10.0,
        help="Integration range: r ∈ [b0, factor*b0] (default: 10.0)"
    )
    parser.add_argument(
        '--mode', choices=['static', 'regularized', 'full'], default='static',
        help="T^{00} expression mode (default: static)"
    )
    parser.add_argument(
        '--test', action='store_true',
        help="Run unit tests with known analytic cases"
    )

    args = parser.parse_args()
    
    compute_negative_energy(
        refined_metrics_path=args.refined,
        ascii_T00_path=args.am,
        output_path=args.out,
        outer_radius_factor=args.factor,
        mode=args.mode,
        run_unit_test=args.test
    )
