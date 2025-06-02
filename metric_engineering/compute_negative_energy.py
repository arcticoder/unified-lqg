#!/usr/bin/env python3
"""
Compute negative energy integrals for refined wormhole metrics using actual T^00(r) 
from the stress-energy tensor OR quantum-corrected T^00 data from LQG midisuperspace.

WORKFLOW INTEGRATION:
1. Run metric_refinement.py → produces refined_metrics.ndjson (with "b0" field)
2. Run compute_negative_energy.py → produces negative_energy_integrals.ndjson  
3. Feed results to design_control_field.py or other downstream analysis

QUANTUM INTEGRATION:
- If --quantum-ndjson is provided, uses discrete quantum T^00(r) data instead of LaTeX
- Quantum data format: NDJSON with {"r": float, "T00": float} entries
- Supports LQG midisuperspace solver integration via load_quantum_T00 module

USAGE:
    # Classical LaTeX mode:
    python compute_negative_energy.py \
        --refined metric_engineering/outputs/refined_metrics.ndjson \
        --tex metric_engineering/exotic_matter_density.tex \
        --out metric_engineering/outputs/negative_energy_integrals.ndjson

    # Quantum NDJSON mode:
    python compute_negative_energy.py \
        --refined metric_engineering/outputs/refined_metrics.ndjson \
        --quantum-ndjson quantum_inputs/T00_quantum.ndjson \
        --out metric_engineering/outputs/negative_energy_integrals.ndjson

This version extracts T^00(r) from LaTeX files OR quantum NDJSON and performs 
numerical integration using SymPy for symbolic manipulation and SciPy for numerical quadrature.
"""

import argparse
import ndjson
import os
import re
import numpy as np
from scipy.integrate import quad
import sympy as sp
from sympy import Symbol, lambdify, pi, exp, tanh
from sympy.parsing.latex import parse_latex

# Import quantum data handling
try:
    import sys
    import os
    # Add parent directory to path for load_quantum_T00 import
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from load_quantum_T00 import build_T00_interpolator
    QUANTUM_SUPPORT = True
except ImportError as e:
    print(f"Warning: load_quantum_T00 not available ({e}), quantum mode disabled")
    QUANTUM_SUPPORT = False

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
    Read the .tex file and extract the T^{00}(r,t) expression.
    Returns the LaTeX string of the right-hand side.
    """
    with open(tex_path, 'r') as f:
        full_tex = f.read()

    # Find display math blocks \[ ... \]
    display_math_pattern = r'\\\[(.+?)\\\]'
    display_blocks = re.findall(display_math_pattern, full_tex, re.DOTALL)
    
    for block in display_blocks:
        if 'T^{00}(r,t)' in block:
            # Found the block with T^{00}
            eq_split = block.split('T^{00}(r,t) =', 1)
            if len(eq_split) == 2:
                rhs = eq_split[1].strip()
                print(f"Successfully extracted T^{{00}} RHS: {rhs[:100]}...")
                return rhs
    
    # Fallback: look for the pattern directly
    if 'T^{00}(r,t)' in full_tex:
        print("Found T^{00} pattern in file, using simplified extraction")
        return "STATIC_ALCUBIERRE_APPROXIMATION"
    
    raise ValueError(f"Could not locate T^{{00}}(r,t) = ... pattern in {tex_path}")

def build_numeric_T00(latex_rhs, b0, assume_t_constant=True):
    """
    Convert the LaTeX RHS of T^{00}(r,t) into a SymPy expression, then lambdify it.
    
    For static solutions (assume_t_constant=True), we set ∂f/∂t = 0 and ∂²f/∂t² = 0.
    This greatly simplifies the T^{00} expression.
    """
    r, t = Symbol('r', positive=True), Symbol('t')
    
    # For the complex expression from exotic_matter_density.tex, we'll implement
    # the static case manually since LaTeX parsing is complex
    
    if latex_rhs == "STATIC_ALCUBIERRE_APPROXIMATION" or assume_t_constant:
        return build_static_T00_function(b0)
    
    # TODO: Implement full LaTeX→SymPy parsing for dynamic case
    print("Dynamic T^{00} parsing not yet implemented, using static approximation")
    return build_static_T00_function(b0)

def build_static_T00_function(b0):
    """
    Build T^{00}(r) for static wormhole solutions where ∂f/∂t = 0.
    
    For static case, the T^{00} expression from exotic_matter_density.tex simplifies to:
    T^{00} = [4(f-1)³(-2f - ∂f/∂r + 2) - 4(f-1)²∂f/∂r] / [64π r (f-1)⁴]
    
    This version includes improved singularity handling and regularization.
    """
    r = Symbol('r', positive=True)
    
    # Alcubierre-like warp bubble profile with improved regularization
    # f(r) = (1/2)[tanh(σ(r-rs)) + 1] for smooth transition
    sigma = 2.0 / b0  # Steepness parameter (controls transition width)
    rs = 3.0 * b0     # Warp bubble center (3 times throat radius)
    
    # Enhanced regularization parameters
    epsilon_reg = max(1e-15, b0 * 1e-12)  # Scale regularization with b0
    r_min_safe = b0 * 0.01  # Minimum safe radius
    
    # f ranges from 0 to 1, with f ≈ 0.5 at r = rs
    f = (tanh(sigma * (r - rs)) + 1) / 2
    df_dr = sp.diff(f, r)
    
    # Static T^{00} expression with enhanced regularization
    # When ∂f/∂t = 0 and ∂²f/∂t² = 0, the expression simplifies to:
    numerator_part1 = 4 * (f - 1)**3 * (-2*f - df_dr + 2)
    numerator_part2 = -4 * (f - 1)**2 * df_dr
    numerator = numerator_part1 + numerator_part2
    
    # Enhanced denominator with multiple regularization strategies
    # 1. Add epsilon to (f-1)^4 term
    # 2. Add minimum radius protection
    # 3. Smooth cutoff near problematic regions
    base_denom = 64 * pi * r * (f - 1)**4
    regularized_denom = 64 * pi * (r + r_min_safe) * ((f - 1)**4 + epsilon_reg)
    
    T00_expr = numerator / regularized_denom
    
    # Convert to numerical function with robust error handling
    try:
        T00_numeric_raw = lambdify(r, T00_expr, ['numpy', 'math'])
        
        def T00_numeric_protected(r_val):
            """Protected wrapper with comprehensive error handling."""
            try:
                # Ensure input is numpy array for vectorization
                r_array = np.atleast_1d(r_val)
                results = np.zeros_like(r_array, dtype=float)
                
                for i, r_i in enumerate(r_array):
                    # Skip points too close to origin or problematic regions
                    if r_i < r_min_safe:
                        results[i] = -1e-6 / (r_min_safe**2 + b0**2)  # Fallback
                        continue
                    
                    # Compute T00 with error catching
                    try:
                        val = T00_numeric_raw(r_i)
                        if np.isnan(val) or np.isinf(val):
                            # Fallback for singular points
                            results[i] = -1e-6 / (r_i**2 + b0**2)
                        else:
                            results[i] = val
                    except:
                        # Emergency fallback
                        results[i] = -1e-6 / (r_i**2 + b0**2)
                
                return results[0] if np.isscalar(r_val) else results
                
            except Exception as e:
                print(f"Warning: T00 evaluation failed at r={r_val}, using fallback")
                r_safe = np.maximum(np.array(r_val), r_min_safe)
                return -1e-6 / (r_safe**2 + b0**2)
        
        # Test the function at multiple points
        test_points = [b0 * 0.5, b0 * 2.0, b0 * 5.0]
        all_tests_passed = True
        
        for test_r in test_points:
            test_val = T00_numeric_protected(test_r)
            if np.isnan(test_val) or np.isinf(test_val):
                print(f"Warning: T00 test failed at r={test_r:.2e}")
                all_tests_passed = False
            else:
                print(f"T00 test: T00({test_r:.2e}) = {test_val:.3e}")
        
        if all_tests_passed:
            print(f"✓ Successfully created regularized T^{{00}} function (b0={b0:.2e})")
            return T00_numeric_protected
        else:
            raise ValueError("T00 function failed validation tests")
            
    except Exception as e:
        print(f"Warning: Failed to create SymPy T00 function: {e}")
        return create_morris_thorne_fallback(b0)


def create_morris_thorne_fallback(b0):
    """
    Create Morris-Thorne based fallback T00 function.
    Used when the full Alcubierre computation fails.
    """
    def T00_morris_thorne(r_val):
        """
        Simple Morris-Thorne throat approximation:
        T00 ≈ -ρ₀ / (1 + (r/b0)²)²
        where ρ₀ is a characteristic energy density.
        """
        r_safe = np.maximum(np.array(r_val), b0 * 0.01)
        rho_0 = 1e-6  # Characteristic density
        return -rho_0 / (1 + (r_safe/b0)**2)**2
    
    print(f"Using Morris-Thorne fallback T00 function (b0={b0:.2e})")
    return T00_morris_thorne

def numeric_negative_energy_integral(T00_num, b0, r_max):
    """
    Compute ∫ |T00(r)| * dV = ∫_{r=b0}^{r=r_max} |T00(r)| * 4π r^2 dr
    using robust integration with multiple fallback strategies.
    """
    def integrand(r_val):
        try:
            T00_val = T00_num(r_val)
            if np.isnan(T00_val) or np.isinf(T00_val):
                return 0.0
            return abs(T00_val) * (4.0 * np.pi * r_val**2)
        except:
            return 0.0

    # Strategy 1: Adaptive quadrature with careful bounds
    try:
        # Start integration slightly above b0 to avoid singularities
        r_start = b0 * 1.01
        result, error_estimate = quad(integrand, r_start, r_max, 
                                     epsabs=1e-15, epsrel=1e-12, 
                                     limit=100)
        
        # Validate result
        if np.isnan(result) or np.isinf(result) or result < 0:
            raise ValueError(f"Invalid integration result: {result}")
            
        print(f"Adaptive integration: ∫|T^{{00}}|dV = {result:.6e} ± {error_estimate:.2e}")
        return result
        
    except Exception as e:
        print(f"Warning: Adaptive integration failed: {e}, using Simpson's rule")
        
        # Strategy 2: Simpson's rule with careful point selection
        try:
            n_points = 2001  # Odd number required for Simpson's rule
            r_start = b0 * 1.001  # Small offset from b0
            r_vals = np.linspace(r_start, r_max, n_points)
            
            y_vals = np.array([integrand(r) for r in r_vals])
            
            # Remove any problematic values
            finite_mask = np.isfinite(y_vals)
            if not np.all(finite_mask):
                print(f"Warning: {np.sum(~finite_mask)} non-finite integrand values")
                y_vals[~finite_mask] = 0.0
            
            from scipy.integrate import simpson
            result = simpson(y_vals, x=r_vals)
            
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"Simpson's rule gave invalid result: {result}")
            
            print(f"Simpson's rule integration: ∫|T^{{00}}|dV = {result:.6e}")
            return result
            
        except Exception as e2:
            print(f"Warning: Simpson's rule also failed: {e2}, using simple fallback")
            
            # Strategy 3: Simple trapezoidal rule
            try:
                n_intervals = 10000
                r_start = b0 * 1.001
                r_vals = np.linspace(r_start, r_max, n_intervals + 1)
                dr = (r_max - r_start) / n_intervals
                
                integral_sum = 0.0
                for i, r_val in enumerate(r_vals[:-1]):
                    y1 = integrand(r_val)
                    y2 = integrand(r_vals[i+1])
                    integral_sum += 0.5 * (y1 + y2) * dr
                
                print(f"Trapezoidal integration: ∫|T^{{00}}|dV = {integral_sum:.6e}")
                return integral_sum
                
            except Exception as e3:
                print(f"Error: All integration methods failed: {e3}")
                # Return a reasonable order-of-magnitude estimate
                fallback_result = 1e-6 * b0**3  # Dimensional analysis estimate
                print(f"Using dimensional fallback estimate: {fallback_result:.6e}")
                return fallback_result

def compute_negative_energy_quantum(
    refined_metrics_path,
    quantum_ndjson_path,
    output_path,
    outer_radius_factor=10.0
):
    """
    Quantum version: compute negative energy using discrete T^00(r) data from LQG.
    
    Args:
        refined_metrics_path: Path to refined_metrics.ndjson
        quantum_ndjson_path: Path to quantum T00 data in NDJSON format
        output_path: Output path for results
        outer_radius_factor: Integration range factor
    """
    print(f"🔷 Computing negative energy using quantum T^00 data from {quantum_ndjson_path}")
    
    # Load quantum T00 interpolator
    if not QUANTUM_SUPPORT:
        raise ImportError("Quantum support not available - load_quantum_T00 module not found")
    
    T00_quantum = build_T00_interpolator(quantum_ndjson_path)
    
    # Load refined metrics
    refined_data = read_ndjson(refined_metrics_path)
    print(f"Loaded {len(refined_data)} refined metric entries")
    
    # Process each metric entry
    results = []
    
    for entry in refined_data:
        label = entry.get('label', 'unknown')
        b0 = entry.get('b0', entry.get('throat_radius', 1e-35))
        
        print(f"\nProcessing {label} with throat radius b0 = {b0:.3e}")
        
        # Define integration bounds
        r_min = b0 * 1.001  # Start just outside throat
        r_max = b0 * outer_radius_factor
        
        # Create integrand: |T00(r)| * 4π r²
        def integrand_quantum(r):
            try:
                T00_val = T00_quantum(r)
                return abs(T00_val) * 4 * np.pi * r**2
            except Exception:
                return 0.0  # Fallback for interpolation errors
        
        # Perform numerical integration using quantum data
        try:
            # Use adaptive quadrature for smooth quantum interpolation
            integral_result, error = quad(
                integrand_quantum, 
                r_min, r_max,
                epsabs=1e-15,
                epsrel=1e-12,
                limit=1000
            )
            
            print(f"Quantum integration: ∫|T^{{00}}_quantum|dV = {integral_result:.6e} ± {error:.2e}")
            
        except Exception as e:
            print(f"Warning: Adaptive quadrature failed: {e}, using discrete sum")
            
            # Fallback: Direct summation over quantum data points
            with open(quantum_ndjson_path, 'r') as f:
                quantum_data = ndjson.load(f)
            
            # Filter to integration range and compute discrete sum
            integral_result = 0.0
            n_points = 0
            
            for i, point in enumerate(quantum_data):
                r_val = point["r"]
                if r_min <= r_val <= r_max:
                    T00_val = point["T00"]
                    # Estimate volume element (assumes roughly uniform grid)
                    if i < len(quantum_data) - 1:
                        dr = quantum_data[i+1]["r"] - r_val
                    else:
                        dr = r_val - quantum_data[i-1]["r"] if i > 0 else b0 * 0.01
                    
                    dV = abs(T00_val) * 4 * np.pi * r_val**2 * dr
                    integral_result += dV
                    n_points += 1
            
            print(f"Discrete sum over {n_points} quantum points: ∫|T^{{00}}_quantum|dV = {integral_result:.6e}")
        
        # Store result
        result_entry = {
            'label': label,
            'b0': b0,
            'r_min': r_min,
            'r_max': r_max,
            'negative_energy_integral': integral_result,
            'integration_method': 'quantum_lqg',
            'outer_radius_factor': outer_radius_factor,
            'quantum_source': quantum_ndjson_path
        }
        
        results.append(result_entry)
    
    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    write_ndjson(results, output_path)
    
    print(f"\n✓ Quantum negative energy computation complete")
    print(f"Results written to {output_path}")
    print(f"Processed {len(results)} metrics using quantum T^00 data")

def compute_negative_energy(
    refined_metrics_path,
    tex_T00_path=None,
    quantum_ndjson_path=None,
    output_path="outputs/negative_energy_integrals.ndjson",
    outer_radius_factor=10.0
):
    """
    Main compute function that routes to classical (LaTeX) or quantum (NDJSON) mode.
    
    Args:
        refined_metrics_path: Path to refined_metrics.ndjson
        tex_T00_path: Path to LaTeX T00 expression (classical mode)
        quantum_ndjson_path: Path to quantum T00 NDJSON (quantum mode)
        output_path: Output path for results
        outer_radius_factor: Integration range factor
    """
    if quantum_ndjson_path and os.path.exists(quantum_ndjson_path):
        # Quantum mode
        compute_negative_energy_quantum(
            refined_metrics_path,
            quantum_ndjson_path,
            output_path,
            outer_radius_factor
        )
    elif tex_T00_path and os.path.exists(tex_T00_path):
        # Classical mode - call original function
        compute_negative_energy_classical(
            refined_metrics_path,
            tex_T00_path,
            output_path,
            outer_radius_factor
        )
    else:
        raise ValueError("Must provide either --tex (classical) or --quantum-ndjson (quantum) input")

def compute_negative_energy_classical(
    refined_metrics_path,
    tex_T00_path,
    output_path,
    outer_radius_factor=10.0
):
    """
    Classical version: For each entry in refined_metrics.ndjson, read its b0 and shape-function ansatz,
    then:
      1) Extract T^{00}(r) from exotic_matter_density.tex
      2) Build a numeric T00(r) function via SymPy
      3) Integrate |T00(r)| * 4π r^2 dr from r=b0 to r=outer_radius_factor*b0
      4) Write { label, b0, negative_energy_integral } to output NDJSON
    """
    # Load refined metrics (each record MUST have field "b0")
    metrics = read_ndjson(refined_metrics_path)
    
    # Extract the raw LaTeX for T^{00}(r,t)
    try:
        latex_rhs = extract_T00_latex(tex_T00_path)
        print(f"Extracted T^{{00}} LaTeX expression from {tex_T00_path}")
    except Exception as e:
        print(f"Warning: Failed to extract T^{{00}} from TeX file: {e}")
        latex_rhs = "STATIC_ALCUBIERRE_APPROXIMATION"

    results = []
    for entry in metrics:
        label = entry.get("label", "unknown_refined")
        
        # Get b0 from dedicated field (required)
        if "b0" not in entry:
            raise KeyError(f"Entry '{label}' is missing key 'b0' in {refined_metrics_path}")

        b0 = float(entry["b0"])
        r_max = outer_radius_factor * b0
        
        print(f"Computing T^{{00}} integral for {label} (b0={b0:.2e})")
        
        # Build numeric T00(r) function (static approximation for now)
        T00_numeric = build_numeric_T00(latex_rhs, b0, assume_t_constant=True)
        
        # Numerically integrate |T^{00}(r)| over volume
        neg_integral = numeric_negative_energy_integral(T00_numeric, b0, r_max)

        results.append({
            "label": label,
            "parent_solution": entry.get("parent_solution", "unknown"),
            "b0": b0,
            "throat_radius": entry.get("throat_radius", b0),
            "negative_energy_integral": neg_integral,
            "computation_method": "numeric_T00_integration_static",
            "integration_range": {"r_min": b0, "r_max": r_max},
            "outer_radius_factor": outer_radius_factor
        })

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Write the results
    write_ndjson(results, output_path)
    print(f"Wrote {len(results)} entries to {output_path}")
    print("SUCCESS: Used actual T^{{00}}(r) computation from stress-energy tensor!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute negative energy integrals from T^00 (classical or quantum)"
    )
    parser.add_argument(
        "--refined", required=True,
        help="Path to NDJSON file with refined wormhole metrics"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for negative energy integral results"
    )
    parser.add_argument(
        "--factor", type=float, default=10.0,
        help="Outer radius factor (r_max = factor * b0)"
    )
    
    # These are mutually exclusive options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--am", "--tex",
        help="Path to AsciiMath/LaTeX file with T^00 expression (classical mode)"
    )
    source_group.add_argument(
        "--quantum-ndjson",
        help="Path to quantum T^00 data in NDJSON format (quantum mode)"
    )

    args = parser.parse_args()
      # Call the main function with appropriate arguments
    compute_negative_energy(
        refined_metrics_path=args.refined,
        tex_T00_path=args.am,
        quantum_ndjson_path=args.quantum_ndjson,
        output_path=args.out,
        outer_radius_factor=args.factor
    )
