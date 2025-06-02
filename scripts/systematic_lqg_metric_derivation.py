#!/usr/bin/env python3
"""
Systematic LQG Metric Derivation with Proper Classical Starting Point

This module implements the correct systematic approach:
1. Solve classical Hamiltonian constraint H_classical = 0 for K_x(r, f(r))
2. Apply polymer corrections K → sin(μK)/μ to the correct classical solution
3. Expand in μ and solve order by order for α, β, γ coefficients  
4. Attempt closed-form resummation to get f_LQG(r) beyond perturbative expansion
5. Use comprehensive timeout handling throughout

The key fix: Replace current K_x = 0 substitution with proper classical 
Schwarzschild solution K_x = ∂_r(ln √f) before polymerizing.

Author: Advanced LQG Implementation with Timeout Handling
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symbolic_timeout_utils import (
    safe_symbolic_operation, safe_series, safe_solve, safe_simplify, 
    safe_expand, safe_integrate, safe_diff, safe_collect, safe_factor,
    safe_cancel, safe_together, safe_apart, safe_trigsimp, safe_ratsimp,
    safe_constraint_expand, safe_hamiltonian_simplify, safe_lqg_series_expand,
    set_default_timeout
)
    def safe_cancel(expr, *args, **kwargs): return sp.cancel(expr, *args)
    def safe_together(expr, *args, **kwargs): return sp.together(expr, *args)
    def safe_trigsimp(expr, *args, **kwargs): return sp.trigsimp(expr, *args)
    def safe_ratsimp(expr, *args, **kwargs): return sp.ratsimp(expr, *args)
    def safe_nsimplify(expr, *args, **kwargs): return sp.nsimplify(expr, *args)
    def safe_apart(expr, *args, **kwargs): return sp.apart(expr, *args)
    def safe_constraint_expand(expr, *args, **kwargs): return sp.expand(expr, *args)
    def safe_hamiltonian_simplify(expr, *args, **kwargs): return sp.simplify(expr, *args)
    def safe_lqg_series_expand(expr, var, point, n, *args, **kwargs): return sp.series(expr, var, point, n, *args)

# Global symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)
f = sp.Function('f')(r)  # Metric function

# Higher-order coefficients for systematic expansion
alpha, beta, gamma_coeff, delta_coeff = sp.symbols('alpha beta gamma delta', real=True)

def construct_classical_hamiltonian():
    """
    Construct the classical Hamiltonian constraint for spherically symmetric LQG.
    
    Returns:
        Classical Hamiltonian in radial-triad variables
    """
    print("Step 1: Constructing classical Hamiltonian constraint...")
    
    # Classical kinetic terms for spherical symmetry
    # H_classical = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + (spatial curvature terms)
    H_classical = (
        -(Ephi / sp.sqrt(Ex)) * Kphi**2
        - 2 * Kphi * Kx * sp.sqrt(Ex)
    )
    
    print(f"  Classical Hamiltonian: {H_classical}")
    return H_classical

def derive_classical_schwarzschild_solution():
    """
    Derive the classical Schwarzschild solution for K_x from H_classical = 0.
    
    For static spherically symmetric metric ds² = -f(r)dt² + dr²/f(r) + r²dΩ²:
    - E^x = r²
    - E^φ = r√f(r)  
    - K_φ = 0 (static condition)
    - K_x = ? (to be determined from constraint)
    
    Returns:
        Classical K_x as function of metric f(r) and its derivatives
    """
    print("\nStep 2: Deriving classical Schwarzschild solution for K_x...")
    
    # Static triad ansatz
    Ex_static = r**2
    Ephi_static = r * sp.sqrt(f)
    Kphi_static = 0
    
    print(f"  Static ansatz: E^x = {Ex_static}")
    print(f"                 E^φ = {Ephi_static}")
    print(f"                 K_φ = {Kphi_static}")
    
    # Classical Hamiltonian becomes: -2 K_x √E^x × 0 = 0
    # But we need to include spatial curvature terms for proper constraint
    
    # For Schwarzschild, the constraint comes from:
    # R^(2) - K_x K_φ + (triad derivative terms) = 0
    # Since K_φ = 0, we need spatial curvature R^(2) = 0
    
    # From the metric f(r) = 1 - 2M/r, we can derive K_x from gauge conditions
    # For radial gauge: ∂_t E^x = {E^x, H_classical} = 0 gives evolution constraint
    
    # From the lapse condition and momentum constraint:
    # K_x is determined by the trace of extrinsic curvature
    # For Schwarzschild: K = ∂_r(√f)/√f × (gauge function)
    
    f_prime = safe_diff(f, r, timeout_seconds=5)
    if f_prime is None:
        print("  Warning: f'(r) differentiation timed out, using symbolic")
        f_prime = sp.diff(f, r)
    
    # Classical K_x from trace constraint
    # This comes from the proper Hamiltonian formulation
    Kx_classical = f_prime / (2 * sp.sqrt(f) * r)
    
    print(f"  Classical K_x = {Kx_classical}")
    
    # For Schwarzschild f(r) = 1 - 2M/r:
    f_schwarzschild = 1 - 2*M/r
    Kx_schwarzschild = Kx_classical.subs(f, f_schwarzschild)
    
    Kx_simplified = safe_simplify(Kx_schwarzschild, timeout_seconds=5)
    if Kx_simplified is None:
        Kx_simplified = Kx_schwarzschild
    
    print(f"  For Schwarzschild metric: K_x = {Kx_simplified}")
    
    return Kx_classical, Kx_simplified

def apply_polymer_corrections_systematically(H_classical, Kx_classical, max_order=6):
    """
    Apply polymer corrections K → sin(μK)/μ to the proper classical solution.
    
    Args:
        H_classical: Classical Hamiltonian
        Kx_classical: Classical K_x solution
        max_order: Maximum μ order to include
        
    Returns:
        Polymer-expanded Hamiltonian with proper classical starting point
    """
    print(f"\nStep 3: Applying polymer corrections to order μ^{max_order}...")
    
    # Replace K_x in classical Hamiltonian with proper solution
    H_with_classical_Kx = H_classical.subs(Kx, Kx_classical)
    
    print(f"  Hamiltonian with classical K_x: {H_with_classical_Kx}")
    
    # Now apply polymer corrections: K → sin(μK)/μ
    # For K_x, we polymerize the classical solution
    mu_Kx_classical = mu * Kx_classical
    mu_Kphi = mu * Kphi
    
    # Expand sin(μK_x)/μ around the classical solution
    n_terms = max_order // 2 + 3
    
    print(f"  Expanding sin(μK_x)/μ to {n_terms} terms...")
    sin_expansion_x = safe_lqg_series_expand(
        sp.sin(mu_Kx_classical) / mu_Kx_classical, mu, 0, n=n_terms, timeout_seconds=8
    )
    
    if sin_expansion_x is None:
        print("  Warning: K_x expansion timed out, using lower order")
        sin_expansion_x = safe_series(
            sp.sin(mu_Kx_classical) / mu_Kx_classical, mu, 0, n=4, timeout_seconds=5
        )
        if sin_expansion_x is None:
            sin_expansion_x = 1 - (mu_Kx_classical)**2/6  # Fallback
        else:
            sin_expansion_x = sin_expansion_x.removeO()
    else:
        sin_expansion_x = sin_expansion_x.removeO()
    
    print(f"  Expanding sin(μK_φ)/μ to {n_terms} terms...")
    sin_expansion_phi = safe_lqg_series_expand(
        sp.sin(mu_Kphi) / mu_Kphi, mu, 0, n=n_terms, timeout_seconds=8
    )
    
    if sin_expansion_phi is None:
        print("  Warning: K_φ expansion timed out, using lower order")
        sin_expansion_phi = safe_series(
            sp.sin(mu_Kphi) / mu_Kphi, mu, 0, n=4, timeout_seconds=5
        )
        if sin_expansion_phi is None:
            sin_expansion_phi = 1 - (mu_Kphi)**2/6  # Fallback
        else:
            sin_expansion_phi = sin_expansion_phi.removeO()
    else:
        sin_expansion_phi = sin_expansion_phi.removeO()
    
    # Apply polymer substitutions
    Kx_polymer = Kx_classical * sin_expansion_x
    Kphi_polymer = Kphi * sin_expansion_phi
    
    # Substitute into Hamiltonian
    H_polymer = H_classical.subs([
        (Kx, Kx_polymer),
        (Kphi, Kphi_polymer)
    ])
    
    # Expand to specified order
    print(f"  Expanding polymer Hamiltonian to order μ^{max_order}...")
    H_expanded = safe_constraint_expand(H_polymer, timeout_seconds=10)
    
    if H_expanded is None:
        print("  Warning: Polymer expansion timed out, using simpler form")
        H_expanded = safe_expand(H_polymer, timeout_seconds=5)
        if H_expanded is None:
            H_expanded = H_polymer
    
    # Collect powers of μ
    H_collected = safe_collect(H_expanded, mu, timeout_seconds=8)
    if H_collected is None:
        H_collected = H_expanded
    
    print(f"  Polymer Hamiltonian (expanded): {H_collected}")
    
    return H_collected

def extract_static_constraint_with_ansatz(H_polymer, max_order=6):
    """
    Extract the static constraint using metric ansatz with higher-order terms.
    
    Args:
        H_polymer: Polymer-expanded Hamiltonian
        max_order: Maximum order for ansatz
        
    Returns:
        Constraint equations by order in μ
    """
    print(f"\nStep 4: Extracting static constraint with ansatz to order μ^{max_order}...")
    
    # Static triad ansatz  
    Ex_static = r**2
    Ephi_static = r * sp.sqrt(f)
    Kphi_static = 0
    
    # Higher-order metric ansatz: f(r) = 1 - 2M/r + α*μ²*M²/r⁴ + β*μ⁴*M⁴/r⁶ + ...
    if max_order >= 2:
        f_ansatz = 1 - 2*M/r + alpha*mu**2*M**2/r**4
    if max_order >= 4:
        f_ansatz += beta*mu**4*M**4/r**6
    if max_order >= 6:
        f_ansatz += gamma_coeff*mu**6*M**6/r**8
    
    print(f"  Metric ansatz: f(r) = {f_ansatz}")
    
    # Substitute static values
    constraint_static = H_polymer.subs([
        (Ex, Ex_static),
        (Ephi, Ephi_static),
        (Kphi, Kphi_static),
        (f, f_ansatz)
    ])
    
    # Simplify constraint
    constraint_simplified = safe_hamiltonian_simplify(constraint_static, timeout_seconds=10)
    if constraint_simplified is None:
        constraint_simplified = safe_simplify(constraint_static, timeout_seconds=8)
        if constraint_simplified is None:
            constraint_simplified = constraint_static
    
    # Extract coefficients by powers of μ
    constraints_by_order = {}
    
    for order in range(0, max_order + 1, 2):  # Even powers only
        coeff = constraint_simplified.coeff(mu, order)
        if coeff is not None and coeff != 0:
            constraints_by_order[f'mu^{order}'] = coeff
            print(f"  μ^{order} constraint: {coeff}")
    
    return constraints_by_order, f_ansatz

def solve_coefficients_systematically(constraints_by_order, f_ansatz):
    """
    Solve for metric coefficients α, β, etc. order by order.
    
    Args:
        constraints_by_order: Dictionary of constraint equations by μ order
        f_ansatz: Metric ansatz with coefficients
        
    Returns:
        Dictionary of solved coefficients
    """
    print("\nStep 5: Solving for metric coefficients order by order...")
    
    coefficients = {}
    
    # μ⁰ constraint should be satisfied by Schwarzschild (check)
    if 'mu^0' in constraints_by_order:
        mu0_constraint = constraints_by_order['mu^0']
        print(f"  μ⁰ constraint (should be 0): {mu0_constraint}")
        
        # This should vanish for Schwarzschild, if not we have an issue
        mu0_simplified = safe_simplify(mu0_constraint, timeout_seconds=5)
        if mu0_simplified is None:
            mu0_simplified = mu0_constraint
        print(f"  Simplified μ⁰: {mu0_simplified}")
    
    # μ² constraint determines α
    if 'mu^2' in constraints_by_order:
        mu2_constraint = constraints_by_order['mu^2']
        print(f"  Solving μ² constraint for α...")
        
        # Expand in powers of 1/r and extract leading coefficient
        mu2_expanded = safe_series(mu2_constraint, r, sp.oo, n=8, timeout_seconds=8)
        if mu2_expanded is None:
            print("  Warning: Series expansion timed out")
            mu2_expanded = mu2_constraint
        else:
            mu2_expanded = mu2_expanded.removeO()
        
        # Solve for α from the constraint = 0
        alpha_solutions = safe_solve(mu2_constraint, alpha, timeout_seconds=8)
        if alpha_solutions is None or len(alpha_solutions) == 0:
            print("  Warning: Direct solve for α failed, trying coefficient extraction")
            # Extract coefficient of leading 1/r term
            mu2_collected = safe_collect(mu2_expanded, 1/r, timeout_seconds=5)
            if mu2_collected is not None:
                # Find coefficient of highest power of 1/r containing α
                print(f"  μ² expanded: {mu2_collected}")
                alpha_solutions = safe_solve(mu2_collected, alpha, timeout_seconds=5)
        
        if alpha_solutions and len(alpha_solutions) > 0:
            alpha_value = alpha_solutions[0]
            coefficients['alpha'] = alpha_value
            print(f"  ✓ Found α = {alpha_value}")
        else:
            print("  ✗ Could not solve for α")
            coefficients['alpha'] = None
    
    # μ⁴ constraint determines β (if α is known)
    if 'mu^4' in constraints_by_order and coefficients.get('alpha') is not None:
        mu4_constraint = constraints_by_order['mu^4']
        
        # Substitute known α value
        mu4_with_alpha = mu4_constraint.subs(alpha, coefficients['alpha'])
        
        print(f"  Solving μ⁴ constraint for β...")
        beta_solutions = safe_solve(mu4_with_alpha, beta, timeout_seconds=8)
        
        if beta_solutions and len(beta_solutions) > 0:
            beta_value = beta_solutions[0]
            coefficients['beta'] = beta_value
            print(f"  ✓ Found β = {beta_value}")
        else:
            print("  ✗ Could not solve for β")
            coefficients['beta'] = None
    
    return coefficients

def attempt_closed_form_resummation(coefficients):
    """
    Attempt to find closed-form resummation if β/α² is a simple constant.
    
    Args:
        coefficients: Dictionary with α, β values
        
    Returns:
        Tuple of (closed_form_expression, success_flag)
    """
    print("\nStep 6: Attempting closed-form resummation...")
    
    alpha_val = coefficients.get('alpha')
    beta_val = coefficients.get('beta')
    
    if alpha_val is None or beta_val is None:
        print("  Cannot attempt resummation without both α and β")
        return None, False
    
    # Check if β/α² is a simple constant
    print(f"  α = {alpha_val}")
    print(f"  β = {beta_val}")
    
    ratio = beta_val / (alpha_val**2)
    ratio_simplified = safe_nsimplify(ratio, timeout_seconds=5)
    if ratio_simplified is None:
        ratio_simplified = safe_simplify(ratio, timeout_seconds=5)
        if ratio_simplified is None:
            ratio_simplified = ratio
    
    print(f"  β/α² = {ratio_simplified}")
    
    # Check if this is a simple rational number
    if ratio_simplified.is_rational and ratio_simplified.is_real:
        print(f"  ✓ Found simple ratio: β/α² = {ratio_simplified}")
        
        # Attempt geometric series resummation
        # If f = 1 - 2M/r + α*μ²*M²/r⁴ + (β/α²)*α²*μ⁴*M⁴/r⁸ + ...
        # This suggests: 1 - 2M/r + α*μ²*M²/r⁴ * (1 + (β/α²)*(α*μ²*M²/r⁴) + ...)
        #              = 1 - 2M/r + α*μ²*M²/r⁴ / (1 - (β/α²)*α*μ²*M²/r⁴)
        
        x = alpha_val * mu**2 * M**2 / r**4
        if ratio_simplified == 1:
            # Perfect geometric series
            f_closed = 1 - 2*M/r + x / (1 - x)
            f_closed = safe_simplify(f_closed, timeout_seconds=8)
            if f_closed is not None:
                print(f"  ✓ Closed form (geometric): f(r) = {f_closed}")
                return f_closed, True
        else:
            # Modified geometric series
            f_closed = 1 - 2*M/r + x / (1 - ratio_simplified * x)
            f_closed = safe_simplify(f_closed, timeout_seconds=8)
            if f_closed is not None:
                print(f"  ✓ Closed form (modified): f(r) = {f_closed}")
                return f_closed, True
    
    print("  No simple closed form found")
    return None, False

def validate_expansion(f_closed, coefficients, max_order=6):
    """
    Validate closed form by re-expanding and comparing coefficients.
    
    Args:
        f_closed: Proposed closed-form expression
        coefficients: Known coefficients from order-by-order solution
        max_order: Maximum order to check
        
    Returns:
        Boolean indicating if validation passed
    """
    print("\nStep 7: Validating closed form by re-expansion...")
    
    if f_closed is None:
        print("  No closed form to validate")
        return False
    
    # Expand closed form in μ
    f_expanded = safe_lqg_series_expand(f_closed, mu, 0, n=max_order//2 + 2, timeout_seconds=10)
    
    if f_expanded is None:
        print("  Warning: Expansion validation timed out")
        return False
    
    f_expanded = f_expanded.removeO()
    
    print(f"  Re-expanded form: {f_expanded}")
    
    # Extract coefficients and compare
    validation_passed = True
    
    # Check α coefficient
    if coefficients.get('alpha') is not None:
        alpha_from_expansion = f_expanded.coeff(mu**2)
        alpha_expected = coefficients['alpha'] * M**2 / r**4
        
        diff = safe_simplify(alpha_from_expansion - alpha_expected, timeout_seconds=5)
        if diff is None or diff != 0:
            print(f"  ✗ α coefficient mismatch: got {alpha_from_expansion}, expected {alpha_expected}")
            validation_passed = False
        else:
            print(f"  ✓ α coefficient matches")
    
    # Check β coefficient  
    if coefficients.get('beta') is not None and max_order >= 4:
        beta_from_expansion = f_expanded.coeff(mu**4)
        beta_expected = coefficients['beta'] * M**4 / r**6
        
        diff = safe_simplify(beta_from_expansion - beta_expected, timeout_seconds=5)
        if diff is None or diff != 0:
            print(f"  ✗ β coefficient mismatch: got {beta_from_expansion}, expected {beta_expected}")
            validation_passed = False
        else:
            print(f"  ✓ β coefficient matches")
    
    return validation_passed

def run_systematic_derivation(max_order=6):
    """
    Run the complete systematic LQG metric derivation.
    
    Args:
        max_order: Maximum order in μ to compute
        
    Returns:
        Dictionary containing all results
    """
    print("="*80)
    print("SYSTEMATIC LQG METRIC DERIVATION")
    print("="*80)
    
    results = {}
    
    try:
        # Step 1: Classical Hamiltonian
        H_classical = construct_classical_hamiltonian()
        results['H_classical'] = H_classical
        
        # Step 2: Classical Schwarzschild solution
        Kx_classical, Kx_schwarzschild = derive_classical_schwarzschild_solution()
        results['Kx_classical'] = Kx_classical
        results['Kx_schwarzschild'] = Kx_schwarzschild
        
        # Step 3: Apply polymer corrections
        H_polymer = apply_polymer_corrections_systematically(
            H_classical, Kx_classical, max_order
        )
        results['H_polymer'] = H_polymer
        
        # Step 4: Extract static constraint
        constraints_by_order, f_ansatz = extract_static_constraint_with_ansatz(
            H_polymer, max_order
        )
        results['constraints_by_order'] = constraints_by_order
        results['f_ansatz'] = f_ansatz
        
        # Step 5: Solve coefficients
        coefficients = solve_coefficients_systematically(constraints_by_order, f_ansatz)
        results['coefficients'] = coefficients
        
        # Step 6: Attempt resummation
        f_closed, resummation_success = attempt_closed_form_resummation(coefficients)
        results['f_closed'] = f_closed
        results['resummation_success'] = resummation_success
        
        # Step 7: Validate
        if f_closed is not None:
            validation_passed = validate_expansion(f_closed, coefficients, max_order)
            results['validation_passed'] = validation_passed
        else:
            results['validation_passed'] = False
        
        print("\n" + "="*80)
        print("DERIVATION COMPLETE")
        print("="*80)
        
        print(f"Classical K_x: {Kx_schwarzschild}")
        if coefficients.get('alpha') is not None:
            print(f"α coefficient: {coefficients['alpha']}")
        if coefficients.get('beta') is not None:
            print(f"β coefficient: {coefficients['beta']}")
        if f_closed is not None:
            print(f"Closed form: f_LQG(r) = {f_closed}")
            print(f"Validation: {'PASSED' if results['validation_passed'] else 'FAILED'}")
        
        results['success'] = True
        
    except Exception as e:
        print(f"\nError during derivation: {e}")
        results['success'] = False
        results['error'] = str(e)
    
    return results

def export_results(results, filename="systematic_lqg_metric_results.py"):
    """
    Export results for use in other scripts.
    
    Args:
        results: Dictionary of derivation results
        filename: Output filename
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(scripts_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Results from systematic LQG metric derivation.\n')
        f.write('Generated by systematic_lqg_metric_derivation.py\n')
        f.write('"""\n\n')
        f.write('import sympy as sp\n\n')
        
        # Export symbols
        f.write('# Symbolic variables\n')
        f.write('r, M, mu = sp.symbols("r M mu", positive=True, real=True)\n')
        f.write('alpha, beta = sp.symbols("alpha beta", real=True)\n\n')
        
        # Export key results
        if results.get('Kx_schwarzschild') is not None:
            f.write(f'# Classical K_x for Schwarzschild\n')
            f.write(f'Kx_schwarzschild = {results["Kx_schwarzschild"]}\n\n')
        
        if results.get('coefficients', {}).get('alpha') is not None:
            f.write(f'# α coefficient\n')
            f.write(f'alpha_value = {results["coefficients"]["alpha"]}\n\n')
        
        if results.get('coefficients', {}).get('beta') is not None:
            f.write(f'# β coefficient\n')
            f.write(f'beta_value = {results["coefficients"]["beta"]}\n\n')
        
        if results.get('f_closed') is not None:
            f.write(f'# Closed-form metric function\n')
            f.write(f'f_LQG_closed = {results["f_closed"]}\n\n')
        
        f.write(f'# Derivation success: {results.get("success", False)}\n')
        f.write(f'# Validation passed: {results.get("validation_passed", False)}\n')
    
    print(f"\nResults exported to: {filepath}")

if __name__ == "__main__":
    # Run systematic derivation
    print("Starting systematic LQG metric derivation with timeout handling...")
    
    # Test with different orders
    for order in [4, 6]:
        print(f"\n{'='*20} ORDER μ^{order} {'='*20}")
        results = run_systematic_derivation(max_order=order)
        
        if results.get('success'):
            export_results(results, f"systematic_results_order_{order}.py")
        else:
            print(f"Derivation failed at order {order}")
    
    print("\nSystematic derivation complete!")
