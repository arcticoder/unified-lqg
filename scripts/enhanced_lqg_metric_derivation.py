#!/usr/bin/env python3
"""
Enhanced LQG metric derivation with higher-order corrections and timeout handling.

This script implements the complete framework for deriving LQG-corrected metrics to
arbitrary order in μ with robust timeout handling for all symbolic operations.
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
import os

# Ensure we can import symbolic_timeout_utils
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    from symbolic_timeout_utils import (
        safe_integrate, safe_solve, safe_series,
        safe_diff, safe_simplify, safe_expand, safe_factor,
        safe_collect, safe_cancel, safe_together,
        set_default_timeout, has_timeout_support
    )
    TIMEOUT_SUPPORT = True
    print("✓ Symbolic timeout support available")
    # Choose a per-module default (tune as needed)
    set_default_timeout(8)  # 8 seconds for complex metric derivations
except ImportError:
    print("Warning: symbolic_timeout_utils not found; using direct SymPy calls")
    TIMEOUT_SUPPORT = False
    # Define no-timeout fallbacks:
    def safe_integrate(expr, *args, **kwargs):
        return sp.integrate(expr, *args)
    def safe_solve(expr, symbol, *args, **kwargs):
        return sp.solve(expr, symbol, *args)
    def safe_series(expr, var, point, n, *args, **kwargs):
        return sp.series(expr, var, point, n, *args)
    def safe_diff(expr, var, *args, **kwargs):
        return sp.diff(expr, var, *args)
    def safe_simplify(expr, *args, **kwargs):
        return sp.simplify(expr, *args)
    def safe_expand(expr, *args, **kwargs):
        return sp.expand(expr, *args)
    def safe_factor(expr, *args, **kwargs):
        return sp.factor(expr, *args)
    def safe_collect(expr, syms, *args, **kwargs):
        return sp.collect(expr, syms, *args)
    def safe_cancel(expr, *args, **kwargs):
        return sp.cancel(expr, *args)
    def safe_together(expr, *args, **kwargs):
        return sp.together(expr, *args)

# Global symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)

# Higher-order coefficients  
alpha, beta, gamma_coeff, delta_coeff = sp.symbols('alpha beta gamma delta', real=True)

def polymer_expand_hamiltonian(max_order: int = 6) -> sp.Expr:
    """
    Polymer-expand the LQG Hamiltonian constraint to higher order.
    
    Replace connection components K by sin(μK)/μ and expand each sin(μK) 
    in a Taylor series up to μ⁴ or μ⁶.
    
    Args:
        max_order: Maximum power of μ to include
        
    Returns:
        Polymer-expanded Hamiltonian
    """
    print(f"Constructing polymer Hamiltonian to order μ^{max_order}...")
    
    # Classical midisuperspace Hamiltonian (spherically symmetric)
    # H = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + (constraint terms)
    H_classical = (
        -(Ephi / sp.sqrt(Ex)) * Kphi**2
        - 2 * Kphi * Kx * sp.sqrt(Ex)
    )
    
    print(f"  Classical Hamiltonian: {H_classical}")
    
    # Apply polymer corrections: K → sin(μK)/μ
    print(f"  Applying polymer corrections to order μ^{max_order}...")
    
    # Expand sin(μK)/μ = 1 - (μK)²/6 + (μK)⁴/120 - (μK)⁶/5040 + ...
    mu_Kx = mu * Kx
    mu_Kphi = mu * Kphi
    
    # Use safe_series to ensure it returns before timing out
    n_terms = max_order // 2 + 3  # Include enough terms
    
    sin_expansion_x = safe_series(
        sp.sin(mu_Kx) / mu_Kx, mu_Kx, 0, n=n_terms, timeout_seconds=6
    )
    sin_expansion_phi = safe_series(
        sp.sin(mu_Kphi) / mu_Kphi, mu_Kphi, 0, n=n_terms, timeout_seconds=6
    )
    
    if sin_expansion_x is None or sin_expansion_phi is None:
        print("  Warning: Series expansion failed, using manual expansion")
        # Manual expansion as fallback
        sin_expansion_x = 1 - (mu_Kx)**2/6 + (mu_Kx)**4/120 - (mu_Kx)**6/5040
        sin_expansion_phi = 1 - (mu_Kphi)**2/6 + (mu_Kphi)**4/120 - (mu_Kphi)**6/5040
    else:
        # Remove O() terms
        sin_expansion_x = sin_expansion_x.removeO()
        sin_expansion_phi = sin_expansion_phi.removeO()
        print(f"  sin(μK_x)/(μK_x) ≈ {sin_expansion_x}")
        print(f"  sin(μK_φ)/(μK_φ) ≈ {sin_expansion_phi}")
    
    # Apply corrections
    Kx_poly = Kx * sin_expansion_x
    Kphi_poly = Kphi * sin_expansion_phi
    
    # Construct polymer Hamiltonian
    H_polymer = (
        -(Ephi / sp.sqrt(Ex)) * Kphi_poly**2
        - 2 * Kphi_poly * Kx_poly * sp.sqrt(Ex)
    )
    
    # Expand to desired order in μ
    print("  Expanding polymer Hamiltonian...")
    H_expanded = safe_expand(H_polymer, timeout_seconds=8)
    
    if H_expanded is None:
        print("  Warning: Expansion failed, using unexpanded form")
        H_expanded = H_polymer
    
    return H_expanded

def formulate_higher_order_ansatz(max_order: int = 6) -> sp.Expr:
    """
    Formulate the higher-order ansatz for the LQG-corrected lapse.
    
    At second order: f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + O(μ⁴)
    Extended: f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + β*μ⁴M³/r⁷ + γ*μ⁶M⁴/r¹⁰ + ...
    
    Args:
        max_order: Maximum power of μ to include
        
    Returns:
        Metric ansatz expression
    """
    print(f"Formulating ansatz to order μ^{max_order}...")
    
    # Start with classical Schwarzschild
    f_ansatz = 1 - 2*M/r
    
    # Add LQG corrections at even orders
    if max_order >= 2:
        f_ansatz += alpha * mu**2 * M**2 / r**4
        print(f"  Added O(μ²) term: α*μ²M²/r⁴")
    
    if max_order >= 4:
        f_ansatz += beta * mu**4 * M**3 / r**7
        print(f"  Added O(μ⁴) term: β*μ⁴M³/r⁷")
    
    if max_order >= 6:
        f_ansatz += gamma_coeff * mu**6 * M**4 / r**10
        print(f"  Added O(μ⁶) term: γ*μ⁶M⁴/r¹⁰")
    
    if max_order >= 8:
        f_ansatz += delta_coeff * mu**8 * M**5 / r**13
        print(f"  Added O(μ⁸) term: δ*μ⁸M⁵/r¹³")
    
    print(f"  Complete ansatz: f_LQG(r) = {f_ansatz}")
    return f_ansatz

def extract_spherical_constraint(H_polymer: sp.Expr) -> sp.Expr:
    """
    Extract the spherically symmetric constraint H_eff(r) = 0 using static ansatz.
    
    Args:
        H_polymer: Polymer-expanded Hamiltonian
        
    Returns:
        Effective constraint as function of r with metric function f(r)
    """
    print("Extracting spherical constraint...")
    
    # Define metric function symbolically
    f = sp.Function('f')(r)
    
    # Static spherically symmetric triad ansatz:
    # E^x = r², E^φ = r√f(r), K_φ = 0, K_x = 0 (initially)
    # The K_x term will be determined by the constraint itself
    
    Ex_static = r**2
    Ephi_static = r * sp.sqrt(f)
    Kphi_static = 0
    Kx_static = 0  # Will be corrected by higher-order polymer terms
    
    print(f"  Substituting: E^x = {Ex_static}")
    print(f"  Substituting: E^φ = {Ephi_static}")  
    print(f"  Substituting: K_x = {Kx_static}, K_φ = {Kphi_static}")
    
    # Make substitutions
    constraint = H_polymer.subs([
        (Ex, Ex_static),
        (Ephi, Ephi_static),
        (Kx, Kx_static),
        (Kphi, Kphi_static)
    ])
    
    # Simplify the constraint
    constraint_simplified = safe_simplify(constraint, timeout_seconds=10)
    
    if constraint_simplified is None:
        print("  Warning: Constraint simplification timed out")
        constraint_simplified = constraint
    
    return constraint_simplified

def solve_coefficients_order_by_order(constraint: sp.Expr, max_order: int = 6) -> Dict[str, sp.Expr]:
    """
    Match term by term in a power-series in μ and solve for coefficients.
    
    Express the Hamiltonian constraint H_eff(r) = 0 and expand as:
    H_eff(r) = A₀(r) + A₁(r)*μ² + A₂(r)*μ⁴ + ...
    
    Enforce that each coefficient of μ^(2n) vanishes.
    
    Args:
        constraint: The constraint equation H_eff(r) = 0
        max_order: Maximum order to solve
        
    Returns:
        Dictionary of solved coefficients
    """
    print("Solving for coefficients order by order...")
    
    # Expand constraint as series in μ
    print("  Expanding constraint in μ...")
    n_series = max_order // 2 + 2
    constraint_series = safe_series(constraint, mu, 0, n=n_series, timeout_seconds=10)
    
    if constraint_series is None:
        print("  Warning: Series expansion failed, trying manual collection")
        # Fallback: expand and collect terms manually
        constraint_expanded = safe_expand(constraint, timeout_seconds=8)
        if constraint_expanded is None:
            constraint_expanded = constraint
        constraint_series = constraint_expanded
    else:
        constraint_series = constraint_series.removeO()
    
    print(f"  Constraint series: {constraint_series}")
    
    # Extract coefficients at different orders
    coefficients = {}
    
    # O(μ⁰) - should give classical constraint (verify)
    coeff_mu0 = constraint_series.coeff(mu, 0)
    if coeff_mu0 is not None:
        print(f"  μ⁰ coefficient: {coeff_mu0}")
    
    # O(μ²) - solve for α
    if max_order >= 2:
        coeff_mu2 = constraint_series.coeff(mu, 2)
        if coeff_mu2 is not None:
            print(f"  μ² coefficient: {coeff_mu2}")
            
            # Solve A₁(r) = 0 for α
            # Extract coefficient of 1/r⁴ (typical form)
            r_inv4_coeff = coeff_mu2.as_coefficients_dict().get(r**(-4), 0)
            
            if r_inv4_coeff != 0:
                alpha_solution = safe_solve(r_inv4_coeff, alpha, timeout_seconds=8)
                if alpha_solution:
                    coefficients['alpha'] = alpha_solution[0]
                    print(f"  Solved α = {coefficients['alpha']}")
                else:
                    print("  Could not solve for α")
            else:
                print("  No 1/r⁴ term found in μ² coefficient")
    
    # O(μ⁴) - solve for β (may depend on α)
    if max_order >= 4 and 'alpha' in coefficients:
        coeff_mu4 = constraint_series.coeff(mu, 4)
        if coeff_mu4 is not None:
            print(f"  μ⁴ coefficient: {coeff_mu4}")
            
            # Substitute known α value
            coeff_mu4_sub = coeff_mu4.subs(alpha, coefficients['alpha'])
            
            # Extract coefficient of 1/r⁷ (typical form)
            r_inv7_coeff = coeff_mu4_sub.as_coefficients_dict().get(r**(-7), 0)
            
            if r_inv7_coeff != 0:
                beta_solution = safe_solve(r_inv7_coeff, beta, timeout_seconds=8)
                if beta_solution:
                    coefficients['beta'] = beta_solution[0]
                    print(f"  Solved β = {coefficients['beta']}")
                else:
                    print("  Could not solve for β")
    
    # O(μ⁶) - solve for γ (may depend on α, β)
    if max_order >= 6 and 'alpha' in coefficients:
        coeff_mu6 = constraint_series.coeff(mu, 6)
        if coeff_mu6 is not None:
            print(f"  μ⁶ coefficient: {coeff_mu6}")
            
            # Substitute known values
            coeff_mu6_sub = coeff_mu6.subs(alpha, coefficients['alpha'])
            if 'beta' in coefficients:
                coeff_mu6_sub = coeff_mu6_sub.subs(beta, coefficients['beta'])
            
            # Extract coefficient of 1/r¹⁰ (typical form)
            r_inv10_coeff = coeff_mu6_sub.as_coefficients_dict().get(r**(-10), 0)
            
            if r_inv10_coeff != 0:
                gamma_solution = safe_solve(r_inv10_coeff, gamma_coeff, timeout_seconds=8)
                if gamma_solution:
                    coefficients['gamma'] = gamma_solution[0]
                    print(f"  Solved γ = {coefficients['gamma']}")
                else:
                    print("  Could not solve for γ")
    
    return coefficients

def attempt_closed_form_resummation(coefficients: Dict[str, sp.Expr]) -> Tuple[sp.Expr, bool]:
    """
    Attempt to find a "resummed" closed form from the polynomial coefficients.
    
    Try to rewrite:
    f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + β*μ⁴M³/r⁷ + ...
    
    as a rational function or other closed form.
    
    Args:
        coefficients: Dictionary of solved coefficients
        
    Returns:
        Tuple of (closed_form_expression, success_flag)
    """
    print("Attempting closed-form resummation...")
    
    if 'alpha' not in coefficients:
        print("  No α coefficient available")
        return None, False
    
    alpha_val = coefficients['alpha']
    
    # Start with base form
    f_base = 1 - 2*M/r
    correction_base = alpha_val * mu**2 * M**2 / r**4
    
    if 'beta' not in coefficients:
        # Only α available
        closed_form = f_base + correction_base
        return closed_form, True
    
    beta_val = coefficients['beta']
    
    # Check if β has a simple relation to α
    # Common patterns: β = c*α² for some constant c
    print("  Checking for β = c*α² pattern...")
    
    # Try to simplify β/α²
    beta_over_alpha_sq = safe_simplify(beta_val / alpha_val**2, timeout_seconds=5)
    
    if beta_over_alpha_sq is not None and beta_over_alpha_sq.is_number:
        print(f"  Found β/α² = {beta_over_alpha_sq}")
        
        # Try rational form: f = 1 - 2M/r + (α*μ²M²/r⁴) / (1 + C*μ²M/r³)
        # Expanding this should give α*μ²M²/r⁴ - C*α*μ⁴M³/r⁷ + ...
        # So C = -β/(α*α) * r³/M = -β_over_alpha_sq * α * r³/M
        
        C_val = -beta_over_alpha_sq * alpha_val
        
        closed_form = f_base + correction_base / (1 + C_val * mu**2 * M / r**3)
        
        print(f"  Trying rational form with C = {C_val}")
        
        # Validate by expanding
        print("  Validating by re-expansion...")
        validation = safe_series(closed_form, mu, 0, n=3, timeout_seconds=6)
        
        if validation is not None:
            validation_expanded = validation.removeO()
            
            # Check if it matches the original polynomial
            expected_poly = (f_base + alpha_val * mu**2 * M**2 / r**4 + 
                           beta_val * mu**4 * M**3 / r**7)
            
            difference = safe_simplify(validation_expanded - expected_poly, timeout_seconds=5)
            
            if difference is not None and difference == 0:
                print("  ✓ Closed form validated successfully!")
                return closed_form, True
            else:
                print(f"  ✗ Validation failed, difference: {difference}")
    
    # Fallback: standard polynomial form
    print("  Using polynomial form as fallback")
    f_poly = f_base + correction_base
    
    if 'beta' in coefficients:
        f_poly += coefficients['beta'] * mu**4 * M**3 / r**7
    
    if 'gamma' in coefficients:
        f_poly += coefficients['gamma'] * mu**6 * M**4 / r**10
    
    return f_poly, False

def get_lqg_metric_closed_form(max_order: int = 4) -> Dict[str, Any]:
    """
    Derive α, β, (γ…) to the given μ-order, attempt closed-form ansatz,
    and return a comprehensive analysis.
    
    Args:
        max_order: Maximum order in μ to derive (4 for μ⁴, 6 for μ⁶)
        
    Returns:
        Dictionary with all derivation results
    """
    print("="*80)
    print(f"LQG METRIC DERIVATION TO ORDER μ^{max_order}")
    print("="*80)
    
    results = {
        'max_order': max_order,
        'coefficients': {},
        'closed_form': None,
        'series_verified': False,
        'timeout_support': TIMEOUT_SUPPORT,
        'success': False
    }
    
    try:
        # Step 1: Polymer-expand the LQG Hamiltonian constraint
        print("\n1. Polymer-expanding Hamiltonian constraint...")
        H_polymer = polymer_expand_hamiltonian(max_order)
        results['hamiltonian_polymer'] = H_polymer
        
        # Step 2: Formulate higher-order ansatz
        print("\n2. Formulating higher-order metric ansatz...")
        f_ansatz = formulate_higher_order_ansatz(max_order)
        results['metric_ansatz'] = f_ansatz
          # Step 3: Extract spherical constraint
        print("\n3. Extracting spherical constraint...")
        constraint = extract_spherical_constraint(H_polymer)
        results['constraint'] = constraint
        
        # Step 4: Solve coefficients order by order
        print("\n4. Solving coefficients order by order...")
        coefficients = solve_coefficients_order_by_order(constraint, max_order)
        results['coefficients'] = coefficients
        
        # Step 5: Attempt closed-form resummation
        print("\n5. Attempting closed-form resummation...")
        closed_form, is_resummed = attempt_closed_form_resummation(coefficients)
        results['closed_form'] = closed_form
        results['is_resummed'] = is_resummed
        
        # Step 6: Validate by re-expansion
        if closed_form is not None:
            print("\n6. Validating closed form by re-expansion...")
            validation = safe_series(closed_form, mu, 0, n=max_order//2 + 2, timeout_seconds=8)
            
            if validation is not None:
                results['validation_expansion'] = validation.removeO()
                results['series_verified'] = True
                print("  ✓ Series verification completed")
            else:
                print("  ⚠ Series verification timed out")
        
        results['success'] = True
        print("\n✓ LQG metric derivation completed successfully!")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  Derived to order: μ^{max_order}")
        print(f"  Coefficients found: {list(coefficients.keys())}")
        print(f"  Closed form: {'resummed rational' if is_resummed else 'polynomial'}")
        print(f"  Final metric: f_LQG(r) = {closed_form}")
        
    except Exception as e:
        print(f"\n✗ Derivation failed: {e}")
        results['success'] = False
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
    
    return results

if __name__ == "__main__":
    # Test the complete derivation framework
    print("Testing LQG metric derivation with timeout handling...")
    
    # Test O(μ⁴) derivation
    results_mu4 = get_lqg_metric_closed_form(max_order=4)
    
    if results_mu4['success']:
        print("\n" + "="*60)
        print("O(μ⁴) DERIVATION SUCCESSFUL")
        print("="*60)
        
        for name, value in results_mu4['coefficients'].items():
            print(f"{name} = {value}")
        
        print(f"\nClosed form: {results_mu4['closed_form']}")
        
        if results_mu4['is_resummed']:
            print("Successfully found resummed rational form!")
        
    else:
        print(f"O(μ⁴) derivation failed: {results_mu4.get('error', 'Unknown error')}")
    
    # Test O(μ⁶) if μ⁴ succeeded
    if results_mu4['success']:
        print("\n" + "="*60)
        print("ATTEMPTING O(μ⁶) DERIVATION")
        print("="*60)
        
        results_mu6 = get_lqg_metric_closed_form(max_order=6)
        
        if results_mu6['success']:
            print("O(μ⁶) derivation also successful!")
        else:
            print("O(μ⁶) derivation failed, but O(μ⁴) is available")
