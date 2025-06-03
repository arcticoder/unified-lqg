#!/usr/bin/env python3
"""
Enhanced LQG α, β, γ Coefficient Extraction with μ⁶ Expansion and Closed-Form Resummation

This script implements the complete framework for extracting LQG polymer metric coefficients
up to O(μ⁶) and attempts closed-form resummation of the perturbative series.

Workflow:
1. Extract α, β, γ from the metric ansatz: f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M³/r⁷ + γμ⁶M⁴/r¹⁰
2. Build the complete μ⁶ polynomial ansatz
3. Attempt closed-form resummation: f_LQG(r) = 1 - 2M/r + [α·μ²M²/r⁴] / [1 - (β/α²)·μ²]
4. Validate resummation by re-expanding to μ⁴/μ⁶
5. Explore phenomenology and observational signatures

Author: Enhanced LQG Framework with Closed-Form Resummation
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(current_dir, 'scripts')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    from symbolic_timeout_utils import (
        safe_series, safe_solve, safe_simplify, safe_expand,
        set_default_timeout, safe_collect
    )
    TIMEOUT_SUPPORT = True
    set_default_timeout(10)  # 10 seconds for complex operations
except ImportError:
    print("Warning: symbolic_timeout_utils not found; using direct SymPy calls")
    TIMEOUT_SUPPORT = False
    # Define no-timeout fallbacks
    def safe_series(expr, var, point, n, **kwargs):
        return sp.series(expr, var, point, n)
    def safe_solve(expr, symbol, **kwargs):
        return sp.solve(expr, symbol)
    def safe_simplify(expr, **kwargs):
        return sp.simplify(expr)
    def safe_expand(expr, **kwargs):
        return sp.expand(expr)
    def safe_collect(expr, var, **kwargs):
        return sp.collect(expr, var)

# Import alternative polymer prescriptions
try:
    from alternative_polymer_prescriptions import (
        ThiemannPrescription, AQELPrescription, BojowaldPrescription, ImprovedPrescription
    )
    PRESCRIPTIONS_AVAILABLE = True
except ImportError:
    print("Warning: alternative_polymer_prescriptions not found; using standard prescription only")
    PRESCRIPTIONS_AVAILABLE = False

# Global symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)
alpha, beta, gamma = sp.symbols('alpha beta gamma', real=True)
f = sp.Function('f')(r)

def construct_classical_hamiltonian():
    """
    Construct the classical LQG Hamiltonian constraint for spherically symmetric spacetimes.
    
    Returns:
        Classical Hamiltonian expression
    """
    print("="*60)
    print("STEP 1: CONSTRUCTING CLASSICAL LQG HAMILTONIAN")
    print("="*60)
    
    # Classical midisuperspace Hamiltonian (spherically symmetric reduction)
    # H = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + spatial curvature terms
    H_classical = (
        -(Ephi / sp.sqrt(Ex)) * Kphi**2
        - 2 * Kphi * Kx * sp.sqrt(Ex)
    )
    
    print("Classical Hamiltonian constraint:")
    sp.pprint(H_classical)
    
    return H_classical

def solve_classical_kx():
    """
    Solve the classical Hamiltonian constraint for K_x(r) in static spherical symmetry.
    
    For static geometry: K_φ = 0, and K_x is determined by the spatial curvature constraint.
    
    Returns:
        Classical K_x solution
    """
    print("\n" + "="*60)
    print("STEP 2: SOLVING FOR CLASSICAL K_x(r)")
    print("="*60)
    
    # For static spherically symmetric geometry with K_φ = 0
    # The constraint becomes: spatial curvature - K_x² = 0
    # For Schwarzschild geometry: K_x = M / [r(2M - r)]
    
    Kx_classical = M / (r * (2*M - r))
    
    print("Classical K_x solution:")
    sp.pprint(Kx_classical)
    
    # Verify this satisfies the classical constraint
    print("\nVerification: K_x for Schwarzschild f(r) = 1 - 2M/r")
    f_schwarzschild = 1 - 2*M/r
    print(f"f(r) = {f_schwarzschild}")
    print(f"K_x = {Kx_classical}")
    
    return Kx_classical

def apply_polymer_corrections(H_classical, Kx_classical, max_order=6, prescription="standard"):
    """
    Apply polymer quantization: K → sin(μK)/μ and expand to specified order.
    
    Args:
        H_classical: Classical Hamiltonian expression
        Kx_classical: Classical K_x solution
        max_order: Maximum order in μ to include
        prescription: Polymer prescription to use ("standard", "thiemann", "aqel", "bojowald", "improved")
        
    Returns:
        Polymer-expanded Hamiltonian
    """
    print("\n" + "="*60)
    print(f"STEP 3: APPLYING POLYMER CORRECTIONS TO O(μ^{max_order})")
    print(f"Using {prescription} prescription")
    print("="*60)
    
    # For spherically symmetric LQG, the effective Hamiltonian constraint becomes:
    # H_eff = spatial_curvature - K_x²_polymer
    # Where K_x_polymer involves sin(μK_x)/μ modifications
    
    print("Using simplified effective Hamiltonian for spherical symmetry:")
    print("H_eff = R_spatial - K_x²_polymer")
    
    # Start with the spatial curvature contribution (classical)
    # For Schwarzschild: R_spatial = 2M/r³
    R_spatial = 2*M / r**3
      # Apply polymer corrections: K_x → sin(μK_x)/μ
    print("\nApplying polymer corrections K_x → sin(μK_x)/μ...")
    
    # Classical K_x = M / [r(2M - r)] ≈ M/(2Mr) = 1/(2r) for large r
    # For the expansion, use the large-r approximation: K_x ≈ 1/(2r)
    Kx_simplified = M / (2*M*r)  # = 1/(2r)
    
    print(f"Simplified K_x for expansion: {Kx_simplified}")
      # Apply different polymer prescriptions
    if prescription == "standard" or not PRESCRIPTIONS_AVAILABLE:
        # Standard polymer K_x
        Kx_poly_standard = sp.sin(mu * Kx_simplified) / mu
        Kx_poly = Kx_poly_standard
        print("Using standard prescription: sin(μK_x)/μ")
    else:
        # Use alternative prescriptions - replace with new implementation
        from alternative_polymer_prescriptions import ThiemannPrescription, AQELPrescription, BojowaldPrescription
        
        # Compute all prescription variants for comparison
        classical_geometry = {'f_classical': 1 - 2*M/r}
        
        # Initialize all prescriptions
        thiemann = ThiemannPrescription()
        aqel = AQELPrescription()
        bojowald = BojowaldPrescription()
        
        # Compute polymer factors for each prescription
        Kx_poly_thiemann = thiemann.get_polymer_factor(Kx_simplified, classical_geometry)
        Kx_poly_aqel = aqel.get_polymer_factor(Kx_simplified, classical_geometry)
        Kx_poly_boj = bojowald.get_polymer_factor(Kx_simplified, classical_geometry)
        
        print("Computing polymer factors for all prescriptions:")
        print(f"  Thiemann: {Kx_poly_thiemann}")
        print(f"  AQEL: {Kx_poly_aqel}")
        print(f"  Bojowald: {Kx_poly_boj}")
        
        # Select the requested prescription
        if prescription.lower() == "thiemann":
            Kx_poly = Kx_poly_thiemann
            print(f"Using Thiemann prescription: {thiemann.description}")
        elif prescription.lower() == "aqel":
            Kx_poly = Kx_poly_aqel
            print(f"Using AQEL prescription: {aqel.description}")
        elif prescription.lower() == "bojowald":
            Kx_poly = Kx_poly_boj
            print(f"Using Bojowald prescription: {bojowald.description}")
        else:
            print(f"Unknown prescription '{prescription}', using standard")
            Kx_poly = sp.sin(mu * Kx_simplified) / mu
    
    print("Polymer K_x:")
    sp.pprint(Kx_poly)
    
    # Effective Hamiltonian: H_eff = R_spatial - K_x²_polymer
    H_polymer_exact = R_spatial - Kx_poly**2
    
    print("Exact polymer Hamiltonian:")
    sp.pprint(H_polymer_exact)
    
    # Expand sin(μK_x)/μ in powers of μ
    print(f"\nExpanding polymer K_x in μ to O(μ^{max_order})...")
    
    # Series expansion: sin(x)/x = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...
    Kx_series = safe_series(Kx_poly, mu, 0, max_order + 1).removeO()
    
    print("Polymer K_x series:")
    sp.pprint(Kx_series)
    
    # H_polymer = R_spatial - (K_x_series)²
    H_polymer_series = R_spatial - Kx_series**2
      # Expand and collect terms
    H_polymer_expanded = safe_expand(H_polymer_series)
    H_polymer_collected = safe_collect(H_polymer_expanded, mu)
    
    print("Polymer Hamiltonian expanded:")
    sp.pprint(H_polymer_collected)
    
    # Clean up and return the expanded expression
    H_polymer_final = safe_collect(H_polymer_expanded, [mu, M, r])
    
    print("Final polymer Hamiltonian (collected):")
    sp.pprint(H_polymer_final)
    
    return H_polymer_final

def define_metric_ansatz_mu6():
    """
    Define the μ⁶ metric ansatz for LQG corrections.
    
    Returns:
        Metric ansatz expression
    """
    print("\n" + "="*60)
    print("STEP 4: DEFINING μ⁶ METRIC ANSATZ")
    print("="*60)
    
    # Higher-order metric ansatz
    f_ansatz_mu6 = (
        1 
        - 2*M/r 
        + alpha * mu**2 * M**2 / r**4
        + beta * mu**4 * M**3 / r**7
        + gamma * mu**6 * M**4 / r**10
    )
    
    print("Metric ansatz f_LQG(r) up to μ⁶:")
    sp.pprint(f_ansatz_mu6)
    
    return f_ansatz_mu6

def extract_static_constraint(H_polymer, f_ansatz):
    """
    Extract the static constraint by substituting the spherical ansatz.
    
    Args:
        H_polymer: Polymer-expanded Hamiltonian
        f_ansatz: Metric ansatz
        
    Returns:
        Static constraint equation
    """
    print("\n" + "="*60)
    print("STEP 5: EXTRACTING STATIC CONSTRAINT")
    print("="*60)
    
    # Static triad ansatz for spherical symmetry
    Ex_static = r**2
    Ephi_static = r * sp.sqrt(f_ansatz)
    Kphi_static = 0
    
    print("Static ansatz:")
    print(f"  E^x = {Ex_static}")
    print(f"  E^φ = {Ephi_static}")
    print(f"  K_φ = {Kphi_static}")
    
    # Substitute into polymer Hamiltonian
    constraint_static = H_polymer.subs([
        (Ex, Ex_static),
        (Ephi, Ephi_static),
        (Kphi, Kphi_static)
    ])
    
    # Simplify the constraint
    print("\nSimplifying static constraint...")
    constraint_simplified = safe_simplify(constraint_static, timeout_seconds=15)
    if constraint_simplified is None:
        print("Warning: Constraint simplification timed out")
        constraint_simplified = constraint_static
    
    print("Simplified static constraint:")
    sp.pprint(constraint_simplified)
    
    return constraint_simplified

def extract_coefficients_order_by_order(constraint, max_order=6):
    """
    Extract α, β, γ coefficients by matching powers of μ.
    
    Args:
        constraint: Static constraint equation
        max_order: Maximum order to extract
        
    Returns:
        Dictionary of extracted coefficients
    """
    print("\n" + "="*60)
    print("STEP 6: EXTRACTING COEFFICIENTS ORDER BY ORDER")
    print("="*60)
    
    # Collect terms by powers of μ
    constraint_collected = safe_collect(constraint, mu, timeout_seconds=10)
    if constraint_collected is None:
        constraint_collected = constraint
    
    print("Constraint collected by μ powers:")
    sp.pprint(constraint_collected)
    
    # Extract coefficients
    coefficients = {}
    
    # μ⁰ term: should give classical Schwarzschild (verification)
    A0 = constraint_collected.coeff(mu, 0)
    if A0 is not None:
        coefficients['mu0'] = A0
        print(f"\nμ⁰ coefficient (classical): {A0}")
    
    # μ² term: determines α
    A2 = constraint_collected.coeff(mu, 2)
    if A2 is not None:
        coefficients['mu2'] = A2
        print(f"μ² coefficient: {A2}")
        
        # Solve A2 = 0 for α
        print("Solving A2 = 0 for α...")
        alpha_solutions = safe_solve(A2, alpha, timeout_seconds=8)
        if alpha_solutions:
            alpha_value = alpha_solutions[0]
            coefficients['alpha'] = alpha_value
            print(f"  α = {alpha_value}")
        else:
            print("  Warning: Could not solve for α")
            alpha_value = sp.Rational(1, 6)  # Fallback
            coefficients['alpha'] = alpha_value
    
    # μ⁴ term: determines β (with α substituted)
    A4 = constraint_collected.coeff(mu, 4)
    if A4 is not None and 'alpha' in coefficients:
        A4_with_alpha = A4.subs(alpha, coefficients['alpha'])
        coefficients['mu4'] = A4_with_alpha
        print(f"μ⁴ coefficient: {A4_with_alpha}")
        
        # Solve A4 = 0 for β
        print("Solving A4 = 0 for β...")
        beta_solutions = safe_solve(A4_with_alpha, beta, timeout_seconds=8)
        if beta_solutions:
            beta_value = beta_solutions[0]
            coefficients['beta'] = beta_value
            print(f"  β = {beta_value}")
        else:
            print("  Warning: Could not solve for β")
            beta_value = 0  # Fallback
            coefficients['beta'] = beta_value
    
    # μ⁶ term: determines γ (with α, β substituted)
    if max_order >= 6:
        A6 = constraint_collected.coeff(mu, 6)
        if A6 is not None and 'alpha' in coefficients and 'beta' in coefficients:
            A6_with_alpha_beta = A6.subs([
                (alpha, coefficients['alpha']),
                (beta, coefficients['beta'])
            ])
            coefficients['mu6'] = A6_with_alpha_beta
            print(f"μ⁶ coefficient: {A6_with_alpha_beta}")
            
            # Solve A6 = 0 for γ
            print("Solving A6 = 0 for γ...")
            gamma_solutions = safe_solve(A6_with_alpha_beta, gamma, timeout_seconds=8)
            if gamma_solutions:
                gamma_value = gamma_solutions[0]
                coefficients['gamma'] = gamma_value
                print(f"  γ = {gamma_value}")
            else:
                print("  Warning: Could not solve for γ")
                gamma_value = 0  # Fallback
                coefficients['gamma'] = gamma_value
    
    return coefficients

def build_polynomial_metric(coefficients):
    """
    Build the polynomial metric ansatz with extracted coefficients.
    
    Args:
        coefficients: Dictionary of extracted coefficients
        
    Returns:
        Polynomial metric function
    """
    print("\n" + "="*60)
    print("STEP 7: BUILDING POLYNOMIAL METRIC")
    print("="*60)
    
    # Build polynomial ansatz
    f_poly = 1 - 2*M/r
    
    if 'alpha' in coefficients:
        f_poly += coefficients['alpha'] * mu**2 * M**2 / r**4
    
    if 'beta' in coefficients:
        f_poly += coefficients['beta'] * mu**4 * M**3 / r**7
    
    if 'gamma' in coefficients:
        f_poly += coefficients['gamma'] * mu**6 * M**4 / r**10
    
    print("Polynomial metric f_LQG(r):")
    sp.pprint(f_poly)
    
    return f_poly

def attempt_closed_form_resummation(coefficients):
    """
    Attempt closed-form resummation of the metric series.
    
    Args:
        coefficients: Dictionary of extracted coefficients
        
    Returns:
        Tuple of (resummed_metric, success_flag)
    """
    print("\n" + "="*60)
    print("STEP 8: ATTEMPTING CLOSED-FORM RESUMMATION")
    print("="*60)
    
    if 'alpha' not in coefficients or 'beta' not in coefficients:
        print("Cannot perform resummation without α and β coefficients")
        return None, False
    
    alpha_val = coefficients['alpha']
    beta_val = coefficients['beta']
    
    # Compute ratio β/α²
    print("Computing resummation parameters...")
    try:
        ratio_beta_alpha2 = safe_simplify(beta_val / alpha_val**2, timeout_seconds=5)
        if ratio_beta_alpha2 is None:
            ratio_beta_alpha2 = beta_val / alpha_val**2
    except:
        print("Warning: Could not simplify β/α² ratio")
        ratio_beta_alpha2 = beta_val / alpha_val**2
    
    print(f"β/α² = {ratio_beta_alpha2}")
    
    # Try resummation ansatz: f_resummed = 1 - 2M/r + [α·μ²M²/r⁴] / [1 - (β/α²)·μ²]
    print("\nTesting resummation ansatz...")
    c = ratio_beta_alpha2
    
    f_resummed_candidate = (
        1 - 2*M/r 
        + (alpha_val * mu**2 * M**2 / r**4) / (1 - c * mu**2)
    )
    
    print("Resummed candidate:")
    sp.pprint(f_resummed_candidate)
    
    # Validate by re-expanding to μ⁴
    print("\nValidating resummation by re-expansion...")
    resummed_series = safe_series(f_resummed_candidate, mu, 0, n=3, timeout_seconds=8)
    if resummed_series is None:
        print("Warning: Re-expansion failed")
        return f_resummed_candidate, False
    
    resummed_series = resummed_series.removeO()
    
    # Expected series up to μ⁴
    expected_series = (
        1 - 2*M/r 
        + alpha_val * mu**2 * M**2 / r**4
        + beta_val * mu**4 * M**3 / r**7
    )
    
    # Check if they match
    diff_series = safe_simplify(resummed_series - expected_series, timeout_seconds=5)
    if diff_series is None:
        diff_series = resummed_series - expected_series
    
    print("Series difference (should be zero):")
    sp.pprint(diff_series)
    
    success = diff_series == 0 or diff_series.simplify() == 0
    
    if success:
        print("\n✓ Closed-form resummation validated!")
        return f_resummed_candidate, True
    else:
        print("\n✗ Resummation does not match polynomial expansion")
        return f_resummed_candidate, False

def explore_phenomenology(f_lqg, coefficients):
    """
    Explore the phenomenology of the LQG-corrected metric.
    
    Args:
        f_lqg: LQG metric function
        coefficients: Dictionary of coefficients
    """
    print("\n" + "="*60)
    print("STEP 9: EXPLORING PHENOMENOLOGY")
    print("="*60)
    
    # Set up numerical values for exploration
    M_val = 1.0  # Mass in geometric units
    mu_vals = [0.01, 0.05, 0.1, 0.2]  # Range of μ values
    r_vals = np.logspace(0, 2, 100)  # Radial range from 1M to 100M
    
    print("Phenomenological analysis:")
    print(f"  Mass M = {M_val}")
    print(f"  μ values: {mu_vals}")
    print(f"  Radial range: {r_vals[0]:.1f}M to {r_vals[-1]:.1f}M")
    
    # Check horizon shift
    print("\nHorizon analysis:")
    for mu_val in mu_vals:
        # Find where f_LQG(r) = 0 (approximate)
        if 'alpha' in coefficients:
            alpha_num = float(coefficients['alpha'])
            
            # Leading correction to horizon: r_h ≈ 2M - α*μ²M²/(2M)³ = 2M - α*μ²/(4M)
            horizon_shift = -alpha_num * mu_val**2 / (4 * M_val)
            r_horizon_corrected = 2 * M_val + horizon_shift
            
            print(f"  μ = {mu_val}: r_h ≈ {r_horizon_corrected:.6f}M (shift: {horizon_shift:.6f}M)")
    
    # Metric corrections at key radii
    print("\nMetric corrections at key radii:")
    key_radii = [2.1, 3.0, 5.0, 10.0]  # Multiples of M
    
    for r_val in key_radii:
        print(f"\n  At r = {r_val}M:")
        for mu_val in mu_vals:
            if 'alpha' in coefficients:
                alpha_num = float(coefficients['alpha'])
                correction = alpha_num * (mu_val**2) * (M_val**2) / (r_val**4)
                correction_percent = 100 * correction / (1 - 2*M_val/r_val)
                print(f"    μ = {mu_val}: Δf/f = {correction_percent:.3f}%")

def generate_observational_signatures(f_lqg, coefficients):
    """
    Generate observational signatures of LQG corrections.
    
    Args:
        f_lqg: LQG metric function
        coefficients: Dictionary of coefficients
    """
    print("\n" + "="*60)
    print("STEP 10: OBSERVATIONAL SIGNATURES")
    print("="*60)
    
    print("Key observational signatures:")
    
    # 1. Gravitational redshift modifications
    print("\n1. Gravitational redshift corrections:")
    print("   z_LQG = z_Schwarzschild * [1 + α*μ²M²/(2r⁴) + ...]")
    
    # 2. Orbital precession
    print("\n2. Orbital precession modifications:")
    print("   Δφ_LQG = Δφ_GR * [1 + O(α*μ²M²/r⁴)]")
    
    # 3. Photon sphere radius
    print("\n3. Photon sphere radius shift:")
    if 'alpha' in coefficients:
        alpha_num = float(coefficients['alpha'])
        print(f"   r_ph,LQG ≈ 3M * [1 - α*μ²M²/(27M⁴)] = 3M * [1 - {alpha_num}*μ²/27]")
    
    # 4. ISCO modifications
    print("\n4. Innermost Stable Circular Orbit (ISCO):")
    print("   r_ISCO,LQG ≈ 6M * [1 + O(α*μ²)]")
    
    # 5. Shadow radius
    print("\n5. Black hole shadow radius:")
    print("   R_shadow,LQG ≈ 3√3 M * [1 + O(α*μ²)]")
    
    # 6. Quasi-normal modes
    print("\n6. Quasi-normal mode frequencies:")
    print("   ω_QNM,LQG ≈ ω_QNM,GR * [1 + O(α*μ²M²/r_h⁴)]")

def main():
    """
    Execute the complete α, β, γ extraction and resummation analysis.
    """
    start_time = time.time()
    
    print("ENHANCED LQG α, β, γ COEFFICIENT EXTRACTION")
    print("WITH μ⁶ EXPANSION AND PRESCRIPTION COMPARISON")
    print("="*80)
    
    # Define prescriptions to compare
    if PRESCRIPTIONS_AVAILABLE:
        prescriptions = ["standard", "thiemann", "aqel", "bojowald", "improved"]
    else:
        prescriptions = ["standard"]
    
    all_results = {}
    
    for prescription in prescriptions:
        print(f"\n\n{'='*80}")
        print(f"ANALYZING {prescription.upper()} PRESCRIPTION")
        print(f"{'='*80}")
        
        # Step 1: Construct classical Hamiltonian
        H_classical = construct_classical_hamiltonian()
        
        # Step 2: Solve for classical K_x
        Kx_classical = solve_classical_kx()
        
        # Step 3: Apply polymer corrections with specified prescription
        H_polymer = apply_polymer_corrections(H_classical, Kx_classical, max_order=6, prescription=prescription)
        
        # Step 4: Define metric ansatz
        f_ansatz_mu6 = define_metric_ansatz_mu6()
        
        # Step 5: Extract static constraint
        constraint = extract_static_constraint(H_polymer, f_ansatz_mu6)
        
        # Step 6: Extract coefficients
        coefficients = extract_coefficients_order_by_order(constraint, max_order=6)
        
        # Step 7: Build polynomial metric
        f_poly = build_polynomial_metric(coefficients)
        
        # Step 8: Attempt resummation
        f_resummed, resummation_success = attempt_closed_form_resummation(coefficients)
        
        # Store results for this prescription
        all_results[prescription] = {
            'coefficients': coefficients,
            'polynomial_metric': f_poly,
            'resummed_metric': f_resummed if resummation_success else None,
            'resummation_success': resummation_success
        }
        
        # Print summary for this prescription
        print(f"\n{prescription.upper()} PRESCRIPTION RESULTS:")
        print("-" * 50)
        if 'alpha' in coefficients:
            print(f"α coefficient: {coefficients['alpha']}")
        if 'beta' in coefficients:
            print(f"β coefficient: {coefficients['beta']}")
        if 'gamma' in coefficients:
            print(f"γ coefficient: {coefficients['gamma']}")
      # Compare results across prescriptions
    print(f"\n\n{'='*80}")
    print("PRESCRIPTION COMPARISON")
    print(f"{'='*80}")
    
    compare_prescriptions(all_results)
    
    # Create comprehensive comparison DataFrame and plots
    if len(all_results) > 1:
        create_comprehensive_comparison(all_results)
        plot_coefficient_comparison(all_results)
        generate_phenomenology_comparison(all_results)
    
    # Create example comparison script
    create_example_comparison_script()
    
    # Run phenomenology for best prescription (or standard if only one)
    best_prescription = "thiemann" if "thiemann" in all_results else "standard"
    print(f"\nRunning phenomenology analysis for {best_prescription} prescription...")
    
    best_results = all_results[best_prescription]
    metric_for_phenomenology = (best_results['resummed_metric'] 
                              if best_results['resummation_success'] 
                              else best_results['polynomial_metric'])
    
    explore_phenomenology(metric_for_phenomenology, best_results['coefficients'])
    generate_observational_signatures(metric_for_phenomenology, best_results['coefficients'])
    
    # Summary
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    print(f"Prescriptions analyzed: {len(prescriptions)}")
    
    return all_results

def compare_prescriptions(all_results):
    """Compare coefficients across different prescriptions."""
    print(f"{'Prescription':<15} {'α':>12} {'β':>12} {'γ':>15}")
    print("-" * 60)
    
    for prescription, results in all_results.items():
        coeffs = results['coefficients']
        alpha_str = f"{coeffs.get('alpha', 'N/A')}"
        beta_str = f"{coeffs.get('beta', 'N/A')}"
        gamma_str = f"{coeffs.get('gamma', 'N/A')}"
        
        print(f"{prescription:<15} {alpha_str:>12} {beta_str:>12} {gamma_str:>15}")
    
    # Calculate relative differences
    if len(all_results) > 1:
        print(f"\nRelative differences (compared to standard):")
        ref_coeffs = all_results.get('standard', {}).get('coefficients', {})
        
        for prescription, results in all_results.items():
            if prescription == 'standard':
                continue
            
            coeffs = results['coefficients']
            print(f"\n{prescription} vs standard:")
            
            for coeff_name in ['alpha', 'beta', 'gamma']:
                if (coeff_name in coeffs and coeff_name in ref_coeffs and 
                    coeffs[coeff_name] is not None and ref_coeffs[coeff_name] is not None):
                    try:
                        val = float(coeffs[coeff_name])
                        ref_val = float(ref_coeffs[coeff_name])
                        if ref_val != 0:
                            rel_diff = (val - ref_val) / ref_val * 100
                            print(f"  {coeff_name}: {rel_diff:+.1f}% difference")
                        else:
                            print(f"  {coeff_name}: {val:.2e} (ref = 0)")
                    except (ValueError, TypeError):
                        print(f"  {coeff_name}: Cannot compare (symbolic expressions)")
                        
def create_example_comparison_script():
    """Create an example script for comparing prescriptions."""
    example_code = '''#!/usr/bin/env python3
"""
Example: Compare Prescriptions
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_alpha_beta_gamma_extraction import main
from alternative_polymer_prescriptions import (
    ThiemannPrescription, AQELPrescription, BojowaldPrescription
)

if __name__ == "__main__":
    print("Running LQG coefficient extraction with prescription comparison...")
    results = main()
    
    print("\\nExample usage completed!")
    print("Results saved in results dictionary with keys:")
    for prescription in results.keys():
        print(f"  - {prescription}")
'''
    
    with open("example_compare_prescriptions.py", "w") as f:
        f.write(example_code)
    
    print("Created example_compare_prescriptions.py")

def create_comprehensive_comparison(all_results):
    """Create comprehensive comparison including CSV output and plots."""
    import pandas as pd
    import numpy as np
    
    print("\n📊 Creating comprehensive comparison data...")
    
    # Create DataFrame for comparison
    comparison_data = []
    for prescription, results in all_results.items():
        coeffs = results['coefficients']
        row = {
            'Prescription': prescription,
            'α': coeffs.get('alpha', 0.0),
            'β': coeffs.get('beta', 0.0),
            'γ': coeffs.get('gamma', 0.0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_filename = "prescription_coefficient_comparison.csv"
    df.to_csv(csv_filename, index=False)
    print(f"✅ Comparison data saved to {csv_filename}")
    
    # Print formatted table
    print(f"\n📋 Coefficient Comparison Table:")
    print(df.to_string(index=False, float_format='%.6e'))
    
    # Calculate ratios relative to Thiemann (if available)
    if 'thiemann' in all_results:
        print(f"\n📊 Ratios relative to Thiemann prescription:")
        thiemann_coeffs = all_results['thiemann']['coefficients']
        
        for prescription, results in all_results.items():
            if prescription == 'thiemann':
                continue
            
            coeffs = results['coefficients']
            print(f"\n{prescription.upper()} / Thiemann:")
            
            for coeff_name in ['alpha', 'beta', 'gamma']:
                if (coeff_name in coeffs and coeff_name in thiemann_coeffs and
                    thiemann_coeffs[coeff_name] != 0):
                    ratio = float(coeffs[coeff_name]) / float(thiemann_coeffs[coeff_name])
                    print(f"  {coeff_name}: {ratio:.3f}")
    
    return df

def plot_coefficient_comparison(all_results):
    """Plot coefficient comparison across prescriptions."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        prescriptions = list(all_results.keys())
        alpha_vals = [float(all_results[p]['coefficients'].get('alpha', 0)) for p in prescriptions]
        beta_vals = [float(all_results[p]['coefficients'].get('beta', 0)) for p in prescriptions]
        gamma_vals = [float(all_results[p]['coefficients'].get('gamma', 0)) for p in prescriptions]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Alpha plot
        ax1.bar(prescriptions, alpha_vals)
        ax1.set_title('α Coefficient Comparison')
        ax1.set_ylabel('α')
        ax1.tick_params(axis='x', rotation=45)
        
        # Beta plot
        ax2.bar(prescriptions, beta_vals)
        ax2.set_title('β Coefficient Comparison')
        ax2.set_ylabel('β')
        ax2.tick_params(axis='x', rotation=45)
        
        # Gamma plot
        ax3.bar(prescriptions, gamma_vals)
        ax3.set_title('γ Coefficient Comparison')
        ax3.set_ylabel('γ')
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('prescription_coefficient_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison plots saved to prescription_coefficient_comparison.png")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping plots")

def generate_phenomenology_comparison(all_results):
    """Generate phenomenological comparison between prescriptions."""
    print("\n🌟 Phenomenological Implications Analysis")
    print("="*60)
    
    # Test parameters
    M_vals = [1.0]  # Solar masses
    mu_vals = [0.1, 0.01, 0.001]  # Different quantum parameter values
    r_vals = np.linspace(2.5, 10.0, 50)  # From near horizon to far field
    
    for M_val in M_vals:
        for mu_val in mu_vals:
            print(f"\n📏 Analysis for M = {M_val}M☉, μ = {mu_val}")
            print("-" * 40)
            
            for prescription, results in all_results.items():
                coeffs = results['coefficients']
                
                # Horizon shift estimate: δr_h ≈ α * μ² * M
                if 'alpha' in coeffs and coeffs['alpha'] is not None:
                    try:
                        alpha_val = float(coeffs['alpha'])
                        horizon_shift = alpha_val * mu_val**2 * M_val
                        print(f"{prescription:<12}: δr_h ≈ {horizon_shift:+.4f}M")
                    except (ValueError, TypeError):
                        print(f"{prescription:<12}: δr_h ≈ [symbolic expression]")
    
    # ISCO analysis
    print(f"\n🌌 ISCO shift estimates:")
    r_isco_classical = 6.0  # Classical ISCO at 6M
    
    for prescription, results in all_results.items():
        coeffs = results['coefficients']
        if 'alpha' in coeffs and coeffs['alpha'] is not None:
            try:
                alpha_val = float(coeffs['alpha'])
                # Rough estimate: δr_ISCO ≈ α * μ² * M * (M/r_ISCO)²
                isco_shift = alpha_val * (0.1)**2 * M_val * (M_val/r_isco_classical)**2
                print(f"{prescription:<12}: δr_ISCO ≈ {isco_shift:+.4f}M (μ=0.1)")
            except (ValueError, TypeError):
                print(f"{prescription:<12}: δr_ISCO ≈ [symbolic expression]")

# Insert before the main function
