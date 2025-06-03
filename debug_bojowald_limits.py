#!/usr/bin/env python3
"""
Debug script for Bojowald prescription limit behavior.
"""

import sympy as sp
import numpy as np
from alternative_polymer_prescriptions import BojowaldPrescription

def debug_bojowald_limits():
    """Debug Bojowald prescription classical and μ² limits."""
    
    print("🔍 Debugging Bojowald Prescription Limits")
    print("=" * 50)
    
    # Create Bojowald prescription
    bojowald = BojowaldPrescription()
    
    # Set up symbols (these are already defined in the prescription)
    mu, r, M = bojowald.mu, bojowald.r, bojowald.M
    
    # Create geometry
    test_classical_geometry = {'r': r, 'M': M}
    
    # Classical extrinsic curvature
    K_classical = M / (r * (2*M - r))
    
    print(f"Bojowald effective μ: {bojowald.compute_effective_mu(test_classical_geometry)}")
    
    # Get polymer factor
    polymer_factor = bojowald.get_polymer_factor(K_classical, test_classical_geometry)
    
    print(f"Polymer factor: {polymer_factor}")
    print()
    
    # Test values
    test_values = {r: 5.0, M: 1.0}
    K_val = K_classical.subs(test_values)
    print(f"K_classical with test values: {K_val}")
    
    # 1. Classical limit test
    print("1. CLASSICAL LIMIT TEST")
    print("-" * 30)
    
    # Symbolic limit
    classical_limit = sp.limit(polymer_factor, mu, 0)
    print(f"Symbolic limit (μ→0): {classical_limit}")
    
    # Small μ test
    small_mu_values = [1e-3, 1e-4, 1e-5, 1e-6]
    
    for small_mu in small_mu_values:
        test_vals_small = test_values.copy()
        test_vals_small[mu] = small_mu
        
        polymer_val = float(polymer_factor.subs(test_vals_small))
        classical_val = float(K_val)
        relative_diff = abs(polymer_val - classical_val) / abs(classical_val)
        
        print(f"  μ = {small_mu:8.1e}: polymer = {polymer_val:10.6f}, classical = {classical_val:10.6f}, rel_diff = {relative_diff:.6f}")
    
    print()
    
    # 2. Series expansion test
    print("2. SERIES EXPANSION TEST")
    print("-" * 30)
    
    # Get series expansion
    expansion = sp.series(polymer_factor, mu, 0, n=6).removeO()
    print(f"Series expansion: {expansion}")
    
    # Check coefficients
    for i in range(6):
        coeff = expansion.coeff(mu, i)
        if coeff is not None and coeff != 0:
            coeff_val = coeff.subs(test_values)
            print(f"  μ^{i} coefficient: {coeff} = {coeff_val}")
    
    print()
    
    # 3. Effective μ analysis
    print("3. EFFECTIVE μ ANALYSIS")
    print("-" * 30)
    
    mu_eff = bojowald.compute_effective_mu(test_classical_geometry)
    print(f"μ_eff = {mu_eff}")
    
    # For Bojowald: μ_eff = μ * sqrt(|K|)
    # The argument to sin becomes: μ_eff * K = μ * sqrt(|K|) * K = μ * K^(3/2)
    argument = mu_eff * K_classical
    print(f"Argument to sin: {argument}")
    
    # Simplify argument
    argument_simplified = sp.simplify(argument)
    print(f"Simplified argument: {argument_simplified}")
    
    # Series expansion of sin(argument)/μ_eff
    sin_factor = sp.sin(argument) / mu_eff
    print(f"sin(argument)/μ_eff = {sin_factor}")
    
    # Expand this
    sin_expansion = sp.series(sin_factor, mu, 0, n=6).removeO()
    print(f"sin expansion: {sin_expansion}")
    
    print()
    
    # 4. Why no μ² term?
    print("4. WHY NO μ² TERM?")
    print("-" * 30)
    
    # For Bojowald: argument = μ * K^(3/2)
    # sin(μ * K^(3/2)) = μ * K^(3/2) - (μ * K^(3/2))³/6 + ...
    #                  = μ * K^(3/2) - μ³ * K^(9/2)/6 + ...
    # So sin(μ * K^(3/2)) / (μ * sqrt(K)) = K - μ² * K⁴/6 + ...
    
    # Wait, let's be more careful:
    # sin(μ * K^(3/2)) / (μ * sqrt(K)) = sin(μ * K^(3/2)) / (μ * K^(1/2))
    
    K_numeric = test_values[M] / (test_values[r] * (2*test_values[M] - test_values[r]))
    K_sqrt = sp.sqrt(K_numeric)
    
    print(f"K = {K_numeric}")
    print(f"sqrt(K) = {K_sqrt}")
    print(f"K^(3/2) = {K_numeric**(3/2)}")
    
    # Manual expansion
    print("\nManual expansion of sin(μ*K^(3/2))/(μ*sqrt(K)):")
    
    # Define x = μ * K^(3/2)
    x = mu * K_numeric**(3/2)
    sin_x = sp.sin(x)
    result = sin_x / (mu * K_sqrt)
    
    manual_expansion = sp.series(result, mu, 0, n=6).removeO()
    print(f"Manual expansion: {manual_expansion}")

if __name__ == "__main__":
    debug_bojowald_limits()
