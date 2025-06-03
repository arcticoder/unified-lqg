#!/usr/bin/env python3
"""
Quick debug for Bojowald test values.
"""

import sympy as sp
from alternative_polymer_prescriptions import BojowaldPrescription

def debug_bojowald_test_values():
    """Debug Bojowald with the exact test values from the failing test."""
    
    print("🔍 Debugging Bojowald with Test Values")
    print("=" * 50)
    
    # Create Bojowald prescription
    bojowald = BojowaldPrescription()
    mu, r, M = bojowald.mu, bojowald.r, bojowald.M
    
    # Test values from the failing test
    test_vals = {r: 10.0, M: 1.0}
    
    # Classical extrinsic curvature with test values
    Kx_classical = M / (r * (2*M - r))
    K_val = Kx_classical.subs(test_vals)
    print(f"Kx_classical = {Kx_classical}")
    print(f"K_val = {K_val}")
    
    # Test classical geometry
    test_classical_geometry = {'r': r, 'M': M}
    
    # Get polymer factor
    polymer_factor = bojowald.get_polymer_factor(Kx_classical, test_classical_geometry)
    print(f"Polymer factor: {polymer_factor}")
    
    # Series expansion to order 4
    series_expansion = sp.series(polymer_factor, mu, 0, 4).removeO()
    print(f"Series expansion: {series_expansion}")
    
    # Check each coefficient
    for i in range(5):
        coeff = series_expansion.coeff(mu, i)
        if coeff is not None and coeff != 0:
            coeff_val = coeff.subs(test_vals)
            print(f"  μ^{i} coefficient: {coeff} = {coeff_val}")
    
    # Specifically check μ³
    mu3_coeff = series_expansion.coeff(mu, 3)
    print(f"\nμ³ coefficient: {mu3_coeff}")
    if mu3_coeff is not None:
        mu3_numerical = float(mu3_coeff.subs(test_vals))
        print(f"μ³ numerical value: {mu3_numerical}")
    else:
        print("μ³ coefficient is None")

if __name__ == "__main__":
    debug_bojowald_test_values()
