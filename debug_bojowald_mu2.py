#!/usr/bin/env python3
"""
Debug the Bojowald μ² issue
"""

import sympy as sp
from alternative_polymer_prescriptions import BojowaldPrescription

def test_bojowald_mu2():
    """Test what happens with μ² expansion for Bojowald."""
    print("🔍 Testing Bojowald μ² behavior...")
    
    # Create prescription
    boj = BojowaldPrescription()
    
    # Test values
    r, M, mu = boj.r, boj.M, boj.mu
    test_vals = {r: 10.0, M: 1.0}
    
    # Classical geometry
    classical_geometry = {'f_classical': 1 - 2*M/r}
    K_classical = M / (r * (2*M - r))
    
    print(f"K_classical = {K_classical}")
    print(f"K_classical substituted = {K_classical.subs(test_vals)}")
    
    # Compute effective mu
    mu_eff = boj.compute_effective_mu(classical_geometry)
    print(f"μ_eff = {mu_eff}")
    print(f"μ_eff simplified = {sp.simplify(mu_eff)}")
    
    # Get polymer factor
    polymer_factor = boj.get_polymer_factor(K_classical, classical_geometry)
    print(f"Polymer factor = {polymer_factor}")
    
    # Test series expansion
    print("\n🧪 Testing series expansion:")
    try:
        series = sp.series(polymer_factor, mu, 0, 4)
        print(f"Series expansion = {series}")
        
        series_no_O = series.removeO()
        print(f"Series without O() = {series_no_O}")
        
        # Check coefficients
        mu0_coeff = series_no_O.coeff(mu, 0)
        mu1_coeff = series_no_O.coeff(mu, 1) 
        mu2_coeff = series_no_O.coeff(mu, 2)
        mu3_coeff = series_no_O.coeff(mu, 3)
        
        print(f"μ⁰ coefficient = {mu0_coeff}")
        print(f"μ¹ coefficient = {mu1_coeff}")
        print(f"μ² coefficient = {mu2_coeff}")
        print(f"μ³ coefficient = {mu3_coeff}")
        
        # Evaluate numerically
        if mu2_coeff:
            mu2_numerical = mu2_coeff.subs(test_vals)
            print(f"μ² coefficient numerical = {mu2_numerical}")
        else:
            print("μ² coefficient is None or 0")
            
    except Exception as e:
        print(f"Series expansion failed: {e}")
    
    # Check argument expansion
    print("\n🧪 Testing argument expansion:")
    argument = mu_eff * K_classical
    print(f"argument = {argument}")
    
    arg_series = sp.series(argument, mu, 0, 4).removeO()
    print(f"argument series = {arg_series}")

if __name__ == "__main__":
    test_bojowald_mu2()
