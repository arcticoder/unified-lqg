#!/usr/bin/env python3
"""
Debug the Improved prescription issue
"""

import sympy as sp
from alternative_polymer_prescriptions import ImprovedPrescription

def test_improved_prescription():
    """Test what happens with the Improved prescription."""
    print("🔍 Testing Improved prescription behavior...")
    
    # Create prescription
    improved = ImprovedPrescription()
    
    # Test values
    r, M, mu = sp.symbols('r M mu', positive=True)
    test_vals = {r: 5.0, M: 1.0}
    
    # Classical geometry
    classical_geometry = {'f_classical': 1 - 2*M/r}
    K_classical = M / (r * (2*M - r))
    
    print(f"K_classical = {K_classical}")
    print(f"K_classical substituted = {K_classical.subs(test_vals)}")
    
    # Compute effective mu
    mu_eff = improved.compute_effective_mu(classical_geometry)
    print(f"μ_eff = {mu_eff}")
    
    # Get polymer factor
    polymer_factor = improved.get_polymer_factor(K_classical, classical_geometry)
    print(f"Polymer factor = {polymer_factor}")
    
    # Test classical limit
    print("\n🧪 Testing μ→0 limit:")
    classical_limit = polymer_factor.subs(mu, 0)
    print(f"Classical limit = {classical_limit}")
    
    # Numerical test
    classical_numerical = classical_limit.subs(test_vals)
    expected_numerical = K_classical.subs(test_vals)
    
    print(f"Classical limit numerical = {classical_numerical}")
    print(f"Expected (K_classical) = {expected_numerical}")
    
    # Test series expansion
    print("\n🧪 Testing series expansion:")
    try:
        series = sp.series(polymer_factor, mu, 0, 4).removeO()
        print(f"Series expansion = {series}")
        
        # Check coefficients
        mu0_coeff = series.coeff(mu, 0)
        mu2_coeff = series.coeff(mu, 2)
        
        print(f"μ⁰ coefficient = {mu0_coeff}")
        print(f"μ² coefficient = {mu2_coeff}")
        
    except Exception as e:
        print(f"Series expansion failed: {e}")
    
    # Test with small mu
    print("\n🧪 Testing with small μ:")
    small_mu_vals = test_vals.copy()
    small_mu_vals[mu] = 1e-6
    
    small_mu_result = polymer_factor.subs(small_mu_vals)
    print(f"Small μ result = {small_mu_result}")

if __name__ == "__main__":
    test_improved_prescription()
