#!/usr/bin/env python3
"""
Quick test for Bojowald prescription fix
"""

import sympy as sp
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_bojowald_classical_limit():
    """Test a simplified Bojowald prescription implementation."""
    
    # Define symbols
    r, M, mu = sp.symbols('r M mu', positive=True)
    
    # Classical extrinsic curvature
    K_classical = M / (r * (2*M - r))
    
    # Bojowald effective μ: μ_eff = μ * sqrt(|K|)
    K_sqrt = sp.sqrt(sp.Abs(K_classical))
    mu_eff = mu * K_sqrt
    
    # Polymer factor with proper limit handling
    argument = mu_eff * K_classical
    
    # Use L'Hôpital's rule: lim(μ→0) sin(μ * sqrt(|K|) * K) / (μ * sqrt(|K|)) = K
    polymer_factor = sp.Piecewise(
        (K_classical, sp.Eq(mu, 0)),
        (sp.sin(argument) / mu_eff, True)
    )
    
    print("Polymer factor:", polymer_factor)
    
    # Test classical limit
    classical_limit = polymer_factor.subs(mu, 0)
    print("Classical limit:", classical_limit)
    
    # Test with numerical values
    test_vals = {r: 5.0, M: 1.0}
    
    # Expected classical value
    expected = K_classical.subs(test_vals)
    print("Expected K_classical:", float(expected))
    
    # Actual limit
    try:
        limit_numerical = classical_limit.subs(test_vals)
        print("Limit numerical:", limit_numerical)
        print("Float value:", float(limit_numerical))
        
        # Check if they match
        if abs(float(limit_numerical) - float(expected)) < 1e-10:
            print("✅ Classical limit test PASSED")
            return True
        else:
            print("❌ Classical limit test FAILED")
            return False
            
    except Exception as e:
        print("❌ Error in evaluation:", e)
        return False

if __name__ == "__main__":
    test_bojowald_classical_limit()
