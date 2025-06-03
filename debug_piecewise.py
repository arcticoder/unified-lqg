#!/usr/bin/env python3
"""
Debug the Piecewise issue with the Improved prescription
"""

import sympy as sp

def test_piecewise_behavior():
    """Test how Piecewise behaves with μ=0."""
    print("🔍 Testing Piecewise behavior...")
    
    # Define symbols
    r, M, mu = sp.symbols('r M mu', positive=True)
    
    # Create a simple test case
    K_classical = M / (r * (2*M - r))
    mu_eff = mu * (1 + mu**2/12)
    argument = mu_eff * K_classical
    
    print(f"K_classical = {K_classical}")
    print(f"μ_eff = {mu_eff}")
    print(f"argument = {argument}")
    
    # Test Piecewise function
    pw = sp.Piecewise(
        (K_classical, sp.Eq(mu, 0)),
        (sp.sin(argument) / mu_eff, True)
    )
    
    print(f"Piecewise function = {pw}")
    
    # Test evaluation at μ=0
    print("\n🧪 Testing μ=0 substitution:")
    result_mu0 = pw.subs(mu, 0)
    print(f"Direct substitution μ=0: {result_mu0}")
    
    # Test evaluation at a numerical point
    test_vals = {r: 5.0, M: 1.0, mu: 0}
    result_numerical = pw.subs(test_vals)
    print(f"Numerical substitution: {result_numerical}")
    
    # Try with limit
    print("\n🧪 Testing limit approach:")
    limit_result = sp.limit(pw, mu, 0)
    print(f"Limit as μ→0: {limit_result}")
    
    # Test what happens with just the second piece
    print("\n🧪 Testing second piece behavior:")
    second_piece = sp.sin(argument) / mu_eff
    print(f"Second piece = {second_piece}")
    print(f"Second piece at μ=0: {second_piece.subs(mu, 0)}")
    print(f"Limit of second piece: {sp.limit(second_piece, mu, 0)}")
    
    # Test series expansion of second piece
    print(f"Series expansion of second piece: {sp.series(second_piece, mu, 0, 3).removeO()}")

if __name__ == "__main__":
    test_piecewise_behavior()
