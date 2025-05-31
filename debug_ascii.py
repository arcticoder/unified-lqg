#!/usr/bin/env python3
"""
Debug script for AsciiMath to SymPy conversion
"""

import numpy as np
from sympy import Symbol, lambdify, pi, sympify, tanh, sech

def debug_ascii_conversion():
    """Debug the AsciiMath to SymPy conversion process"""
    
    # Original AsciiMath expression
    ascii_rhs = """(
    4*(f(r) - 1)^3 * (-2*f(r) - df_dr(r) + 2) 
    - 4*(f(r) - 1)^2 * df_dr(r)
  ) / (64*pi*r*(f(r) - 1)^4)"""
    
    print("Original AsciiMath expression:")
    print(ascii_rhs)
    print()
    
    # Step 1: Basic replacements
    expr = ascii_rhs.replace("^", "**")
    print("After ^ -> **:")
    print(expr)
    print()
    
    # Step 2: Replace function calls
    expr = expr.replace("f(r)", "((tanh((2/b0)*(r - 3*b0)) + 1)/2)")
    expr = expr.replace("df_dr(r)", "((1/b0) * sech((2/b0)*(r - 3*b0))**2)")
    print("After function replacements:")
    print(expr)
    print()
    
    # Step 3: Try to sympify with proper locals
    r = Symbol('r', positive=True)
    b0 = Symbol('b0', positive=True)
    
    try:
        T00_sym = sympify(expr, locals={"pi": pi, "tanh": tanh, "sech": sech, "r": r, "b0": b0})
        print("✓ SymPy conversion successful!")
        print("T00_sym type:", type(T00_sym))
        print("T00_sym:", T00_sym)
        
        # Test with a specific b0 value
        b0_val = 1e-35
        T00_sym_sub = T00_sym.subs(b0, b0_val)
        print(f"\nAfter substituting b0 = {b0_val}:")
        print("T00_sym_sub:", T00_sym_sub)
        
        # Try lambdify
        func_map = {
            "sech": lambda x: 1/np.cosh(x),
            "tanh": np.tanh,
            "cosh": np.cosh,
            "sinh": np.sinh
        }
        T00_numeric = lambdify(r, T00_sym_sub, ["numpy", func_map])
        print("\n✓ Lambdify successful!")
        
        # Test at a point
        test_r = b0_val * 2.0
        test_val = T00_numeric(test_r)
        print(f"T00({test_r:.2e}) = {test_val:.3e}")
        
    except Exception as e:
        print(f"✗ Error during SymPy conversion: {e}")
        print(f"Error type: {type(e)}")
        
        # Try a simpler approach - direct substitution
        print("\nTrying direct substitution approach...")
        try:
            # Define the Alcubierre function components directly
            sigma = 2/b0_val
            rs = 3*b0_val
            
            def f_alcubierre(r_val):
                return 0.5 * (np.tanh(sigma * (r_val - rs)) + 1)
            
            def df_dr_alcubierre(r_val):
                return (1/b0_val) * (1/np.cosh(sigma * (r_val - rs)))**2
            
            def T00_direct(r_val):
                f = f_alcubierre(r_val)
                df_dr = df_dr_alcubierre(r_val)
                numerator = (4*(f - 1)**3 * (-2*f - df_dr + 2) - 4*(f - 1)**2 * df_dr)
                denominator = 64 * np.pi * r_val * (f - 1)**4
                return numerator / denominator
            
            test_val = T00_direct(test_r)
            print(f"✓ Direct computation: T00({test_r:.2e}) = {test_val:.3e}")
            
        except Exception as e2:
            print(f"✗ Direct approach also failed: {e2}")

if __name__ == "__main__":
    debug_ascii_conversion()
