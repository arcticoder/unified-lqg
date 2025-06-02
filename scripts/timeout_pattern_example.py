#!/usr/bin/env python3
"""
Example demonstrating the complete timeout pattern for SymPy operations.

This template shows how to properly integrate symbolic_timeout_utils
into any module that uses SymPy operations.
"""

import sympy as sp
import numpy as np
import os
import sys

# Step 1: Ensure we can import symbolic_timeout_utils
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

try:
    from symbolic_timeout_utils import (
        safe_integrate, safe_solve, safe_series,
        safe_diff, safe_simplify, safe_expand, 
        safe_factor, safe_collect, safe_cancel,
        set_default_timeout
    )
    TIMEOUT_SUPPORT = True
    # Choose a per-module default (tune as needed)
    set_default_timeout(8)  # or 10, or 6, depending on complexity
    print("✓ Symbolic timeout support available")
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

def example_differentiation():
    """Example: Replace direct sp.diff calls with safe_diff"""
    print("\n=== Differentiation Example ===")
    
    x, r = sp.symbols('x r')
    f = sp.sin(x**2) * sp.exp(-r/x)
    
    # OLD WAY (can hang):
    # df_dr = sp.diff(f, r)
    
    # NEW WAY (with timeout):
    df_dr = safe_diff(f, r, timeout_seconds=5)
    
    if df_dr is not None:
        print(f"df/dr = {df_dr}")
    else:
        print("Differentiation timed out, using fallback or skipping")

def example_simplification():
    """Example: Replace direct sp.simplify calls with safe_simplify"""
    print("\n=== Simplification Example ===")
    
    x = sp.Symbol('x')
    expr = (x**2 + 2*x + 1) / (x + 1)
    
    # OLD WAY:
    # simplified = sp.simplify(expr)
    
    # NEW WAY:
    simplified = safe_simplify(expr, timeout_seconds=8)
    
    if simplified is not None:
        print(f"Simplified: {simplified}")
    else:
        # Timeout → fall back:
        print("Simplification timed out, using original expression")
        simplified = expr

def example_series_expansion():
    """Example: Replace direct sp.series calls with safe_series"""
    print("\n=== Series Expansion Example ===")
    
    x, mu = sp.symbols('x mu')
    expr = sp.sin(mu * x) / (mu * x)
    
    # OLD WAY:
    # ser = sp.series(expr, mu, 0, 4).removeO()
    
    # NEW WAY:
    ser = safe_series(expr, mu, 0, n=4, timeout_seconds=6)
    
    if ser is not None:
        ser = ser.removeO()
        print(f"Series expansion: {ser}")
    else:
        # Fallback or log timeout
        print("Series expansion timed out, using manual expansion")
        # Provide manual fallback if possible
        ser = 1 - (mu*x)**2/6 + (mu*x)**4/120

def example_equation_solving():
    """Example: Replace direct sp.solve calls with safe_solve"""
    print("\n=== Equation Solving Example ===")
    
    r, M, alpha = sp.symbols('r M alpha')
    
    # Some complex equation from LQG
    r_inv4_coeff = alpha * M**2 - r**2 / 6
    
    # OLD WAY:
    # alpha_solution = sp.solve(r_inv4_coeff, alpha)
    
    # NEW WAY:
    alpha_solution = safe_solve(r_inv4_coeff, alpha, timeout_seconds=8)
    
    if alpha_solution and len(alpha_solution) > 0:
        print(f"Solved α = {alpha_solution[0]}")
    else:
        print("Could not solve for α or solving timed out")

def example_integration():
    """Example: Replace direct sp.integrate calls with safe_integrate"""
    print("\n=== Integration Example ===")
    
    x, r = sp.symbols('x r')
    integrand = sp.exp(-x**2) * x**2
    
    # OLD WAY (can hang on complex integrals):
    # result = sp.integrate(integrand, (x, 0, sp.oo))
    
    # NEW WAY:
    result = safe_integrate(integrand, (x, 0, sp.oo), timeout_seconds=10)
    
    if result is not None:
        print(f"Integral result: {result}")
    else:
        print("Integration timed out, using numerical fallback")
        # Switch to numerical integration as fallback
        from scipy.integrate import quad
        numerical_result, _ = quad(lambda x: np.exp(-x**2) * x**2, 0, 10)
        print(f"Numerical fallback: {numerical_result}")

def example_comprehensive_computation():
    """Example of a complete computation using multiple safe operations"""
    print("\n=== Comprehensive Example ===")
    
    r, M, mu, alpha = sp.symbols('r M mu alpha', real=True, positive=True)
    
    # Define a complex LQG-related expression
    f_metric = 1 - 2*M/r + alpha*mu**2*M**2/r**4
    
    print(f"Starting metric: f(r) = {f_metric}")
    
    # Step 1: Differentiate
    df_dr = safe_diff(f_metric, r, timeout_seconds=5)
    if df_dr is None:
        print("  Differentiation failed")
        return
    
    print(f"  df/dr = {df_dr}")
    
    # Step 2: Simplify the derivative
    df_dr_simplified = safe_simplify(df_dr, timeout_seconds=5)
    if df_dr_simplified is None:
        df_dr_simplified = df_dr
    
    print(f"  Simplified: {df_dr_simplified}")
    
    # Step 3: Expand in small μ
    expansion = safe_series(df_dr_simplified, mu, 0, n=3, timeout_seconds=6)
    if expansion is not None:
        expansion = expansion.removeO()
        print(f"  μ-expansion: {expansion}")
    else:
        print("  Series expansion timed out")
    
    # Step 4: Solve a related constraint
    constraint = df_dr_simplified.subs(r, 2*M)  # Evaluate at horizon
    alpha_solution = safe_solve(constraint, alpha, timeout_seconds=8)
    
    if alpha_solution:
        print(f"  Constraint solution: α = {alpha_solution[0]}")
    else:
        print("  Constraint solving failed or timed out")

def test_error_handling():
    """Test that error handling works properly"""
    print("\n=== Error Handling Test ===")
    
    # This should handle errors gracefully
    x = sp.Symbol('x')
    problematic_expr = sp.log(x) / (x - 1)
    
    # Test differentiation at problematic point
    result = safe_diff(problematic_expr, x, timeout_seconds=2)
    print(f"Differentiation result: {result}")
    
    # Test solving impossible equation
    impossible_eq = x**2 + 1  # No real solutions
    solution = safe_solve(impossible_eq, x, timeout_seconds=2)
    print(f"Solution to x² + 1 = 0: {solution}")

def main():
    """Run all examples to demonstrate the timeout pattern"""
    print("SYMBOLIC TIMEOUT UTILITIES - USAGE EXAMPLES")
    print("="*60)
    
    print(f"Platform timeout support: {TIMEOUT_SUPPORT}")
    
    # Run all examples
    example_differentiation()
    example_simplification()
    example_series_expansion()
    example_equation_solving()
    example_integration()
    example_comprehensive_computation()
    test_error_handling()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Now you can apply this pattern to any SymPy-using module.")

if __name__ == "__main__":
    main()
