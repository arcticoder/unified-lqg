#!/usr/bin/env python3
"""
Timeout Pattern Example

This file demonstrates the correct pattern for importing and using symbolic timeout utilities
in any script that performs SymPy operations. Use this as a template for new scripts.

Key Features:
1. Robust import block with fallbacks
2. All symbolic operations wrapped with safe_* functions
3. Proper timeout handling and fallbacks
4. Clear error reporting
"""

import os
import sys
import sympy as sp
import numpy as np

# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Standard timeout import pattern - copy this block to any new script
try:
    from symbolic_timeout_utils import (
        # Core safe operations
        safe_integrate, safe_solve, safe_series, safe_diff, 
        safe_simplify, safe_expand, safe_factor,
        
        # Algebraic operations
        safe_collect, safe_apart, safe_cancel, safe_together,
        
        # Simplification operations
        safe_trigsimp, safe_ratsimp, safe_nsimplify, safe_powsimp,
        safe_logcombine, safe_radsimp, safe_separatevars,
        
        # Advanced operations
        safe_summation, safe_product, safe_dsolve,
        safe_solve_univariate_inequality, safe_limit,
        
        # Matrix operations
        safe_matrix_det, safe_matrix_inv, safe_matrix_eigenvals, safe_matrix_eigenvects,
        
        # Polynomial operations
        safe_roots, safe_factor_list, safe_gcd, safe_lcm,
        safe_solve_poly_system, safe_groebner, safe_resultant,
        
        # Solver operations
        safe_linsolve, safe_nonlinsolve, safe_nsolve,
        
        # Special functions
        safe_hyperexpand, safe_combsimp, safe_gammasimp,
        safe_besselsimp, safe_fu, safe_refine, safe_ask,
        
        # LQG-specific operations
        safe_constraint_expand, safe_hamiltonian_simplify,
        safe_lqg_series_expand, safe_tensor_expand,
        
        # Utility functions
        set_default_timeout, has_timeout_support, DEFAULT_SYMBOLIC_TIMEOUT
    )
    TIMEOUT_SUPPORT = True
    set_default_timeout(8)  # Set timeout appropriate for this module's complexity
    print(f"✓ Timeout support available: {has_timeout_support()}")
    
except ImportError as e:
    print(f"Warning: symbolic_timeout_utils not found ({e}); using direct SymPy calls")
    TIMEOUT_SUPPORT = False
    
    # Define no-timeout fallbacks exactly as specified
    def safe_integrate(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)  # Remove timeout args
        return sp.integrate(expr, *args, **kwargs)
    
    def safe_solve(expr, symbol, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.solve(expr, symbol, *args, **kwargs)
    
    def safe_series(expr, var, point, n, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.series(expr, var, point, n, *args, **kwargs)
    
    def safe_diff(expr, var, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.diff(expr, var, *args, **kwargs)
    
    def safe_simplify(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.simplify(expr, *args, **kwargs)
    
    def safe_expand(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.expand(expr, *args, **kwargs)
    
    def safe_factor(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.factor(expr, *args, **kwargs)
    
    def safe_collect(expr, syms, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.collect(expr, syms, *args, **kwargs)
    
    def safe_cancel(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.cancel(expr, *args, **kwargs)
    
    def safe_together(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.together(expr, *args, **kwargs)
    
    def safe_trigsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.trigsimp(expr, *args, **kwargs)
    
    def safe_ratsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.ratsimp(expr, *args, **kwargs)
    
    def safe_nsimplify(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.nsimplify(expr, *args, **kwargs)
    
    # Additional fallbacks for new wrappers
    def safe_powsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.powsimp(expr, *args, **kwargs)
    
    def safe_apart(expr, var=None, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        if var is None:
            return sp.apart(expr, *args, **kwargs)
        else:
            return sp.apart(expr, var, *args, **kwargs)
    
    def safe_summation(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.summation(expr, *args, **kwargs)
    
    def safe_product(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.product(expr, *args, **kwargs)
    
    def safe_dsolve(eq, func, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.dsolve(eq, func, *args, **kwargs)
    
    def safe_limit(expr, var, point, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.limit(expr, var, point, *args, **kwargs)
    
    def safe_solve_univariate_inequality(inequality, symbol, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.solve_univariate_inequality(inequality, symbol, *args, **kwargs)
    
    # Matrix operation fallbacks
    def safe_matrix_det(matrix, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return matrix.det()
    
    def safe_matrix_inv(matrix, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return matrix.inv()    
    def safe_matrix_eigenvals(matrix, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return matrix.eigenvals()
    
    def safe_matrix_eigenvects(matrix, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return matrix.eigenvects()
    
    # Additional fallbacks for polynomial operations
    def safe_roots(poly, var=None, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        if var is None:
            return sp.roots(poly, *args, **kwargs)
        else:
            return sp.roots(poly, var, *args, **kwargs)
    
    def safe_factor_list(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.factor_list(expr, *args, **kwargs)
    
    def safe_gcd(a, b, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.gcd(a, b, *args, **kwargs)
    
    def safe_lcm(a, b, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.lcm(a, b, *args, **kwargs)
    
    def safe_resultant(f, g, var, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.resultant(f, g, var, *args, **kwargs)
    
    def safe_groebner(polys, *gens, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.groebner(polys, *gens, **kwargs)
    
    def safe_solve_poly_system(polys, *gens, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.solve_poly_system(polys, *gens, **kwargs)
    
    # Solver operations fallbacks
    def safe_linsolve(system, variables=None, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        if variables is None:
            return sp.linsolve(system, *args, **kwargs)
        else:
            return sp.linsolve(system, variables, *args, **kwargs)
    
    def safe_nonlinsolve(system, variables, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.nonlinsolve(system, variables, *args, **kwargs)
    
    def safe_nsolve(equations, variables, initial_guess, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.nsolve(equations, variables, initial_guess, *args, **kwargs)
    
    # Special functions fallbacks
    def safe_hyperexpand(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.hyperexpand(expr, *args, **kwargs)
    
    def safe_combsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.combsimp(expr, *args, **kwargs)
    
    def safe_gammasimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.gammasimp(expr, *args, **kwargs)
    
    def safe_besselsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.besselsimp(expr, *args, **kwargs)
    
    def safe_fu(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.fu(expr, *args, **kwargs)
    
    def safe_refine(expr, assumptions, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.refine(expr, assumptions, *args, **kwargs)
    
    def safe_ask(expr, assumptions=None, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        if assumptions is None:
            return sp.ask(expr, *args, **kwargs)
        else:
            return sp.ask(expr, assumptions, *args, **kwargs)
    
    # Additional simplification operations
    def safe_logcombine(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.logcombine(expr, *args, **kwargs)
    
    def safe_radsimp(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.radsimp(expr, *args, **kwargs)
    
    def safe_separatevars(expr, symbols=None, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        if symbols is None:
            return sp.separatevars(expr, *args, **kwargs)
        else:
            return sp.separatevars(expr, symbols, *args, **kwargs)
    
    # LQG-specific operation fallbacks
    def safe_constraint_expand(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.expand(expr, trig=True, complex=True, power_base=True)
    
    def safe_hamiltonian_simplify(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        # Apply multiple simplification strategies
        x = sp.trigsimp(expr)
        x = sp.cancel(x)
        x = sp.simplify(x)
        return x
    
    def safe_lqg_series_expand(expr, param, point, order, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        series_result = sp.series(expr, param, point, order, *args, **kwargs)
        return series_result.removeO() if hasattr(series_result, 'removeO') else series_result
    
    def safe_tensor_expand(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        return sp.expand(expr, deep=True, power_base=False, power_exp=False)
def demonstrate_timeout_usage():
    """
    Demonstrate proper usage patterns for symbolic operations with timeouts.
    """
    print("\n=== Timeout Pattern Example ===")
    
    # Define symbolic variables
    x, y, z = sp.symbols('x y z', real=True)
    mu, alpha, M, r = sp.symbols('mu alpha M r', positive=True)
    
    print("\n1. Basic symbolic operations:")
    
    # Differentiation
    expr = sp.sin(x**2) + sp.cos(y**3)
    diff_result = safe_diff(expr, x, timeout_seconds=5)
    if diff_result is not None:
        print(f"  ✓ d/dx({expr}) = {diff_result}")
    else:
        print(f"  ⏱ Differentiation timed out")
    
    # Simplification
    complex_expr = sp.sqrt(x**2) * sp.sin(x)**2 + sp.sqrt(x**2) * sp.cos(x)**2
    simp_result = safe_simplify(complex_expr, timeout_seconds=6)
    if simp_result is not None:
        print(f"  ✓ Simplified: {complex_expr} → {simp_result}")
    else:
        print(f"  ⏱ Simplification timed out, using original: {complex_expr}")
    
    # Expansion
    expand_expr = (x + y + z)**3
    expand_result = safe_expand(expand_expr, timeout_seconds=3)
    if expand_result is not None:
        print(f"  ✓ Expanded: {expand_expr} → {expand_result}")
    else:
        print(f"  ⏱ Expansion timed out")
    
    print("\n2. Series expansions (common in LQG work):")
    
    # Series expansion like sin(μ*K)/(μ*K)
    K = sp.Symbol('K')
    sin_expansion = safe_series(sp.sin(mu*K)/(mu*K), mu*K, 0, n=4, timeout_seconds=5)
    if sin_expansion is not None:
        sin_expansion = sin_expansion.removeO()
        print(f"  ✓ sin(μK)/(μK) ≈ {sin_expansion}")
    else:
        print(f"  ⏱ Series expansion timed out")
    
    # Higher-order series for metric function
    f_lqg = 1 - 2*M/r + alpha*mu**2*M**2/r**4
    f_series = safe_series(f_lqg, mu, 0, n=3, timeout_seconds=4)
    if f_series is not None:
        f_series = f_series.removeO()
        print(f"  ✓ f_LQG(r) series: {f_series}")
    else:
        print(f"  ⏱ Metric series expansion timed out")
    
    print("\n3. Equation solving:")
    
    # Solve for LQG coefficient
    equation = alpha*mu**2 - sp.Rational(1,6)*mu**2
    alpha_sol = safe_solve(equation, alpha, timeout_seconds=5)
    if alpha_sol:
        print(f"  ✓ Solved α: {alpha_sol[0]}")
    else:
        print(f"  ⏱ Solve timed out or no solution found")
    
    print("\n4. Integration (often problematic):")
    
    # Simple integration
    integral1 = safe_integrate(x**2, (x, 0, 1), timeout_seconds=3)
    if integral1 is not None:
        print(f"  ✓ ∫₀¹ x² dx = {integral1}")
    else:
        print(f"  ⏱ Integration timed out")
    
    # More complex integration that might timeout
    integral2 = safe_integrate(sp.exp(-x**2)*sp.sin(x**3), (x, 0, sp.oo), timeout_seconds=2)
    if integral2 is not None:
        print(f"  ✓ Complex integral = {integral2}")
    else:
        print(f"  ⏱ Complex integration timed out (expected)")
    
    print("\n5. Advanced operations:")
    
    # Collecting terms
    poly_expr = x**3 + 2*x**2*y + x*y**2 + y**3
    collected = safe_collect(poly_expr, x, timeout_seconds=3)
    if collected is not None:
        print(f"  ✓ Collected in x: {collected}")
    else:
        print(f"  ⏱ Collection timed out")
    
    # Factoring
    factor_expr = x**4 - 1
    factored = safe_factor(factor_expr, timeout_seconds=3)
    if factored is not None:
        print(f"  ✓ Factored: {factor_expr} = {factored}")
    else:
        print(f"  ⏱ Factoring timed out")
    
    print("\n6. Matrix operations:")
    
    # Create test matrix
    A = sp.Matrix([[1, 2], [3, 4]])
    
    # Determinant
    det_A = safe_matrix_det(A, timeout_seconds=3)
    if det_A is not None:
        print(f"  ✓ det(A) = {det_A}")
    else:
        print(f"  ⏱ Matrix determinant timed out")
    
    # Eigenvalues
    eigenvals = safe_matrix_eigenvals(A, timeout_seconds=4)
    if eigenvals:
        print(f"  ✓ Eigenvalues: {eigenvals}")
    else:
        print(f"  ⏱ Eigenvalue computation timed out")
    
    print("\n=== Pattern Example Complete ===")


def demonstrate_lqg_workflow():
    """
    Show how to use timeouts in a typical LQG metric derivation workflow.
    """
    print("\n=== LQG Workflow Example ===")
    
    # Symbolic variables for LQG
    r, M, mu, alpha, beta, gamma = sp.symbols('r M mu alpha beta gamma', real=True, positive=True)
    K_x, K_phi = sp.symbols('K_x K_phi', real=True)
    
    print("\n1. Polymer quantization of curvature:")
    
    # Original Hamiltonian term: K_x^2 
    K_x_squared = K_x**2
    print(f"  Classical term: {K_x_squared}")
    
    # Polymer correction: sin(μK_x)/(μK_x) ≈ 1 - (μK_x)²/6 + ...
    sin_factor = sp.sin(mu*K_x)/(mu*K_x)
    sin_expansion = safe_series(sin_factor, mu*K_x, 0, n=4, timeout_seconds=6)
    
    if sin_expansion is not None:
        sin_expansion = sin_expansion.removeO()
        print(f"  Polymer correction: sin(μK_x)/(μK_x) ≈ {sin_expansion}")
        
        # Apply correction to get polymer Hamiltonian term
        H_polymer_term = K_x**2 * sin_expansion**2
        H_expanded = safe_expand(H_polymer_term, timeout_seconds=8)
        
        if H_expanded is not None:
            print(f"  Polymer term: K_x² × [sin(μK_x)/(μK_x)]² ≈ {H_expanded}")
        else:
            print("  ⏱ Polymer term expansion timed out")
    else:
        print("  ⏱ sin expansion timed out")
    
    print("\n2. LQG metric ansatz:")
    
    # Define extended ansatz
    f_ansatz = 1 - 2*M/r + alpha*mu**2*M**2/r**4 + beta*mu**4*M**3/r**7 + gamma*mu**6*M**4/r**10
    print(f"  f_LQG(r) = {f_ansatz}")
    
    # Series expansion to check structure
    f_series = safe_series(f_ansatz, mu, 0, n=4, timeout_seconds=5)
    if f_series is not None:
        f_series = f_series.removeO()
        print(f"  Series form: {f_series}")
    else:
        print("  ⏱ Ansatz series expansion timed out")
    
    print("\n3. Constraint equation setup:")
    
    # Mock constraint equation: H_eff = f'(r)/f(r) + corrections = 0
    f_prime = safe_diff(f_ansatz, r, timeout_seconds=5)
    if f_prime is not None:
        constraint = f_prime/f_ansatz + mu**2*M**2/r**6  # Mock additional term
        constraint_simp = safe_simplify(constraint, timeout_seconds=8)
        
        if constraint_simp is not None:
            print(f"  Constraint: {constraint_simp} = 0")
        else:
            print(f"  ⏱ Constraint simplification timed out")
            constraint_simp = constraint
        
        # Expand in μ to match coefficients
        constraint_series = safe_series(constraint_simp, mu, 0, n=3, timeout_seconds=8)
        if constraint_series is not None:
            constraint_series = constraint_series.removeO()
            print(f"  Series: {constraint_series}")
            
            # Extract coefficients
            mu2_coeff = constraint_series.coeff(mu, 2)
            if mu2_coeff is not None:
                # Extract coefficient of 1/r^4 term to solve for alpha
                r_neg4_coeff = mu2_coeff.as_coefficients_dict().get(r**(-4), 0)
                if r_neg4_coeff != 0:
                    alpha_solution = safe_solve(r_neg4_coeff, alpha, timeout_seconds=5)
                    if alpha_solution:
                        print(f"  ✓ Solved α = {alpha_solution[0]}")
                    else:
                        print("  ⏱ Alpha solve timed out")
                else:
                    print("  No 1/r⁴ coefficient found")
            else:
                print("  No μ² coefficient found")
        else:
            print("  ⏱ Constraint series expansion timed out")
    else:
        print("  ⏱ Derivative computation timed out")
    
    print("\n4. Validation and closed form:")
    
    # Try to find patterns in coefficients for closed form
    if 'alpha_solution' in locals() and alpha_solution:
        alpha_val = alpha_solution[0]
        
        # Mock beta value for demonstration
        beta_val = -alpha_val**2 / 3  # Example relationship
        
        # Check if there's a pattern
        ratio = safe_simplify(beta_val / alpha_val**2, timeout_seconds=4)
        if ratio is not None:
            print(f"  Coefficient ratio β/α² = {ratio}")
            
            # Try closed form
            if ratio == sp.Rational(-1, 3):
                closed_form = (1 - 2*M/r) + (alpha_val*mu**2*M**2/r**4) / (1 + mu**2*M/(3*r**3))
                
                # Validate by expanding
                validation = safe_series(closed_form, mu, 0, n=3, timeout_seconds=6)
                if validation is not None:
                    validation = validation.removeO()
                    print(f"  ✓ Closed form validation: {validation}")
                else:
                    print("  ⏱ Closed form validation timed out")
            else:
                print("  No simple closed form found")
        else:
            print("  ⏱ Ratio computation timed out")
    
    print("\n=== LQG Workflow Complete ===")


if __name__ == "__main__":
    # Demonstrate basic timeout patterns
    demonstrate_timeout_usage()
    
    # Show LQG-specific workflow
    demonstrate_lqg_workflow()
    
    print(f"\n✓ Timeout support active: {TIMEOUT_SUPPORT}")
    if TIMEOUT_SUPPORT:
        print("  All symbolic operations are protected by timeouts")
    else:
        print("  Running with direct SymPy calls (no timeout protection)")


# Additional examples for comprehensive demonstration

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
    
    # Run core demonstrations
    demonstrate_timeout_usage()
    demonstrate_lqg_workflow()
    
    # Run additional examples  
    example_differentiation()
    example_simplification()
    example_series_expansion()
    example_equation_solving()
    example_integration()
    example_comprehensive_computation()
    test_error_handling()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Copy the import pattern from this file to use in your own scripts.")


# Run the demonstration when called directly
if __name__ == "__main__":
    main()
