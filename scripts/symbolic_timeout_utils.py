#!/usr/bin/env python3
"""
Robust timeout handling utilities for all symbolic operations.

This module provides a unified approach to handling timeouts in SymPy operations
to prevent indefinite hanging during symbolic computations.
"""

import sympy as sp
import signal
import time
import sys
import warnings
from typing import Any, Callable, Optional, Union
from contextlib import contextmanager

# Global timeout setting - can be adjusted per module
DEFAULT_SYMBOLIC_TIMEOUT = 6  # seconds (updated default)

class SymbolicTimeoutError(Exception):
    """Custom exception for symbolic operation timeouts."""
    pass

@contextmanager
def timeout_context(seconds: int):
    """
    Context manager that raises SymbolicTimeoutError after specified seconds.
    Only works on Unix-like systems with SIGALRM support.
    """
    if not hasattr(signal, 'SIGALRM'):
        # Windows fallback - no timeout
        yield
        return
        
    def timeout_handler(signum, frame):
        raise SymbolicTimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def safe_symbolic_operation(
    operation: Callable,
    *args,
    timeout_seconds: int = DEFAULT_SYMBOLIC_TIMEOUT,
    description: str = "symbolic operation",
    fallback_result: Any = None,
    raise_on_timeout: bool = False,
    **kwargs
) -> Any:
    """
    Safely execute a symbolic operation with timeout and error handling.
    
    Args:
        operation: SymPy function to execute (e.g., sp.integrate, sp.solve)
        *args: Positional arguments for the operation
        timeout_seconds: Maximum time to allow for the operation
        description: Human-readable description for logging
        fallback_result: Value to return if operation fails/times out
        raise_on_timeout: If True, raise exception on timeout instead of returning fallback
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of operation, or fallback_result if timeout/error occurs
        
    Raises:
        SymbolicTimeoutError: If raise_on_timeout=True and operation times out
    """
    start_time = time.time()
    
    try:
        # Use timeout context if on Unix-like system
        if hasattr(signal, 'SIGALRM'):
            with timeout_context(timeout_seconds):
                result = operation(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"  ✓ {description} completed in {elapsed:.2f}s")
                return result
        else:
            # Windows fallback: just try the operation with warning
            warnings.warn(f"Timeout not available on Windows. Attempting {description}...")
            result = operation(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"  ✓ {description} completed in {elapsed:.2f}s (no timeout)")
            return result
            
    except SymbolicTimeoutError:
        print(f"  ⏱ {description} timed out after {timeout_seconds}s")
        if raise_on_timeout:
            raise
        return fallback_result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ✗ {description} failed after {elapsed:.2f}s: {e}")
        if raise_on_timeout:
            raise
        return fallback_result

# Convenient wrappers for common SymPy operations
def safe_integrate(expr, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.integrate with timeout."""
    return safe_symbolic_operation(
        sp.integrate, expr, *args,
        timeout_seconds=timeout_seconds,
        description=f"integrate({expr})",
        **kwargs
    )

def safe_solve(equations, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.solve with timeout."""
    return safe_symbolic_operation(
        sp.solve, equations, *args,
        timeout_seconds=timeout_seconds,
        description=f"solve({equations})",
        fallback_result=[],
        **kwargs
    )

def safe_series(expr, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.series with timeout."""
    return safe_symbolic_operation(
        sp.series, expr, *args,
        timeout_seconds=timeout_seconds,
        description=f"series({expr})",
        **kwargs
    )

def safe_simplify(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.simplify with timeout."""
    return safe_symbolic_operation(
        sp.simplify, expr,
        timeout_seconds=timeout_seconds,
        description=f"simplify({expr})",
        fallback_result=expr,  # Return original expression if simplification fails
        **kwargs
    )

def safe_expand(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.expand with timeout."""
    return safe_symbolic_operation(
        sp.expand, expr,
        timeout_seconds=timeout_seconds,
        description=f"expand({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_factor(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.factor with timeout."""
    return safe_symbolic_operation(
        sp.factor, expr,
        timeout_seconds=timeout_seconds,
        description=f"factor({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_dsolve(eq, func, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.dsolve with timeout (uses longer default timeout)."""
    return safe_symbolic_operation(
        sp.dsolve, eq, func,
        timeout_seconds=timeout_seconds,
        description=f"dsolve({eq})",
        **kwargs
    )

def safe_solve_univariate_inequality(inequality, symbol, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.solve_univariate_inequality with timeout."""
    return safe_symbolic_operation(
        sp.solve_univariate_inequality, inequality, symbol,
        timeout_seconds=timeout_seconds,
        description=f"solve_univariate_inequality({inequality})",
        **kwargs
    )

def safe_limit(expr, var, point, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.limit with timeout."""
    return safe_symbolic_operation(
        sp.limit, expr, var, point,
        timeout_seconds=timeout_seconds,
        description=f"limit({expr}, {var} -> {point})",
        **kwargs
    )

def safe_diff(expr, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.diff with timeout."""
    return safe_symbolic_operation(
        sp.diff, expr, *args,
        timeout_seconds=timeout_seconds,
        description=f"diff({expr})",
        **kwargs
    )

# Additional safe wrappers for more SymPy operations

def safe_collect(expr, syms, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.collect with timeout."""
    return safe_symbolic_operation(
        sp.collect, expr, syms,
        timeout_seconds=timeout_seconds,
        description=f"collect({expr}, {syms})",
        fallback_result=expr,
        **kwargs
    )

def safe_apart(expr, var=None, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.apart with timeout."""
    if var is None:
        return safe_symbolic_operation(
            sp.apart, expr,
            timeout_seconds=timeout_seconds,
            description=f"apart({expr})",
            fallback_result=expr,
            **kwargs
        )
    else:
        return safe_symbolic_operation(
            sp.apart, expr, var,
            timeout_seconds=timeout_seconds,
            description=f"apart({expr}, {var})",
            fallback_result=expr,
            **kwargs
        )

def safe_cancel(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.cancel with timeout."""
    return safe_symbolic_operation(
        sp.cancel, expr,
        timeout_seconds=timeout_seconds,
        description=f"cancel({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_together(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.together with timeout."""
    return safe_symbolic_operation(
        sp.together, expr,
        timeout_seconds=timeout_seconds,
        description=f"together({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_trigsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.trigsimp with timeout."""
    return safe_symbolic_operation(
        sp.trigsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"trigsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_ratsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.ratsimp with timeout."""
    return safe_symbolic_operation(
        sp.ratsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"ratsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_nsimplify(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.nsimplify with timeout."""
    return safe_symbolic_operation(
        sp.nsimplify, expr,
        timeout_seconds=timeout_seconds,
        description=f"nsimplify({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_powsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.powsimp with timeout."""
    return safe_symbolic_operation(
        sp.powsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"powsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_logcombine(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.logcombine with timeout."""
    return safe_symbolic_operation(
        sp.logcombine, expr,
        timeout_seconds=timeout_seconds,
        description=f"logcombine({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_radsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.radsimp with timeout."""
    return safe_symbolic_operation(
        sp.radsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"radsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_separatevars(expr, symbols=None, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.separatevars with timeout."""
    if symbols is None:
        return safe_symbolic_operation(
            sp.separatevars, expr,
            timeout_seconds=timeout_seconds,
            description=f"separatevars({expr})",
            fallback_result=expr,
            **kwargs
        )
    else:
        return safe_symbolic_operation(
            sp.separatevars, expr, symbols,
            timeout_seconds=timeout_seconds,
            description=f"separatevars({expr}, {symbols})",
            fallback_result=expr,
            **kwargs
        )

def safe_simplify_complex(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.simplify with complex expressions."""
    return safe_symbolic_operation(
        lambda x: sp.simplify(x.expand(complex=True)),
        expr,
        timeout_seconds=timeout_seconds,
        description=f"simplify_complex({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_fibonacci(n, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.fibonacci with timeout."""
    return safe_symbolic_operation(
        sp.fibonacci, n,
        timeout_seconds=timeout_seconds,
        description=f"fibonacci({n})",
        **kwargs
    )

def safe_lucas(n, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.lucas with timeout."""
    return safe_symbolic_operation(
        sp.lucas, n,
        timeout_seconds=timeout_seconds,
        description=f"lucas({n})",
        **kwargs
    )

def safe_summation(expr, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.summation with timeout (uses longer default timeout)."""
    return safe_symbolic_operation(
        sp.summation, expr, *args,
        timeout_seconds=timeout_seconds,
        description=f"summation({expr})",
        **kwargs
    )

def safe_product(expr, *args, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.product with timeout (uses longer default timeout)."""
    return safe_symbolic_operation(
        sp.product, expr, *args,
        timeout_seconds=timeout_seconds,
        description=f"product({expr})",
        **kwargs
    )

def safe_Matrix_operations(matrix, operation_name, operation_func, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for Matrix operations with timeout."""
    return safe_symbolic_operation(
        operation_func, matrix,
        timeout_seconds=timeout_seconds,
        description=f"{operation_name}(Matrix)",
        **kwargs
    )

def safe_matrix_det(matrix, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for Matrix.det() with timeout."""
    return safe_symbolic_operation(
        lambda m: m.det(), matrix,
        timeout_seconds=timeout_seconds,
        description=f"det({matrix})",
        **kwargs
    )

def safe_matrix_inv(matrix, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for Matrix.inv() with timeout."""
    return safe_symbolic_operation(
        lambda m: m.inv(), matrix,
        timeout_seconds=timeout_seconds,
        description=f"inv({matrix})",
        **kwargs
    )

def safe_matrix_eigenvals(matrix, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*3, **kwargs):
    """Safe wrapper for Matrix.eigenvals() with timeout."""
    return safe_symbolic_operation(
        lambda m: m.eigenvals(), matrix,
        timeout_seconds=timeout_seconds,
        description=f"eigenvals({matrix})",
        fallback_result={},
        **kwargs
    )

def safe_matrix_eigenvects(matrix, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*3, **kwargs):
    """Safe wrapper for Matrix.eigenvects() with timeout."""
    return safe_symbolic_operation(
        lambda m: m.eigenvects(), matrix,
        timeout_seconds=timeout_seconds,
        description=f"eigenvects({matrix})",
        fallback_result=[],
        **kwargs
    )

# Additional specialized wrappers for LQG and symbolic computation

def safe_tensor_expand(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for tensor expansion operations."""
    return safe_symbolic_operation(
        lambda x: sp.expand(x, deep=True, power_base=False, power_exp=False),
        expr,
        timeout_seconds=timeout_seconds,
        description=f"tensor_expand({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_poly_expand(expr, *gens, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for polynomial expansion with generators."""
    return safe_symbolic_operation(
        sp.expand, expr, *gens,
        timeout_seconds=timeout_seconds,
        description=f"poly_expand({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_factor_list(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.factor_list with timeout."""
    return safe_symbolic_operation(
        sp.factor_list, expr,
        timeout_seconds=timeout_seconds,
        description=f"factor_list({expr})",
        fallback_result=(1, []),
        **kwargs
    )

def safe_gcd(a, b, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.gcd with timeout."""
    return safe_symbolic_operation(
        sp.gcd, a, b,
        timeout_seconds=timeout_seconds,
        description=f"gcd({a}, {b})",
        fallback_result=1,
        **kwargs
    )

def safe_lcm(a, b, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.lcm with timeout."""
    return safe_symbolic_operation(
        sp.lcm, a, b,
        timeout_seconds=timeout_seconds,
        description=f"lcm({a}, {b})",
        fallback_result=a*b,
        **kwargs
    )

def safe_resultant(f, g, var, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.resultant with timeout."""
    return safe_symbolic_operation(
        sp.resultant, f, g, var,
        timeout_seconds=timeout_seconds,
        description=f"resultant({f}, {g}, {var})",
        **kwargs
    )

def safe_groebner(polys, *gens, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*3, **kwargs):
    """Safe wrapper for sp.groebner with timeout."""
    return safe_symbolic_operation(
        sp.groebner, polys, *gens,
        timeout_seconds=timeout_seconds,
        description=f"groebner({polys})",
        fallback_result=[],
        **kwargs
    )

def safe_solve_poly_system(polys, *gens, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.solve_poly_system with timeout."""
    return safe_symbolic_operation(
        sp.solve_poly_system, polys, *gens,
        timeout_seconds=timeout_seconds,
        description=f"solve_poly_system({polys})",
        fallback_result=[],
        **kwargs
    )

def safe_roots(poly, var=None, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.roots with timeout."""
    if var is None:
        return safe_symbolic_operation(
            sp.roots, poly,
            timeout_seconds=timeout_seconds,
            description=f"roots({poly})",
            fallback_result={},
            **kwargs
        )
    else:
        return safe_symbolic_operation(
            sp.roots, poly, var,
            timeout_seconds=timeout_seconds,
            description=f"roots({poly}, {var})",
            fallback_result={},
            **kwargs
        )

def safe_real_root(poly, index, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.real_root with timeout."""
    return safe_symbolic_operation(
        sp.real_root, poly, index,
        timeout_seconds=timeout_seconds,
        description=f"real_root({poly}, {index})",
        **kwargs
    )

def safe_nsolve(equations, variables, initial_guess, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.nsolve with timeout."""
    return safe_symbolic_operation(
        sp.nsolve, equations, variables, initial_guess,
        timeout_seconds=timeout_seconds,
        description=f"nsolve({equations})",
        **kwargs
    )

def safe_linsolve(system, variables=None, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.linsolve with timeout."""
    if variables is None:
        return safe_symbolic_operation(
            sp.linsolve, system,
            timeout_seconds=timeout_seconds,
            description=f"linsolve({system})",
            fallback_result=set(),
            **kwargs
        )
    else:
        return safe_symbolic_operation(
            sp.linsolve, system, variables,
            timeout_seconds=timeout_seconds,
            description=f"linsolve({system}, {variables})",
            fallback_result=set(),
            **kwargs
        )

def safe_nonlinsolve(system, variables, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe wrapper for sp.nonlinsolve with timeout."""
    return safe_symbolic_operation(
        sp.nonlinsolve, system, variables,
        timeout_seconds=timeout_seconds,
        description=f"nonlinsolve({system}, {variables})",
        fallback_result=set(),
        **kwargs
    )

# Advanced simplification operations for LQG computations

def safe_fu(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.fu (Fu algorithm) with timeout."""
    return safe_symbolic_operation(
        sp.fu, expr,
        timeout_seconds=timeout_seconds,
        description=f"fu({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_posify(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.posify with timeout."""
    return safe_symbolic_operation(
        sp.posify, expr,
        timeout_seconds=timeout_seconds,
        description=f"posify({expr})",
        fallback_result=(expr, {}),
        **kwargs
    )

def safe_refine(expr, assumptions, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.refine with timeout."""
    return safe_symbolic_operation(
        sp.refine, expr, assumptions,
        timeout_seconds=timeout_seconds,
        description=f"refine({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_ask(expr, assumptions=None, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.ask with timeout."""
    if assumptions is None:
        return safe_symbolic_operation(
            sp.ask, expr,
            timeout_seconds=timeout_seconds,
            description=f"ask({expr})",
            **kwargs
        )
    else:
        return safe_symbolic_operation(
            sp.ask, expr, assumptions,
            timeout_seconds=timeout_seconds,
            description=f"ask({expr}, {assumptions})",
            **kwargs
        )

# Special functions and series operations

def safe_hyperexpand(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.hyperexpand with timeout."""
    return safe_symbolic_operation(
        sp.hyperexpand, expr,
        timeout_seconds=timeout_seconds,
        description=f"hyperexpand({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_combsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.combsimp with timeout."""
    return safe_symbolic_operation(
        sp.combsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"combsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_gammasimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.gammasimp with timeout."""
    return safe_symbolic_operation(
        sp.gammasimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"gammasimp({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_besselsimp(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT, **kwargs):
    """Safe wrapper for sp.besselsimp with timeout."""
    return safe_symbolic_operation(
        sp.besselsimp, expr,
        timeout_seconds=timeout_seconds,
        description=f"besselsimp({expr})",
        fallback_result=expr,
        **kwargs
    )

# LQG-specific operations for constraint algebra

def safe_constraint_expand(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe expansion specifically for constraint expressions."""
    return safe_symbolic_operation(
        lambda x: sp.expand(x, trig=True, complex=True, power_base=True),
        expr,
        timeout_seconds=timeout_seconds,
        description=f"constraint_expand({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_hamiltonian_simplify(expr, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe simplification specifically for Hamiltonian expressions."""
    def hamiltonian_simp(x):
        # Apply multiple simplification strategies
        x = sp.trigsimp(x)
        x = sp.cancel(x)
        x = sp.simplify(x)
        return x
    
    return safe_symbolic_operation(
        hamiltonian_simp, expr,
        timeout_seconds=timeout_seconds,
        description=f"hamiltonian_simplify({expr})",
        fallback_result=expr,
        **kwargs
    )

def safe_lqg_series_expand(expr, param, point, order, timeout_seconds=DEFAULT_SYMBOLIC_TIMEOUT*2, **kwargs):
    """Safe series expansion for LQG parameter expansions."""
    def lqg_series(x, p, pt, n):
        series_result = sp.series(x, p, pt, n)
        return series_result.removeO() if hasattr(series_result, 'removeO') else series_result
    
    return safe_symbolic_operation(
        lqg_series, expr, param, point, order,
        timeout_seconds=timeout_seconds,
        description=f"lqg_series_expand({expr}, {param}, order={order})",
        **kwargs
    )

# Utility function to check if timeout is available
def has_timeout_support() -> bool:
    """Check if the current platform supports SIGALRM-based timeouts."""
    return hasattr(signal, 'SIGALRM')

# Configuration function
def set_default_timeout(seconds: int):
    """Set the global default timeout for symbolic operations."""
    global DEFAULT_SYMBOLIC_TIMEOUT
    DEFAULT_SYMBOLIC_TIMEOUT = seconds
    print(f"Default symbolic timeout set to {seconds} seconds")

# Test function
def test_timeout_functionality():
    """Test the timeout functionality with a deliberately slow operation."""
    print("Testing timeout functionality...")
    
    if not has_timeout_support():
        print("  Warning: Running on platform without timeout support")
    
    # Test with a fast operation
    result = safe_symbolic_operation(
        sp.expand, (sp.Symbol('x') + 1)**3,
        timeout_seconds=1,
        description="fast expansion test"
    )
    
    if result is not None:
        print("  ✓ Fast operation completed successfully")
    else:
        print("  ✗ Fast operation failed unexpectedly")
    
    # Test with a potentially slow operation
    x = sp.Symbol('x')
    slow_expr = sp.integrate(sp.exp(-x**2) * sp.sin(x**3), (x, 0, sp.oo))
    result = safe_symbolic_operation(
        sp.simplify, slow_expr,
        timeout_seconds=1,
        description="potentially slow simplification test"
    )
    
    print("  ✓ Timeout handling test completed")

if __name__ == "__main__":
    test_timeout_functionality()
