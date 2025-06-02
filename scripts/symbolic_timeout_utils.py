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
DEFAULT_SYMBOLIC_TIMEOUT = 5  # seconds

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
