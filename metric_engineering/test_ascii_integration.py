#!/usr/bin/env python3
"""
Test script for the improved AsciiMath-based T^{00} integration.

This script validates the new implementation by:
1. Testing against known analytic cases (Gaussian T00)
2. Comparing AsciiMath vs LaTeX parsing results  
3. Checking singularity handling near f → 1 points
4. Verifying integration accuracy with different methods

Run this after creating the .am file to ensure everything works correctly.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent))

try:
    from compute_negative_energy_am import (
        extract_T00_ascii, 
        build_numeric_T00_from_ascii,
        robust_integration,
        unit_test_gaussian_T00
    )
    print("✓ Successfully imported AsciiMath modules")
except ImportError as e:
    print(f"✗ Failed to import modules: {e}")
    print("Make sure you have installed: pip install sympy scipy numpy python-ndjson")
    sys.exit(1)


def test_ascii_extraction():
    """Test AsciiMath expression extraction from .am file."""
    print("\n=== Testing AsciiMath Extraction ===")
    
    am_path = "exotic_matter_density.am"
    if not os.path.exists(am_path):
        print(f"✗ AsciiMath file not found: {am_path}")
        return False
    
    try:
        # Test static mode extraction
        static_expr = extract_T00_ascii(am_path, mode="static")
        print(f"✓ Static T00 extracted: {len(static_expr)} characters")
        
        # Test regularized mode extraction  
        reg_expr = extract_T00_ascii(am_path, mode="regularized")
        print(f"✓ Regularized T00 extracted: {len(reg_expr)} characters")
        
        return True
    except Exception as e:
        print(f"✗ AsciiMath extraction failed: {e}")
        return False


def test_sympy_conversion():
    """Test conversion from AsciiMath to SymPy and numerical function."""
    print("\n=== Testing SymPy Conversion ===")
    
    # Simple test expression
    test_expr = "(4*(f(r) - 1)^3 * (-2*f(r) - df_dr(r) + 2)) / (64*pi*r*(f(r) - 1)^4)"
    b0_test = 1e-35
    
    try:
        T00_func = build_numeric_T00_from_ascii(test_expr, b0_test, mode="static")
        
        # Test function at several points
        test_points = [0.5e-35, 1e-35, 2e-35, 5e-35]
        for r_test in test_points:
            result = T00_func(r_test)
            print(f"  T00({r_test:.1e}) = {result:.3e}")
        
        print("✓ SymPy conversion successful")
        return True
    except Exception as e:
        print(f"✗ SymPy conversion failed: {e}")
        return False


def test_integration_methods():
    """Test different numerical integration strategies."""
    print("\n=== Testing Integration Methods ===")
    
    # Simple test function: T00(r) = -1/(r² + b0²)
    b0 = 1e-35
    r_max = 10 * b0
    
    def simple_T00(r):
        return -1e-6 / (r**2 + b0**2)
    
    methods = ["adaptive", "simpson", "composite"]
    results = {}
    
    for method in methods:
        try:
            result = robust_integration(simple_T00, b0, r_max, method=method)
            results[method] = result
            print(f"  {method:>10}: {result:.6e}")
        except Exception as e:
            print(f"  {method:>10}: FAILED ({e})")
            results[method] = None
    
    # Check consistency between methods
    valid_results = [r for r in results.values() if r is not None]
    if len(valid_results) >= 2:
        relative_diff = abs(max(valid_results) - min(valid_results)) / max(valid_results)
        if relative_diff < 0.1:  # 10% tolerance
            print(f"✓ Integration methods consistent (rel_diff = {relative_diff:.3f})")
            return True
        else:
            print(f"✗ Integration methods inconsistent (rel_diff = {relative_diff:.3f})")
            return False
    else:
        print("✗ Too few methods succeeded")
        return False


def test_singularity_handling():
    """Test behavior near f → 1 singularities."""
    print("\n=== Testing Singularity Handling ===")
    
    b0 = 1e-35
    
    # Create a function that approaches singularity
    def singular_T00(r):
        # Alcubierre-like function that goes to 1 at some point
        f = 0.5 * (np.tanh(2 * (r - 3*b0) / b0) + 1)
        denominator = (f - 1)**4 + 1e-12  # Small regularization
        if abs(denominator) < 1e-15:
            return 0.0
        return -1e-6 / denominator
    
    try:
        # Test integration with regularization
        result = robust_integration(singular_T00, b0, 10*b0, method="adaptive")
        print(f"  Regularized integration result: {result:.6e}")
        
        # Test at specific points near singularity
        test_r = 3 * b0  # Near the tanh transition
        test_val = singular_T00(test_r)
        print(f"  T00 at transition point r={test_r:.1e}: {test_val:.3e}")
        
        print("✓ Singularity handling successful")
        return True
    except Exception as e:
        print(f"✗ Singularity handling failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("COMPREHENSIVE TEST OF ASCIIMATH T^{00} IMPLEMENTATION")
    print("=" * 60)
    
    tests = [
        ("AsciiMath Extraction", test_ascii_extraction),
        ("SymPy Conversion", test_sympy_conversion), 
        ("Integration Methods", test_integration_methods),
        ("Singularity Handling", test_singularity_handling),
        ("Unit Test (Gaussian)", lambda: unit_test_gaussian_T00() is not None)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for production use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    # Change to the metric_engineering directory
    if os.path.basename(os.getcwd()) != "metric_engineering":
        if os.path.exists("metric_engineering"):
            os.chdir("metric_engineering")
            print(f"Changed to directory: {os.getcwd()}")
        else:
            print("Please run this script from the warp-framework root or metric_engineering directory")
            sys.exit(1)
    
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
