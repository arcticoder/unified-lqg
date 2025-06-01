#!/usr/bin/env python3
"""
Validation script for T^{00} integration using known analytic test cases.

This script validates the numerical integration by comparing against
simple cases where we can compute the integral analytically.
"""

import numpy as np
from scipy.integrate import quad
import math

def test_gaussian_integral():
    """
    Test case: Gaussian T00(r) = -A * exp(-(r-r0)²/σ²)
    
    Analytic: ∫_{-∞}^{∞} exp(-(r-r0)²/σ²) * r² dr = σ³ * sqrt(π) * [(r0/σ)² + 1/2] (approximately)
    For volume integral: multiply by 4π
    """
    print("\n=== GAUSSIAN T00 VALIDATION ===")
    
    # Parameters
    A = 1e-6
    r0 = 2e-35  # Center
    sigma = 0.5e-35  # Width
    b0 = 0.1e-35  # Integration start
    r_max = 10e-35  # Integration end
    
    def gaussian_T00(r):
        return -A * np.exp(-((r - r0)**2) / (sigma**2))
    
    def integrand(r):
        return abs(gaussian_T00(r)) * 4 * np.pi * r**2
    
    # Numerical integration
    numerical, error = quad(integrand, b0, r_max, epsabs=1e-15, epsrel=1e-12)
    
    # Approximate analytic (for large integration range)
    # ∫ |T00| * 4πr² dr ≈ A * 4π * σ³ * sqrt(π) * scaling_factor
    # where scaling_factor accounts for the r² weight and finite bounds
    scaling_factor = (r0**2 + sigma**2/2) / sigma**2  # Rough approximation
    analytic_approx = A * 4 * np.pi * sigma**3 * np.sqrt(np.pi) * scaling_factor
    
    print(f"Gaussian parameters: A={A:.1e}, r0={r0:.1e}, σ={sigma:.1e}")
    print(f"Integration range: [{b0:.1e}, {r_max:.1e}]")
    print(f"Numerical result: {numerical:.6e} ± {error:.2e}")
    print(f"Analytic approx:  {analytic_approx:.6e}")
    print(f"Ratio (num/analytic): {numerical/analytic_approx:.3f}")
    
    return numerical, analytic_approx


def test_power_law_integral():
    """
    Test case: Power law T00(r) = -A / r^n
    
    Analytic: ∫ r^(-n) * r² dr = ∫ r^(2-n) dr
    For n ≠ 3: integral = [r^(3-n)] / (3-n) evaluated from b0 to r_max
    """
    print("\n=== POWER LAW T00 VALIDATION ===")
    
    # Parameters
    A = 1e-6
    n = 2  # Power law exponent (avoid n=3 to prevent divergence)
    b0 = 1e-35
    r_max = 10e-35
    
    def power_law_T00(r):
        return -A / (r**n)
    
    def integrand(r):
        return abs(power_law_T00(r)) * 4 * np.pi * r**2
    
    # Numerical integration
    numerical, error = quad(integrand, b0, r_max, epsabs=1e-15, epsrel=1e-12)
    
    # Analytic result
    # ∫ |A/r^n| * 4π * r² dr = 4πA * ∫ r^(2-n) dr
    # For n=2: = 4πA * ∫ 1 dr = 4πA * (r_max - b0)
    if n == 2:
        analytic = 4 * np.pi * A * (r_max - b0)
    elif n != 3:
        analytic = 4 * np.pi * A * (r_max**(3-n) - b0**(3-n)) / (3-n)
    else:
        analytic = 4 * np.pi * A * np.log(r_max/b0)  # Special case n=3
    
    print(f"Power law parameters: A={A:.1e}, n={n}")
    print(f"Integration range: [{b0:.1e}, {r_max:.1e}]")
    print(f"Numerical result: {numerical:.6e} ± {error:.2e}")
    print(f"Analytic result:  {analytic:.6e}")
    print(f"Ratio (num/analytic): {numerical/analytic:.6f}")
    print(f"Relative error: {abs(numerical-analytic)/analytic:.2e}")
    
    return numerical, analytic


def test_exponential_integral():
    """
    Test case: Exponential decay T00(r) = -A * exp(-r/λ)
    
    This doesn't have a simple closed form for ∫ exp(-r/λ) * r² dr,
    but we can use series expansion or numerical reference.
    """
    print("\n=== EXPONENTIAL T00 VALIDATION ===")
    
    # Parameters
    A = 1e-6
    lambda_param = 2e-35  # Decay length
    b0 = 0.1e-35
    r_max = 10e-35
    
    def exponential_T00(r):
        return -A * np.exp(-r/lambda_param)
    
    def integrand(r):
        return abs(exponential_T00(r)) * 4 * np.pi * r**2
    
    # Numerical integration
    numerical, error = quad(integrand, b0, r_max, epsabs=1e-15, epsrel=1e-12)
    
    # For comparison, integrate analytically using integration by parts twice
    # ∫ exp(-r/λ) * r² dr = -λ³ * exp(-r/λ) * (r²/λ² + 2r/λ + 2)
    def antiderivative(r):
        x = r / lambda_param
        return -lambda_param**3 * np.exp(-x) * (x**2 + 2*x + 2)
    
    analytic = 4 * np.pi * A * (antiderivative(r_max) - antiderivative(b0))
    
    print(f"Exponential parameters: A={A:.1e}, λ={lambda_param:.1e}")
    print(f"Integration range: [{b0:.1e}, {r_max:.1e}]")
    print(f"Numerical result: {numerical:.6e} ± {error:.2e}")
    print(f"Analytic result:  {analytic:.6e}")
    print(f"Ratio (num/analytic): {numerical/analytic:.6f}")
    print(f"Relative error: {abs(numerical-analytic)/analytic:.2e}")
    
    return numerical, analytic


def run_validation_suite():
    """Run all validation tests and report summary."""
    print("=" * 60)
    print("T^{00} INTEGRATION VALIDATION SUITE")
    print("=" * 60)
    
    tests = [
        ("Gaussian", test_gaussian_integral),
        ("Power Law", test_power_law_integral), 
        ("Exponential", test_exponential_integral)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            numerical, analytic = test_func()
            rel_error = abs(numerical - analytic) / abs(analytic)
            results.append((test_name, numerical, analytic, rel_error))
        except Exception as e:
            print(f"\n✗ {test_name} test failed: {e}")
            results.append((test_name, 0, 0, float('inf')))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"{'Test':<12} {'Numerical':<12} {'Analytic':<12} {'Rel. Error':<10} {'Status'}")
    print("-" * 60)
    
    all_passed = True
    for test_name, numerical, analytic, rel_error in results:
        if rel_error < 1e-6:
            status = "✓ PASS"
        elif rel_error < 1e-3:
            status = "~ OK"
        else:
            status = "✗ FAIL"
            all_passed = False
        
        print(f"{test_name:<12} {numerical:<12.3e} {analytic:<12.3e} {rel_error:<10.2e} {status}")
    
    print("-" * 60)
    if all_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("The numerical integration is working correctly.")
    else:
        print("⚠️  Some validation tests failed.")
        print("Check the integration algorithm for potential issues.")
    
    return all_passed


if __name__ == "__main__":
    success = run_validation_suite()
    exit(0 if success else 1)
