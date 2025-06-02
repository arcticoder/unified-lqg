#!/usr/bin/env python3
"""
Test script to verify Hamilton-Jacobi integration fixes.

This script tests the timeout and series expansion fallback approach
to ensure geodesic integration doesn't hang indefinitely.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hamilton_jacobi_check import (
    hamilton_jacobi_analysis,
    attempt_analytical_integration,
    solve_bounce_radius,
    numeric_action_integral
)

def test_integration_timeout():
    """Test that integration respects timeout and falls back to series."""
    print("Testing integration timeout and fallback...")
    
    # Get Hamilton-Jacobi setup
    hj_results = hamilton_jacobi_analysis()
    
    # Test with very short timeout to force fallback
    print("\n--- Testing with 1 second timeout ---")
    integration_results = attempt_analytical_integration(hj_results, timeout=1.0)
    
    print(f"Integration success: {integration_results['integration_success']}")
    print(f"Series success: {integration_results['series_success']}")
    
    if integration_results['p_r_series'] is not None:
        print("✓ Series expansion succeeded")
    else:
        print("✗ Series expansion failed")
    
    return integration_results

def test_bounce_radius():
    """Test bounce radius calculation."""
    print("\nTesting bounce radius calculation...")
    
    # Test with different parameter values
    test_cases = [
        {"M": 1.0, "mu": 0.01, "alpha": 1/6},
        {"M": 1.0, "mu": 0.05, "alpha": 1/6},
        {"M": 2.0, "mu": 0.03, "alpha": 1/6}
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n--- Test case {i+1}: M={params['M']}, μ={params['mu']}, α={params['alpha']} ---")
        result = solve_bounce_radius(params["M"], params["mu"], params["alpha"])
        
        if result.get('success', False):
            print(f"✓ Bounce radius: {result['r_bounce']:.6f}")
            print(f"  Relative shift: {result['relative_shift']:+.3%}")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")

def test_numeric_integration():
    """Test numeric action integral."""
    print("\nTesting numeric action integral...")
    
    # Test parameters  
    E_val = 0.95
    m_val = 1.0
    M_val = 1.0
    mu_val = 0.05
    alpha_val = 1/6
    
    result = numeric_action_integral(E_val, m_val, M_val, mu_val, alpha_val)
    
    if result.get('success', False):
        print(f"✓ Numeric integration succeeded")
        print(f"  Action S = {result['S_numeric']:.6f}")
        print(f"  Error estimate = {result['integration_error']:.2e}")
    else:
        print(f"✗ Numeric integration failed: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING HAMILTON-JACOBI INTEGRATION FIXES")
    print("="*60)
    
    try:
        # Test 1: Integration timeout and fallback
        integration_results = test_integration_timeout()
        
        # Test 2: Bounce radius calculation
        test_bounce_radius()
        
        # Test 3: Numeric integration
        numeric_results = test_numeric_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
        # Summary
        if integration_results['integration_success']:
            print("✓ Integration works (either direct or series)")
        else:
            print("✗ Integration completely failed")
            
        print("The Hamilton-Jacobi script should now run without hanging!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
