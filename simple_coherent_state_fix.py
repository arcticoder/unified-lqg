#!/usr/bin/env python3
"""
Demonstrate how to fix coherent state issues in LQG framework.
This script shows how to use integer-valued E_x, E_phi and
matching K_x, K_phi to ensure successful coherent state construction.
"""

import numpy as np
from lqg_fixed_components import (
    LQGParameters,
    LatticeConfiguration,
    KinematicalHilbertSpace,
    MuBarScheme,
    run_lqg_quantization
)

def main():
    print("=== LQG COHERENT STATE FIX DEMONSTRATION ===\n")
    
    # Create custom LQG parameters
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        # Increase widths for better coherent state construction
        coherent_width_E=1.5,
        coherent_width_K=1.5
    )
    
    # Use the integer-valued flux JSON file
    print("Solution 1: Using integer-valued JSON file")
    print("Running with examples/lqg_example_integer_values.json")
    print("This file has integer E_x, E_phi values and matching K_x, K_phi values:")
    print("  E_x = [2, 1, 0, -1, -2]")
    print("  E_phi = [1, 1, 0, -1, -1]")
    print("  K_x = [0.2, 0.1, 0.0, -0.1, -0.2]  # Exactly 0.1 * μ")
    print("  K_phi = [0.1, 0.05, 0.0, -0.05, -0.1]  # Matches K_phi_approx scaling")
    print("\n")
    
    # Run quantization using the integer values JSON
    result = run_lqg_quantization(
        classical_data_file="examples/lqg_example_integer_values.json",
        output_file="quantum_inputs/T00_quantum_fixed.json",
        lqg_params=params
    )
    
    # Handle eigenvalues properly when printing
    eigenvalues = result["eigenvalues"]
    print("\nEigenvalues (properly handled):")
    for i, eigen in enumerate(eigenvalues[:2]):
        print(f"  λ_{i} = {np.real(eigen):.6e} + {np.imag(eigen):.6e}j (|λ| = {np.abs(eigen):.6e})")
    
    print("\nSummary of Fixes:")
    print("1. Use JSON with integer values that match discrete flux eigenvalues")
    print("   - E_x, E_phi should be integers within [-μ_max, μ_max], [-ν_max, ν_max]")
    print("   - K_x, K_phi should match the 0.1 * μ scaling used in the code")
    print("2. Increase coherent state widths for better overlap")
    print("   - coherent_width_E = 1.5 (from default 0.5)")
    print("   - coherent_width_K = 1.5 (from default 0.5)")
    print("3. Fix ComplexWarning by using np.abs() or np.real() when handling eigenvalues")
    
    return 0

if __name__ == "__main__":
    main()
