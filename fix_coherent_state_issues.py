#!/usr/bin/env python3
"""
Fix the coherent state failure by directly using values that match the discrete flux basis.

This script demonstrates how to:
1. Use values that match discrete flux basis eigenvalues
2. Demonstrate how to fix the ComplexWarning for eigenvalues
"""

import numpy as np
import json
import sys
from lqg_fixed_components import LQGParameters, run_lqg_quantization, MuBarScheme

def main():
    print("\n=== FIXING COHERENT STATE CONSTRUCTION IN LQG FRAMEWORK ===")
    
    # Create parameters with good widths
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        coherent_width_E=1.5,  # Increased from 0.5
        coherent_width_K=1.5,  # Increased from 0.5
        mu_bar_scheme=MuBarScheme.MINIMAL_AREA
    )
    
    # We'll directly use our pre-prepared integer JSON file
    classical_data_file = "examples/lqg_example_integer_values.json"
    output_file = "quantum_inputs/T00_quantum_fixed.json"
    
    print("APPROACH 1: Using Integer-Valued JSON File")
    print(f"Using flux values from: {classical_data_file}")
    print("This file contains E_x, E_phi values that match integer μ, ν eigenvalues")
    print("And K_x, K_phi values that match the K_x_approx = 0.1 * μ scaling\n")
    
    # Run the quantization
    result = run_lqg_quantization(
        classical_data_file=classical_data_file,
        output_file=output_file,
        lqg_params=params
    )
    
    # Create recommendations for users
    print("\n=== RECOMMENDATIONS FOR LQG COHERENT STATE CONSTRUCTION ===")
    print("When constructing coherent states in the flux basis:")
    print("1. Choose classical E_x, E_phi values that match integer μ, ν eigenvalues:")
    print("   E.g., E_x = [2, 1, 0, -1, -2] for μ ∈ [-2, 2]")
    print("2. Choose classical K_x, K_phi values that match the K-operator scaling:")
    print("   In the code, K_x_approx = 0.1 * μ, so use K_x = [0.2, 0.1, 0.0, -0.1, -0.2]")
    print("3. If you must use non-integer E values, increase coherent_width_E and coherent_width_K")
    print("   E.g., coherent_width_E = 2.0, coherent_width_K = 2.0")
    print("4. To fix ComplexWarning, use float(np.abs(eigenvals[0])) or float(np.real(eigenvals[0]))")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
