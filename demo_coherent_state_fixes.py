#!/usr/bin/env python3
"""
Minimal demonstration of fixed LQG coherent state construction.
"""

from lqg_fixed_components import LQGParameters, run_lqg_quantization, MuBarScheme
import numpy as np
import json

def main():
    # === OPTION 1: Use integer-valued JSON file that matches flux eigenvalues ===
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        coherent_width_E=1.0,
        coherent_width_K=1.0
    )
    
    result = run_lqg_quantization(
        classical_data_file="examples/lqg_example_integer_values.json",
        output_file="quantum_inputs/T00_quantum_fixed.json",
        lqg_params=params
    )
    
    # Fix eigenvalue complex warning when printing
    eigenvalues = result["eigenvalues"]
    print("\nFixed eigenvalues (properly handled):")
    for i, eigen in enumerate(eigenvalues[:2]):
        print(f"  λ_{i} = {np.real(eigen):.6e} + {np.imag(eigen):.6e}j (|λ| = {np.abs(eigen):.6e})")
    
    print("\nSee LQG_COHERENT_STATE_GUIDE.md for detailed explanation of the fixes.")

if __name__ == "__main__":
    main()
