#!/usr/bin/env python3
"""
Run LQG quantization with integer-valued flux inputs to ensure coherent state peaking.

This script addresses the issue where coherent state construction fails when classical
values don't align with discrete flux eigenvalues. It uses the integer-valued example file
and increases the coherent state width parameters.
"""

from lqg_fixed_components import LQGParameters, run_lqg_quantization
from enum import Enum
import numpy as np

class MuBarScheme(Enum):
    """μ̄-schemes for holonomy corrections in LQG"""
    MINIMAL_AREA = "minimal_area"
    IMPROVED_DYNAMICS = "improved_dynamics"
    SIMPLIFIED = "simplified"

def main():
    # Use the integer-valued json file that matches the discrete flux basis
    classical_data_file = "examples/lqg_example_integer_values.json"
    output_file = "quantum_inputs/T00_quantum_integer.json"
    
    # Create LQG parameters with appropriate settings
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        # The scaling between mu and K_x is 0.1 in the code
        # K_x_approx = 0.1 * mu
        # With classical K_x = [0.2, 0.1, 0.0, -0.1, -0.2]
        # and mu = [2, 1, 0, -1, -2], they match perfectly
        coherent_width_E=1.0,  # Widen this from default 0.5
        coherent_width_K=1.0,  # Widen this from default 0.5
        mu_bar_scheme=MuBarScheme.MINIMAL_AREA
    )

    # Run the quantization
    print("\n============ Running LQG quantization with integer flux values ============")
    print(f"Using classical data from: {classical_data_file}")
    print(f"These values are specifically chosen to match the discrete flux basis:")
    print(f"  μ ∈ [-{params.mu_max}, {params.mu_max}], ν ∈ [-{params.nu_max}, {params.nu_max}]")
    print(f"  K_x_approx = 0.1 * μ, K_phi_approx = 0.1 * ν")
    print("====================================================================\n")
    
    result = run_lqg_quantization(
        classical_data_file=classical_data_file,
        output_file=output_file,
        lqg_params=params
    )
    
    # Print eigenvalues with proper handling of complex numbers
    eigenvalues = result["eigenvalues"]
    print("\nEigenvalues (properly handled):")
    for i, eigen in enumerate(eigenvalues[:5]):
        if np.iscomplex(eigen):
            print(f"  λ_{i} = {np.real(eigen):.6e} + {np.imag(eigen):.6e}j (|λ| = {np.abs(eigen):.6e})")
        else:
            print(f"  λ_{i} = {eigen:.6e}")
    
    print(f"\nOutput saved to: {output_file}")
    
    return result

if __name__ == "__main__":
    main()
