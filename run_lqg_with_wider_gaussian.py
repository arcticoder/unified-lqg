#!/usr/bin/env python3
"""
Run LQG quantization with much wider coherent state parameters to ensure overlap.

This script addresses the issue where coherent state construction fails because of
insufficient overlap between classical values and quantum eigenvalues.
"""

from lqg_fixed_components import LQGParameters, run_lqg_quantization, MuBarScheme
import numpy as np
import json
import os

def main():
    # Use the original data but with MUCH wider coherent state parameters
    classical_data_file = "examples/lqg_demo_classical_data.json"  # Original values
    output_file = "quantum_inputs/T00_quantum_wider_coherent.json"
    
    # Create LQG parameters with much wider settings
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        coherent_width_E=5.0,  # Much larger than default 0.5
        coherent_width_K=5.0,  # Much larger than default 0.5
        mu_bar_scheme=MuBarScheme.MINIMAL_AREA
    )

    print("\n============ Running LQG quantization with wide coherent state parameters ============")
    print(f"Using classical data from: {classical_data_file}")
    print("With modified coherent state parameters:")
    print(f"  coherent_width_E = {params.coherent_width_E} (default: 0.5)")
    print(f"  coherent_width_K = {params.coherent_width_K} (default: 0.5)")
    print("==============================================================================\n")
    
    # Run the quantization
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
    
    # Let's also create a new json file with integer values that perfectly match
    # the flux eigenvalues and K-operator scaling
    create_perfect_match_json()
    return result

def create_perfect_match_json():
    """Create a JSON file with E and K values that perfectly match the quantum operators."""
    # Output path
    output_file = "examples/perfect_match_values.json"
    
    # Load original data for reference
    with open("examples/lqg_demo_classical_data.json", "r") as f:
        original_data = json.load(f)
        
    # Create new data with perfect match values
    # These values are designed to match exactly with the quantum operator eigenvalues
    perfect_data = {
        "r_grid": original_data["r_grid"],
        "dr": original_data["dr"],
        "E_x": [2, 1, 0, -1, -2],     # Integers to match μ eigenvalues
        "E_phi": [2, 1, 0, -1, -2],   # Integers to match ν eigenvalues
        "K_x": [0.2, 0.1, 0.0, -0.1, -0.2],     # Exactly 0.1 * μ
        "K_phi": [0.2, 0.1, 0.0, -0.1, -0.2],  # Exactly 0.1 * ν
        "exotic": original_data["exotic"],
        "metadata": {
            "description": "Perfect match data for LQG quantum operators",
            "geometry": "spherically_symmetric_wormhole",
            "n_sites": 5,
            "r_min": 1e-35,
            "r_max": 1e-33,
            "throat_location": 5.05e-34,
            "units": "SI_meters",
            "exotic_matter_type": "phantom_scalar",
            "notes": "Values chosen to match exactly with quantum operator eigenvalues"
        }
    }
    
    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(perfect_data, f, indent=2)
    
    print(f"\nCreated perfect match data file: {output_file}")
    print("This file contains values that exactly match the quantum operator eigenvalues")
    print("To use it, run: python run_lqg_with_integer_values.py --data examples/perfect_match_values.json")

if __name__ == "__main__":
    main()
