#!/usr/bin/env python3
"""
Direct test of coherent state construction with different configurations
to verify the fix for the normalization issue.
"""

import numpy as np
import json
from lqg_fixed_components import (
    LQGParameters, 
    LatticeConfiguration, 
    KinematicalHilbertSpace,
    MuBarScheme
)

def test_coherent_state_construction(
    classical_E_x,
    classical_E_phi,
    classical_K_x,
    classical_K_phi,
    width_E=0.5,
    width_K=0.5,
    mu_max=2,
    nu_max=2
):
    """Test coherent state construction with the given parameters"""
    print(f"\nTesting coherent state with widths E={width_E}, K={width_K}:")
    print(f"  Classical E_x: {classical_E_x}")
    print(f"  Classical E_phi: {classical_E_phi}")
    print(f"  Classical K_x: {classical_K_x}")
    print(f"  Classical K_phi: {classical_K_phi}")
    
    # Create parameters
    lqg_params = LQGParameters(
        mu_max=mu_max,
        nu_max=nu_max,
        basis_truncation=100,
        coherent_width_E=width_E,
        coherent_width_K=width_K
    )
    
    # Create lattice configuration
    n_sites = len(classical_E_x)
    lattice_config = LatticeConfiguration(
        n_sites=n_sites,
        r_min=1e-35,
        r_max=1e-33
    )
    setattr(lattice_config, "E_x_classical", list(classical_E_x))
    setattr(lattice_config, "E_phi_classical", list(classical_E_phi))
    
    # Create kinematical Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
    
    # Generate and check coherent state
    print("Generating coherent state and verifying ⟨K⟩...")
    psi_coh, errors = kin_space.create_coherent_state_with_Kcheck(
        E_x_target=classical_E_x,
        E_phi_target=classical_E_phi,
        K_x_target=classical_K_x,
        K_phi_target=classical_K_phi,
        width=width_E
    )
    
    print("\nSummary of deviations:")
    print(f"  max |⟨E⟩−E_classical| = {errors['max_E_error']:.2e}")
    print(f"  max |⟨K⟩−K_classical| = {errors['max_K_error']:.2e}")
    
    return errors


# Test 1: Original values from lqg_demo_classical_data.json with default widths
print("===== Test 1: Original values with default widths =====")
# Load original values
with open('examples/lqg_demo_classical_data.json', 'r') as f:
    orig_data = json.load(f)

orig_E_x = np.array(orig_data["E_x"])
orig_E_phi = np.array(orig_data["E_phi"])
orig_K_x = np.array(orig_data["K_x"])
orig_K_phi = np.array(orig_data["K_phi"])

# Should fail with uniform state fallback
errors1 = test_coherent_state_construction(
    orig_E_x, orig_E_phi, orig_K_x, orig_K_phi, 
    width_E=0.5, width_K=0.5
)

# Test 2: Original values but with wider Gaussians
print("\n===== Test 2: Original values with wider Gaussians =====")
# Should work better but still have some error
errors2 = test_coherent_state_construction(
    orig_E_x, orig_E_phi, orig_K_x, orig_K_phi, 
    width_E=2.0, width_K=2.0
)

# Test 3: Integer values that exactly match the basis
print("\n===== Test 3: Integer values matching the basis =====")
# These should give perfect matching
integer_E_x = np.array([2, 1, 0, -1, -2])
integer_E_phi = np.array([1, 1, 0, -1, -1])
integer_K_x = np.array([0.2, 0.1, 0.0, -0.1, -0.2])
integer_K_phi = np.array([0.1, 0.05, 0.0, -0.05, -0.1])

errors3 = test_coherent_state_construction(
    integer_E_x, integer_E_phi, integer_K_x, integer_K_phi, 
    width_E=0.5, width_K=0.5
)

# Print comparison
print("\n===== Comparison of all tests =====")
print(f"Test 1 (original, narrow): max E error = {errors1['max_E_error']:.2e}, max K error = {errors1['max_K_error']:.2e}")
print(f"Test 2 (original, wide):   max E error = {errors2['max_E_error']:.2e}, max K error = {errors2['max_K_error']:.2e}")
print(f"Test 3 (integer values):   max E error = {errors3['max_E_error']:.2e}, max K error = {errors3['max_K_error']:.2e}")
