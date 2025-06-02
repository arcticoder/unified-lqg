#!/usr/bin/env python3
"""
Specific test for the 5-site integer coherent state construction.
"""

import json
import numpy as np
from lqg_fixed_components import (
    LQGParameters, LatticeConfiguration, KinematicalHilbertSpace,
    FluxBasisState
)

# Load the integer data
with open("examples/lqg_example_integer_values.json", 'r') as f:
    data = json.load(f)

E_x = data['E_x']        # [2, 1, 0, -1, -2]
E_phi = data['E_phi']    # [1, 1, 0, -1, -1]
K_x = data['K_x']        # [0.2, 0.1, 0.0, -0.1, -0.2]
K_phi = data['K_phi']    # [0.1, 0.05, 0.0, -0.05, -0.1]

print("Target configuration:")
print(f"  E_x (μ): {E_x}")
print(f"  E_phi (ν): {E_phi}")
print(f"  K_x: {K_x}")
print(f"  K_phi: {K_phi}")

# Create the exact target state
mu_config = np.array(E_x, dtype=int)
nu_config = np.array(E_phi, dtype=int)
target_state = FluxBasisState(mu_config, nu_config)

print(f"\nTarget FluxBasisState:")
print(f"  μ_config: {target_state.mu_config}")
print(f"  ν_config: {target_state.nu_config}")

# Set up LQG parameters
lqg_params = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=1000  # Large enough to include our state
)

# Create lattice configuration for 5 sites
lattice_config = LatticeConfiguration(
    n_sites=5,
    r_min=1e-35,
    r_max=1e-33
)

print(f"\nBuilding kinematical Hilbert space...")
kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
print(f"  Dimension: {kin_space.dim}")
print(f"  Sites: {kin_space.n_sites}")
print(f"  μ_max: {lqg_params.mu_max}")
print(f"  ν_max: {lqg_params.nu_max}")

# Check if target state exists
if target_state in kin_space.state_to_index:
    index = kin_space.state_to_index[target_state]
    print(f"✓ Target state FOUND at index {index}")
else:
    print("✗ Target state NOT FOUND in basis")
    
    # Show some sample states to understand the structure
    print("\nFirst 10 basis states:")
    for i in range(min(10, len(kin_space.basis_states))):
        state = kin_space.basis_states[i]
        print(f"  {i:3d}: μ={state.mu_config}, ν={state.nu_config}")
    
    print("\nLast 10 basis states:")
    for i in range(max(0, len(kin_space.basis_states) - 10), len(kin_space.basis_states)):
        state = kin_space.basis_states[i]
        print(f"  {i:3d}: μ={state.mu_config}, ν={state.nu_config}")

# Try to construct coherent state regardless
print(f"\nConstructing coherent state with default widths...")
try:
    psi, checks = kin_space.create_coherent_state_with_Kcheck(
        np.array(E_x, dtype=float),
        np.array(E_phi, dtype=float),
        np.array(K_x, dtype=float),
        np.array(K_phi, dtype=float)
    )
    
    print(f"✓ Coherent state constructed successfully")
    print(f"  Max E error: {checks['max_E_error']:.6f}")
    print(f"  Max K error: {checks['max_K_error']:.6f}")
    print(f"  State norm: {np.linalg.norm(psi):.6f}")
    
    # Show which basis state has maximum amplitude
    max_amplitude_idx = np.argmax(np.abs(psi))
    max_state = kin_space.basis_states[max_amplitude_idx]
    print(f"  Max amplitude at state {max_amplitude_idx}: μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"  Amplitude: {psi[max_amplitude_idx]:.6f}")
    
except Exception as e:
    print(f"✗ Error constructing coherent state: {e}")

# Try with wider Gaussians
print(f"\nTrying with wider Gaussian widths...")
lqg_params_wide = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=1000,
    coherent_width_E=2.0,
    coherent_width_K=2.0
)

kin_space_wide = KinematicalHilbertSpace(lattice_config, lqg_params_wide)
try:
    psi_wide, checks_wide = kin_space_wide.create_coherent_state_with_Kcheck(
        np.array(E_x, dtype=float),
        np.array(E_phi, dtype=float),
        np.array(K_x, dtype=float),
        np.array(K_phi, dtype=float)
    )
    
    print(f"✓ Wide-Gaussian coherent state constructed successfully")
    print(f"  Max E error: {checks_wide['max_E_error']:.6f}")
    print(f"  Max K error: {checks_wide['max_K_error']:.6f}")
    print(f"  State norm: {np.linalg.norm(psi_wide):.6f}")
    
    # Show which basis state has maximum amplitude
    max_amplitude_idx_wide = np.argmax(np.abs(psi_wide))
    max_state_wide = kin_space_wide.basis_states[max_amplitude_idx_wide]
    print(f"  Max amplitude at state {max_amplitude_idx_wide}: μ={max_state_wide.mu_config}, ν={max_state_wide.nu_config}")
    print(f"  Amplitude: {psi_wide[max_amplitude_idx_wide]:.6f}")
    
except Exception as e:
    print(f"✗ Error constructing wide-Gaussian coherent state: {e}")
