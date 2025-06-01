#!/usr/bin/env python3
"""
Script to verify if specific states are present in the kinematical Hilbert space basis.
This helps diagnose why the coherent state isn't peaking where expected.
"""

import numpy as np
from lqg_fixed_components import (
    LQGParameters, 
    LatticeConfiguration,
    KinematicalHilbertSpace,
    FluxBasisState
)

# Create a small Hilbert space
lqg_params = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=1000
)

lattice_config = LatticeConfiguration(
    n_sites=3,
    r_min=1.0,
    r_max=3.0
)

print("Creating kinematical Hilbert space...")
kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
print(f"Hilbert space dimension: {kin_space.dim}")

# Define the states we want to check
target_states = [
    ([2, 0, -2], [1, 0, -1]),    # Our desired state
    ([2, 1, 0], [1, 0, 0]),      # Another state
    ([0, 0, 0], [0, 0, 0]),      # Zero state
    ([-2, -1, 0], [1, 0, -1])    # Random state
]

# Check if each target state is in the basis
print("\nChecking for target states in the basis...")
for i, (mu_config, nu_config) in enumerate(target_states):
    target_state = FluxBasisState(np.array(mu_config), np.array(nu_config))
    
    if target_state in kin_space.state_to_index:
        idx = kin_space.state_to_index[target_state]
        print(f"State {i+1}: μ={mu_config}, ν={nu_config} - FOUND at index {idx}")
    else:
        print(f"State {i+1}: μ={mu_config}, ν={nu_config} - NOT FOUND")

# List the first few basis states to see what's actually in there
print("\nFirst 10 basis states:")
for i in range(min(10, kin_space.dim)):
    state = kin_space.basis_states[i]
    print(f"{i}: {state}")

# Check the mu/nu ranges actually used
mu_min = min([min(state.mu_config) for state in kin_space.basis_states])
mu_max = max([max(state.mu_config) for state in kin_space.basis_states])
nu_min = min([min(state.nu_config) for state in kin_space.basis_states])
nu_max = max([max(state.nu_config) for state in kin_space.basis_states])

print(f"\nActual μ range: [{mu_min}, {mu_max}]")
print(f"Actual ν range: [{nu_min}, {nu_max}]")
print(f"Expected μ range: [-{lqg_params.mu_max}, {lqg_params.mu_max}]")
print(f"Expected ν range: [-{lqg_params.nu_max}, {lqg_params.nu_max}]")
