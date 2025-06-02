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

# Add more comprehensive checks
print("\n=== Comprehensive Analysis ===")

# Print basis statistics
print(f"\nBasis Statistics:")
print(f"  Total dimension: {kin_space.dim}")
print(f"  μ_max: {kin_space.lqg_params.mu_max}")
print(f"  ν_max: {kin_space.lqg_params.nu_max}")

# Count μ and ν ranges in actual basis
mu_values = set()
nu_values = set()

for state in kin_space.basis_states:
    mu_values.update(state.mu_config)
    nu_values.update(state.nu_config)

print(f"  μ values present: {sorted(mu_values)}")
print(f"  ν values present: {sorted(nu_values)}")

# Test integer JSON configurations
print("\n=== Testing Integer JSON Configurations ===")

# Perfect match configuration from lqg_example_integer_values.json
perfect_mu = [2, 1, 0, -1, -2]
perfect_nu = [1, 1, 0, -1, -1]

print(f"\nChecking perfect match from JSON:")
print(f"  μ = {perfect_mu}")
print(f"  ν = {perfect_nu}")

# This needs to be checked for a 5-site configuration
if kin_space.n_sites == len(perfect_mu):
    target_perfect = FluxBasisState(np.array(perfect_mu), np.array(perfect_nu))
    if target_perfect in kin_space.state_to_index:
        idx = kin_space.state_to_index[target_perfect]
        print(f"✓ Perfect match state FOUND at index {idx}")
    else:
        print("✗ Perfect match state NOT FOUND in basis")
        print("  Try increasing basis_truncation or check μ_max/ν_max")
else:
    print(f"  Skipping - need {len(perfect_mu)} sites, have {kin_space.n_sites}")

print("\n=== Sample States from Basis ===")
print("First 10 basis states:")
for i in range(min(10, kin_space.dim)):
    state = kin_space.basis_states[i]
    print(f"  {i:3d}: μ={state.mu_config}, ν={state.nu_config}")

# Recommendations
print("\n=== Recommendations ===")
print("To ensure target states exist:")
print("1. Verify μ and ν values are within [-μ_max, μ_max] and [-ν_max, ν_max]")
print("2. Increase basis_truncation if needed")
print("3. Use states shown above that exist in the basis")
print("4. For coherent states, use wider Gaussian widths if target state not in basis")
