#!/usr/bin/env python3
"""
Modified coherent state test that works with a state actually present in the basis.
"""

import numpy as np
import time
from lqg_fixed_components import (
    LQGParameters,
    LatticeConfiguration,
    KinematicalHilbertSpace,
    MuBarScheme
)

# 1. Create LQG parameters with wider widths
lqg_params = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=1000,
    coherent_width_E=3.0,   # Wider width
    coherent_width_K=3.0    # Wider width
)

# 2. Create lattice configuration
lattice_config = LatticeConfiguration(
    n_sites=3,
    r_min=1.0,
    r_max=3.0
)

# 3. Create kinematical Hilbert space
print("Creating kinematical Hilbert space...")
kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
print(f"Hilbert space dimension: {kin_space.dim}")

# 4. Find a basis state near the middle of the spectrum that's actually in the basis
print("\nFinding a suitable basis state to target...")
target_idx = kin_space.dim // 2  # Start near the middle
target_state = kin_space.basis_states[target_idx]
print(f"Selected target state: {target_state}")

# 5. Extract μ and ν configurations from the basis state
target_mu = target_state.mu_config
target_nu = target_state.nu_config

# 6. Create classical E and K values that match this basis state
# E_x should correspond to μ, E_phi to ν
# Use the same convention as in construct_coherent_state: K_x ≈ 0.1 * μ
classical_E_x = target_mu.astype(float)
classical_E_phi = target_nu.astype(float)
classical_K_x = 0.1 * target_mu
classical_K_phi = 0.1 * target_nu

print(f"Target μ: {target_mu}")
print(f"Target ν: {target_nu}")
print(f"Classical E_x: {classical_E_x}")
print(f"Classical E_phi: {classical_E_phi}")
print(f"Classical K_x: {classical_K_x}")
print(f"Classical K_phi: {classical_K_phi}")

# 7. Generate coherent state using these values
print("\nGenerating coherent state...")
start_time = time.time()
psi_coh, errors = kin_space.create_coherent_state_with_Kcheck(
    E_x_target=classical_E_x,
    E_phi_target=classical_E_phi,
    K_x_target=classical_K_x,
    K_phi_target=classical_K_phi,
    width=3.0  # Wide width helps ensure overlap
)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

# 8. Show results
print("\nSummary of deviations:")
print(f"  max |⟨E⟩−E_classical| = {errors['max_E_error']:.2e}")
print(f"  max |⟨K⟩−K_classical| = {errors['max_K_error']:.2e}")

# 9. Check if our target state has the highest coefficient
max_coef_idx = np.argmax(np.abs(psi_coh))
max_coef = psi_coh[max_coef_idx]
max_state = kin_space.basis_states[max_coef_idx]

target_coef = psi_coh[target_idx]

print("\nDominant basis state analysis:")
print(f"Target state index: {target_idx}")
print(f"Target coefficient: {np.abs(target_coef):.6f}")

print(f"Max coefficient state: {max_state}")
print(f"Max coefficient: {np.abs(max_coef):.6f}")

# 10. Check distribution of coefficients
sorted_indices = np.argsort(np.abs(psi_coh))[::-1]  # Descending order
top_n = 5
print(f"\nTop {top_n} states by coefficient magnitude:")
for i in range(top_n):
    idx = sorted_indices[i]
    state = kin_space.basis_states[idx]
    coef = psi_coh[idx]
    print(f"{i+1}: State {state}, coefficient: {np.abs(coef):.6f}")

# Verify the normalization
print(f"\nState vector norm: {np.linalg.norm(psi_coh):.10f}")
