#!/usr/bin/env python3
"""
Testing coherent state construction with wider Gaussian widths
to address the normalization issue.
"""

import numpy as np
from lqg_fixed_components import (
    LQGParameters,
    LatticeConfiguration,
    KinematicalHilbertSpace,
    MuBarScheme
)

# === Setup with integer-valued flux basis and WIDER GAUSSIANS ===

# 1. Define a simple 3-site lattice
n_sites = 3

# 2. Choose classical values that exactly match integer μ and ν
classical_E_x = np.array([2, 0, -2])      # Peaks at μ = 2, 0, -2
classical_E_phi = np.array([1, 0, -1])    # Peaks at ν = 1, 0, -1

# 3. Choose K values compatible with μ, ν via K_x_approx = 0.1 * μ
classical_K_x = np.array([0.2, 0.0, -0.2])  # = 0.1 * classical_E_x / 1.0
classical_K_phi = np.array([0.1, 0.0, -0.1])  # = 0.1 * classical_E_phi / 1.0

# 4. Create LQG parameters with MUCH WIDER Gaussians
lqg_params = LQGParameters(
    gamma=1.0,              # Simplified values for the demo
    planck_length=1.0,      # Using natural units
    planck_area=1.0,
    mu_bar_scheme=MuBarScheme.MINIMAL_AREA,
    holonomy_correction=True,
    inverse_triad_regularization=True,
    mu_max=2,
    nu_max=2,
    coherent_width_E=3.0,   # MUCH WIDER than default
    coherent_width_K=3.0    # MUCH WIDER than default
)

# 5. Create lattice configuration
lattice_config = LatticeConfiguration(
    n_sites=n_sites,
    r_min=1.0,      # Dummy values for the demo
    r_max=3.0
)
setattr(lattice_config, "E_x_classical", list(classical_E_x))
setattr(lattice_config, "E_phi_classical", list(classical_E_phi))

# 6. Instantiate the Hilbert space
print("Creating kinematical Hilbert space...")
kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
print(f"Hilbert-space dimension: {kin_space.dim}")

# 7. Generate and check a coherent state
print("\nGenerating coherent state with WIDER GAUSSIANS (width=3.0)...")
psi_coh, errors = kin_space.create_coherent_state_with_Kcheck(
    E_x_target=classical_E_x,
    E_phi_target=classical_E_phi,
    K_x_target=classical_K_x,
    K_phi_target=classical_K_phi,
    width=3.0  # Use wider width
)

# 8. Show results
print("\nSummary of deviations:")
print(f"  max |⟨E⟩−E_classical| = {errors['max_E_error']:.2e}")
print(f"  max |⟨K⟩−K_classical| = {errors['max_K_error']:.2e}")

# 9. Verify dominant basis state coefficient
print("\nAnalyzing coherent state vector:")
max_coef_idx = np.argmax(np.abs(psi_coh))
max_coef = psi_coh[max_coef_idx]
max_state = kin_space.basis_states[max_coef_idx]

print(f"Dominant basis state: {max_state}")
print(f"Coefficient magnitude: {np.abs(max_coef):.6f}")

# 10. Now try with the intended target basis state
target_mu = [2, 0, -2]
target_nu = [1, 0, -1]

# Find the basis state that matches our target
target_state_idx = None
for i, state in enumerate(kin_space.basis_states):
    if (np.array_equal(state.mu_config, target_mu) and 
        np.array_equal(state.nu_config, target_nu)):
        target_state_idx = i
        break

if target_state_idx is not None:
    target_coef = psi_coh[target_state_idx]
    target_state = kin_space.basis_states[target_state_idx]
    print(f"\nTarget basis state: {target_state}")
    print(f"Target coefficient magnitude: {np.abs(target_coef):.6f}")
    print(f"Target coefficient rank: {sorted(np.abs(psi_coh), reverse=True).index(np.abs(target_coef)) + 1} of {len(psi_coh)}")
else:
    print("\nTarget state not found in basis")
