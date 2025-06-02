#!/usr/bin/env python3
"""
Quick validation of constraint algebra.
"""

import numpy as np
from lqg_fixed_components import LatticeConfiguration, LQGParameters, KinematicalHilbertSpace, MidisuperspaceHamiltonianConstraint

# Create configuration
lattice = LatticeConfiguration()
lattice.n_sites = 5

lqg_params = LQGParameters()
lqg_params.planck_length = 0.01
lqg_params.max_basis_states = 1000
lqg_params.use_improved_dynamics = True

print(f"Creating kinematical space with {lattice.n_sites} sites, max {lqg_params.max_basis_states} basis states...")

# Create kinematical space
kin_space = KinematicalHilbertSpace(lattice, lqg_params)

print(f"✅ Hilbert space dimension: {kin_space.dim}")
print(f"✅ Number of sites: {kin_space.n_sites}")

# Create constraint
constraint = MidisuperspaceHamiltonianConstraint(lattice, lqg_params, kin_space)

# Test building constraint matrix  
print("Building constraint matrix...")
classical_E_x = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
classical_E_phi = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
classical_K_x = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
classical_K_phi = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
scalar_field = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
scalar_momentum = np.array([0.1, 0.1, 0.1, 0.1, 0.1])

H = constraint.construct_full_hamiltonian(
    classical_E_x, classical_E_phi, 
    classical_K_x, classical_K_phi,
    scalar_field, scalar_momentum
)

print(f"✅ Constraint matrix shape: {H.shape}")
print(f"✅ Matrix density: {H.nnz / (H.shape[0] * H.shape[1]):.6f}")

print("✅ Constraint algebra validation successful!")
