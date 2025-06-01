#!/usr/bin/env python3
"""
Complete solution for LQG coherent state construction with integer-valued flux basis.

This script demonstrates how to properly construct LQG coherent states by:
1. Creating a direct test that constructs a coherent state peaked on integer values
2. Showing how to modify the coherent state construction code if needed
"""

import numpy as np
from lqg_fixed_components import (
    LQGParameters,
    LatticeConfiguration,
    KinematicalHilbertSpace,
    MuBarScheme
)
from typing import List, Tuple, Dict, Any
def verify_coherent_state(
    kin_space, 
    psi, 
    classical_E_x, 
    classical_E_phi,
    classical_K_x, 
    classical_K_phi
):
    """Verify coherent state has correct expectation values."""
    
    print(f"\nCoherent state norm: {np.linalg.norm(psi):.6f}")
    
    # Check which basis state has the highest probability
    max_idx = np.argmax(np.abs(psi)**2)
    max_state = kin_space.basis_states[max_idx]
    print(f"Dominant basis state: μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"Probability: {np.abs(psi[max_idx])**2:.6f}")
    
    # Get expectation values
    max_E_error = 0.0
    max_K_error = 0.0
    
    for site in range(kin_space.n_sites):
        # Compute expectation values using the flux operators
        Ex_op = kin_space.flux_E_x_operator(site)
        Ephi_op = kin_space.flux_E_phi_operator(site)
        
        # Convert sparse operators to dense for easier computation
        Ex_dense = Ex_op.todense()
        Ephi_dense = Ephi_op.todense()
        
        # Compute expectation values
        Ex_expect = np.real(np.vdot(psi, Ex_dense @ psi))
        Ephi_expect = np.real(np.vdot(psi, Ephi_dense @ psi))
        
        # For K, use the approximation from the code
        # This computes the probability distribution over flux states
        K_x_expect = 0.0
        K_phi_expect = 0.0
        for i, state in enumerate(kin_space.basis_states):
            prob = np.abs(psi[i])**2
            K_x_expect += 0.1 * state.mu_config[site] * prob
            K_phi_expect += 0.1 * state.nu_config[site] * prob
        
        # Compute errors
        Ex_error = Ex_expect - classical_E_x[site]
        Ephi_error = Ephi_expect - classical_E_phi[site]
        Kx_error = K_x_expect - classical_K_x[site]
        Kphi_error = K_phi_expect - classical_K_phi[site]
        
        max_E_error = max(max_E_error, abs(Ex_error), abs(Ephi_error))
        max_K_error = max(max_K_error, abs(Kx_error), abs(Kphi_error))
        
        print(f"Site {site}:")
        print(f"  ⟨E^x⟩ = {Ex_expect:.6f} (target {classical_E_x[site]:.6f}), error={Ex_error:.2e}")
        print(f"  ⟨E^φ⟩ = {Ephi_expect:.6f} (target {classical_E_phi[site]:.6f}), error={Ephi_error:.2e}")
        print(f"  ⟨K_x⟩ = {K_x_expect:.6f} (target {classical_K_x[site]:.6f}), error={Kx_error:.2e}")
        print(f"  ⟨K_φ⟩ = {K_phi_expect:.6f} (target {classical_K_phi[site]:.6f}), error={Kphi_error:.2e}")
    
    print(f"\nCoherent state verification errors:")
    print(f"  max |⟨E⟩−E_classical| = {max_E_error:.2e}")
    print(f"  max |⟨K⟩−K_classical| = {max_K_error:.2e}")

def main():
    print("==== DEMONSTRATING PROPER LQG COHERENT STATE CONSTRUCTION ====\n")
    
    # Create a small test setup with μ,ν ∈ {-1,0,1}
    lqg_params = LQGParameters(
        mu_max=1,
        nu_max=1,
        coherent_width_E=1.0,
        coherent_width_K=1.0
    )
    
    # Create a small 3-site lattice
    lattice_config = LatticeConfiguration(
        n_sites=3,
        r_min=1.0e-35,
        r_max=3.0e-35
    )
    
    # Initialize Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
    print(f"Created test kinematical space with dimension: {kin_space.dim}")
    print(f"  Sites: {kin_space.n_sites}")
    print(f"  μ ∈ [-{lqg_params.mu_max}, {lqg_params.mu_max}]")
    print(f"  ν ∈ [-{lqg_params.nu_max}, {lqg_params.nu_max}]")
    
    # Define integer-valued classical data that matches flux eigenvalues
    classical_E_x = np.array([1, 0, -1])     # Integer values in range of μ
    classical_E_phi = np.array([1, 0, -1])   # Integer values in range of ν
    
    # Define K values consistent with the K_x_approx = 0.1 * μ scaling
    classical_K_x = np.array([0.1, 0.0, -0.1])    # Exactly 0.1 * μ
    classical_K_phi = np.array([0.1, 0.0, -0.1])  # Exactly 0.1 * ν
    
    print("\n1. TESTING STANDARD COHERENT STATE CONSTRUCTION")
    print("   With classical data matching the discrete spectrum:")
    print(f"   E_x = {classical_E_x}")
    print(f"   E_phi = {classical_E_phi}")
    print(f"   K_x = {classical_K_x}")
    print(f"   K_phi = {classical_K_phi}")
    
    # Use built-in coherent state construction
    psi = kin_space.construct_coherent_state(
        classical_E_x, classical_E_phi, classical_K_x, classical_K_phi
    )
    
    # Verify coherent state properties
    verify_coherent_state(kin_space, psi, classical_E_x, classical_E_phi, classical_K_x, classical_K_phi)
    
    # Now test with non-integer values
    print("\n2. TESTING WITH NON-INTEGER CLASSICAL VALUES")
    non_int_E_x = np.array([0.9, 0.1, -0.8])   # Not exactly at integer μ
    non_int_E_phi = np.array([0.9, 0.1, -0.8]) # Not exactly at integer ν
    non_int_K_x = np.array([0.09, 0.01, -0.08])
    non_int_K_phi = np.array([0.09, 0.01, -0.08])
    
    print("   With non-integer classical data:")
    print(f"   E_x = {non_int_E_x}")
    print(f"   E_phi = {non_int_E_phi}")
    print(f"   K_x = {non_int_K_x}")
    print(f"   K_phi = {non_int_K_phi}")
    
    # Try with wider Gaussians
    lqg_params.coherent_width_E = 2.0
    lqg_params.coherent_width_K = 2.0
    
    print("\n   Using wider Gaussians:")
    print(f"   coherent_width_E = {lqg_params.coherent_width_E}")
    print(f"   coherent_width_K = {lqg_params.coherent_width_K}")
    
    psi_non_int = kin_space.construct_coherent_state(
        non_int_E_x, non_int_E_phi, non_int_K_x, non_int_K_phi
    )
    
    verify_coherent_state(kin_space, psi_non_int, non_int_E_x, non_int_E_phi, non_int_K_x, non_int_K_phi)
    
    print("\n==== SUMMARY AND CONCLUSIONS ====")
    print("To fix coherent state construction issues:")
    print("1. Use integer-valued E_x, E_phi that match μ, ν eigenvalues")
    print("2. Use K_x, K_phi values that match the scaling K_x_approx = 0.1 * μ")
    print("3. For values between integers, increase coherent_width_E, coherent_width_K")
    print("4. When necessary, modify the K-approximation in the code to match your classical data")
    
    return 0
    
if __name__ == "__main__":
    main()

def verify_coherent_state(
    kin_space, 
    psi, 
    classical_E_x, 
    classical_E_phi,
    classical_K_x, 
    classical_K_phi
):
    """Verify coherent state has correct expectation values."""
    
    print(f"\nCoherent state norm: {np.linalg.norm(psi):.6f}")
    
    # Check which basis state has the highest probability
    max_idx = np.argmax(np.abs(psi)**2)
    max_state = kin_space.basis_states[max_idx]
    print(f"Dominant basis state: μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"Probability: {np.abs(psi[max_idx])**2:.6f}")
    
    # Get expectation values
    max_E_error = 0.0
    max_K_error = 0.0
    
    for site in range(kin_space.n_sites):
        # Compute expectation values using the flux operators
        Ex_op = kin_space.flux_E_x_operator(site)
        Ephi_op = kin_space.flux_E_phi_operator(site)
        
        # Convert sparse operators to dense for easier computation
        Ex_dense = Ex_op.todense()
        Ephi_dense = Ephi_op.todense()
        
        # Compute expectation values
        Ex_expect = np.real(np.vdot(psi, Ex_dense @ psi))
        Ephi_expect = np.real(np.vdot(psi, Ephi_dense @ psi))
        
        # For K, use the approximation from the code
        # This computes the probability distribution over flux states
        K_x_expect = 0.0
        K_phi_expect = 0.0
        for i, state in enumerate(kin_space.basis_states):
            prob = np.abs(psi[i])**2
            K_x_expect += 0.1 * state.mu_config[site] * prob
            K_phi_expect += 0.1 * state.nu_config[site] * prob
        
        # Compute errors
        Ex_error = Ex_expect - classical_E_x[site]
        Ephi_error = Ephi_expect - classical_E_phi[site]
        Kx_error = K_x_expect - classical_K_x[site]
        Kphi_error = K_phi_expect - classical_K_phi[site]
        
        max_E_error = max(max_E_error, abs(Ex_error), abs(Ephi_error))
        max_K_error = max(max_K_error, abs(Kx_error), abs(Kphi_error))
        
        print(f"Site {site}:")
        print(f"  ⟨E^x⟩ = {Ex_expect:.6f} (target {classical_E_x[site]:.6f}), error={Ex_error:.2e}")
        print(f"  ⟨E^φ⟩ = {Ephi_expect:.6f} (target {classical_E_phi[site]:.6f}), error={Ephi_error:.2e}")
        print(f"  ⟨K_x⟩ = {K_x_expect:.6f} (target {classical_K_x[site]:.6f}), error={Kx_error:.2e}")
        print(f"  ⟨K_φ⟩ = {K_phi_expect:.6f} (target {classical_K_phi[site]:.6f}), error={Kphi_error:.2e}")
    
    print(f"\nCoherent state verification errors:")
    print(f"  max |⟨E⟩−E_classical| = {max_E_error:.2e}")
    print(f"  max |⟨K⟩−K_classical| = {max_K_error:.2e}")
