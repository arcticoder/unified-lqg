#!/usr/bin/env python3
"""
Simple demonstration of coherent state construction with integer values.
"""

import numpy as np
from lqg_fixed_components import (
    LQGParameters,
    LatticeConfiguration,
    KinematicalHilbertSpace,
    MuBarScheme
)

# Define helper function to verify coherent state properties
def verify_expectation_values(kin_space, psi, classical_E_x, classical_E_phi):
    """Compute and display expectation values <E^x>, <E^phi> of coherent state."""
    
    print(f"Coherent state norm: {np.linalg.norm(psi):.6f}")
    
    for site in range(kin_space.n_sites):
        # Compute expectation values of flux operators
        Ex_op = kin_space.flux_E_x_operator(site)
        Ephi_op = kin_space.flux_E_phi_operator(site)
        
        # Dense matrices for simple computation
        Ex_dense = Ex_op.todense()
        Ephi_dense = Ephi_op.todense()
        
        # Expectation values
        Ex_expect = np.real(np.vdot(psi, Ex_dense @ psi))
        Ephi_expect = np.real(np.vdot(psi, Ephi_dense @ psi))
        
        # Errors
        Ex_error = Ex_expect - classical_E_x[site]
        Ephi_error = Ephi_expect - classical_E_phi[site]
        
        print(f"Site {site}:")
        print(f"  ⟨E^x⟩ = {Ex_expect:.6f} (target {classical_E_x[site]:.6f}), error={Ex_error:.2e}")
        print(f"  ⟨E^φ⟩ = {Ephi_expect:.6f} (target {classical_E_phi[site]:.6f}), error={Ephi_error:.2e}")
    
    # Find dominant basis state
    max_idx = np.argmax(np.abs(psi)**2)
    max_state = kin_space.basis_states[max_idx]
    print(f"\nDominant basis state: μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"Coefficient magnitude: {np.abs(psi[max_idx]):.6f}")

def main():
    print("=== DEMONSTRATING COHERENT STATE CONSTRUCTION WITH INTEGER VALUES ===\n")
    
    # Create parameters with suitable coherent state width
    params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=100,
        coherent_width_E=1.0,
        coherent_width_K=1.0
    )
    
    # Create a small 3-site lattice for efficiency
    lattice_config = LatticeConfiguration(
        n_sites=3,
        r_min=1.0e-35,
        r_max=3.0e-35
    )
    
    # Initialize Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, params)
    print(f"Created kinematical space with {kin_space.dim} basis states")
    print(f"μ ∈ [-{params.mu_max}, {params.mu_max}], ν ∈ [-{params.nu_max}, {params.nu_max}]\n")
    
    # Test 1: Integer values matching flux eigenvalues
    print("TEST 1: INTEGER VALUES")
    classical_E_x = np.array([2, 1, 0])     # Integers matching μ eigenvalues
    classical_E_phi = np.array([1, 0, -1])  # Integers matching ν eigenvalues
    classical_K_x = np.array([0.2, 0.1, 0.0])   # 0.1 * μ
    classical_K_phi = np.array([0.1, 0.0, -0.1]) # 0.1 * ν
    
    print(f"Using values:")
    print(f"E_x = {classical_E_x} (integers matching μ eigenvalues)")
    print(f"E_phi = {classical_E_phi} (integers matching ν eigenvalues)")
    print(f"K_x = {classical_K_x} (matching K_x_approx = 0.1*μ scaling)")
    print(f"K_phi = {classical_K_phi} (matching K_phi_approx = 0.1*ν scaling)\n")
    
    # Create coherent state
    psi_int = kin_space.construct_coherent_state(
        classical_E_x, classical_E_phi, classical_K_x, classical_K_phi
    )
    verify_expectation_values(kin_space, psi_int, classical_E_x, classical_E_phi)
    
    # Test 2: Non-integer values with wide Gaussians
    print("\n\nTEST 2: NON-INTEGER VALUES WITH WIDER GAUSSIANS")
    classical_E_x2 = np.array([1.2, 0.7, -0.3])     # Non-integers
    classical_E_phi2 = np.array([0.8, -0.1, -1.7])  # Non-integers
    classical_K_x2 = np.array([0.12, 0.07, -0.03])  # 0.1 * E_x2
    classical_K_phi2 = np.array([0.08, -0.01, -0.17]) # 0.1 * E_phi2
    
    # Increase width for non-integer case
    params.coherent_width_E = 2.0
    params.coherent_width_K = 2.0
    
    print(f"Using non-integer values:")
    print(f"E_x = {classical_E_x2}")
    print(f"E_phi = {classical_E_phi2}")
    print(f"K_x = {classical_K_x2}")
    print(f"K_phi = {classical_K_phi2}")
    print(f"with coherent_width_E = {params.coherent_width_E}, coherent_width_K = {params.coherent_width_K}\n")
    
    # Create coherent state with wider Gaussians
    psi_non_int = kin_space.construct_coherent_state(
        classical_E_x2, classical_E_phi2, classical_K_x2, classical_K_phi2
    )
    verify_expectation_values(kin_space, psi_non_int, classical_E_x2, classical_E_phi2)
    
    # Summary
    print("\n=== SUMMARY ===")
    print("For successful coherent state construction:")
    print("1. Use integer E_x, E_phi values matching the μ, ν eigenvalues")
    print("2. Use K_x, K_phi values that match the K_x_approx = 0.1 * μ scaling")
    print("3. For non-integer classical values, increase coherent_width_E and coherent_width_K")
    print("4. To fix ComplexWarning, use float(np.abs(eigenvals[0])) or float(np.real(eigenvals[0]))")
    
if __name__ == "__main__":
    main()
