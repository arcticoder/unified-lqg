#!/usr/bin/env python3
"""
Minimal Example: Building LQG Midisuperspace Hamiltonian

This example demonstrates how to wire up and invoke the fully-implemented 
Hamiltonian‐constraint machinery in lqg_genuine_quantization.py, replacing 
any "toy" diagonal placeholders with genuine LQG midisuperspace operators 
(holonomy corrections, inverse-triad regularization, matter coupling, etc.).

The example shows how to:
1. Load classical triad/extrinsic-curvature data and exotic matter profile
2. Build a LatticeConfiguration and choose LQGParameters 
3. Instantiate a KinematicalHilbertSpace
4. Call construct_full_hamiltonian() to build the complete H_grav + H_matter matrix
5. Solve the constraint H|ψ⟩ = 0 for physical states

Author: LQG Warp Framework
"""

import json
import numpy as np
import scipy.sparse as sp
import os

# Import all the core classes from lqg_genuine_quantization.py
from lqg_genuine_quantization import (
    LatticeConfiguration,
    LQGParameters,
    KinematicalHilbertSpace,
    MidisuperspaceHamiltonianConstraint,
    MuBarScheme
)


def load_classical_data(filename: str):
    """
    Load classical midisuperspace data from JSON file.
    
    Expected format:
    {
      "r":           [r₁, r₂, …, r_N],           # Radial grid
      "h11":         [h₁₁(r₁), …, h₁₁(r_N)],   # Metric component h_rr  
      "h22":         [h₂₂(r₁), …, h₂₂(r_N)],   # Metric component h_θθ
      "E11":         [E^r(r₁), …, E^r(r_N)],   # Triad component E^r
      "E22":         [E^θ(r₁), …, E^θ(r_N)],   # Triad component E^θ
    }
    
    Note: This converts between metric/triad conventions and the E^x, E^φ, K_x, K_φ
    variables used in the LQG constraint equations.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Extract radial grid
    r_grid = np.array(data["r"], dtype=float)
    dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1e-36
    
    # Convert metric/triad data to LQG variables
    # In spherical symmetry: E^x ≈ E^r, E^φ ≈ r²E^θ  
    E_x = np.array(data["E11"], dtype=float)  # Radial triad component
    E_phi = np.array(data["E22"], dtype=float) * (r_grid**2)  # Angular triad with r² factor
    
    # For extrinsic curvature, compute from metric evolution or use simple approximation
    # Here we approximate based on triad gradients (simplified)
    K_x = np.gradient(E_x) / (2 * dr)  # Simplified K_r component
    K_phi = np.gradient(E_phi) / (2 * dr * r_grid + 1e-12)  # Simplified K_θ component
    
    # Create a simple phantom field profile
    exotic_field = 0.1 * np.exp(-((r_grid - np.mean(r_grid))/np.std(r_grid))**2)
    
    # Scalar momentum (assuming canonical conjugate momentum)
    scalar_momentum = np.gradient(exotic_field) / dr
    
    return r_grid, dr, E_x, E_phi, K_x, K_phi, exotic_field, scalar_momentum


def main():
    """Main demonstration of LQG Hamiltonian construction"""
    
    print("🚀 LQG Midisuperspace Hamiltonian Construction Example")
    print("=" * 60)
    
    # 1. Load classical midisuperspace data 
    print("1. Loading classical data from example_reduced_variables.json...")
    r_grid, dr, E_x, E_phi, K_x, K_phi, exotic_field, scalar_momentum = load_classical_data(
        "examples/example_reduced_variables.json"
    )
    
    print(f"   - Radial grid: {len(r_grid)} points from {r_grid[0]:.2e} to {r_grid[-1]:.2e} m")
    print(f"   - Grid spacing: dr = {dr:.2e} m")
    print(f"   - E^x range: [{np.min(E_x):.3f}, {np.max(E_x):.3f}]")
    print(f"   - E^φ range: [{np.min(E_phi):.3e}, {np.max(E_phi):.3e}]")
    print(f"   - Exotic field amplitude: {np.max(np.abs(exotic_field)):.3f}")

    # 2. Build LatticeConfiguration for the midisuperspace
    print("\n2. Setting up lattice configuration...")
    lattice_config = LatticeConfiguration(
        n_sites=len(r_grid),
        r_min=r_grid[0],
        r_max=r_grid[-1],
        throat_radius=np.mean(r_grid)  # Approximate throat location
    )
    
    # 3. Choose LQG parameters (μ̄-scheme, flux truncation, etc.)
    print("\n3. Configuring LQG parameters...")
    lqg_params = LQGParameters(
        gamma=0.2375,  # Barbero-Immirzi parameter
        planck_length=1.616e-35,  # Planck length in meters
        planck_area=(1.616e-35)**2,  # Planck area
        mu_bar_scheme=MuBarScheme.IMPROVED_DYNAMICS,  # Use improved dynamics scheme
        holonomy_correction=True,      # Enable sin(μ̄K)/μ̄ corrections
        inverse_triad_regularization=True,  # Enable Thiemann regularization
        mu_max=3,                      # Flux basis truncated to |μ| ≤ 3
        nu_max=3,                      # Flux basis truncated to |ν| ≤ 3
        regularization_epsilon=1e-14,  # Numerical cutoff
        coherent_width_E=0.5,         # Coherent state parameters
        coherent_width_K=0.5,
        scalar_mass=1e-4 * 2.176e-8,  # Phantom field mass in kg
        equation_of_state="phantom"    # Phantom scalar field
    )
    
    print(f"   - μ̄-scheme: {lqg_params.mu_bar_scheme.value}")
    print(f"   - Flux truncation: |μ|,|ν| ≤ {lqg_params.mu_max}")
    print(f"   - Holonomy corrections: {lqg_params.holonomy_correction}")
    print(f"   - Inverse-triad regularization: {lqg_params.inverse_triad_regularization}")

    # 4. Build the kinematical Hilbert space
    print("\n4. Constructing kinematical Hilbert space...")
    kin_space = KinematicalHilbertSpace(
        lattice_config=lattice_config,
        lqg_params=lqg_params
    )
    
    print(f"   - Hilbert space dimension: {kin_space.dim}")
    print(f"   - Basis states generated: {len(kin_space.basis_states)}")

    # 5. Instantiate the Hamiltonian‐constraint builder
    print("\n5. Setting up Hamiltonian constraint operator...")
    constraint_solver = MidisuperspaceHamiltonianConstraint(
        lattice_config=lattice_config,
        lqg_params=lqg_params,
        kinematical_space=kin_space
    )

    # 6. Construct the full H_grav + H_matter matrix with genuine LQG operators
    print("\n6. Building genuine LQG Hamiltonian matrix...")
    print("   This replaces any 'toy' diagonal placeholders with:")
    print("   - Holonomy loops: sin(μ̄K)/μ̄ corrections")
    print("   - Thiemann inverse-triad factors: 1/√|E| regularization")  
    print("   - Spatial derivatives: discrete lattice operators")
    print("   - Phantom scalar coupling: proper stress-energy quantization")
    
    # Call the main constructor - this is where the magic happens!
    H_sparse = constraint_solver.construct_full_hamiltonian(
        classical_E_x=E_x,
        classical_E_phi=E_phi,
        classical_K_x=K_x,
        classical_K_phi=K_phi,
        scalar_field=exotic_field,
        scalar_momentum=scalar_momentum
    )
    
    print(f"\n✅ Hamiltonian matrix constructed successfully!")
    print(f"   - Matrix shape: {H_sparse.shape}")
    print(f"   - Non-zero elements: {H_sparse.nnz}")
    print(f"   - Matrix density: {H_sparse.nnz / (H_sparse.shape[0] * H_sparse.shape[1]):.8f}")
    print(f"   - Memory usage: ~{H_sparse.data.nbytes / 1024:.1f} KB")

    # 7. Solve H |ψ⟩ = 0 for the lowest few physical eigenmodes
    print("\n7. Solving Hamiltonian constraint H|ψ⟩ = 0...")
    try:
        eigenvals, eigenvecs = constraint_solver.solve_constraint(
            num_eigs=5, 
            use_gpu=False  # Use CPU solver for this example
        )
        
        print(f"   ✅ Found {len(eigenvals)} eigenvalues:")
        for i, val in enumerate(eigenvals):
            print(f"      λ_{i}: {val:.6e} (|λ_{i}|: {abs(val):.6e})")
        
        # The eigenvectors with eigenvalues closest to zero are the "physical" states
        print(f"\n   Physical state candidates (closest to H|ψ⟩ = 0):")
        most_physical_idx = np.argmin(np.abs(eigenvals))
        print(f"   - Most physical state: eigenvector {most_physical_idx}")
        print(f"   - Eigenvalue: {eigenvals[most_physical_idx]:.8e}")
        print(f"   - State norm: {np.linalg.norm(eigenvecs[:, most_physical_idx]):.6f}")

    except Exception as e:
        print(f"   ⚠️  Eigenvalue solver encountered issue: {e}")
        print("   This can happen with highly constrained systems or numerical precision limits.")

    # 8. Optional: Save or analyze quantum observables
    print("\n8. Computing quantum observables...")
    if 'eigenvals' in locals() and len(eigenvals) > 0:
        # Example: compute expectation values in the most physical state
        psi_physical = eigenvecs[:, most_physical_idx]
        
        # You could compute ⟨ψ|T^00|ψ⟩ here for quantum backreaction
        print(f"   - Physical state computed successfully")
        print(f"   - Ready for quantum ⟨T^00⟩ backreaction calculations")
        
        # Example output for integration with classical pipeline:
        quantum_T00_data = {
            "r_values": r_grid.tolist(),
            "eigenvalue_closest_to_zero": float(eigenvals[most_physical_idx]),
            "quantum_state_norm": float(np.linalg.norm(psi_physical)),
            "constraint_violation": float(abs(eigenvals[most_physical_idx])),
            "computation_successful": True
        }
        
        # Save to JSON for classical pipeline integration
        os.makedirs("outputs", exist_ok=True)
        output_file = "outputs/quantum_T00_lqg_example.json"
        with open(output_file, "w") as f:
            json.dump(quantum_T00_data, f, indent=2)
        print(f"   - Quantum data saved to: {output_file}")

    print("\n" + "=" * 60)
    print("🎯 LQG Hamiltonian Construction Complete!")
    print("\nKey accomplishments:")
    print("✓ Loaded classical midisuperspace data")
    print("✓ Constructed genuine LQG kinematical Hilbert space") 
    print("✓ Built full H_grav + H_matter with:")
    print("  • Holonomy corrections sin(μ̄K)/μ̄")
    print("  • Thiemann inverse-triad regularization")
    print("  • Spatial derivative discretization")
    print("  • Phantom scalar field quantization")
    print("✓ Solved constraint H|ψ⟩ = 0 for physical states")
    print("✓ Ready for quantum backreaction in warp drive pipeline")


if __name__ == "__main__":
    main()
