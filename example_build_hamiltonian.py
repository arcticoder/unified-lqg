# example_build_hamiltonian.py

import json
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg

# (A) import all the core classes from lqg_genuine_quantization.py
from lqg_genuine_quantization import (
    LatticeConfiguration,
    LQGParameters,
    KinematicalHilbertSpace,
    MidisuperspaceHamiltonianConstraint,
    MuBarScheme
)


def load_classical_data(filename: str):
    """
    Load classical data and convert to LQG format.
    
    The actual JSON format has:
    {
      "r": [r₁, r₂, …, r_N],
      "h11": [h₁₁(r₁), …, h₁₁(r_N)],  # metric component
      "h22": [h₂₂(r₁), …, h₂₂(r_N)],  # metric component
      "E11": [E¹₁(r₁), …, E¹₁(r_N)],  # triad component
      "E22": [E²₂(r₁), …, E²₂(r_N)],  # triad component
      ...
    }
    
    We'll map this to LQG midisuperspace variables:
    - E_x ← sqrt(E11) (radial triad)
    - E_phi ← sqrt(E22) (angular triad)  
    - K_x ← derived from h11 evolution
    - K_phi ← derived from h22 evolution
    - exotic ← phantom field (synthesized)
    """
    with open(filename, "r") as f:
        data = json.load(f)

    # Convert to numpy arrays and map to LQG variables
    r_grid = np.array(data["r"], dtype=float)
    dr = r_grid[1] - r_grid[0] if len(r_grid) > 1 else 1e-36
    
    # Map triad components (take square root to get triad from densitized triad)
    E_x = np.sqrt(np.abs(np.array(data["E11"], dtype=float)))
    E_phi = np.sqrt(np.abs(np.array(data["E22"], dtype=float)))
    
    # Synthesize extrinsic curvature from metric evolution
    # K ~ ∂h/∂t, but we don't have time evolution, so use spatial derivatives as proxy
    h11 = np.array(data["h11"], dtype=float)
    h22 = np.array(data["h22"], dtype=float)
    
    # Simple finite difference approximation for K
    K_x = np.zeros_like(r_grid)
    K_phi = np.zeros_like(r_grid)
    
    for i in range(len(r_grid)):
        if i > 0 and i < len(r_grid) - 1:
            # Use centered difference
            dh11_dr = (h11[i+1] - h11[i-1]) / (2 * dr)
            dh22_dr = (h22[i+1] - h22[i-1]) / (2 * dr)
            K_x[i] = 0.1 * dh11_dr  # scaled approximation
            K_phi[i] = 0.1 * dh22_dr
        elif i == 0:
            # Forward difference
            dh11_dr = (h11[i+1] - h11[i]) / dr
            dh22_dr = (h22[i+1] - h22[i]) / dr
            K_x[i] = 0.1 * dh11_dr
            K_phi[i] = 0.1 * dh22_dr
        else:
            # Backward difference
            dh11_dr = (h11[i] - h11[i-1]) / dr
            dh22_dr = (h22[i] - h22[i-1]) / dr
            K_x[i] = 0.1 * dh11_dr
            K_phi[i] = 0.1 * dh22_dr
    
    # Synthesize phantom scalar field
    # Create a profile concentrated near the throat
    throat_radius = data.get("throat_radius", r_grid[len(r_grid)//2])
    phantom_amplitude = 1.0
    phantom_width = 2 * dr
    
    exotic_field = phantom_amplitude * np.exp(-((r_grid - throat_radius) / phantom_width)**2)
    
    print(f"Loaded and converted classical data from {filename}:")
    print(f"  Grid: {len(r_grid)} sites, r ∈ [{r_grid[0]:.2e}, {r_grid[-1]:.2e}]")
    print(f"  dr = {dr:.2e}")
    print(f"  E^x range: [{np.min(E_x):.3f}, {np.max(E_x):.3f}]")
    print(f"  E^φ range: [{np.min(E_phi):.3f}, {np.max(E_phi):.3f}]")
    print(f"  K_x range: [{np.min(K_x):.3f}, {np.max(K_x):.3f}]")
    print(f"  K_φ range: [{np.min(K_phi):.3f}, {np.max(K_phi):.3f}]")
    print(f"  Phantom field range: [{np.min(exotic_field):.3f}, {np.max(exotic_field):.3f}]")

    return r_grid, dr, E_x, E_phi, K_x, K_phi, exotic_field


def main():
    # 1. Load classical midisuperspace data (example_reduced_variables.json)
    r_grid, dr, E_x, E_phi, K_x, K_phi, exotic_field = load_classical_data(
        "examples/example_reduced_variables.json"    )    # 2. Build a LatticeConfiguration for the midisuperspace (REDUCED FOR DEMO):
    # Reduce to 3 sites instead of full grid for manageable computation
    n_demo_sites = 3
    lattice_config = LatticeConfiguration(
        n_sites=n_demo_sites,
        r_min=float(r_grid[0]),
        r_max=float(r_grid[-1]),
        throat_radius=float(r_grid[len(r_grid)//2])  # Use middle point as throat
    )

    # Truncate classical data to match demo size
    indices = np.linspace(0, len(r_grid)-1, n_demo_sites, dtype=int)
    E_x = E_x[indices]
    E_phi = E_phi[indices]
    K_x = K_x[indices]
    K_phi = K_phi[indices]
    exotic_field = exotic_field[indices]
    
    print(f"Reduced to {n_demo_sites} sites for demo: r ∈ [{r_grid[indices[0]]:.2e}, {r_grid[indices[-1]]:.2e}]")

    # 3. Pick your LQG parameters (μ̄-scheme, flux truncation, etc.):
    lqg_params = LQGParameters(
        gamma             = 1.0,
        planck_length     = 1.0,   # using natural units c = G = ℏ = 1,
        planck_area       = 1.0,   # so that μ̄ ∼ √|E| is dimensionless,
        mu_bar_scheme     = MuBarScheme.IMPROVED_DYNAMICS,
        holonomy_correction       = True,
        inverse_triad_regularization = True,
        mu_max            = 3,     # flux-basis truncated to |μ|,|ν| ≤ 3
        nu_max            = 3,
        basis_truncation  = 1000,  # (optional)
        coherent_width_E  = 0.5,
        coherent_width_K  = 0.5,
        scalar_mass       = 1e-4,  # phantom field mass (Planck units),
        equation_of_state = "phantom"
    )

    # 4. Build the kinematical Hilbert space:
    kin_space = KinematicalHilbertSpace(
        lattice_config = lattice_config,
        lqg_params     = lqg_params
    )

    # 5. Instantiate the Hamiltonian‐constraint builder:
    constraint_solver = MidisuperspaceHamiltonianConstraint(
        lattice_config = lattice_config,
        lqg_params     = lqg_params,
        kinematical_space = kin_space
    )    # 6. Construct the full H_grav + H_matter matrix:
    # Need to add scalar momentum (conjugate to scalar field)
    scalar_momentum = np.zeros_like(exotic_field)  # Start with zero momentum
    
    H_sparse: sp.csr_matrix = constraint_solver.construct_full_hamiltonian(
        classical_E_x     = E_x,
        classical_E_phi   = E_phi,
        classical_K_x     = K_x,
        classical_K_phi   = K_phi,
        scalar_field      = exotic_field,
        scalar_momentum   = scalar_momentum
    )
    # Now H_sparse is a (dim × dim) sparse matrix whose nonzero structure
    # comes from genuine holonomy loops, Thiemann inverse-triad terms,
    # spatial derivative couplings, and real phantom‐scalar couplings.

    print(f"Hamiltonian matrix constructed: shape = {H_sparse.shape}, nnz = {H_sparse.nnz}")

    # 7. (Optionally) solve H |ψ⟩ = 0 for the lowest few eigenmodes:
    eigenvals, eigenvecs = constraint_solver.solve_constraint(num_eigs=5, use_gpu=False)
    print("Lowest five eigenvalues (closest to zero):", eigenvals)

    # 8. You can now save or analyze those "physical" states ψ_i(r) further.
    # For example, compute ⟨T^00(r)⟩ from eigenvecs etc., or feed back into your warp‐pipeline.

    # If you want to export the quantum T^00 data in JSON for the classical pipeline:
    #    backreaction_data = {
    #       "r_values":       list(r_grid),
    #       "quantum_T00":    list( computed_T00_array ),
    #       "total_mass_energy": ...,
    #       "peak_energy_density": ...,
    #       "peak_location": ...,
    #    }
    #    constraint_solver.save_quantum_T00_for_pipeline(backreaction_data,
    #                                                   "quantum_inputs/T00_quantum_refined.json")

    # 5. DEMONSTRATE NEW CONSTRAINT FEATURES
    print("\n" + "="*60)
    print("5. CONSTRAINT VERIFICATION DEMONSTRATION")
    print("="*60)
    
    # 5a. Verify Gauss constraint in spherical symmetry
    print("\n5a. Gauss Constraint Verification:")
    gauss_results = constraint_solver.verify_gauss_constraint()
    for key, value in gauss_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2e}")
        else:
            print(f"  {key}: {value}")
    
    # 5b. Test diffeomorphism constraint implementations
    print("\n5b. Diffeomorphism Constraint Construction:")
    
    print("  Testing gauge-fixing approach...")
    C_diffeo_gauge = constraint_solver.construct_diffeomorphism_constraint(gauge_fixing=True)
    print(f"    Gauge-fixing matrix: {C_diffeo_gauge.nnz} non-zero elements")
    
    print("  Testing discrete operator approach...")
    C_diffeo_discrete = constraint_solver.construct_diffeomorphism_constraint(gauge_fixing=False)
    print(f"    Discrete operator matrix: {C_diffeo_discrete.nnz} non-zero elements")
    
    # 5c. Complete constraint algebra verification
    print("\n5c. Constraint Algebra Verification:")
    constraint_solver.C_diffeo_matrix = C_diffeo_gauge  # Use gauge-fixing for verification
    algebra_results = constraint_solver.verify_constraint_algebra()
    
    print("  Constraint algebra results:")
    for key, value in algebra_results.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.2e}")
        else:
            print(f"    {key}: {value}")
    
    # Summary of constraint implementation
    print(f"\n📋 CONSTRAINT IMPLEMENTATION SUMMARY:")
    print(f"  ✅ Gauss constraint: {'SATISFIED' if gauss_results.get('gauss_constraint_satisfied', False) else 'ISSUES'}")
    print(f"  ✅ Diffeomorphism constraint: IMPLEMENTED (2 approaches)")
    print(f"  ✅ Constraint algebra: {'SATISFIED' if algebra_results.get('constraint_algebra_satisfied', False) else 'ISSUES'}")


if __name__ == "__main__":
    main()
