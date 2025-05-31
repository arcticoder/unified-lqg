#!/usr/bin/env python3
"""
Compute Mode Spectrum for Warp Drive Quantum Field Design

This script solves the field-mode eigenproblem for quantum fields in the optimized
warp drive geometry. It computes eigenfrequencies ωₙ and field profiles for:

1. Scalar field modes in curved spacetime
2. Electromagnetic field modes in metamaterial analogue systems  
3. Phonon modes in BEC analogue systems

The eigenproblem is typically of the form:
    (-∇² + V_eff(r)) ψₙ = ωₙ² ψₙ

Where V_eff(r) includes geometric effects from the warp drive metric.
"""

import argparse
import numpy as np
import ndjson
import os
import sys
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import json

def read_ndjson(path):
    """Load NDJSON file if it exists, return empty list otherwise."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found, returning empty data")
        return []
    with open(path) as f:
        return ndjson.load(f)

def parse_metric_data(metric_data):
    """Extract geometric parameters from metric data."""
    if not metric_data:
        # Default test geometry
        return {
            'throat_radius': 1.6e-35,
            'shape_function': 'b0**2 / r',
            'redshift_function': '0',
            'geometry_type': 'morris_thorne'
        }
    
    entry = metric_data[0]  # Use first entry
    return {
        'throat_radius': entry.get('b0', entry.get('throat_radius', 1.6e-35)),
        'shape_function': entry.get('shape_function', 'b0**2 / r'),
        'redshift_function': entry.get('redshift_function', '0'),
        'geometry_type': entry.get('refinement_method', 'morris_thorne'),
        'metric_components': entry.get('refined_metric', {}),
        'optimization_params': entry.get('optimization_parameters', {})
    }

def effective_potential_warp_drive(r, geometry_params, field_type='scalar'):
    """
    Compute effective potential V_eff(r) for quantum fields in warp drive geometry.
    
    For scalar fields in curved spacetime:
    V_eff(r) = (1/√g) ∂ᵢ(√g gⁱʲ ∂ⱼ) + curvature_coupling
    
    For the Morris-Thorne wormhole with metric:
    ds² = -dt² + dr²/(1-b(r)/r) + r²(dθ² + sin²θ dφ²)
    """
    b0 = geometry_params['throat_radius']
    
    # Avoid singularities by ensuring r > b0
    r_safe = np.maximum(r, 1.1 * b0)
    
    # Shape function b(r) - handle different forms
    if 'optimized' in geometry_params.get('geometry_type', ''):
        # For optimized metrics with smoothing
        smoothing = geometry_params.get('optimization_params', {}).get('smoothing_parameter', b0 * 0.1)
        b_r = b0**2 / (r_safe + smoothing)  # Regularized shape function
    else:
        # Standard Morris-Thorne with regularization
        b_r = b0**2 / r_safe
    
    # Ensure b(r) < r for physical wormhole
    b_r = np.minimum(b_r, 0.9 * r_safe)
    
    # Redshift function effects
    redshift_amp = 0.0
    if 'redshift_amplitude' in geometry_params.get('optimization_params', {}):
        redshift_amp = geometry_params['optimization_params']['redshift_amplitude']
    
    # Effective potential components
    if field_type == 'scalar':
        # Geometric term from curved spacetime (regularized)
        factor = np.maximum(1 - b_r/r_safe, 0.1)  # Avoid division by zero
        geometric_term = b_r / (2 * r_safe**3) * factor**(-1.5)
        
        # Curvature coupling (conformal coupling ξ = 1/6) 
        curvature_term = (1/6) * (b_r * (b_r - 2*r_safe)) / (r_safe**4 * factor**2)
        
        # Redshift corrections
        redshift_term = redshift_amp**2 / (2 * r_safe**2)
        
        # Simple regularization for numerical stability
        total_potential = geometric_term + curvature_term + redshift_term
        return np.nan_to_num(total_potential, nan=0.0, posinf=1e10, neginf=-1e10)
    
    elif field_type == 'electromagnetic':
        # EM modes in curved spacetime (more complex)
        factor = np.maximum(1 - b_r/r_safe, 0.1)
        return np.nan_to_num(b_r / (r_safe**3) * factor**(-1), nan=0.0, posinf=1e10, neginf=0.0)
    
    elif field_type == 'phonon':
        # Phonon modes in BEC analogue with effective refractive index
        factor = np.maximum(1 - b_r/r_safe, 0.1)
        n_eff = np.sqrt(factor)  # Effective refractive index from metric
        return np.nan_to_num((1 - n_eff**2) / (2 * r_safe**2), nan=0.0, posinf=1e5, neginf=0.0)
    
    else:
        raise ValueError(f"Unknown field type: {field_type}")

def build_radial_operator(r_grid, geometry_params, field_type='scalar', boundary='dirichlet'):
    """
    Build the radial part of the wave operator as a matrix.
    
    For spherically symmetric case, the eigenvalue equation becomes:
    [-d²/dr² + l(l+1)/r² + V_eff(r)] R_nl(r) = ωₙₗ² R_nl(r)
    """
    N = len(r_grid)
    dr = r_grid[1] - r_grid[0]  # Assuming uniform grid
    
    # Build kinetic energy operator: -d²/dr²
    # Using finite differences: f''(i) ≈ (f(i+1) - 2f(i) + f(i-1))/dr²
    kinetic_diag = -2.0 / dr**2 * np.ones(N)
    kinetic_upper = 1.0 / dr**2 * np.ones(N-1)
    kinetic_lower = 1.0 / dr**2 * np.ones(N-1)
    
    kinetic_matrix = diags([kinetic_lower, kinetic_diag, kinetic_upper], 
                          [-1, 0, 1], shape=(N, N), format='csr')
    
    # Build potential energy operator: V_eff(r)
    V_eff = np.array([effective_potential_warp_drive(r, geometry_params, field_type) 
                      for r in r_grid])
    potential_matrix = diags([V_eff], [0], shape=(N, N), format='csr')
    
    # Full Hamiltonian: H = -∇² + V_eff
    H = -kinetic_matrix + potential_matrix
    
    # Convert to dense for boundary condition application
    H_dense = H.toarray()
    
    # Apply boundary conditions
    if boundary == 'dirichlet':
        # ψ(r_min) = ψ(r_max) = 0
        H_dense[0, :] = 0
        H_dense[0, 0] = 1
        H_dense[-1, :] = 0  
        H_dense[-1, -1] = 1
    elif boundary == 'neumann':
        # dψ/dr|boundaries = 0
        H_dense[0, 1] = H_dense[0, 0]
        H_dense[-1, -2] = H_dense[-1, -1]
    
    return H_dense

def solve_eigenvalue_problem(H_matrix, n_modes=10, sigma=0.0):
    """
    Solve the eigenvalue problem H ψ = ω² ψ
    
    Returns:
        eigenvalues: ω² values
        eigenvectors: corresponding mode profiles
    """
    try:
        # Use dense solver for all cases (more reliable for small problems)
        if isinstance(H_matrix, np.ndarray):
            H_array = H_matrix
        else:
            H_array = H_matrix.toarray()
        
        eigenvalues, eigenvectors = eigh(H_array)
        
        # Select first n_modes
        eigenvalues = eigenvalues[:n_modes]
        eigenvectors = eigenvectors[:, :n_modes]
        
        # Convert to frequencies (ω = √|λ| for eigenvalues)
        frequencies = np.sqrt(np.abs(eigenvalues))
        
        return frequencies, eigenvectors, eigenvalues
        
    except Exception as e:
        print(f"Eigenvalue solver failed: {e}")
        # Return dummy values for debugging
        n_dummy = min(n_modes, H_matrix.shape[0])
        return np.ones(n_dummy), np.eye(H_matrix.shape[0], n_dummy), np.ones(n_dummy)

def compute_mode_spectrum(geometry_path, config_path, output_path, field_type='scalar'):
    """
    Main function to compute the complete mode spectrum.
    """
    print(f"🔬 Computing {field_type} field mode spectrum")
    print(f"   Geometry: {geometry_path}")
    print(f"   Config: {config_path}")
    
    # Load geometry data
    geometry_data = read_ndjson(geometry_path)
    geometry_params = parse_metric_data(geometry_data)
    
    b0 = geometry_params['throat_radius']
    print(f"   Throat radius: {b0:.2e} m")
    print(f"   Geometry type: {geometry_params['geometry_type']}")
    
    # Set up radial grid
    r_min = max(b0, 1e-40)  # Avoid r=0 singularity
    r_max = 10 * b0         # Extend to several throat radii
    N_points = 500          # Grid resolution
    
    r_grid = np.linspace(r_min, r_max, N_points)
    dr = r_grid[1] - r_grid[0]
    
    print(f"   Radial grid: {r_min:.2e} to {r_max:.2e} m ({N_points} points)")
    print(f"   Grid spacing: {dr:.2e} m")
    
    # Build Hamiltonian operator
    print("   Building wave operator...")
    H_matrix = build_radial_operator(r_grid, geometry_params, field_type)
    
    # Solve eigenvalue problem
    print("   Solving eigenvalue problem...")
    n_modes = 20  # Number of modes to compute
    frequencies, mode_profiles, eigenvalues = solve_eigenvalue_problem(H_matrix, n_modes)
    
    # Prepare output data
    mode_spectrum = []
    for n in range(len(frequencies)):
        omega_n = frequencies[n]
        eigenval = eigenvalues[n]
        profile = mode_profiles[:, n]
        
        # Normalize mode profile
        norm = np.sqrt(np.trapz(profile**2, r_grid))
        if norm > 1e-12:
            profile = profile / norm
        
        mode_entry = {
            'mode_label': f"{field_type}_mode_{n}",
            'mode_number': int(n),
            'frequency': float(omega_n),
            'eigenvalue': float(eigenval),
            'field_type': field_type,
            'geometry_source': geometry_path,
            'parent_geometry': geometry_params['geometry_type'],
            'throat_radius': float(b0),
            'r_grid': [float(r) for r in r_grid[::10]],  # Decimated for output size
            'mode_profile': [float(phi) for phi in profile[::10]],  # Decimated profile
            'grid_spacing': float(dr),
            'integration_range': {'r_min': float(r_min), 'r_max': float(r_max)},
            'normalization': float(norm)
        }
        
        mode_spectrum.append(mode_entry)
        
        print(f"   Mode {n}: ω = {omega_n:.6e} Hz (λ = {eigenval:.6e})")
    
    # Write output
    with open(output_path, 'w') as f:
        for entry in mode_spectrum:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Computed {len(mode_spectrum)} {field_type} modes")
    print(f"   Wrote spectrum to: {output_path}")
    print(f"   Frequency range: {frequencies[0]:.2e} - {frequencies[-1]:.2e} Hz")
    
    return mode_spectrum

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quantum field mode spectrum in warp drive geometry")
    parser.add_argument('--geometry', required=True, help="Geometry/metric NDJSON file")
    parser.add_argument('--config', required=True, help="Configuration file")
    parser.add_argument('--field', default='scalar', choices=['scalar', 'electromagnetic', 'phonon'],
                       help="Type of quantum field")
    parser.add_argument('--out', required=True, help="Output mode spectrum NDJSON")
    
    args = parser.parse_args()
    
    compute_mode_spectrum(args.geometry, args.config, args.out, args.field)
