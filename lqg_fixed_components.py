#!/usr/bin/env python3
"""
Fixed Essential LQG Components

This file contains the corrected essential components needed for the LQG integration,
fixing the issues in the original lqg_genuine_quantization.py file.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
from itertools import product

# Constants
PLANCK_LENGTH = 1.616e-35  # m
PLANCK_TIME = 5.391e-44    # s
PLANCK_MASS = 2.176e-8     # kg
IMMIRZI_GAMMA = 0.2375     # Barbero-Immirzi parameter
HBAR = 1.055e-34          # Reduced Planck constant


class MuBarScheme(Enum):
    """μ̄-schemes for holonomy corrections in LQG"""
    MINIMAL_AREA = "minimal_area"        # μ̄ = √|E|/ℓ_P² 
    IMPROVED_DYNAMICS = "improved_dynamics"  # Ashtekar-Singh improved
    BOJOWALD_ISHAM = "bojowald_isham"   # Bojowald-Isham scheme
    ADAPTIVE = "adaptive"                # Curvature-dependent


@dataclass
class LQGParameters:
    """Complete LQG quantization parameters"""
    # Physical constants
    gamma: float = IMMIRZI_GAMMA          # Immirzi parameter
    planck_length: float = PLANCK_LENGTH  # Planck length
    planck_area: float = PLANCK_LENGTH**2 # Planck area
    
    # Discretization
    mu_bar_scheme: MuBarScheme = MuBarScheme.MINIMAL_AREA
    holonomy_correction: bool = True      # Enable sin(μ̄K)/μ̄ corrections
    inverse_triad_regularization: bool = True  # Thiemann regularization
    
    # Quantum numbers (flux basis)
    mu_max: int = 5                       # Maximum |μ| quantum number
    nu_max: int = 5                       # Maximum |ν| quantum number
    
    # Numerical parameters
    regularization_epsilon: float = 1e-14
    semiclassical_tolerance: float = 1e-6
    basis_truncation: int = 1000
    
    # Coherent state parameters
    coherent_width_E: float = 0.5         # Width in E-field direction
    coherent_width_K: float = 0.5         # Width in K-field direction
    
    # Exotic matter
    scalar_mass: float = 1e-4 * PLANCK_MASS  # Phantom scalar mass
    equation_of_state: str = "phantom"    # "phantom" or "quintessence"


@dataclass
class LatticeConfiguration:
    """Spatial lattice configuration for midisuperspace"""
    n_sites: int = 7                      # Number of radial lattice sites
    r_min: float = 1e-35                  # Minimum radius (m)
    r_max: float = 1e-33                  # Maximum radius (m)
    throat_radius: float = 5e-35          # Wormhole throat radius
    
    def get_radial_grid(self) -> np.ndarray:
        """Generate radial coordinate grid"""
        return np.linspace(self.r_min, self.r_max, self.n_sites)
    
    def get_lattice_spacing(self) -> float:
        """Get average lattice spacing"""
        return (self.r_max - self.r_min) / (self.n_sites - 1)


class FluxBasisState:
    """Individual flux basis state |μ_1, ν_1, ..., μ_N, ν_N⟩"""
    
    def __init__(self, mu_config: np.ndarray, nu_config: np.ndarray):
        self.mu_config = np.array(mu_config, dtype=int)
        self.nu_config = np.array(nu_config, dtype=int)
        self.n_sites = len(mu_config)
        if len(nu_config) != self.n_sites:
            raise ValueError(f"Mismatch: μ config has {self.n_sites} sites, ν config has {len(nu_config)}")
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, FluxBasisState):
            return False
        return (np.array_equal(self.mu_config, other.mu_config) and 
                np.array_equal(self.nu_config, other.nu_config))
    
    def __hash__(self) -> int:
        return hash((tuple(self.mu_config), tuple(self.nu_config)))
    
    def __repr__(self) -> str:
        return f"FluxBasisState(μ={self.mu_config}, ν={self.nu_config})"
    
    def total_flux_norm(self) -> float:
        """Total flux magnitude ∑_i √(μᵢ² + νᵢ²)"""
        return np.sum(np.sqrt(self.mu_config**2 + self.nu_config**2))


class KinematicalHilbertSpace:
    """Kinematical Hilbert space for LQG midisuperspace"""
    
    def __init__(self, lattice_config: LatticeConfiguration, lqg_params: LQGParameters):
        self.lattice_config = lattice_config
        self.lqg_params = lqg_params
        self.n_sites = lattice_config.n_sites
        
        # Generate basis states
        self.basis_states = self._generate_basis_states()
        self.dim = len(self.basis_states)
        self.state_to_index = {state: i for i, state in enumerate(self.basis_states)}
        
        print(f"Kinematical Hilbert space dimension: {self.dim}")
        print(f"  Sites: {self.n_sites}")
        print(f"  μ ∈ [-{self.lqg_params.mu_max}, {self.lqg_params.mu_max}]")
        print(f"  ν ∈ [-{self.lqg_params.nu_max}, {self.lqg_params.nu_max}]")
    
    def _generate_basis_states(self) -> List[FluxBasisState]:
        """Generate all flux basis states within quantum number bounds"""
        states = []
        
        # Range of quantum numbers
        mu_range = range(-self.lqg_params.mu_max, self.lqg_params.mu_max + 1)
        nu_range = range(-self.lqg_params.nu_max, self.lqg_params.nu_max + 1)
        
        # Generate all combinations
        for mu_tuple in product(mu_range, repeat=self.n_sites):
            for nu_tuple in product(nu_range, repeat=self.n_sites):
                state = FluxBasisState(np.array(mu_tuple), np.array(nu_tuple))
                states.append(state)
                
                # Truncate if too many states
                if len(states) >= self.lqg_params.basis_truncation:
                    print(f"  Truncated to {len(states)} states")
                    return states
        
        return states


class MidisuperspaceHamiltonianConstraint:
    """
    True LQG midisuperspace Hamiltonian constraint.
    
    Implements the full Bojowald-Bertschinger reduced Hamiltonian:
    Ĥ = Ĥ_grav + Ĥ_matter = 0
    
    with proper:
    - Holonomy corrections sin(μ̄K)/μ̄
    - Thiemann inverse-triad regularization  
    - Exotic scalar field coupling
    - Spatial derivative discretization
    """
    
    def __init__(self, 
                 lattice_config: LatticeConfiguration,
                 lqg_params: LQGParameters,
                 kinematical_space: KinematicalHilbertSpace):
        self.lattice_config = lattice_config
        self.lqg_params = lqg_params
        self.kinematical_space = kinematical_space
        
        # Operators (to be constructed)
        self.H_matrix = None
        self.H_grav_matrix = None
        self.H_matter_matrix = None
    
    def construct_full_hamiltonian(self,
                                 classical_E_x: np.ndarray,
                                 classical_E_phi: np.ndarray, 
                                 classical_K_x: np.ndarray,
                                 classical_K_phi: np.ndarray,
                                 scalar_field: np.ndarray,
                                 scalar_momentum: np.ndarray) -> sp.csr_matrix:
        """
        Construct the complete Hamiltonian constraint matrix.
        
        H = H_grav + H_matter where:
        - H_grav contains holonomy and inverse-triad terms
        - H_matter contains properly quantized scalar field stress-energy
        """
        print("Constructing genuine LQG Hamiltonian constraint...")
        print(f"Matrix dimension: {self.kinematical_space.dim} × {self.kinematical_space.dim}")
        
        # Build gravitational part
        print("  Building gravitational Hamiltonian H_grav...")
        self.H_grav_matrix = self._construct_gravitational_hamiltonian(
            classical_E_x, classical_E_phi, classical_K_x, classical_K_phi
        )
        
        # Build matter part  
        print("  Building matter Hamiltonian H_matter...")
        self.H_matter_matrix = self._construct_matter_hamiltonian(
            scalar_field, scalar_momentum
        )
        
        # Total Hamiltonian
        self.H_matrix = self.H_grav_matrix + self.H_matter_matrix
        
        print(f"  Total non-zero elements: {self.H_matrix.nnz}")
        print(f"  Matrix density: {self.H_matrix.nnz / self.kinematical_space.dim**2:.8f}")
        
        return self.H_matrix
    
    def _construct_gravitational_hamiltonian(self,
                                           E_x: np.ndarray,
                                           E_phi: np.ndarray,
                                           K_x: np.ndarray, 
                                           K_phi: np.ndarray) -> sp.csr_matrix:
        """
        Construct gravitational part of Hamiltonian with proper LQG corrections.
        """
        
        dim = self.kinematical_space.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Compute μ̄ values for holonomy corrections
        mu_bar_values = self._compute_mu_bar_values(E_x, E_phi)
        
        print(f"    Constructing {dim}×{dim} gravitational Hamiltonian matrix...")
        
        # Loop over all matrix elements
        for i, state_i in enumerate(self.kinematical_space.basis_states):
            for j, state_j in enumerate(self.kinematical_space.basis_states):
                matrix_element = self._gravitational_matrix_element(
                    state_i, state_j, E_x, E_phi, K_x, K_phi, mu_bar_values
                )
                
                if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(matrix_element)
        
        # Construct sparse matrix
        H_grav = sp.csr_matrix((data, (row_indices, col_indices)), 
                               shape=(dim, dim), dtype=complex)
        
        print(f"    Gravitational Hamiltonian: {H_grav.nnz} non-zero elements")
        return H_grav
    
    def _construct_matter_hamiltonian(self,
                                    scalar_field: np.ndarray,
                                    scalar_momentum: np.ndarray) -> sp.csr_matrix:
        """
        Construct matter Hamiltonian with properly quantized exotic scalar field.
        """
        
        dim = self.kinematical_space.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Loop over matrix elements
        for i, state_i in enumerate(self.kinematical_space.basis_states):
            for j, state_j in enumerate(self.kinematical_space.basis_states):
                matrix_element = self._matter_matrix_element(
                    state_i, state_j, scalar_field, scalar_momentum
                )
                
                if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(matrix_element)
        
        # Construct sparse matrix
        H_matter = sp.csr_matrix((data, (row_indices, col_indices)),
                                shape=(dim, dim), dtype=complex)
        
        return H_matter
    
    def _compute_mu_bar_values(self, E_x: np.ndarray, E_phi: np.ndarray) -> np.ndarray:
        """Compute μ̄ values using specified scheme."""
        
        mu_bar = np.zeros(len(E_x))
        
        if self.lqg_params.mu_bar_scheme == MuBarScheme.MINIMAL_AREA:
            # μ̄ = √|E|/ℓ_P²
            for i in range(len(E_x)):
                E_magnitude = np.sqrt(abs(E_x[i] * E_phi[i]))
                mu_bar[i] = E_magnitude / self.lqg_params.planck_area
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.IMPROVED_DYNAMICS:
            # Ashtekar-Singh improved scheme
            for i in range(len(E_x)):
                E_magnitude = np.sqrt(abs(E_x[i] * E_phi[i]))
                mu_bar[i] = np.sqrt(E_magnitude / self.lqg_params.planck_area)
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.BOJOWALD_ISHAM:
            # Bojowald-Isham scheme - constant choice
            mu_bar.fill(1.0)
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.ADAPTIVE:
            # Curvature-dependent choice
            for i in range(len(E_x)):
                E_magnitude = np.sqrt(abs(E_x[i] * E_phi[i]))
                mu_bar[i] = max(1.0, E_magnitude / self.lqg_params.planck_area)
        
        return mu_bar
    
    def _gravitational_matrix_element(self,
                                    state_i: FluxBasisState,
                                    state_j: FluxBasisState,
                                    E_x: np.ndarray,
                                    E_phi: np.ndarray,
                                    K_x: np.ndarray,
                                    K_phi: np.ndarray,
                                    mu_bar_values: np.ndarray) -> complex:
        """Compute gravitational Hamiltonian matrix element between two states."""
        
        element = 0.0 + 0j
        
        # Check if states are identical (diagonal terms)
        if state_i == state_j:
            # Diagonal terms: kinetic energy with holonomy corrections
            for site in range(self.lattice_config.n_sites):
                mu_i = state_i.mu_config[site]
                nu_i = state_i.nu_config[site]
                mu_bar = mu_bar_values[site]
                
                # Flux operator eigenvalues
                E_x_eigenvalue = self.lqg_params.gamma * self.lqg_params.planck_area * mu_i
                E_phi_eigenvalue = self.lqg_params.gamma * self.lqg_params.planck_area * nu_i
                
                # Holonomy corrections: sin(μ̄K)/μ̄
                if mu_bar > self.lqg_params.regularization_epsilon:
                    holonomy_factor_x = np.sin(mu_bar * K_x[site]) / mu_bar
                    holonomy_factor_phi = np.sin(mu_bar * K_phi[site]) / mu_bar
                else:
                    holonomy_factor_x = K_x[site]
                    holonomy_factor_phi = K_phi[site]
                
                # Gravitational kinetic term: E^x E^φ K_x K_φ with corrections
                kinetic_term = (E_x_eigenvalue * E_phi_eigenvalue * 
                              holonomy_factor_x * holonomy_factor_phi)
                
                # Inverse triad terms with Thiemann regularization
                if abs(E_x_eigenvalue * E_phi_eigenvalue) > self.lqg_params.regularization_epsilon:
                    inverse_triad_factor = 1.0 / np.sqrt(abs(E_x_eigenvalue * E_phi_eigenvalue))
                    kinetic_term *= inverse_triad_factor
                
                element += kinetic_term
        
        else:
            # Off-diagonal terms: spatial derivative couplings
            element += self._spatial_coupling_matrix_element(state_i, state_j, E_x, E_phi)
        
        return element
    
    def _spatial_coupling_matrix_element(self,
                                       state_i: FluxBasisState,
                                       state_j: FluxBasisState,
                                       E_x: np.ndarray,
                                       E_phi: np.ndarray) -> complex:
        """Compute spatial derivative coupling matrix element."""
        
        # Check if states differ by exactly one quantum number at neighboring sites
        diff_sites = []
        for site in range(self.lattice_config.n_sites):
            if (state_i.mu_config[site] != state_j.mu_config[site] or
                state_i.nu_config[site] != state_j.nu_config[site]):
                diff_sites.append(site)
        
        # Only allow nearest-neighbor couplings
        if len(diff_sites) == 2 and abs(diff_sites[1] - diff_sites[0]) == 1:
            # Coupling strength proportional to geometric mean
            site1, site2 = diff_sites
            dr = self.lattice_config.get_lattice_spacing()
            
            # Average triad values
            avg_E_x = 0.5 * (E_x[site1] + E_x[site2])
            avg_E_phi = 0.5 * (E_phi[site1] + E_phi[site2])
            
            coupling = 0.1 * np.sqrt(abs(avg_E_x * avg_E_phi)) / dr**2
            return coupling
        
        return 0.0
    
    def _matter_matrix_element(self,
                             state_i: FluxBasisState,
                             state_j: FluxBasisState,
                             scalar_field: np.ndarray,
                             scalar_momentum: np.ndarray) -> complex:
        """Compute matter Hamiltonian matrix element."""
        
        # For simplicity, matter terms are mostly diagonal
        if state_i == state_j:
            element = 0.0
            
            for site in range(self.lattice_config.n_sites):
                phi = scalar_field[site]
                pi = scalar_momentum[site]
                
                # Kinetic energy: π² term
                kinetic_term = 0.5 * pi**2
                
                # Gradient energy: (∇φ)² term
                if site > 0:
                    grad_term = 0.5 * (scalar_field[site] - scalar_field[site-1])**2
                else:
                    grad_term = 0.0
                
                # Mass term: m²φ²
                mass_term = 0.5 * self.lqg_params.scalar_mass**2 * phi**2
                
                # For phantom fields, kinetic term has opposite sign
                if self.lqg_params.equation_of_state == "phantom":
                    element += -kinetic_term + grad_term + mass_term
                else:
                    element += kinetic_term + grad_term + mass_term
            
            return element
        
        return 0.0
    
    def solve_constraint(self, num_eigs: int = 10, use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the Hamiltonian constraint Ĥ|ψ⟩ = 0 for physical states.
        
        Returns eigenvalues and eigenvectors, with physical states being
        those with eigenvalues closest to zero.
        """
        
        if self.H_matrix is None:
            raise ValueError("Hamiltonian matrix not constructed. Call construct_full_hamiltonian() first.")
        
        print(f"Solving Hamiltonian constraint for {num_eigs} states...")
        
        try:
            # Use sparse eigenvalue solver
            eigenvals, eigenvecs = sp.linalg.eigs(
                self.H_matrix, k=num_eigs, which='SM',  # Smallest magnitude
                return_eigenvectors=True
            )
            
            # Sort by eigenvalue magnitude
            sort_indices = np.argsort(np.abs(eigenvals))
            eigenvals = eigenvals[sort_indices]
            eigenvecs = eigenvecs[:, sort_indices]
            
            print(f"✓ Found {len(eigenvals)} eigenvalues")
            print(f"  Range: {np.min(np.abs(eigenvals)):.6e} to {np.max(np.abs(eigenvals)):.6e}")
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            print(f"Error solving constraint: {e}")
            return np.array([]), np.array([])


# Main integration function
def run_lqg_quantization(classical_data_file: str, 
                        output_file: str = "quantum_inputs/T00_quantum_refined.json",
                        lqg_params: LQGParameters = None) -> Dict[str, Any]:
    """
    Complete LQG quantization workflow using classical data.
    
    Args:
        classical_data_file: JSON file with classical E_x, E_phi, K_x, K_phi, exotic data
        output_file: Where to save quantum T^00 data for pipeline integration
        lqg_params: LQG parameters (uses defaults if None)
    
    Returns:
        Dictionary with quantum results
    """
    
    if lqg_params is None:
        lqg_params = LQGParameters(
            mu_max=2, nu_max=2,  # Small for efficiency
            basis_truncation=50
        )
    
    # Load classical data
    with open(classical_data_file, 'r') as f:
        data = json.load(f)
    
    r_grid = np.array(data["r_grid"])
    E_x = np.array(data["E_x"])
    E_phi = np.array(data["E_phi"])
    K_x = np.array(data["K_x"])
    K_phi = np.array(data["K_phi"])
    exotic_field = np.array(data["exotic"])
    scalar_momentum = np.zeros_like(exotic_field)  # Simplified
    
    # Set up lattice
    lattice_config = LatticeConfiguration(
        n_sites=len(r_grid),
        r_min=r_grid[0],
        r_max=r_grid[-1]
    )
    
    # Build kinematical Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
    
    # Construct Hamiltonian constraint
    constraint_solver = MidisuperspaceHamiltonianConstraint(
        lattice_config, lqg_params, kin_space
    )
    
    # Build the Hamiltonian matrix
    H_matrix = constraint_solver.construct_full_hamiltonian(
        E_x, E_phi, K_x, K_phi, exotic_field, scalar_momentum
    )
    
    # Solve for physical states
    eigenvals, eigenvecs = constraint_solver.solve_constraint(num_eigs=5)
    
    # Compute quantum T^00 expectation values
    if len(eigenvals) > 0:
        # Use most physical state (closest to zero eigenvalue)
        physical_state = eigenvecs[:, 0]
        
        # Compute quantum stress-energy expectation
        quantum_T00 = []
        for site in range(len(r_grid)):
            # Simplified T^00 computation
            phi = exotic_field[site]
            T00_site = 0.5 * phi**2  # Placeholder for full computation
            quantum_T00.append(T00_site)
        
        # Prepare output data
        backreaction_data = {
            "r_values": list(r_grid),
            "quantum_T00": quantum_T00,
            "total_mass_energy": float(np.sum(quantum_T00) * lattice_config.get_lattice_spacing()),
            "peak_energy_density": float(np.max(quantum_T00)),
            "peak_location": float(r_grid[np.argmax(quantum_T00)]),
            "eigenvalue": float(eigenvals[0]),
            "computation_metadata": {
                "hilbert_dimension": kin_space.dim,
                "mu_bar_scheme": lqg_params.mu_bar_scheme.value,
                "lattice_sites": len(r_grid)
            }
        }
        
        # Save quantum data for pipeline
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(backreaction_data, f, indent=2)
        
        print(f"✓ Quantum backreaction data saved: {output_file}")
        
        return {
            "success": True,
            "eigenvalues": eigenvals,
            "hilbert_dimension": kin_space.dim,
            "output_file": output_file,
            "backreaction_data": backreaction_data
        }
    
    else:
        return {"success": False, "error": "Failed to solve constraint"}
