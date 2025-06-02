#!/usr/bin/env python3
"""
Beyond Spherical Symmetry: Angular Perturbations in LQG Midisuperspace

This module extends the LQG framework to include non-radial degrees of freedom:
1. Spherical harmonic perturbations Y_lm(θ,φ)
2. Extended kinematical Hilbert space with angular quantum numbers
3. Modified constraint operators mixing radial and angular shifts
4. Gauge-fixing strategies for diffeomorphism constraint

Author: LQG Framework Team
Date: June 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from enum import Enum

# Import base LQG components
from lqg_fixed_components import (
    LatticeConfiguration,
    LQGParameters,
    FluxBasisState
)

class SphericalHarmonicMode(Enum):
    """Supported spherical harmonic modes for perturbations."""
    Y_10 = "Y_10"  # Dipole mode
    Y_11 = "Y_11"  # Dipole mode (complex)
    Y_20 = "Y_20"  # Quadrupole mode
    Y_21 = "Y_21"  # Quadrupole mode
    Y_22 = "Y_22"  # Quadrupole mode

@dataclass
class AngularPerturbationConfig:
    """Configuration for angular perturbations."""
    enabled_modes: List[SphericalHarmonicMode]
    perturbation_amplitudes: Dict[str, float]
    max_angular_quantum_number: int = 2
    angular_truncation_energy: float = 1e-30
    
@dataclass
class ExtendedFluxBasisState:
    """
    Extended basis state including angular quantum numbers.
    
    Extends FluxBasisState with angular degrees of freedom:
    |μ, ν, l, m⟩ where (l,m) are spherical harmonic quantum numbers.
    """
    radial_mu_config: np.ndarray    # Radial flux quantum numbers
    radial_nu_config: np.ndarray    # Radial flux quantum numbers  
    angular_l_config: np.ndarray    # Angular momentum quantum numbers
    angular_m_config: np.ndarray    # Magnetic quantum numbers
    
    def __post_init__(self):
        """Validate quantum number consistency."""
        assert len(self.radial_mu_config) == len(self.radial_nu_config)
        assert len(self.angular_l_config) == len(self.angular_m_config)
        
        # Check angular momentum constraints |m| ≤ l
        for l, m in zip(self.angular_l_config, self.angular_m_config):
            assert abs(m) <= l, f"Invalid angular quantum numbers: l={l}, m={m}"
    
    def __eq__(self, other):
        """Equality comparison for basis states."""
        if not isinstance(other, ExtendedFluxBasisState):
            return False
        
        return (np.array_equal(self.radial_mu_config, other.radial_mu_config) and
                np.array_equal(self.radial_nu_config, other.radial_nu_config) and
                np.array_equal(self.angular_l_config, other.angular_l_config) and
                np.array_equal(self.angular_m_config, other.angular_m_config))
    
    def __hash__(self):
        """Hash for use in dictionaries."""
        return hash((tuple(self.radial_mu_config), tuple(self.radial_nu_config),
                    tuple(self.angular_l_config), tuple(self.angular_m_config)))

class ExtendedKinematicalHilbertSpace:
    """
    Extended kinematical Hilbert space including angular degrees of freedom.
    
    Basis states: |μᵢ, νᵢ, lᵢ, mᵢ⟩ for i = 1,...,N lattice sites
    where (μᵢ, νᵢ) are radial flux quantum numbers and (lᵢ, mᵢ) are angular.
    """
    
    def __init__(self, 
                 lattice_config: LatticeConfiguration,
                 lqg_params: LQGParameters,
                 angular_config: AngularPerturbationConfig):
        self.lattice_config = lattice_config
        self.lqg_params = lqg_params
        self.angular_config = angular_config
        
        # Basis states
        self.basis_states: List[ExtendedFluxBasisState] = []
        self.dim = 0
        
        # Angular truncation parameters
        self.l_max = angular_config.max_angular_quantum_number
        
    def generate_extended_flux_basis(self) -> None:
        """
        Generate extended flux basis including angular quantum numbers.
        
        Creates tensor product of radial flux states with angular momentum states.
        """
        print("Generating extended flux basis with angular perturbations...")
        print(f"  Maximum angular momentum: l_max = {self.l_max}")
        print(f"  Enabled modes: {[mode.value for mode in self.angular_config.enabled_modes]}")
        
        # Generate radial quantum number combinations
        radial_states = self._generate_radial_flux_combinations()
        print(f"  Radial basis states: {len(radial_states)}")
        
        # Generate angular quantum number combinations
        angular_states = self._generate_angular_momentum_combinations()
        print(f"  Angular basis states: {len(angular_states)}")
        
        # Create tensor product basis
        self.basis_states = []
        
        for radial_state in radial_states:
            for angular_state in angular_states:
                
                # Check if this angular configuration is allowed
                if self._is_angular_state_allowed(angular_state):
                    
                    extended_state = ExtendedFluxBasisState(
                        radial_mu_config=radial_state['mu'],
                        radial_nu_config=radial_state['nu'],
                        angular_l_config=angular_state['l'],
                        angular_m_config=angular_state['m']
                    )
                    
                    self.basis_states.append(extended_state)
        
        self.dim = len(self.basis_states)
        print(f"✓ Extended Hilbert space dimension: {self.dim}")
        
        # Check if dimension is manageable
        if self.dim > self.lqg_params.max_basis_states:
            print(f"⚠️ Dimension {self.dim} exceeds maximum {self.lqg_params.max_basis_states}")
            print("   Consider reducing l_max or enabled modes")
    
    def _generate_radial_flux_combinations(self) -> List[Dict[str, np.ndarray]]:
        """Generate radial flux quantum number combinations."""
        
        radial_states = []
        n_sites = self.lattice_config.n_sites
        
        # Simple truncation: μ, ν ∈ {-2, -1, 0, 1, 2}
        mu_range = range(-2, 3)
        nu_range = range(-2, 3)
        
        # Generate all combinations (with truncation for manageable size)
        count = 0
        max_states = 1000  # Prevent explosion
        
        for mu_config in self._generate_integer_configs(n_sites, mu_range):
            for nu_config in self._generate_integer_configs(n_sites, nu_range):
                
                radial_states.append({
                    'mu': np.array(mu_config),
                    'nu': np.array(nu_config)
                })
                
                count += 1
                if count > max_states:
                    break
            
            if count > max_states:
                break
        
        return radial_states
    
    def _generate_angular_momentum_combinations(self) -> List[Dict[str, np.ndarray]]:
        """Generate angular momentum quantum number combinations."""
        
        angular_states = []
        n_sites = self.lattice_config.n_sites
        
        # For each site, allow l ∈ {0, 1, 2, ..., l_max}
        # and corresponding m ∈ {-l, -l+1, ..., l-1, l}
        
        # Start simple: only include specific enabled modes
        allowed_lm_pairs = []
        
        for mode in self.angular_config.enabled_modes:
            if mode == SphericalHarmonicMode.Y_10:
                allowed_lm_pairs.append((1, 0))
            elif mode == SphericalHarmonicMode.Y_11:
                allowed_lm_pairs.append((1, 1))
                allowed_lm_pairs.append((1, -1))
            elif mode == SphericalHarmonicMode.Y_20:
                allowed_lm_pairs.append((2, 0))
            elif mode == SphericalHarmonicMode.Y_21:
                allowed_lm_pairs.append((2, 1))
                allowed_lm_pairs.append((2, -1))
            elif mode == SphericalHarmonicMode.Y_22:
                allowed_lm_pairs.append((2, 2))
                allowed_lm_pairs.append((2, -2))
        
        # Add spherically symmetric state (l=0, m=0)
        allowed_lm_pairs.append((0, 0))
        
        # Generate combinations for all sites
        max_angular_states = 100  # Limit for manageable computation
        count = 0
        
        for l_config in self._generate_angular_configs(n_sites, allowed_lm_pairs):
            if count < max_angular_states:
                angular_states.append(l_config)
                count += 1
        
        return angular_states
    
    def _generate_integer_configs(self, n_sites: int, value_range: range) -> List[List[int]]:
        """Generate all combinations of integer values for n_sites."""
        
        if n_sites == 1:
            return [[val] for val in value_range]
        
        configs = []
        for val in value_range:
            for sub_config in self._generate_integer_configs(n_sites - 1, value_range):
                configs.append([val] + sub_config)
                if len(configs) > 1000:  # Prevent explosion
                    return configs
        
        return configs
    
    def _generate_angular_configs(self, n_sites: int, allowed_lm_pairs: List[Tuple[int, int]]) -> List[Dict[str, np.ndarray]]:
        """Generate angular momentum configurations for all sites."""
        
        configs = []
        
        # Simple approach: uniform angular state across all sites
        for l, m in allowed_lm_pairs:
            config = {
                'l': np.full(n_sites, l),
                'm': np.full(n_sites, m)
            }
            configs.append(config)
        
        # Add mixed configurations (only for small n_sites)
        if n_sites <= 3 and len(allowed_lm_pairs) > 1:
            # Try some mixed configurations
            for i, (l1, m1) in enumerate(allowed_lm_pairs):
                for j, (l2, m2) in enumerate(allowed_lm_pairs):
                    if i != j:
                        mixed_config = {
                            'l': np.array([l1] + [l2] * (n_sites - 1)),
                            'm': np.array([m1] + [m2] * (n_sites - 1))
                        }
                        configs.append(mixed_config)
                        
                        if len(configs) > 50:  # Limit mixed configurations
                            break
                if len(configs) > 50:
                    break
        
        return configs
    
    def _is_angular_state_allowed(self, angular_state: Dict[str, np.ndarray]) -> bool:
        """Check if angular momentum state satisfies constraints."""
        
        l_config = angular_state['l']
        m_config = angular_state['m']
        
        # Check |m| ≤ l constraint
        for l, m in zip(l_config, m_config):
            if abs(m) > l:
                return False
        
        # Check energy cutoff (simplified)
        total_angular_energy = np.sum(l_config * (l_config + 1))
        if total_angular_energy > self.angular_config.angular_truncation_energy * 1e30:
            return False
        
        return True

class ExtendedMidisuperspaceHamiltonianConstraint:
    """
    Extended Hamiltonian constraint including angular perturbations.
    
    Implements coupling between radial and angular degrees of freedom
    in the constraint operators.
    """
    
    def __init__(self,
                 lattice_config: LatticeConfiguration,
                 lqg_params: LQGParameters,
                 extended_kinematical_space: ExtendedKinematicalHilbertSpace,
                 angular_config: AngularPerturbationConfig):
        self.lattice_config = lattice_config
        self.lqg_params = lqg_params
        self.kinematical_space = extended_kinematical_space
        self.angular_config = angular_config
        
        # Extended constraint matrices
        self.H_matrix = None
        self.H_radial_matrix = None
        self.H_angular_matrix = None
        self.H_coupling_matrix = None
        
        # Extended diffeomorphism constraint
        self.C_diffeo_radial_matrix = None
        self.C_diffeo_angular_matrix = None
    
    def construct_extended_hamiltonian(self,
                                     classical_data: Dict[str, np.ndarray],
                                     angular_perturbations: Dict[str, np.ndarray]) -> 'sp.csr_matrix':
        """
        Construct extended Hamiltonian constraint including angular degrees.
        
        H_total = H_radial + H_angular + H_coupling
        """
        import scipy.sparse as sp
        
        print("Constructing extended Hamiltonian with angular perturbations...")
        
        dim = self.kinematical_space.dim
        print(f"Extended matrix dimension: {dim} × {dim}")
        
        # Build radial part (standard LQG constraint)
        print("  Building radial Hamiltonian...")
        self.H_radial_matrix = self._construct_radial_hamiltonian(classical_data)
        
        # Build angular part (spherical Laplacian on angular variables)
        print("  Building angular Hamiltonian...")
        self.H_angular_matrix = self._construct_angular_hamiltonian(angular_perturbations)
        
        # Build radial-angular coupling
        print("  Building radial-angular coupling...")
        self.H_coupling_matrix = self._construct_coupling_hamiltonian(
            classical_data, angular_perturbations
        )
        
        # Total Hamiltonian
        self.H_matrix = (self.H_radial_matrix + 
                        self.H_angular_matrix + 
                        self.H_coupling_matrix)
        
        print(f"✓ Extended Hamiltonian constructed:")
        print(f"  Total non-zero elements: {self.H_matrix.nnz}")
        print(f"  Matrix density: {self.H_matrix.nnz / dim**2:.8f}")
        
        return self.H_matrix
    
    def _construct_radial_hamiltonian(self, classical_data: Dict[str, np.ndarray]) -> 'sp.csr_matrix':
        """Construct radial part of extended Hamiltonian."""
        import scipy.sparse as sp
        
        dim = self.kinematical_space.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Loop over extended basis states
        for i, state_i in enumerate(self.kinematical_space.basis_states):
            for j, state_j in enumerate(self.kinematical_space.basis_states):
                
                # Radial Hamiltonian only couples states with same angular quantum numbers
                if (np.array_equal(state_i.angular_l_config, state_j.angular_l_config) and
                    np.array_equal(state_i.angular_m_config, state_j.angular_m_config)):
                    
                    # Compute radial matrix element
                    matrix_element = self._radial_matrix_element(
                        state_i, state_j, classical_data
                    )
                    
                    if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(matrix_element)
        
        H_radial = sp.csr_matrix((data, (row_indices, col_indices)), 
                                shape=(dim, dim), dtype=complex)
        
        print(f"    Radial Hamiltonian: {H_radial.nnz} non-zero elements")
        return H_radial
    
    def _construct_angular_hamiltonian(self, angular_perturbations: Dict[str, np.ndarray]) -> 'sp.csr_matrix':
        """Construct angular part of Hamiltonian (spherical Laplacian)."""
        import scipy.sparse as sp
        
        dim = self.kinematical_space.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Angular Laplacian: -∇²_angular acting on (l,m) quantum numbers
        for i, state_i in enumerate(self.kinematical_space.basis_states):
            for j, state_j in enumerate(self.kinematical_space.basis_states):
                
                # Angular Laplacian only couples states with same radial quantum numbers
                if (np.array_equal(state_i.radial_mu_config, state_j.radial_mu_config) and
                    np.array_equal(state_i.radial_nu_config, state_j.radial_nu_config)):
                    
                    matrix_element = self._angular_laplacian_matrix_element(
                        state_i, state_j, angular_perturbations
                    )
                    
                    if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                        row_indices.append(i)
                        col_indices.append(j)
                        data.append(matrix_element)
        
        H_angular = sp.csr_matrix((data, (row_indices, col_indices)), 
                                 shape=(dim, dim), dtype=complex)
        
        print(f"    Angular Hamiltonian: {H_angular.nnz} non-zero elements")
        return H_angular
    
    def _construct_coupling_hamiltonian(self,
                                      classical_data: Dict[str, np.ndarray],
                                      angular_perturbations: Dict[str, np.ndarray]) -> 'sp.csr_matrix':
        """Construct radial-angular coupling terms."""
        import scipy.sparse as sp
        
        dim = self.kinematical_space.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Coupling terms mix radial and angular derivatives
        for i, state_i in enumerate(self.kinematical_space.basis_states):
            for j, state_j in enumerate(self.kinematical_space.basis_states):
                
                matrix_element = self._coupling_matrix_element(
                    state_i, state_j, classical_data, angular_perturbations
                )
                
                if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(matrix_element)
        
        H_coupling = sp.csr_matrix((data, (row_indices, col_indices)), 
                                  shape=(dim, dim), dtype=complex)
        
        print(f"    Coupling Hamiltonian: {H_coupling.nnz} non-zero elements")
        return H_coupling
    
    def _radial_matrix_element(self,
                              state_i: ExtendedFluxBasisState,
                              state_j: ExtendedFluxBasisState,
                              classical_data: Dict[str, np.ndarray]) -> complex:
        """Compute radial Hamiltonian matrix element."""
        
        # Use standard LQG radial matrix element computation
        # (This would call the original _gravitational_matrix_element function)
        
        # Simplified version for demonstration
        if np.array_equal(state_i.radial_mu_config, state_j.radial_mu_config):
            # Diagonal term
            return sum(state_i.radial_mu_config * classical_data['classical_K_x'])
        else:
            # Off-diagonal coupling
            return 0.1 * np.sum(np.abs(state_i.radial_mu_config - state_j.radial_mu_config))
    
    def _angular_laplacian_matrix_element(self,
                                         state_i: ExtendedFluxBasisState,
                                         state_j: ExtendedFluxBasisState,
                                         angular_perturbations: Dict[str, np.ndarray]) -> complex:
        """Compute angular Laplacian matrix element."""
        
        # Spherical Laplacian eigenvalues: -l(l+1) for Y_lm
        if np.array_equal(state_i.angular_l_config, state_j.angular_l_config):
            # Diagonal in angular quantum numbers
            l_total = np.sum(state_i.angular_l_config * (state_i.angular_l_config + 1))
            return -l_total * 1e-32  # Scale for appropriate units
        else:
            # Off-diagonal angular coupling (simplified)
            return 0.0
    
    def _coupling_matrix_element(self,
                                state_i: ExtendedFluxBasisState,
                                state_j: ExtendedFluxBasisState,
                                classical_data: Dict[str, np.ndarray],
                                angular_perturbations: Dict[str, np.ndarray]) -> complex:
        """Compute radial-angular coupling matrix element."""
        
        # Simplified coupling: radial derivatives coupled to angular momentum
        radial_diff = np.sum(np.abs(state_i.radial_mu_config - state_j.radial_mu_config))
        angular_diff = np.sum(np.abs(state_i.angular_l_config - state_j.angular_l_config))
        
        if radial_diff == 1 and angular_diff == 1:
            # Non-trivial radial-angular coupling
            coupling_strength = 0.01  # Small perturbative coupling
            return coupling_strength
        
        return 0.0

def create_angular_perturbation_demo():
    """Demonstrate extended LQG framework with angular perturbations."""
    
    print("🌐 LQG Beyond Spherical Symmetry: Angular Perturbations Demo")
    print("=" * 60)
    
    # Setup base configuration
    lattice_config = LatticeConfiguration()
    lattice_config.n_sites = 3  # Keep small for demonstration
    lattice_config.r_min = 1e-35
    lattice_config.r_max = 1e-33
    
    lqg_params = LQGParameters(
        planck_length=0.01,
        max_basis_states=500,  # Increased for extended space
        regularization_epsilon=1e-10
    )
    
    # Setup angular perturbation configuration
    angular_config = AngularPerturbationConfig(
        enabled_modes=[SphericalHarmonicMode.Y_10, SphericalHarmonicMode.Y_20],
        perturbation_amplitudes={'Y_10': 0.1, 'Y_20': 0.05},
        max_angular_quantum_number=2
    )
    
    print(f"Enabled angular modes: {[mode.value for mode in angular_config.enabled_modes]}")
    
    # Build extended kinematical Hilbert space
    print("\n1. Building Extended Kinematical Hilbert Space")
    extended_kin_space = ExtendedKinematicalHilbertSpace(
        lattice_config, lqg_params, angular_config
    )
    extended_kin_space.generate_extended_flux_basis()
    
    # Setup classical and angular perturbation data
    print("\n2. Setting up Classical Background and Angular Perturbations")
    r_sites = np.linspace(lattice_config.r_min, lattice_config.r_max, lattice_config.n_sites)
    
    classical_data = {
        'classical_E_x': np.ones(lattice_config.n_sites) * 1e-32,
        'classical_E_phi': np.ones(lattice_config.n_sites) * 1e-32,
        'classical_K_x': np.sin(r_sites / 1e-34) * 1e2,
        'classical_K_phi': np.cos(r_sites / 1e-34) * 1e2,
        'scalar_field': np.tanh(r_sites / 1e-34),
        'scalar_momentum': np.zeros(lattice_config.n_sites)
    }
    
    angular_perturbations = {
        'Y_10_amplitude': np.full(lattice_config.n_sites, angular_config.perturbation_amplitudes['Y_10']),
        'Y_20_amplitude': np.full(lattice_config.n_sites, angular_config.perturbation_amplitudes['Y_20'])
    }
    
    # Build extended constraint
    print("\n3. Building Extended Hamiltonian Constraint")
    extended_constraint = ExtendedMidisuperspaceHamiltonianConstraint(
        lattice_config, lqg_params, extended_kin_space, angular_config
    )
    
    H_extended = extended_constraint.construct_extended_hamiltonian(
        classical_data, angular_perturbations
    )
    
    print(f"\n✓ Extended framework demonstration complete!")
    print(f"  Extended Hilbert space dimension: {extended_kin_space.dim}")
    print(f"  Hamiltonian matrix size: {H_extended.shape}")
    print(f"  Non-zero matrix elements: {H_extended.nnz}")
    print(f"  Matrix sparsity: {H_extended.nnz / extended_kin_space.dim**2:.6f}")
    
    return extended_constraint, H_extended

if __name__ == "__main__":
    extended_constraint, H_extended = create_angular_perturbation_demo()
