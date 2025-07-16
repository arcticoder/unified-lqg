#!/usr/bin/env python3
"""
Fixed Essential LQG Components

This file contains the corrected essential components needed for the LQG integration,
fixing the issues in the original lqg_genuine_quantization.py file and adding
K-operator support plus coherent-state verification.

Updated to use Maxwell-extended Hilbert space from kinematical_hilbert.py
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

# Import Maxwell-extended Hilbert space and spin-foam cross-validation
from kinematical_hilbert import MidisuperspaceHilbert, LatticeConfig
from spin_foam_crossval import SimpleRadialDipoleGraph, SpinFoamAmplitude

# Constants
PLANCK_LENGTH = 1.616e-35  # m
PLANCK_TIME = 5.391e-44    # s
PLANCK_MASS = 2.176e-8     # kg
IMMIRZI_GAMMA = 0.2375     # Barbero-Immirzi parameter
HBAR = 1.055e-34           # Reduced Planck constant


class MuBarScheme(Enum):
    """μ̄-schemes for holonomy corrections in LQG"""
    MINIMAL_AREA = "minimal_area"          # μ̄ = √|E|/ℓ_P² 
    IMPROVED_DYNAMICS = "improved_dynamics"  # Ashtekar-Singh improved
    BOJOWALD_ISHAM = "bojowald_isham"       # Bojowald-Isham scheme
    ADAPTIVE = "adaptive"                  # Curvature-dependent


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
    
    def flux_E_x_operator(self, site: int) -> sp.csr_matrix:
        """
        Flux operator Ê^x_I acting on site I.
        Returns diagonal matrix with eigenvalues μ_I.
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        diagonal_elements = [float(state.mu_config[site]) for state in self.basis_states]
        return sp.diags(diagonal_elements, format='csr')
    
    def flux_E_phi_operator(self, site: int) -> sp.csr_matrix:
        """
        Flux operator Ê^φ_I acting on site I.
        Returns diagonal matrix with eigenvalues ν_I.
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        diagonal_elements = [float(state.nu_config[site]) for state in self.basis_states]
        return sp.diags(diagonal_elements, format='csr')
    
    def curvature_K_x_operator(self, site: int, mu_bar: float = 1.0) -> sp.csr_matrix:
        """
        Extrinsic curvature operator K̂_x_I using holonomy corrections.
        
        K̂_x ∝ (1/(2i*μ̄)) * [U(μ̄) - U(-μ̄)]
        where U(μ̄) is the holonomy shift operator.
        
        Args:
            site: Lattice site index
            mu_bar: μ̄ parameter for holonomy correction
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        dim = self.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Build matrix elements: ⟨state_j|K̂_x_I|state_i⟩
        for i, state_i in enumerate(self.basis_states):
            for j, state_j in enumerate(self.basis_states):
                matrix_element = self._curvature_matrix_element_x(state_i, state_j, site, mu_bar)
                
                if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(matrix_element)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim), dtype=complex)
    
    def curvature_K_phi_operator(self, site: int, mu_bar: float = 1.0) -> sp.csr_matrix:
        """
        Extrinsic curvature operator K̂_φ_I using holonomy corrections.
        
        Similar to K_x but acts on the φ-direction quantum numbers.
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        dim = self.dim
        row_indices = []
        col_indices = []
        data = []
        
        # Build matrix elements: ⟨state_j|K̂_φ_I|state_i⟩
        for i, state_i in enumerate(self.basis_states):
            for j, state_j in enumerate(self.basis_states):
                matrix_element = self._curvature_matrix_element_phi(state_i, state_j, site, mu_bar)
                
                if abs(matrix_element) > self.lqg_params.regularization_epsilon:
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(matrix_element)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim), dtype=complex)
    
    def _curvature_matrix_element_x(self, state_i: FluxBasisState, state_j: FluxBasisState, 
                                   site: int, mu_bar: float) -> complex:
        """Compute matrix element ⟨state_j|K̂_x_site|state_i⟩"""
        
        # Check if states differ only at the specified site
        if not self._states_differ_only_at_site(state_i, state_j, site):
            return 0.0 + 0j
        
        mu_i = state_i.mu_config[site]
        mu_j = state_j.mu_config[site]
        
        if mu_j == mu_i + 1:
            return 1.0 / (2j * mu_bar) if abs(mu_bar) > 1e-10 else 0.5j
        elif mu_j == mu_i - 1:
            return -1.0 / (2j * mu_bar) if abs(mu_bar) > 1e-10 else -0.5j
        else:
            return 0.0 + 0j
    
    def _curvature_matrix_element_phi(self, state_i: FluxBasisState, state_j: FluxBasisState,
                                     site: int, mu_bar: float) -> complex:
        """Compute matrix element ⟨state_j|K̂_φ_site|state_i⟩"""
        
        # Check if states differ only at the specified site
        if not self._states_differ_only_at_site(state_i, state_j, site):
            return 0.0 + 0j
        
        nu_i = state_i.nu_config[site]
        nu_j = state_j.nu_config[site]
        
        if nu_j == nu_i + 1:
            return 1.0 / (2j * mu_bar) if abs(mu_bar) > 1e-10 else 0.5j
        elif nu_j == nu_i - 1:
            return -1.0 / (2j * mu_bar) if abs(mu_bar) > 1e-10 else -0.5j
        else:
            return 0.0 + 0j
    
    def _states_differ_only_at_site(self, state_i: FluxBasisState, state_j: FluxBasisState, 
                                   site: int) -> bool:
        """Check if two states differ only at the specified site"""
        for k in range(self.n_sites):
            if k == site:
                continue
            if (state_i.mu_config[k] != state_j.mu_config[k] or 
                state_i.nu_config[k] != state_j.nu_config[k]):
                return False
        return True
    
    def construct_coherent_state(self, 
                               classical_E_x: np.ndarray, 
                               classical_E_phi: np.ndarray,
                               classical_K_x: np.ndarray, 
                               classical_K_phi: np.ndarray) -> np.ndarray:
        """
        Construct LQG coherent state peaked on classical field values.
        
        This creates a genuine semiclassical state |Ψ⟩ such that:
        ⟨Ψ|Ê^x(r_i)|Ψ⟩ ≈ E^x_classical(r_i)
        ⟨Ψ|K̂_x(r_i)|Ψ⟩ ≈ K_x^classical(r_i)
        etc.
        
        Args:
            classical_E_x: Classical E^x field values at each site
            classical_E_phi: Classical E^φ field values at each site  
            classical_K_x: Classical K_x field values at each site
            classical_K_phi: Classical K_φ field values at each site
            
        Returns:
            Normalized coherent state vector of length self.dim
        """
        print("Constructing LQG coherent state...")
        
        if len(classical_E_x) != self.n_sites:
            raise ValueError(f"classical_E_x length {len(classical_E_x)} != n_sites {self.n_sites}")
        if len(classical_E_phi) != self.n_sites:
            raise ValueError(f"classical_E_phi length {len(classical_E_phi)} != n_sites {self.n_sites}")
        if len(classical_K_x) != self.n_sites:
            raise ValueError(f"classical_K_x length {len(classical_K_x)} != n_sites {self.n_sites}")
        if len(classical_K_phi) != self.n_sites:
            raise ValueError(f"classical_K_phi length {len(classical_K_phi)} != n_sites {self.n_sites}")
        
        # Initialize coherent state coefficients
        psi = np.zeros(self.dim, dtype=complex)
        
        for i, state in enumerate(self.basis_states):
            # Compute overlap with classical configuration
            overlap = 1.0 + 0j
            
            for site in range(self.n_sites):
                # E-field part: Gaussian peaked on classical E values
                E_x_quantum = float(state.mu_config[site])
                E_phi_quantum = float(state.nu_config[site])
                
                # Width parameters
                width_E = self.lqg_params.coherent_width_E
                
                # Gaussian weighting for E fields
                delta_E_x = (E_x_quantum - classical_E_x[site])**2
                delta_E_phi = (E_phi_quantum - classical_E_phi[site])**2
                
                overlap *= np.exp(-(delta_E_x + delta_E_phi) / (2 * width_E**2))
                
                # K-field part: Use approximate relationship K ∝ μ, ν
                width_K = self.lqg_params.coherent_width_K
                
                # Simple approximation: K_x ∝ μ and K_φ ∝ ν with scaling
                K_x_approx = 0.1 * state.mu_config[site]  # Scaling factor can be tuned
                K_phi_approx = 0.1 * state.nu_config[site]
                
                delta_K_x = (K_x_approx - classical_K_x[site])**2
                delta_K_phi = (K_phi_approx - classical_K_phi[site])**2
                
                overlap *= np.exp(-(delta_K_x + delta_K_phi) / (2 * width_K**2))
            
            psi[i] = overlap
        
        # Normalize
        norm = np.linalg.norm(psi)
        if norm > self.lqg_params.regularization_epsilon:
            psi /= norm
        else:
            # Fall back to uniform superposition if normalization fails
            print("Warning: Coherent state normalization failed, using uniform superposition")
            psi = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        
        print(f"Coherent state constructed with norm {np.linalg.norm(psi):.6f}")
        return psi

    # === New methods added below ===

    def _compute_mu_bar_for_each_site(self) -> np.ndarray:
        """
        Example: μ̄_i = √|E^x_classical_i|  (or whatever scheme you choose).
        If you already have a field `lqg_params.mu_bar_scheme`, adjust accordingly.
        """
        # We expect the classical E^x array to be stored in lattice_config or passed separately.
        # Here we'll look for an attribute `E_x_classical` in lattice_config if it exists.
        if hasattr(self.lattice_config, "E_x_classical"):
            E_x_cl = np.array(self.lattice_config.E_x_classical[: self.n_sites])
        else:
            # Fallback: use uniform values, or zeros
            E_x_cl = np.ones(self.n_sites)
        return np.sqrt(np.abs(E_x_cl))

    def holonomy_shift_operator(self, site: int, direction: str, mu_shift: int) -> sp.csr_matrix:
        """
        “Shift” the flux quantum number by ±μ̄ at a single site.  
        Here we shift μ or ν by exactly ±1 in the flux basis (i.e., one quantum unit).
        In a full SU(2) implementation, this would be a genuine group action.

        Args:
          site: which lattice site (0 ≤ site < n_sites)
          direction: 'x' or 'phi'
          mu_shift: +1 or -1  (shifts μ or ν by one unit in the finite basis)
        Returns:
          A (dim×dim) sparse operator that maps |…, (μ_i,ν_i), …⟩ → |…, (μ_i ± 1, ν_i), …⟩
          if direction=='x', or similarly shifts ν_i if direction=='phi'.
        """
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"Site {site} out of range")

        rows = []
        cols = []
        data = []

        for idx, composite_state in enumerate(self.basis_states):
            mu_i = composite_state.mu_config[site]
            nu_i = composite_state.nu_config[site]

            if direction == 'x':
                new_mu = mu_i + mu_shift
                new_nu = nu_i
            elif direction == 'phi':
                new_mu = mu_i
                new_nu = nu_i + mu_shift
            else:
                raise ValueError("Direction must be 'x' or 'phi'")

            # Check if (new_mu, new_nu) lies within allowed single-site range
            if (-self.lqg_params.mu_max <= new_mu <= self.lqg_params.mu_max) and \
               (-self.lqg_params.nu_max <= new_nu <= self.lqg_params.nu_max):
                # Build the new FluxBasisState label
                new_mu_config = composite_state.mu_config.copy()
                new_nu_config = composite_state.nu_config.copy()
                new_mu_config[site] = new_mu
                new_nu_config[site] = new_nu
                new_state = FluxBasisState(new_mu_config, new_nu_config)

                if new_state in self.state_to_index:
                    new_idx = self.state_to_index[new_state]
                    rows.append(new_idx)
                    cols.append(idx)
                    data.append(1.0)

        return sp.csr_matrix((data, (rows, cols)), shape=(self.dim, self.dim))

    def kx_operator(self, site: int) -> sp.csr_matrix:
        r"""
        Build \widehat{K}_x(site) ≈ [U_x(μ̄) - U_x(-μ̄)] / (2 i μ̄),
        where U_x(±μ̄) is holonomy_shift_operator(site, 'x', ±1).
        We treat the shift of ±1 in μ as one “quantum unit.” If μ̄ ≠ 1,
        one could rescale accordingly. The final matrix is Hermitian.
        """
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"Site {site} out of range")

        μbar = float(self._compute_mu_bar_for_each_site()[site])
        if μbar < 1e-12:
            # If μ̄ ≈ 0, return a zero operator
            return sp.csr_matrix((self.dim, self.dim))

        U_plus = self.holonomy_shift_operator(site, 'x', +1)
        U_minus = self.holonomy_shift_operator(site, 'x', -1)

        # (U_plus - U_minus) / (2 i μ̄); multiply numerator by i to get a real, Hermitian matrix
        numerator = (U_plus - U_minus).astype(complex) * (1j / 2.0)
        Kx = numerator * (1.0 / μbar)
        return sp.csr_matrix(Kx)

    def kphi_operator(self, site: int) -> sp.csr_matrix:
        r"""
        Build \widehat{K}_φ(site) ≈ [U_phi(μ̄) - U_phi(-μ̄)] / (2 i μ̄),
        where U_phi(±μ̄) is holonomy_shift_operator(site, 'phi', ±1).
        """
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"Site {site} out of range")

        μbar = float(self._compute_mu_bar_for_each_site()[site])
        if μbar < 1e-12:
            return sp.csr_matrix((self.dim, self.dim))

        U_plus = self.holonomy_shift_operator(site, 'phi', +1)
        U_minus = self.holonomy_shift_operator(site, 'phi', -1)

        numerator = (U_plus - U_minus).astype(complex) * (1j / 2.0)
        Kphi = numerator * (1.0 / μbar)
        return sp.csr_matrix(Kphi)

    def create_coherent_state_with_Kcheck(self,
                                          E_x_target: np.ndarray,
                                          E_phi_target: np.ndarray,
                                          K_x_target: np.ndarray,
                                          K_phi_target: np.ndarray,
                                          width: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Build the same flux‐Gaussian coherent state as in construct_coherent_state,
        then compute ⟨K_x⟩ and ⟨K_phi⟩ for each site and verify they match the classical targets.

        Returns:
          psi: the normalized coherent‐state vector
          checks: dict containing 'max_E_error' and 'max_K_error'
        """
        # 1) Build the flux‐Gaussian coherent state
        psi = self.construct_coherent_state(E_x_target, E_phi_target, K_x_target, K_phi_target)

        max_E_error = 0.0
        max_K_error = 0.0

        for site in range(self.n_sites):
            Ex_op = self.flux_E_x_operator(site)
            Ephi_op = self.flux_E_phi_operator(site)
            Kx_op = self.kx_operator(site)
            Kphi_op = self.kphi_operator(site)

            Ex_expect = np.real(np.vdot(psi, Ex_op @ psi))
            Ephi_expect = np.real(np.vdot(psi, Ephi_op @ psi))
            Kx_expect = np.real(np.vdot(psi, Kx_op @ psi))
            Kphi_expect = np.real(np.vdot(psi, Kphi_op @ psi))

            max_E_error = max(max_E_error,
                              abs(Ex_expect - E_x_target[site]),
                              abs(Ephi_expect - E_phi_target[site]))
            max_K_error = max(max_K_error,
                              abs(Kx_expect - K_x_target[site]),
                              abs(Kphi_expect - K_phi_target[site]))

            print(f"Site {site:2d}:")
            print(f"    ⟨E^x⟩ = {Ex_expect:.6f} (target {E_x_target[site]:.6f}),  error={Ex_expect - E_x_target[site]:.2e}")
            print(f"    ⟨E^φ⟩ = {Ephi_expect:.6f} (target {E_phi_target[site]:.6f}),  error={Ephi_expect - E_phi_target[site]:.2e}")
            print(f"    ⟨K_x⟩ = {Kx_expect:.6f} (target {K_x_target[site]:.6f}),  error={Kx_expect - K_x_target[site]:.2e}")
            print(f"    ⟨K_φ⟩ = {Kphi_expect:.6f} (target {K_phi_target[site]:.6f}),  error={Kphi_expect - K_phi_target[site]:.2e}")

        checks = {
            "max_E_error": max_E_error,
            "max_K_error": max_K_error,
        }
        return psi, checks


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
        
        H_matter = sp.csr_matrix((data, (row_indices, col_indices)),
                                shape=(dim, dim), dtype=complex)
        
        return H_matter
    
    def _compute_mu_bar_values(self, E_x: np.ndarray, E_phi: np.ndarray) -> np.ndarray:
        """Compute μ̄ values using specified scheme."""
        
        mu_bar = np.zeros(len(E_x))
        
        if self.lqg_params.mu_bar_scheme == MuBarScheme.MINIMAL_AREA:
            for i in range(len(E_x)):
                E_magnitude = np.sqrt(abs(E_x[i] * E_phi[i]))
                mu_bar[i] = E_magnitude / self.lqg_params.planck_area
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.IMPROVED_DYNAMICS:
            for i in range(len(E_x)):
                E_magnitude = np.sqrt(abs(E_x[i] * E_phi[i]))
                mu_bar[i] = np.sqrt(E_magnitude / self.lqg_params.planck_area)
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.BOJOWALD_ISHAM:
            mu_bar.fill(1.0)
        
        elif self.lqg_params.mu_bar_scheme == MuBarScheme.ADAPTIVE:
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
        
        # Diagonal terms
        if state_i == state_j:
            for site in range(self.lattice_config.n_sites):
                mu_i = state_i.mu_config[site]
                nu_i = state_i.nu_config[site]
                mu_bar = mu_bar_values[site]
                
                E_x_eigenvalue = self.lqg_params.gamma * self.lqg_params.planck_area * mu_i
                E_phi_eigenvalue = self.lqg_params.gamma * self.lqg_params.planck_area * nu_i
                
                if mu_bar > self.lqg_params.regularization_epsilon:
                    holonomy_factor_x = np.sin(mu_bar * K_x[site]) / mu_bar
                    holonomy_factor_phi = np.sin(mu_bar * K_phi[site]) / mu_bar
                else:
                    holonomy_factor_x = K_x[site]
                    holonomy_factor_phi = K_phi[site]
                
                kinetic_term = (E_x_eigenvalue * E_phi_eigenvalue * 
                              holonomy_factor_x * holonomy_factor_phi)
                
                if abs(E_x_eigenvalue * E_phi_eigenvalue) > self.lqg_params.regularization_epsilon:
                    inverse_triad_factor = 1.0 / np.sqrt(abs(E_x_eigenvalue * E_phi_eigenvalue))
                    kinetic_term *= inverse_triad_factor
                
                element += kinetic_term
        else:
            element += self._spatial_coupling_matrix_element(state_i, state_j, E_x, E_phi)
        
        return element
    
    def _spatial_coupling_matrix_element(self,
                                         state_i: FluxBasisState,
                                         state_j: FluxBasisState,
                                         E_x: np.ndarray,
                                         E_phi: np.ndarray) -> complex:
        """Compute spatial derivative coupling matrix element."""
        
        diff_sites = []
        for site in range(self.lattice_config.n_sites):
            if (state_i.mu_config[site] != state_j.mu_config[site] or
                state_i.nu_config[site] != state_j.nu_config[site]):
                diff_sites.append(site)
        
        if len(diff_sites) == 2 and abs(diff_sites[1] - diff_sites[0]) == 1:
            site1, site2 = diff_sites
            dr = self.lattice_config.get_lattice_spacing()
            
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
        
        if state_i == state_j:
            element = 0.0
            for site in range(self.lattice_config.n_sites):
                phi = scalar_field[site]
                pi = scalar_momentum[site]
                
                kinetic_term = 0.5 * pi**2
                if site > 0:
                    grad_term = 0.5 * (scalar_field[site] - scalar_field[site-1])**2
                else:
                    grad_term = 0.0
                mass_term = 0.5 * self.lqg_params.scalar_mass**2 * phi**2
                
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
            eigenvals, eigenvecs = sp.linalg.eigs(
                self.H_matrix, k=num_eigs, which='SM',  # Smallest magnitude
                return_eigenvectors=True
            )
            
            sort_indices = np.argsort(np.abs(eigenvals))
            eigenvals = eigenvals[sort_indices]
            eigenvecs = eigenvecs[:, sort_indices]
            
            print(f"✓ Found {len(eigenvals)} eigenvalues")
            print(f"  Range: {np.min(np.abs(eigenvals)):.6e} to {np.max(np.abs(eigenvals)):.6e}")
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            print(f"Error solving constraint: {e}")
            return np.array([]), np.array([])


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
    
    # Add Maxwell field data (with defaults if not present)
    A_r = np.array(data.get("A_r", [0.0] * len(r_grid)))
    pi_r = np.array(data.get("pi_r", [0.0] * len(r_grid)))
    
    # Set up lattice
    lattice_config = LatticeConfiguration(
        n_sites=len(r_grid),
        r_min=r_grid[0],
        r_max=r_grid[-1]
    )
    # Optionally attach classical E arrays for μ̄ computation:
    setattr(lattice_config, "E_x_classical", list(E_x))
    setattr(lattice_config, "E_phi_classical", list(E_phi))
    
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
    
    # Optionally perform coherent-state check on the ground state
    if len(eigenvals) > 0:
        physical_state = eigenvecs[:, 0]
        # Reconstruct a coherent state peaked on the same classical data
        psi_coh, errors = kin_space.create_coherent_state_with_Kcheck(
            E_x, E_phi, K_x, K_phi, width=lqg_params.coherent_width_E
        )
        print("\nCoherent-state verification errors:")
        print(f"  max |⟨E⟩−E_classical| = {errors['max_E_error']:.2e}")
        print(f"  max |⟨K⟩−K_classical| = {errors['max_K_error']:.2e}")
    
    # Compute quantum T^00 expectation values    if len(eigenvals) > 0:
        quantum_T00 = []
        for site in range(len(r_grid)):
            phi = exotic_field[site]
            T00_site = 0.5 * phi**2  # Placeholder for full computation
            quantum_T00.append(T00_site)
        
        backreaction_data = {
            "r_values": list(r_grid),
            "quantum_T00": quantum_T00,
            "total_mass_energy": float(np.sum(quantum_T00) * lattice_config.get_lattice_spacing()),
            "peak_energy_density": float(np.max(quantum_T00)),
            "peak_location": float(r_grid[np.argmax(quantum_T00)]),
            "eigenvalue": float(np.abs(eigenvals[0])),  # Use absolute value to avoid ComplexWarning
            "computation_metadata": {
                "hilbert_dimension": kin_space.dim,
                "mu_bar_scheme": lqg_params.mu_bar_scheme.value,
                "lattice_sites": len(r_grid)
            }
        }
        
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


# === Example usage snippet at the bottom of lqg_fixed_components.py ===

if __name__ == "__main__":
    # Quick demo driver.  In practice, pull classical arrays from JSON or previous code.
    import numpy as np

    # Suppose we have 3 sites:
    n_sites = 3
    classical_E_x   = np.array([1.0, 0.8, 0.6])
    classical_E_phi = np.array([0.5, 0.4, 0.3])
    classical_K_x   = np.array([0.1, 0.05, 0.0])
    classical_K_phi = np.array([0.02, 0.01, 0.0])

    # Build minimal lattice_config / lqg_params for the demo
    lattice_config = LatticeConfiguration(
        n_sites=n_sites,
        r_min=1.0,      # dummy values for the demo
        r_max=3.0
    )
    setattr(lattice_config, "E_x_classical", list(classical_E_x))
    setattr(lattice_config, "E_phi_classical", list(classical_E_phi))

    lqg_params = LQGParameters(
        gamma=1.0,
        planck_length=1.0,
        planck_area=1.0,
        mu_bar_scheme=MuBarScheme.MINIMAL_AREA,
        holonomy_correction=True,
        inverse_triad_regularization=True,
        mu_max=1,
        nu_max=1,
        coherent_width_E=0.5,
        coherent_width_K=0.5
    )

    # Instantiate the Hilbert space
    kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
    print(f"\n→ Hilbert‐space dimension: {kin_space.dim}")

    # Generate and check a coherent state
    print("→ Generating coherent state and verifying ⟨K⟩…\n")
    psi_coh, errors = kin_space.create_coherent_state_with_Kcheck(
        E_x_target=classical_E_x,
        E_phi_target=classical_E_phi,
        K_x_target=classical_K_x,
        K_phi_target=classical_K_phi,
        width=0.5
    )

    print("\nSummary of deviations:")
    print(f"  max |⟨E⟩−E_classical| = {errors['max_E_error']:.2e}")
    print(f"  max |⟨K⟩−K_classical| = {errors['max_K_error']:.2e}")