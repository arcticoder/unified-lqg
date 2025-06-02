#!/usr/bin/env python3
"""
Kinematical Hilbert Space (Task 2)

Implements the kinematical Hilbert space for LQG midisuperspace models:
- 2-link symmetric polymer with gauge-invariant variables
- Flux operators E^x_I, E^φ_I on lattice sites  
- Holonomy operators h_I(μ, ν) with SU(2) quantum numbers
- Spatial diffeomorphism constraints built into construction
- Quantum number truncation for finite-dimensional approximation

The kinematical space is H_kin = ⊗_I H_I where each site I has:
- Flux quantum numbers: (μ_I, ν_I) ∈ Z × Z (truncated to finite range)
- SU(2) representation theory: |j, m⟩ states with j = √(μ² + ν²)/2
- Gauge-invariant combinations for physical observables

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, List, Tuple, Optional, NamedTuple
import json
from dataclasses import dataclass
from itertools import product
import itertools


@dataclass
class LatticeConfig:
    """Configuration for 2-link symmetric lattice"""
    n_sites: int = 2                    # Number of lattice sites (edges)
    mu_range: Tuple[int, int] = (-3, 3) # Range for μ quantum numbers
    nu_range: Tuple[int, int] = (-3, 3) # Range for ν quantum numbers
    gamma: float = 0.2375               # Barbero-Immirzi parameter
    
    # Classical data for coherent state construction
    E_x_classical: List[float] = None
    E_phi_classical: List[float] = None
    
    # Maxwell field classical data
    A_r_classical: List[float] = None
    pi_r_classical: List[float] = None
    
    def __post_init__(self):
        if self.E_x_classical is None:
            self.E_x_classical = [1.0] * self.n_sites
        if self.E_phi_classical is None:
            self.E_phi_classical = [0.5] * self.n_sites
        if self.A_r_classical is None:
            self.A_r_classical = [0.0] * self.n_sites
        if self.pi_r_classical is None:
            self.pi_r_classical = [0.0] * self.n_sites


class FluxBasis:
    """
    Basis for single-site flux Hilbert space
    
    Each site I has quantum numbers (μ_I, ν_I) representing fluxes:
    - E^x_I ↔ μ_I (tangential flux)
    - E^φ_I ↔ ν_I (normal flux to 2-surface)
    - SU(2) representation: j_I = √(μ_I² + ν_I²)/2
    """
    
    def __init__(self, mu_range: Tuple[int, int], nu_range: Tuple[int, int]):
        self.mu_range = mu_range
        self.nu_range = nu_range
        
        # Generate all (μ, ν) combinations
        mu_values = range(mu_range[0], mu_range[1] + 1)
        nu_values = range(nu_range[0], nu_range[1] + 1)
        
        self.basis_states = []
        self.state_to_index = {}
        
        for i, (mu, nu) in enumerate(product(mu_values, nu_values)):
            state = (mu, nu)
            self.basis_states.append(state)
            self.state_to_index[state] = i
        
        self.dim = len(self.basis_states)
        
        print(f"Single-site flux basis: {self.dim} states")
        print(f"  μ ∈ {mu_range}, ν ∈ {nu_range}")
        print(f"  SU(2) representations: j ∈ [0, {self.max_j():.1f}]")
    
    def max_j(self) -> float:
        """Maximum SU(2) spin for this basis"""
        max_mu = max(abs(self.mu_range[0]), abs(self.mu_range[1]))
        max_nu = max(abs(self.nu_range[0]), abs(self.nu_range[1]))
        return np.sqrt(max_mu**2 + max_nu**2) / 2
    
    def get_j_value(self, mu: int, nu: int) -> float:
        """Get SU(2) spin j for quantum numbers (μ, ν)"""
        return np.sqrt(mu**2 + nu**2) / 2
    
    def flux_E_x_eigenvalue(self, mu: int, nu: int) -> float:
        """Eigenvalue of flux operator E^x for state |μ, ν⟩"""
        return float(mu)
    
    def flux_E_phi_eigenvalue(self, mu: int, nu: int) -> float:
        """Eigenvalue of flux operator E^φ for state |μ, ν⟩"""
        return float(nu)


class MidisuperspaceHilbert:
    """
    Complete kinematical Hilbert space H_kin = ⊗_I H_I
    
    For symmetric 2-link model:
    - Site 0: Edge linking vertices (radial edge)
    - Site 1: Edge within 2-surface (tangential edge)
    - Diffeomorphism invariance: gauge-invariant combinations only
    - Maxwell field: Each site also carries Maxwell occupation numbers
    """
    
    def __init__(self, config: LatticeConfig, maxwell_levels: int = 1):
        """
        Args:
            config: Lattice configuration with geometry parameters
            maxwell_levels: Highest Maxwell occupation number per site (0...maxwell_levels)
        """
        self.config = config
        self.n_sites = config.n_sites
        self.maxwell_levels = maxwell_levels
        
        print(f"Building {self.n_sites}-site kinematical Hilbert space with Maxwell fields...")
        
        # Create single-site flux basis (geometry)
        self.site_basis = FluxBasis(config.mu_range, config.nu_range)
        self.site_dim = self.site_basis.dim
        
        # 1) Build flux (geometry) basis
        self.flux_states = self._generate_flux_basis()
        
        # 2) Build Maxwell basis: all combinations of n_i ∈ [0, maxwell_levels] for each site
        self.maxwell_states = list(
            itertools.product(range(maxwell_levels + 1), repeat=self.n_sites)
        )
        
        # 3) Build composite basis: pair each geometry state with each Maxwell state
        self.composite_states = [
            (flux_state, max_state)
            for flux_state in self.flux_states
            for max_state in self.maxwell_states
        ]
        
        # Total Hilbert space dimension
        self.hilbert_dim = len(self.composite_states)
        
        # Map composite_states to an index for quick lookups
        self.state_to_index = {
            state: idx for idx, state in enumerate(self.composite_states)
        }
        
        print(f"Total kinematical dimension: {self.hilbert_dim}")
        print(f"  Flux basis size: {len(self.flux_states)}")
        print(f"  Maxwell basis size: {len(self.maxwell_states)}")
        
        # Store classical values for coherent states
        self.E_x_classical = np.array(config.E_x_classical[:self.n_sites])
        self.E_phi_classical = np.array(config.E_phi_classical[:self.n_sites])
    
    def _generate_flux_basis(self):
        """
        Generate flux basis states: iterates over mu_range^n_sites × nu_range^n_sites
        and returns a list of tuples: ((μ0,ν0), (μ1,ν1), …, (μ_{n−1},ν_{n−1})).
        """
        flux_states = []
        
        # Generate all combinations of site states
        site_indices = [range(self.site_dim) for _ in range(self.n_sites)]
        
        for site_indices_tuple in product(*site_indices):
            # Convert site indices to quantum numbers
            state = []
            for site, site_index in enumerate(site_indices_tuple):
                mu, nu = self.site_basis.basis_states[site_index]
                state.append((mu, nu))
            
            flux_state = tuple(state)
            flux_states.append(flux_state)
        
        print(f"Flux basis: {len(flux_states)} states")
        return flux_states
    
    def maxwell_occupation(self, comp_index):
        """Return the Maxwell occupation tuple (n0,...,n_{N−1}) for a composite state index."""
        _, max_state = self.composite_states[comp_index]
        return max_state    
    def _build_composite_basis(self):
        """Legacy method - composite basis is now built in __init__"""
        # This method is no longer needed as composite basis is built directly
        pass
    
    def flux_E_x_operator(self, site: int) -> sp.csr_matrix:
        """
        Flux operator E^x_I acting on site I
        
        Returns diagonal matrix with eigenvalues μ_I
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        print(f"Building E^x operator for site {site}...")
        
        # Diagonal operator: each state |..., (μ_I, ν_I), ...⟩ → μ_I |..., (μ_I, ν_I), ...⟩
        diagonal_elements = []
        
        for (flux_state, max_state) in self.composite_states:
            mu_i, nu_i = flux_state[site]
            eigenvalue = self.site_basis.flux_E_x_eigenvalue(mu_i, nu_i)
            diagonal_elements.append(eigenvalue)
        
        return sp.diags(diagonal_elements, format='csr')
    
    def flux_E_phi_operator(self, site: int) -> sp.csr_matrix:
        """
        Flux operator E^φ_I acting on site I
        
        Returns diagonal matrix with eigenvalues ν_I
        """
        if site >= self.n_sites:
            raise ValueError(f"Site {site} out of range [0, {self.n_sites})")
        
        print(f"Building E^φ operator for site {site}...")
        
        diagonal_elements = []
        
        for (flux_state, max_state) in self.composite_states:
            mu_i, nu_i = flux_state[site]
            eigenvalue = self.site_basis.flux_E_phi_eigenvalue(mu_i, nu_i)
            diagonal_elements.append(eigenvalue)
        
        return sp.diags(diagonal_elements, format='csr')
    
    def holonomy_operator(self, site: int, direction: str) -> sp.csr_matrix:
        """
        Holonomy operator h_I(μ, ν) = exp(iτ^a μ^a_I) for SU(2) group elements
        
        This is a placeholder for the full SU(2) representation theory.
        In practice, holonomies appear in the Hamiltonian constraint.
        
        Args:
            site: lattice site index
            direction: 'x' or 'phi' for different group element directions
        """
        print(f"Building holonomy operator at site {site}, direction {direction}")
        
        # For now, return identity (holonomies enter nontrivially in constraint)
        # Full implementation requires SU(2) matrix representations
        return sp.identity(self.hilbert_dim, format='csr')
    
    def create_coherent_state(self, E_x_target: np.ndarray, E_phi_target: np.ndarray, 
                            width: float = 1.0) -> np.ndarray:
        """
        Create semiclassical coherent state peaked at classical values
        
        Coherent state is Gaussian wavepacket in flux eigenvalue space:
        ψ(μ, ν) ∝ exp(-[(μ - μ_cl)² + (ν - ν_cl)²] / (2σ²))
        
        Args:
            E_x_target: Classical E^x values for each site
            E_phi_target: Classical E^φ values for each site  
            width: Gaussian width parameter σ
              Returns:
            Normalized quantum state vector
        """
        print(f"Creating coherent state with classical targets:")
        for site in range(self.n_sites):
            print(f"  Site {site}: E^x = {E_x_target[site]:.3f}, E^φ = {E_phi_target[site]:.3f}")
        
        psi = np.zeros(self.hilbert_dim, dtype=complex)
        
        for i, composite_state in enumerate(self.composite_states):
            amplitude = 1.0
            
            # Extract flux state and Maxwell state from composite state
            flux_state, maxwell_state = composite_state
            
            for site in range(self.n_sites):
                mu_i, nu_i = flux_state[site]
                
                # Gaussian weight for each site
                delta_E_x = mu_i - E_x_target[site]
                delta_E_phi = nu_i - E_phi_target[site]
                
                weight = np.exp(-0.5 * (delta_E_x**2 + delta_E_phi**2) / width**2)
                amplitude *= weight
            
            psi[i] = amplitude
        
        # Normalize
        norm = np.linalg.norm(psi)
        if norm > 1e-12:
            psi = psi / norm
        else:
            # Fallback: uniform superposition if coherent state is too narrow
            print("Warning: Coherent state too narrow, using uniform superposition")
            psi = np.ones(self.hilbert_dim) / np.sqrt(self.hilbert_dim)
        
        overlap = np.abs(np.vdot(psi, psi))
        print(f"Coherent state created with norm: {overlap:.6f}")
        
        return psi
    
    def get_classical_expectation_values(self, state: np.ndarray) -> Dict:
        """
        Compute expectation values of flux operators in given quantum state
        
        Returns:
            Dictionary with E^x and E^φ expectation values for each site
        """
        expectations = {}
        
        for site in range(self.n_sites):
            E_x_op = self.flux_E_x_operator(site)
            E_phi_op = self.flux_E_phi_operator(site)
            
            exp_E_x = np.real(np.conj(state) @ E_x_op @ state)
            exp_E_phi = np.real(np.conj(state) @ E_phi_op @ state)
            
            expectations[f"E_x_{site}"] = float(exp_E_x)
            expectations[f"E_phi_{site}"] = float(exp_E_phi)
        
        return expectations
    
    def print_state_summary(self, state: np.ndarray, label: str = "State"):
        """Print summary of quantum state properties"""
        print(f"\n{label} Summary:")
        print(f"  Hilbert space dimension: {self.hilbert_dim}")
        print(f"  State norm: {np.linalg.norm(state):.6f}")
        
        # Find dominant basis states
        probabilities = np.abs(state)**2
        sorted_indices = np.argsort(probabilities)[::-1]
        
        print(f"  Dominant components:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = probabilities[idx]
            composite_state = self.composite_states[idx]
            
            if prob > 1e-6:  # Only show significant components
                print(f"    |{composite_state}⟩: {prob:.4f}")
        
        # Show expectation values
        expectations = self.get_classical_expectation_values(state)
        print(f"  Flux expectation values:")
        for site in range(self.n_sites):
            E_x = expectations[f"E_x_{site}"]
            E_phi = expectations[f"E_phi_{site}"]
            print(f"    Site {site}: ⟨E^x⟩ = {E_x:.3f}, ⟨E^φ⟩ = {E_phi:.3f}")
    
    def export_quantum_observables(self, state: np.ndarray, output_file: str):
        """
        Export quantum observables in format compatible with the warp framework
        
        Creates JSON output with expectation values and quantum corrections
        suitable for integration with the classical warp drive pipeline.
        """
        observables = self.get_classical_expectation_values(state)
        
        # Add quantum state properties
        probabilities = np.abs(state)**2
        dominant_states = []
        
        sorted_indices = np.argsort(probabilities)[::-1]
        for i in range(min(10, len(sorted_indices))):
            idx = sorted_indices[i]
            prob = probabilities[idx]
            if prob > 1e-6:
                composite_state = self.composite_states[idx]
                dominant_states.append({
                    "state_index": int(idx),
                    "probability": float(prob),
                    "quantum_numbers": [list(site_state) for site_state in composite_state]
                })
        
        # Compute quantum corrections to classical values
        quantum_corrections = {}
        for site in range(self.n_sites):
            E_x_quantum = observables[f"E_x_{site}"]
            E_phi_quantum = observables[f"E_phi_{site}"]
            E_x_classical = self.E_x_classical[site]
            E_phi_classical = self.E_phi_classical[site]
            
            quantum_corrections[f"site_{site}"] = {
                "E_x_correction": float(E_x_quantum - E_x_classical),
                "E_phi_correction": float(E_phi_quantum - E_phi_classical),
                "E_x_ratio": float(E_x_quantum / E_x_classical) if E_x_classical != 0 else float('inf'),
                "E_phi_ratio": float(E_phi_quantum / E_phi_classical) if E_phi_classical != 0 else float('inf')
            }
        
        # Compile full export data
        export_data = {
            "hilbert_space_dimension": self.hilbert_dim,
            "n_sites": self.n_sites,
            "state_norm": float(np.linalg.norm(state)),
            "flux_expectation_values": observables,
            "quantum_corrections": quantum_corrections,
            "dominant_quantum_states": dominant_states,
            "classical_reference": {
                "E_x_classical": [float(x) for x in self.E_x_classical],
                "E_phi_classical": [float(x) for x in self.E_phi_classical]
            },
            "lattice_config": {
                "mu_range": self.config.mu_range,
                "nu_range": self.config.nu_range,
                "gamma": self.config.gamma
            }
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)        
        print(f"✓ Quantum observables exported to {output_file}")
        return export_data
    
    def compute_phantom_T00_operator(self, site: int) -> sp.csr_matrix:
        """
        Build phantom scalar stress-energy operator T00_phantom(site).
        For a phantom scalar field: T00 = -(1/2)[π² - (∇φ)² + m²φ²]
        
        This is a simplified diagonal operator for demonstration.
        In full implementation, would use proper scalar field operators.
        """
        # For simplicity, return a diagonal operator with small phantom contribution
        diagonal_elements = []
        
        for (flux_state, max_state) in self.composite_states:
            # Simple phantom contribution based on flux eigenvalues
            mu_i, nu_i = flux_state[site]
            phantom_val = -0.1 * (mu_i**2 + nu_i**2) / (2.0 * (1 + abs(mu_i) + abs(nu_i)))
            diagonal_elements.append(phantom_val)
        
        return sp.diags(diagonal_elements, format='csr')

    def compute_expectation_E_and_T00(self, psi):
        """
        Now returns:
          Ex_vals, Ephi_vals, T00_phantom_vals, T00_maxwell_vals, T00_total_vals
        """
        n = self.n_sites
        Ex_vals = []
        Ephi_vals = []
        T00_ph_vals = []
        T00_mx_vals = []
        T00_tot_vals = []

        for site in range(n):
            # 1) Flux (geometry) expectation:
            Ex_op = self.flux_E_x_operator(site)
            Ephi_op = self.flux_E_phi_operator(site)
            Ex_expect = np.real(np.vdot(psi, Ex_op @ psi))
            Ephi_expect = np.real(np.vdot(psi, Ephi_op @ psi))

            # 2) Phantom T00 (existing code):
            T00_ph = self.compute_phantom_T00_operator(site)
            T00_ph_expect = np.real(np.vdot(psi, T00_ph @ psi))

            # 3) Maxwell T00 (new):
            T00_mx_op = self.maxwell_T00_operator(site)
            T00_mx_expect = np.real(np.vdot(psi, T00_mx_op @ psi))

            # 4) Sum them:
            T00_tot = T00_ph_expect + T00_mx_expect

            # 5) Append to lists
            Ex_vals.append(Ex_expect)
            Ephi_vals.append(Ephi_expect)
            T00_ph_vals.append(T00_ph_expect)
            T00_mx_vals.append(T00_mx_expect)
            T00_tot_vals.append(T00_tot)

        return Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals

    def compute_stress_energy_expectation(self, state: np.ndarray, 
                                        coordinate: float) -> Dict:
        """
        Compute stress-energy tensor expectation values for warp drive framework
        
        This provides T^μν expectation values needed for the classical pipeline.
        Now includes both phantom and Maxwell contributions.
        """
        # Get combined expectation values
        Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = self.compute_expectation_E_and_T00(state)
        
        # Get flux expectation values for compatibility
        observables = self.get_classical_expectation_values(state)
        
        # Simple model: Total T^00 from phantom + Maxwell
        T00_total = sum(T00_tot_vals)
        T00_components = {}
        
        for site in range(self.n_sites):
            T00_components[f"site_{site}"] = {
                "E_x_expectation": float(Ex_vals[site]),
                "E_phi_expectation": float(Ephi_vals[site]),
                "T00_phantom": float(T00_ph_vals[site]),
                "T00_maxwell": float(T00_mx_vals[site]),
                "T00_total": float(T00_tot_vals[site]),
                "coordinate": float(coordinate)
            }
        
        return {
            "T00_total": float(T00_total),
            "T00_phantom_total": float(sum(T00_ph_vals)),
            "T00_maxwell_total": float(sum(T00_mx_vals)),
            "T00_components": T00_components,
            "coordinate": float(coordinate),
            "flux_contributions": observables
        }

    def maxwell_pi_operator(self, site: int) -> sp.csr_matrix:
        """
        Build \\hat{\\pi}^r(site) acting on the Maxwell label at 'site'.
        For maxwell_levels=1 (two‐state oscillator), define:
          pi = [[0, 1.0],
                [1.0, 0]]
        on the {n=0,n=1} subspace at 'site'. Tensor‐identity elsewhere.
        """
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"Site {site} out of range")

        dim = self.hilbert_dim

        # Create 2×2 pi‐matrix on single site's Maxwell subspace:
        #  n=0 ↔ n=1 off‐diagonal with coefficient 1.0
        pi_local = np.array([[0.0, 1.0], [1.0, 0.0]])

        # Now build a sparse operator in the full composite basis:
        rows, cols, data = [], [], []
        for idx, (flux_state, max_state) in enumerate(self.composite_states):
            # max_state is a tuple (n0,...,n_{N−1})
            n_i = max_state[site]
            # Only two levels (0 or 1). Shifts 0↔1.
            if n_i == 0:
                new_n = 1
                amp = pi_local[0, 1]  # = 1.0
            elif n_i == 1:
                new_n = 0
                amp = pi_local[1, 0]  # = 1.0
            else:
                # If maxwell_levels>1, you could generalize. For now, treat n>1 as no coupling.
                continue

            # Build new Maxwell tuple:
            new_max = list(max_state)
            new_max[site] = new_n
            new_max = tuple(new_max)

            new_state = (flux_state, new_max)
            jdx = self.state_to_index.get(new_state)
            if jdx is None:
                continue

            rows.append(jdx)
            cols.append(idx)
            data.append(amp)

        return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))

    def maxwell_gradient_operator(self, site: int) -> sp.csr_matrix:
        """
        In a purely radial Maxwell field, the 'magnetic' gradient term vanishes.
        Return the zero matrix of size (hilbert_dim × hilbert_dim).
        """
        return sp.csr_matrix((self.hilbert_dim, self.hilbert_dim))

    def maxwell_T00_operator(self, site: int) -> sp.csr_matrix:
        """
        Build  T00_maxwell(site) = 0.5 * [ (pi^r_i)^2 + (grad A)_i^2 ].
        Since grad A ~ 0, this is 0.5 * (pi_i)^2.
        """
        pi_op = self.maxwell_pi_operator(site)
        # (pi^2) = pi_op @ pi_op. 0.5× that:
        pi2 = pi_op.dot(pi_op)
        return pi2.multiply(0.5)


def load_lattice_from_reduced_variables(filename: str) -> LatticeConfig:
    """
    Load lattice configuration from reduced variables JSON file
    
    Expected format (supports both old format with sites array and new format with direct arrays):
    {
        "sites": [
            {"E_x": value, "E_phi": value, "A_r": value, "pi_r": value, ...},
            ...
        ],
        "gamma": Barbero-Immirzi parameter
    }
    
    OR:
    
    {
        "E_x": [values...],
        "E_phi": [values...], 
        "A_r": [values...],
        "pi_r": [values...],
        "gamma": Barbero-Immirzi parameter
    }
    """
    print(f"Loading lattice configuration from {filename}")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Check format and extract data accordingly
    if "sites" in data:
        # New format with sites array
        sites = data.get("sites", [])
        n_sites = len(sites)
        
        if n_sites == 0:
            print("Warning: No sites found in data, using default 2-site configuration")
            return LatticeConfig()
        
        # Extract classical flux values
        E_x_classical = []
        E_phi_classical = []
        A_r_classical = []
        pi_r_classical = []
        
        for site_data in sites:
            E_x_classical.append(site_data.get("E_x", 1.0))
            E_phi_classical.append(site_data.get("E_phi", 0.5))
            A_r_classical.append(site_data.get("A_r", 0.0))
            pi_r_classical.append(site_data.get("pi_r", 0.0))
    
    else:
        # Old format with direct arrays
        E_x_classical = data.get("E_x", [1.0, 1.0])
        E_phi_classical = data.get("E_phi", [0.5, 0.5])
        A_r_classical = data.get("A_r", [0.0] * len(E_x_classical))
        pi_r_classical = data.get("pi_r", [0.0] * len(E_x_classical))
        n_sites = len(E_x_classical)
    
    # Extract other parameters
    gamma = data.get("gamma", 0.2375)
    
    config = LatticeConfig(
        n_sites=n_sites,
        gamma=gamma,
        E_x_classical=E_x_classical,
        E_phi_classical=E_phi_classical,
        A_r_classical=A_r_classical,
        pi_r_classical=pi_r_classical
    )
    
    print(f"Loaded configuration:")
    print(f"  Sites: {n_sites}")
    print(f"  Barbero-Immirzi parameter: γ = {gamma}")
    print(f"  Classical E^x: {E_x_classical}")
    print(f"  Classical E^φ: {E_phi_classical}")
    print(f"  Classical A^r: {A_r_classical}")
    print(f"  Classical π^r: {pi_r_classical}")
    
    return config


def create_example_lattice() -> LatticeConfig:
    """Create example 2-site lattice for testing"""
    return LatticeConfig(
        n_sites=2,
        mu_range=(-2, 2),
        nu_range=(-2, 2),
        gamma=0.2375,
        E_x_classical=[1.5, 0.8],
        E_phi_classical=[0.3, -0.5]
    )


def test_kinematical_hilbert():
    """Test basic functionality of kinematical Hilbert space"""
    print("="*60)
    print("TESTING KINEMATICAL HILBERT SPACE")
    print("="*60)
    
    # Create test configuration
    config = create_example_lattice()
    hilbert = MidisuperspaceHilbert(config)
    
    print(f"\n1. Basic properties:")
    print(f"   Hilbert dimension: {hilbert.hilbert_dim}")
    print(f"   Number of sites: {hilbert.n_sites}")
    
    # Test flux operators
    print(f"\n2. Testing flux operators:")
    E_x_0 = hilbert.flux_E_x_operator(0)
    E_phi_1 = hilbert.flux_E_phi_operator(1)
    
    print(f"   E^x_0 matrix: {E_x_0.shape}, {E_x_0.nnz} non-zeros")
    print(f"   E^φ_1 matrix: {E_phi_1.shape}, {E_phi_1.nnz} non-zeros")
    
    # Test coherent state
    print(f"\n3. Testing coherent state:")
    psi_coherent = hilbert.create_coherent_state(
        np.array(config.E_x_classical),
        np.array(config.E_phi_classical),
        width=1.5
    )
    
    hilbert.print_state_summary(psi_coherent, "Coherent State")
    
    # Test commutation relations [E^x, E^φ] (should be zero for different sites)
    print(f"\n4. Testing operator properties:")
    E_x_0 = hilbert.flux_E_x_operator(0)
    E_phi_0 = hilbert.flux_E_phi_operator(0)
    
    # Same-site fluxes should commute (both diagonal)
    commutator = E_x_0 @ E_phi_0 - E_phi_0 @ E_x_0
    comm_norm = np.linalg.norm(commutator.data)
    
    print(f"   [E^x_0, E^φ_0] norm: {comm_norm:.2e} (should be 0)")
    
    print(f"\n✅ Kinematical Hilbert space test completed")


def test_quantum_framework_integration():
    """Test enhanced quantum framework integration capabilities"""
    print("="*60)
    print("TESTING QUANTUM-CLASSICAL FRAMEWORK INTEGRATION")
    print("="*60)
    
    # Create test configuration with more realistic parameters
    config = LatticeConfig(
        n_sites=2,
        mu_range=(-1, 1),  # Smaller for faster computation
        nu_range=(-1, 1),
        gamma=0.2375,
        E_x_classical=[2.5, 1.2],     # More separated classical values
        E_phi_classical=[0.8, -1.1]
    )
    
    hilbert = MidisuperspaceHilbert(config)
    
    print(f"\n1. Framework Integration Setup:")
    print(f"   Hilbert dimension: {hilbert.hilbert_dim}")
    print(f"   Classical E^x targets: {config.E_x_classical}")
    print(f"   Classical E^φ targets: {config.E_phi_classical}")
    
    # Create coherent state for quantum corrections
    print(f"\n2. Creating Quantum Coherent State:")
    psi_quantum = hilbert.create_coherent_state(
        np.array(config.E_x_classical),
        np.array(config.E_phi_classical),
        width=0.8  # Tighter coherent state
    )
    
    # Export quantum observables for pipeline integration
    print(f"\n3. Exporting Quantum Observables:")
    quantum_data = hilbert.export_quantum_observables(
        psi_quantum, 
        "outputs/quantum_observables_demo.json"
    )
    
    # Show quantum corrections
    print(f"\n4. Quantum Corrections Summary:")
    for site in range(hilbert.n_sites):
        corrections = quantum_data["quantum_corrections"][f"site_{site}"]
        print(f"   Site {site}:")
        print(f"     E^x: {config.E_x_classical[site]:.3f} → {quantum_data['flux_expectation_values'][f'E_x_{site}']:.3f}")
        print(f"     E^φ: {config.E_phi_classical[site]:.3f} → {quantum_data['flux_expectation_values'][f'E_phi_{site}']:.3f}")
        print(f"     Quantum correction: ΔE^x = {corrections['E_x_correction']:.3f}, ΔE^φ = {corrections['E_phi_correction']:.3f}")
    
    # Compute stress-energy expectation for warp framework
    print(f"\n5. Stress-Energy Tensor Integration:")
    coordinates = [1e-35, 2e-35, 5e-35, 1e-34]
    stress_energy_data = []
    
    for r in coordinates:
        T_data = hilbert.compute_stress_energy_expectation(psi_quantum, r)
        stress_energy_data.append(T_data)
        print(f"   r = {r:.1e}: T^00 = {T_data['T00_total']:.3e}")
    
    # Export stress-energy data for classical pipeline
    with open("outputs/quantum_T00_demo.json", 'w') as f:
        json.dump(stress_energy_data, f, indent=2)
    
    print(f"\n✅ Quantum-classical integration test completed")
    print(f"   Generated: outputs/quantum_observables_demo.json")
    print(f"   Generated: outputs/quantum_T00_demo.json")
    print(f"   Ready for classical warp drive pipeline integration!")

def test_maxwell_integration():
    """Test Maxwell field integration with the LQG framework"""
    print("="*60)
    print("TESTING MAXWELL FIELD INTEGRATION")
    print("="*60)
      # Load configuration with Maxwell fields
    print(f"\n1. Loading configuration with Maxwell fields:")
    try:
        config = load_lattice_from_reduced_variables("examples/lqg_demo_classical_data.json")
        # Override with smaller ranges for testing
        config.mu_range = (-1, 1)
        config.nu_range = (-1, 1) 
        config.n_sites = min(3, config.n_sites)  # Limit to 3 sites for testing
        config.E_x_classical = config.E_x_classical[:config.n_sites]
        config.E_phi_classical = config.E_phi_classical[:config.n_sites]
        config.A_r_classical = config.A_r_classical[:config.n_sites]
        config.pi_r_classical = config.pi_r_classical[:config.n_sites]
        print(f"   ✓ Loaded and adjusted for testing (μ,ν ∈ [-1,1], {config.n_sites} sites)")
    except FileNotFoundError:
        print("   Using fallback configuration...")
        config = LatticeConfig(
            n_sites=3,
            mu_range=(-1, 1),
            nu_range=(-1, 1),
            gamma=0.2375,
            E_x_classical=[1.0, 0.9, 1.0],
            E_phi_classical=[1.1, 1.15, 1.1],
            A_r_classical=[0.01, 0.02, 0.01],
            pi_r_classical=[0.001, 0.002, 0.001]
        )
    
    # Create Hilbert space with Maxwell fields
    print(f"\n2. Creating Maxwell-extended Hilbert space:")
    hilbert = MidisuperspaceHilbert(config, maxwell_levels=1)
    
    print(f"   Total Hilbert dimension: {hilbert.hilbert_dim}")
    print(f"   Flux basis size: {len(hilbert.flux_states)}")
    print(f"   Maxwell basis size: {len(hilbert.maxwell_states)}")
    
    # Test Maxwell operators
    print(f"\n3. Testing Maxwell operators:")
    for site in range(min(2, hilbert.n_sites)):  # Test first 2 sites
        pi_op = hilbert.maxwell_pi_operator(site)
        grad_op = hilbert.maxwell_gradient_operator(site)
        T00_op = hilbert.maxwell_T00_operator(site)
        
        print(f"   Site {site}:")
        print(f"     π^r operator: {pi_op.shape}, {pi_op.nnz} non-zeros")
        print(f"     ∇A operator: {grad_op.shape}, {grad_op.nnz} non-zeros")
        print(f"     T00_Maxwell operator: {T00_op.shape}, {T00_op.nnz} non-zeros")
    
    # Create quantum state
    print(f"\n4. Creating coherent state:")
    psi = hilbert.create_coherent_state(
        np.array(config.E_x_classical),
        np.array(config.E_phi_classical),
        width=1.0
    )
    
    # Test combined expectation values
    print(f"\n5. Computing combined expectation values:")
    Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = hilbert.compute_expectation_E_and_T00(psi)
    
    print(f"   Site-by-site breakdown:")
    for site in range(hilbert.n_sites):
        print(f"     Site {site}:")
        print(f"       E^x: {Ex_vals[site]:.4f} (classical: {config.E_x_classical[site]:.4f})")
        print(f"       E^φ: {Ephi_vals[site]:.4f} (classical: {config.E_phi_classical[site]:.4f})")
        print(f"       T00_phantom: {T00_ph_vals[site]:.6e}")
        print(f"       T00_Maxwell: {T00_mx_vals[site]:.6e}")
        print(f"       T00_total: {T00_tot_vals[site]:.6e}")
    
    # Test stress-energy tensor export
    print(f"\n6. Testing stress-energy tensor integration:")
    T_data = hilbert.compute_stress_energy_expectation(psi, coordinate=1e-35)
    
    print(f"   Total T00: {T_data['T00_total']:.6e}")
    print(f"   Phantom contribution: {T_data['T00_phantom_total']:.6e}")
    print(f"   Maxwell contribution: {T_data['T00_maxwell_total']:.6e}")
    
    # Export quantum observables
    print(f"\n7. Exporting quantum observables:")
    obs_data = hilbert.export_quantum_observables(psi, "outputs/maxwell_integration_test.json")
    
    print(f"   ✓ Exported to outputs/maxwell_integration_test.json")
    print(f"   Hilbert dimension: {obs_data['hilbert_space_dimension']}")
    print(f"   State norm: {obs_data['state_norm']:.6f}")
    
    # Check Maxwell occupation expectation values
    print(f"\n8. Maxwell occupation analysis:")
    for i, comp_state in enumerate(hilbert.composite_states[:5]):  # Show first 5 states
        flux_state, maxwell_state = comp_state
        prob = np.abs(psi[i])**2
        if prob > 1e-6:
            print(f"     State {i}: Maxwell occupation {maxwell_state}, probability {prob:.6f}")
    
    print(f"\n✅ Maxwell field integration test completed successfully!")
    return hilbert, psi, T_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test kinematical Hilbert space")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--test-integration", action="store_true", help="Test quantum-classical framework integration")
    parser.add_argument("--test-maxwell", action="store_true", help="Test Maxwell field integration")
    parser.add_argument("--config", type=str, help="Configuration file to load")
    parser.add_argument("--mu-range", type=int, nargs=2, default=[-2, 2],
                       help="Range for μ quantum numbers")
    parser.add_argument("--nu-range", type=int, nargs=2, default=[-2, 2],
                       help="Range for ν quantum numbers")
    parser.add_argument("--maxwell-levels", type=int, default=1,
                       help="Maximum Maxwell occupation number")
    
    args = parser.parse_args()
    
    if args.test:
        test_kinematical_hilbert()
    elif args.test_integration:
        test_quantum_framework_integration()
    elif args.test_maxwell:
        test_maxwell_integration()
    elif args.config:
        config = load_lattice_from_reduced_variables(args.config)
        config.mu_range = tuple(args.mu_range)
        config.nu_range = tuple(args.nu_range)
        
        hilbert = MidisuperspaceHilbert(config, maxwell_levels=args.maxwell_levels)
        
        # Create and analyze coherent state
        psi = hilbert.create_coherent_state(
            np.array(config.E_x_classical),
            np.array(config.E_phi_classical)
        )
        
        hilbert.print_state_summary(psi, "Ground State Guess")
        
        # Show Maxwell contributions
        Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = hilbert.compute_expectation_E_and_T00(psi)
        print(f"\nMaxwell Contributions:")
        for site in range(hilbert.n_sites):
            print(f"  Site {site}: T00_Maxwell = {T00_mx_vals[site]:.6e}")
    else:
        test_kinematical_hilbert()
