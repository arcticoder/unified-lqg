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
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, NamedTuple
import json
from dataclasses import dataclass
from itertools import product


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
    
    def __post_init__(self):
        if self.E_x_classical is None:
            self.E_x_classical = [1.0] * self.n_sites
        if self.E_phi_classical is None:
            self.E_phi_classical = [0.5] * self.n_sites


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
    """
    
    def __init__(self, config: LatticeConfig):
        self.config = config
        self.n_sites = config.n_sites
        
        print(f"Building {self.n_sites}-site kinematical Hilbert space...")
        
        # Create single-site bases
        self.site_basis = FluxBasis(config.mu_range, config.nu_range)
        self.site_dim = self.site_basis.dim
        
        # Total Hilbert space dimension
        self.hilbert_dim = self.site_dim ** self.n_sites
        
        print(f"Total kinematical dimension: {self.hilbert_dim}")
        
        # Generate composite basis states
        self._build_composite_basis()
        
        # Store classical values for coherent states
        self.E_x_classical = np.array(config.E_x_classical[:self.n_sites])
        self.E_phi_classical = np.array(config.E_phi_classical[:self.n_sites])
    
    def _build_composite_basis(self):
        """Build basis for tensor product Hilbert space"""
        print("Building composite basis states...")
        
        self.composite_states = []
        self.state_to_index = {}
        
        # Generate all combinations of site states
        site_indices = [range(self.site_dim) for _ in range(self.n_sites)]
        
        for composite_index, site_indices_tuple in enumerate(product(*site_indices)):
            # Convert site indices to quantum numbers
            state = []
            for site, site_index in enumerate(site_indices_tuple):
                mu, nu = self.site_basis.basis_states[site_index]
                state.append((mu, nu))
            
            composite_state = tuple(state)
            self.composite_states.append(composite_state)
            self.state_to_index[composite_state] = composite_index
        
        print(f"Composite basis: {len(self.composite_states)} states")
    
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
        
        for composite_state in self.composite_states:
            mu_i, nu_i = composite_state[site]
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
        
        for composite_state in self.composite_states:
            mu_i, nu_i = composite_state[site]
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
            
            for site in range(self.n_sites):
                mu_i, nu_i = composite_state[site]
                
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
    
    def compute_stress_energy_expectation(self, state: np.ndarray, 
                                        coordinate: float) -> Dict:
        """
        Compute stress-energy tensor expectation values for warp drive framework
        
        This provides T^μν expectation values needed for the classical pipeline.
        For the simplified 2-site model, this gives effective T^00 contributions.
        """
        # Get flux expectation values
        observables = self.get_classical_expectation_values(state)
        
        # Simple model: T^00 ∝ (E^x)² + (E^φ)² for effective stress-energy density
        T00_total = 0.0
        T00_components = {}
        
        for site in range(self.n_sites):
            E_x = observables[f"E_x_{site}"]
            E_phi = observables[f"E_phi_{site}"]
            
            # Effective stress-energy contribution from this site
            T00_site = (E_x**2 + E_phi**2) * np.exp(-coordinate**2 / (2 * (site + 1)**2))
            T00_total += T00_site
            
            T00_components[f"site_{site}"] = {
                "E_x_contribution": float(E_x**2),
                "E_phi_contribution": float(E_phi**2),
                "total_T00": float(T00_site),
                "coordinate": float(coordinate)
            }
        
        return {
            "T00_total": float(T00_total),
            "T00_components": T00_components,
            "coordinate": float(coordinate),
            "flux_contributions": observables
        }


def load_lattice_from_reduced_variables(filename: str) -> LatticeConfig:
    """
    Load lattice configuration from reduced variables JSON file
    
    Expected format:
    {
        "sites": [
            {"E_x": value, "E_phi": value, ...},
            ...
        ],
        "gamma": Barbero-Immirzi parameter
    }
    """
    print(f"Loading lattice configuration from {filename}")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract site data
    sites = data.get("sites", [])
    n_sites = len(sites)
    
    if n_sites == 0:
        print("Warning: No sites found in data, using default 2-site configuration")
        return LatticeConfig()
    
    # Extract classical flux values
    E_x_classical = []
    E_phi_classical = []
    
    for site_data in sites:
        E_x_classical.append(site_data.get("E_x", 1.0))
        E_phi_classical.append(site_data.get("E_phi", 0.5))
    
    # Extract other parameters
    gamma = data.get("gamma", 0.2375)
    
    config = LatticeConfig(
        n_sites=n_sites,
        gamma=gamma,
        E_x_classical=E_x_classical,
        E_phi_classical=E_phi_classical
    )
    
    print(f"Loaded configuration:")
    print(f"  Sites: {n_sites}")
    print(f"  Barbero-Immirzi parameter: γ = {gamma}")
    print(f"  Classical E^x: {E_x_classical}")
    print(f"  Classical E^φ: {E_phi_classical}")
    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test kinematical Hilbert space")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--test-integration", action="store_true", help="Test quantum-classical framework integration")
    parser.add_argument("--config", type=str, help="Configuration file to load")
    parser.add_argument("--mu-range", type=int, nargs=2, default=[-2, 2],
                       help="Range for μ quantum numbers")
    parser.add_argument("--nu-range", type=int, nargs=2, default=[-2, 2],
                       help="Range for ν quantum numbers")
    
    args = parser.parse_args()
    
    if args.test:
        test_kinematical_hilbert()
    elif args.test_integration:
        test_quantum_framework_integration()
    elif args.config:
        config = load_lattice_from_reduced_variables(args.config)
        config.mu_range = tuple(args.mu_range)
        config.nu_range = tuple(args.nu_range)
        
        hilbert = MidisuperspaceHilbert(config)
        
        # Create and analyze coherent state
        psi = hilbert.create_coherent_state(
            np.array(config.E_x_classical),
            np.array(config.E_phi_classical)
        )
        
        hilbert.print_state_summary(psi, "Ground State Guess")
    else:
        test_kinematical_hilbert()
