#!/usr/bin/env python3
"""
Enhanced LQG system with strategic basis generation.
This provides the final fix for coherent state construction issues.
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Set, Optional
import sys
import os

sys.path.append('.')

from lqg_fixed_components import (
    FluxBasisState, LQGParameters, KinematicalHilbertSpace, 
    LatticeConfiguration, MidisuperspaceHamiltonianConstraint
)

class EnhancedKinematicalHilbertSpace(KinematicalHilbertSpace):
    """
    Enhanced KinematicalHilbertSpace with strategic basis generation.
    
    This class ensures that target states for coherent state construction
    are included in the basis, even with truncation.
    """
    
    def __init__(self, lattice_config: LatticeConfiguration, lqg_params: LQGParameters, 
                 target_states: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None):
        """
        Initialize with optional target states to ensure inclusion.
        
        Args:
            lattice_config: Lattice configuration
            lqg_params: LQG parameters
            target_states: List of (mu_config, nu_config) tuples to ensure inclusion
        """
        self.target_states = target_states or []
        super().__init__(lattice_config, lqg_params)
    
    def _generate_basis_states(self) -> List[FluxBasisState]:
        """
        Generate basis states with strategic inclusion of target states.
        
        Strategy:
        1. Include all target states first
        2. Add neighbor states around targets
        3. Fill remaining space with low-energy states (sorted by quantum number magnitude)
        """
        states = []
        states_set = set()
        
        mu_range = list(range(-self.lqg_params.mu_max, self.lqg_params.mu_max + 1))
        nu_range = list(range(-self.lqg_params.nu_max, self.lqg_params.nu_max + 1))
        
        # Step 1: Ensure target states are included
        if self.target_states:
            print(f"Ensuring {len(self.target_states)} target states are included...")
            for mu_config, nu_config in self.target_states:
                # Validate quantum numbers are within bounds
                if (np.abs(mu_config) <= self.lqg_params.mu_max).all() and \
                   (np.abs(nu_config) <= self.lqg_params.nu_max).all():
                    state = FluxBasisState(mu_config, nu_config)
                    state_tuple = (tuple(mu_config), tuple(nu_config))
                    if state_tuple not in states_set:
                        states.append(state)
                        states_set.add(state_tuple)
                        print(f"  ✓ Added target: μ={mu_config}, ν={nu_config}")
                else:
                    print(f"  ✗ Target out of bounds: μ={mu_config}, ν={nu_config}")
        
        # Step 2: Add neighbor states around targets
        if self.target_states:
            neighbors_added = 0
            for mu_config, nu_config in self.target_states:
                # Add states with small perturbations to each site
                for delta_mu in [-1, 0, 1]:
                    for delta_nu in [-1, 0, 1]:
                        for site in range(self.lattice_config.n_sites):
                            if len(states) >= self.lqg_params.basis_truncation:
                                break
                            
                            # Perturb one site
                            new_mu = mu_config.copy()
                            new_nu = nu_config.copy()
                            
                            new_mu[site] = np.clip(new_mu[site] + delta_mu, 
                                                 -self.lqg_params.mu_max, self.lqg_params.mu_max)
                            new_nu[site] = np.clip(new_nu[site] + delta_nu,
                                                 -self.lqg_params.nu_max, self.lqg_params.nu_max)
                            
                            state_tuple = (tuple(new_mu), tuple(new_nu))
                            if state_tuple not in states_set:
                                states.append(FluxBasisState(new_mu, new_nu))
                                states_set.add(state_tuple)
                                neighbors_added += 1
            
            if neighbors_added > 0:
                print(f"  Added {neighbors_added} neighbor states around targets")
        
        # Step 3: Fill remaining space with low-energy states
        if len(states) < self.lqg_params.basis_truncation:
            print(f"Filling remaining basis space ({len(states)}/{self.lqg_params.basis_truncation})...")
            
            # Create list of all possible states with their "energy" (sum of squares)
            energy_priority = []
            for mu_tuple in product(mu_range, repeat=self.lattice_config.n_sites):
                for nu_tuple in product(nu_range, repeat=self.lattice_config.n_sites):
                    state_tuple = (mu_tuple, nu_tuple)
                    if state_tuple not in states_set:
                        # Energy estimate: sum of squares of quantum numbers
                        energy = sum(m**2 for m in mu_tuple) + sum(n**2 for n in nu_tuple)
                        energy_priority.append((energy, mu_tuple, nu_tuple))
            
            # Sort by energy (low energy first) and add states
            energy_priority.sort(key=lambda x: x[0])
            
            for energy, mu_tuple, nu_tuple in energy_priority:
                if len(states) >= self.lqg_params.basis_truncation:
                    print(f"  Truncated to {len(states)} states")
                    break
                    
                state_tuple = (mu_tuple, nu_tuple)
                if state_tuple not in states_set:
                    states.append(FluxBasisState(np.array(mu_tuple), np.array(nu_tuple)))
                    states_set.add(state_tuple)
        
        print(f"Enhanced basis generated: {len(states)} states")
        
        # Verify target states are present
        if self.target_states:
            for i, (target_mu, target_nu) in enumerate(self.target_states):
                found = False
                for j, state in enumerate(states):
                    if (state.mu_config == target_mu).all() and (state.nu_config == target_nu).all():
                        print(f"  ✓ Target {i+1} found at position {j}")
                        found = True
                        break
                if not found:
                    print(f"  ✗ Target {i+1} NOT found in final basis!")
        
        return states

def create_enhanced_lqg_system(classical_data: dict, 
                             basis_size: int = 5000,
                             ensure_target_states: bool = True) -> Tuple[EnhancedKinematicalHilbertSpace, np.ndarray]:
    """
    Create enhanced LQG system with coherent state that actually works.
    
    Args:
        classical_data: Dictionary with E_x, E_phi, K_x, K_phi arrays
        basis_size: Size of basis to generate
        ensure_target_states: Whether to use strategic basis generation
    
    Returns:
        (hilbert_space, coherent_state) tuple
    """
    
    print("=== Creating Enhanced LQG System ===")
    
    # Extract classical data
    E_x = np.array(classical_data['E_x'])
    E_phi = np.array(classical_data['E_phi'])
    K_x = np.array(classical_data.get('K_x', 0.1 * E_x))
    K_phi = np.array(classical_data.get('K_phi', 0.05 * E_phi))
    n_sites = len(E_x)
    
    print(f"Classical configuration:")
    print(f"  E_x: {E_x}")
    print(f"  E_phi: {E_phi}")
    print(f"  K_x: {K_x}")
    print(f"  K_phi: {K_phi}")
    
    # Convert to quantum numbers (assuming integer-valued E fields)
    target_mu = E_x.astype(int)
    target_nu = E_phi.astype(int)
    
    print(f"Target quantum numbers:")
    print(f"  μ: {target_mu}")
    print(f"  ν: {target_nu}")
    
    # Set up LQG parameters
    mu_max = max(2, int(np.max(np.abs(target_mu))))
    nu_max = max(2, int(np.max(np.abs(target_nu))))
    
    lqg_params = LQGParameters(
        mu_max=mu_max,
        nu_max=nu_max,
        basis_truncation=basis_size,
        coherent_width_E=1.0,
        coherent_width_K=1.0
    )
    
    lattice_config = LatticeConfiguration(n_sites=n_sites)
    
    print(f"LQG parameters:")
    print(f"  μ_max: {mu_max}, ν_max: {nu_max}")
    print(f"  Basis size: {basis_size}")
    
    # Create Hilbert space
    if ensure_target_states:
        target_states = [(target_mu, target_nu)]
        hilbert_space = EnhancedKinematicalHilbertSpace(
            lattice_config, lqg_params, target_states
        )
    else:
        hilbert_space = KinematicalHilbertSpace(lattice_config, lqg_params)
    
    # Construct coherent state
    print("\nConstructing coherent state...")
    coherent_state = hilbert_space.construct_coherent_state(E_x, E_phi, K_x, K_phi)
    
    # Analyze the result
    print(f"Coherent state analysis:")
    max_amp_idx = np.argmax(np.abs(coherent_state))
    max_state = hilbert_space.basis_states[max_amp_idx]
    max_amplitude = coherent_state[max_amp_idx]
    
    print(f"  Max amplitude at state {max_amp_idx}:")
    print(f"    μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"    Amplitude: {abs(max_amplitude):.6f}")
    
    # Check if it's the target state
    is_target = (max_state.mu_config == target_mu).all() and (max_state.nu_config == target_nu).all()
    print(f"    Is target state: {is_target} {'✓' if is_target else '✗'}")
    
    # Compute expectation values
    print(f"\nExpectation value errors:")
    max_E_error = 0.0
    max_K_error = 0.0
    
    for site in range(n_sites):
        # E_x expectation
        E_x_op = hilbert_space.flux_E_x_operator(site)
        exp_E_x = np.real(np.conj(coherent_state) @ E_x_op @ coherent_state)
        E_x_error = abs(exp_E_x - E_x[site])
        max_E_error = max(max_E_error, E_x_error)
        
        # E_phi expectation
        E_phi_op = hilbert_space.flux_E_phi_operator(site)
        exp_E_phi = np.real(np.conj(coherent_state) @ E_phi_op @ coherent_state)
        E_phi_error = abs(exp_E_phi - E_phi[site])
        max_E_error = max(max_E_error, E_phi_error)
        
        print(f"  Site {site}: E_x error={E_x_error:.2e}, E_phi error={E_phi_error:.2e}")
    
    print(f"  Maximum E-field error: {max_E_error:.2e}")
    
    success = is_target and max_E_error < 0.1
    print(f"\n{'✓ SUCCESS' if success else '✗ FAILED'}: Coherent state construction {'works!' if success else 'needs improvement'}")
    
    return hilbert_space, coherent_state

def test_enhanced_system():
    """Test the enhanced LQG system with various configurations."""
    
    print("=== Testing Enhanced LQG System ===\n")
    
    # Test 1: Original failing case
    print("Test 1: Original 5-site integer configuration")
    test_data_1 = {
        'E_x': [2.0, 1.0, 0.0, -1.0, -2.0],
        'E_phi': [1.0, 1.0, 0.0, -1.0, -1.0]
    }
    
    hilbert_space_1, coherent_state_1 = create_enhanced_lqg_system(test_data_1)
    
    print(f"\n" + "="*50 + "\n")
    
    # Test 2: Smaller configuration
    print("Test 2: Smaller 3-site configuration")
    test_data_2 = {
        'E_x': [1.0, 0.0, -1.0],
        'E_phi': [1.0, 0.0, -1.0]
    }
    
    hilbert_space_2, coherent_state_2 = create_enhanced_lqg_system(test_data_2, basis_size=1000)
    
    print(f"\n" + "="*50 + "\n")
    
    # Test 3: Without strategic basis (for comparison)
    print("Test 3: Original system without strategic basis (for comparison)")
    hilbert_space_3, coherent_state_3 = create_enhanced_lqg_system(
        test_data_1, basis_size=1000, ensure_target_states=False
    )

if __name__ == "__main__":
    test_enhanced_system()
