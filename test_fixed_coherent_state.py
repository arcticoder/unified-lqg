#!/usr/bin/env python3
"""
Fixed coherent state test using strategic basis generation.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append('.')

from lqg_fixed_components import (
    FluxBasisState, LQGParameters, KinematicalHilbertSpace, 
    MidisuperspaceHamiltonianConstraint, LatticeConfiguration
)

class StrategicKinematicalHilbertSpace(KinematicalHilbertSpace):
    """Enhanced KinematicalHilbertSpace with strategic basis generation."""
    
    def __init__(self, lattice_config, lqg_params, target_states=None):
        self.target_states = target_states or []
        super().__init__(lattice_config, lqg_params)
    
    def _generate_basis_states(self):
        """Generate basis states with strategic inclusion of target states."""
        from itertools import product
        
        states = []
        states_set = set()
        
        mu_range = list(range(-self.lqg_params.mu_max, self.lqg_params.mu_max + 1))
        nu_range = list(range(-self.lqg_params.nu_max, self.lqg_params.nu_max + 1))
        
        # First, ensure target states are included
        if self.target_states:
            print(f"Ensuring {len(self.target_states)} target states are included...")
            for mu_config, nu_config in self.target_states:
                state = FluxBasisState(mu_config, nu_config)
                state_tuple = (tuple(mu_config), tuple(nu_config))
                if state_tuple not in states_set:
                    states.append(state)
                    states_set.add(state_tuple)
                    print(f"  Added target: μ={mu_config}, ν={nu_config}")
        
        # Add neighbor states around targets
        if self.target_states:
            for mu_config, nu_config in self.target_states:
                for delta_mu in [-1, 0, 1]:
                    for delta_nu in [-1, 0, 1]:
                        for site in range(self.n_sites):
                            new_mu = mu_config.copy()
                            new_nu = nu_config.copy()
                            
                            new_mu[site] = max(-self.lqg_params.mu_max, 
                                             min(self.lqg_params.mu_max, new_mu[site] + delta_mu))
                            new_nu[site] = max(-self.lqg_params.nu_max, 
                                             min(self.lqg_params.nu_max, new_nu[site] + delta_nu))
                            
                            state_tuple = (tuple(new_mu), tuple(new_nu))
                            if (state_tuple not in states_set and 
                                len(states) < self.lqg_params.basis_truncation):
                                states.append(FluxBasisState(new_mu, new_nu))
                                states_set.add(state_tuple)
        
        # Fill remaining space with low-energy states
        energy_priority = []
        for mu_tuple in product(mu_range, repeat=self.n_sites):
            for nu_tuple in product(nu_range, repeat=self.n_sites):
                state_tuple = (mu_tuple, nu_tuple)
                if state_tuple not in states_set:
                    energy = sum(m**2 for m in mu_tuple) + sum(n**2 for n in nu_tuple)
                    energy_priority.append((energy, mu_tuple, nu_tuple))
        
        energy_priority.sort(key=lambda x: x[0])
        
        for energy, mu_tuple, nu_tuple in energy_priority:
            if len(states) >= self.lqg_params.basis_truncation:
                print(f"  Truncated to {len(states)} states")
                break
            state_tuple = (mu_tuple, nu_tuple)
            if state_tuple not in states_set:
                states.append(FluxBasisState(np.array(mu_tuple), np.array(nu_tuple)))
                states_set.add(state_tuple)
        
        print(f"Strategic basis generated: {len(states)} states")
        return states

def test_fixed_coherent_state():
    """Test coherent state construction with strategic basis."""
    
    print("=== Testing Fixed Coherent State Construction ===")
    
    # Target configuration
    target_E_x = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    target_E_phi = np.array([1.0, 1.0, 0.0, -1.0, -1.0])
    target_K_x = 0.1 * target_E_x
    target_K_phi = 0.05 * target_E_phi
    
    print(f"Target E_x: {target_E_x}")
    print(f"Target E_phi: {target_E_phi}")
    
    # Convert to integer quantum numbers
    target_mu = target_E_x.astype(int)
    target_nu = target_E_phi.astype(int)
    target_state = FluxBasisState(target_mu, target_nu)
    
    print(f"Target μ: {target_mu}")
    print(f"Target ν: {target_nu}")
      # LQG parameters with larger basis
    lqg_params = LQGParameters(
        mu_max=2,
        nu_max=2,
        basis_truncation=5000,  # Increased basis size
        coherent_width_E=1.0,
        coherent_width_K=1.0
    )
      # Create strategic Hilbert space
    target_states = [(target_mu, target_nu)]
    lattice_config = LatticeConfiguration(n_sites=5)
    hilbert_space = StrategicKinematicalHilbertSpace(
        lattice_config=lattice_config,
        lqg_params=lqg_params,
        target_states=target_states
    )
    
    print(f"\nStrategic Hilbert space dimension: {hilbert_space.dim}")
    
    # Check if target state exists
    target_found = False
    target_index = None
    
    for i, state in enumerate(hilbert_space.basis_states):
        if (state.mu_config == target_mu).all() and (state.nu_config == target_nu).all():
            target_found = True
            target_index = i
            break
    
    print(f"✓ Target state found: {target_found}")
    if target_found:
        print(f"  Position in basis: {target_index}")
    
    if not target_found:
        print("✗ Cannot proceed without target state in basis")
        return
      # Build coherent state directly from Hilbert space
    print("\nConstructing coherent state...")
    
    coherent_state = hilbert_space.construct_coherent_state(
        target_E_x, target_E_phi, target_K_x, target_K_phi
    )
    
    print(f"Coherent state norm: {np.linalg.norm(coherent_state):.6f}")
    
    # Analyze the coherent state
    print(f"\nCoherent state analysis:")
    
    # Find max amplitude
    max_amp_idx = np.argmax(np.abs(coherent_state))
    max_state = hilbert_space.basis_states[max_amp_idx]
    max_amplitude = coherent_state[max_amp_idx]
    
    print(f"Max amplitude at state {max_amp_idx}:")
    print(f"  μ={max_state.mu_config}, ν={max_state.nu_config}")
    print(f"  Amplitude: {abs(max_amplitude):.6f}")
    print(f"  Is target state: {max_amp_idx == target_index}")
      # Compute expectation values
    print(f"\nExpectation values:")
    for site in range(5):
        # E_x expectation
        E_x_op = hilbert_space.flux_E_x_operator(site)
        exp_E_x = np.real(np.conj(coherent_state) @ E_x_op @ coherent_state)
        
        # E_phi expectation  
        E_phi_op = hilbert_space.flux_E_phi_operator(site)
        exp_E_phi = np.real(np.conj(coherent_state) @ E_phi_op @ coherent_state)
        
        print(f"Site {site}:")
        print(f"  ⟨E^x⟩ = {exp_E_x:.6f} (target {target_E_x[site]:.6f}), error={exp_E_x - target_E_x[site]:.2e}")
        print(f"  ⟨E^φ⟩ = {exp_E_phi:.6f} (target {target_E_phi[site]:.6f}), error={exp_E_phi - target_E_phi[site]:.2e}")
    
    # Show top amplitudes
    print(f"\nTop 5 amplitudes:")
    sorted_indices = np.argsort(np.abs(coherent_state))[::-1]
    for i, idx in enumerate(sorted_indices[:5]):
        state = hilbert_space.basis_states[idx]
        amp = coherent_state[idx]
        target_marker = " <- TARGET" if idx == target_index else ""
        print(f"  {i+1}. State {idx}: μ={state.mu_config}, ν={state.nu_config}")
        print(f"     Amplitude: {abs(amp):.6f}{target_marker}")
    
    return coherent_state, hilbert_space

if __name__ == "__main__":
    test_fixed_coherent_state()
