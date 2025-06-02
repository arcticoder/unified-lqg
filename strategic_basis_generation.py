#!/usr/bin/env python3
"""
Enhanced basis generation that ensures target states are included.
"""

import numpy as np
from itertools import product
from typing import List, Tuple, Set
import sys
import os

# Add current directory to path to import lqg modules
sys.path.append('.')

from lqg_fixed_components import FluxBasisState, LQGParameters, KinematicalHilbertSpace

def generate_strategic_basis(n_sites: int, mu_max: int, nu_max: int, 
                           target_states: List[Tuple[np.ndarray, np.ndarray]] = None,
                           max_basis_size: int = 10000) -> List[FluxBasisState]:
    """
    Generate basis states with strategic inclusion of target states.
    
    Args:
        n_sites: Number of lattice sites
        mu_max, nu_max: Maximum quantum numbers
        target_states: List of (mu_config, nu_config) tuples to ensure inclusion
        max_basis_size: Maximum number of basis states to generate
    
    Returns:
        List of FluxBasisState objects including target states
    """
    
    mu_range = list(range(-mu_max, mu_max + 1))
    nu_range = list(range(-nu_max, nu_max + 1))
    
    states = []
    states_set = set()  # Track unique states to avoid duplicates
    
    # First, ensure target states are included
    if target_states:
        print(f"Ensuring {len(target_states)} target states are included...")
        for mu_config, nu_config in target_states:
            state = FluxBasisState(mu_config, nu_config)
            state_tuple = (tuple(mu_config), tuple(nu_config))
            if state_tuple not in states_set:
                states.append(state)
                states_set.add(state_tuple)
                print(f"  Added target: μ={mu_config}, ν={nu_config}")
    
    # Then add states around targets (neighbors)
    if target_states:
        print("Adding neighbor states around targets...")
        for mu_config, nu_config in target_states:
            # Add states with small perturbations
            for delta_mu in [-1, 0, 1]:
                for delta_nu in [-1, 0, 1]:
                    for site in range(n_sites):
                        # Perturb one site at a time
                        new_mu = mu_config.copy()
                        new_nu = nu_config.copy()
                        
                        new_mu[site] = max(-mu_max, min(mu_max, new_mu[site] + delta_mu))
                        new_nu[site] = max(-nu_max, min(nu_max, new_nu[site] + delta_nu))
                        
                        state_tuple = (tuple(new_mu), tuple(new_nu))
                        if state_tuple not in states_set and len(states) < max_basis_size:
                            states.append(FluxBasisState(new_mu, new_nu))
                            states_set.add(state_tuple)
    
    # Fill remaining space with systematic states
    print(f"Filling remaining basis space (current: {len(states)}/{max_basis_size})...")
    
    # Add low-energy states (smaller quantum numbers first)
    energy_priority = []
    for mu_tuple in product(mu_range, repeat=n_sites):
        for nu_tuple in product(nu_range, repeat=n_sites):
            state_tuple = (mu_tuple, nu_tuple)
            if state_tuple not in states_set:
                # Energy estimate: sum of squares
                energy = sum(m**2 for m in mu_tuple) + sum(n**2 for n in nu_tuple)
                energy_priority.append((energy, mu_tuple, nu_tuple))
    
    # Sort by energy and add states
    energy_priority.sort(key=lambda x: x[0])
    
    for energy, mu_tuple, nu_tuple in energy_priority:
        if len(states) >= max_basis_size:
            break
        state_tuple = (mu_tuple, nu_tuple)
        if state_tuple not in states_set:
            states.append(FluxBasisState(np.array(mu_tuple), np.array(nu_tuple)))
            states_set.add(state_tuple)
    
    print(f"Generated {len(states)} basis states")
    return states

def test_strategic_basis():
    """Test the strategic basis generation."""
    
    # Parameters
    n_sites = 5
    mu_max = nu_max = 2
    
    # Target configuration from our test
    target_mu = np.array([2, 1, 0, -1, -2])
    target_nu = np.array([1, 1, 0, -1, -1])
    target_states = [(target_mu, target_nu)]
    
    print("=== Testing Strategic Basis Generation ===")
    print(f"Target: μ={target_mu}, ν={target_nu}")
    
    # Generate strategic basis
    basis_states = generate_strategic_basis(
        n_sites=n_sites,
        mu_max=mu_max,
        nu_max=nu_max,
        target_states=target_states,
        max_basis_size=5000  # Reasonable size
    )
    
    # Check if target is included
    target_found = False
    target_index = None
    
    for i, state in enumerate(basis_states):
        if (state.mu_config == target_mu).all() and (state.nu_config == target_nu).all():
            target_found = True
            target_index = i
            break
    
    print(f"\n✓ Target state found: {target_found}")
    if target_found:
        print(f"  Position in basis: {target_index}")
    
    # Show first few states
    print(f"\nFirst 10 basis states:")
    for i in range(min(10, len(basis_states))):
        state = basis_states[i]
        marker = " <- TARGET" if i == target_index else ""
        print(f"  {i:3d}: μ={state.mu_config}, ν={state.nu_config}{marker}")
    
    # Show diversity of μ configurations
    mu_configs = set()
    for state in basis_states:
        mu_configs.add(tuple(state.mu_config))
    
    print(f"\nBasis diversity:")
    print(f"  Total states: {len(basis_states)}")
    print(f"  Unique μ configurations: {len(mu_configs)}")
    print(f"  Sample μ configs: {list(mu_configs)[:10]}")
    
    return basis_states, target_found

if __name__ == "__main__":
    test_strategic_basis()
