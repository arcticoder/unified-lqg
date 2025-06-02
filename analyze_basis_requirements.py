#!/usr/bin/env python3
"""
Analyze basis requirements for LQG coherent state construction.
"""

import numpy as np
from itertools import product

def analyze_basis_order():
    """Analyze the order of basis state generation to understand truncation."""
    
    n_sites = 5
    mu_max = nu_max = 2
    
    mu_range = list(range(-mu_max, mu_max + 1))  # [-2, -1, 0, 1, 2]
    nu_range = list(range(-nu_max, nu_max + 1))
    
    print(f"μ range: {mu_range}")
    print(f"ν range: {nu_range}")
    print(f"Total μ combinations: {len(mu_range)**n_sites}")
    print(f"Total basis states: {len(mu_range)**n_sites * len(nu_range)**n_sites}")
    
    # Target configuration
    target_mu = np.array([2, 1, 0, -1, -2])
    target_nu = np.array([1, 1, 0, -1, -1])
    
    print(f"\nTarget μ: {target_mu}")
    print(f"Target ν: {target_nu}")
    
    # Find where target appears in generation order
    states_checked = 0
    found_target = False
    
    for i, mu_tuple in enumerate(product(mu_range, repeat=n_sites)):
        for j, nu_tuple in enumerate(product(nu_range, repeat=n_sites)):
            states_checked += 1
            
            if (np.array(mu_tuple) == target_mu).all() and (np.array(nu_tuple) == target_nu).all():
                print(f"\n✓ Target found at position {states_checked}")
                print(f"  μ iteration: {i}")
                print(f"  ν iteration: {j}")
                found_target = True
                break
                
            # Show first few and target area
            if states_checked <= 10:
                print(f"  {states_checked:4d}: μ={mu_tuple}, ν={nu_tuple}")
            elif states_checked == 1000:
                print(f"  {states_checked:4d}: μ={mu_tuple}, ν={nu_tuple} <- TRUNCATION POINT")
                
        if found_target:
            break
    
    if not found_target and states_checked >= 1000:
        print(f"\n✗ Target NOT found within first 1000 states")
        
        # Let's check where μ=[2,1,0,-1,-2] appears
        print(f"\nLooking for μ=[2,1,0,-1,-2] position...")
        for i, mu_tuple in enumerate(product(mu_range, repeat=n_sites)):
            if (np.array(mu_tuple) == target_mu).all():
                print(f"  μ=[2,1,0,-1,-2] is at μ-position {i}")
                print(f"  This corresponds to basis states {i * len(nu_range)**n_sites} to {(i+1) * len(nu_range)**n_sites - 1}")
                break
    
    # Calculate minimum basis size needed
    print(f"\nTo include target state, need at least {states_checked} basis states")
    
    # Show some basis states around the target
    print(f"\nSample of μ configurations (first 20):")
    for i, mu_tuple in enumerate(product(mu_range, repeat=n_sites)):
        if i < 20:
            print(f"  {i:3d}: μ={mu_tuple}")
        elif i == 20:
            print("  ...")
            break

if __name__ == "__main__":
    analyze_basis_order()
