#!/usr/bin/env python3
"""
Debug version to understand what's happening in the GPU search.
"""

import numpy as np
import torch

def debug_gpu_search():
    """Debug the GPU search logic with simple values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple test case
    n_sites = 3  # Smaller for easier debugging
    mu_max = 2
    nu_max = 2
    
    # Values that should definitely match
    classical_E_x = np.array([2, 1, 0])
    classical_E_phi = np.array([1, 0, -1])
    classical_K_x = np.array([0.2, 0.1, 0.0])  # = 0.1 × μ
    classical_K_phi = np.array([0.1, 0.0, -0.1])  # = 0.1 × ν
    
    print(f"Target μ: {classical_E_x}")
    print(f"Target ν: {classical_E_phi}")
    print(f"Target K_x: {classical_K_x}")
    print(f"Target K_phi: {classical_K_phi}")
    
    # Build search space
    mu_vals = torch.arange(-mu_max, mu_max + 1, device=device)    # [-2, -1, 0, 1, 2]
    nu_vals = torch.arange(-nu_max, nu_max + 1, device=device)
    
    print(f"μ values: {mu_vals}")
    print(f"ν values: {nu_vals}")
    
    # Create grids
    grids = [mu_vals] * n_sites
    mu_grid = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)
    mu_grid = mu_grid.reshape(-1, n_sites)  # shape: [125, 3]
    
    grids_nu = [nu_vals] * n_sites
    nu_grid = torch.stack(torch.meshgrid(*grids_nu, indexing="ij"), dim=-1)
    nu_grid = nu_grid.reshape(-1, n_sites)
    
    print(f"μ grid shape: {mu_grid.shape}")
    print(f"ν grid shape: {nu_grid.shape}")
    
    # Convert classical arrays to tensors
    classical_E_x_t = torch.tensor(classical_E_x, device=device, dtype=torch.int64)
    classical_E_phi_t = torch.tensor(classical_E_phi, device=device, dtype=torch.int64)
    classical_K_x_t = torch.tensor(classical_K_x, device=device, dtype=torch.float32)
    classical_K_phi_t = torch.tensor(classical_K_phi, device=device, dtype=torch.float32)
    
    print(f"Classical E_x tensor: {classical_E_x_t}")
    print(f"Classical E_phi tensor: {classical_E_phi_t}")
    
    # Look for exact μ match
    print("\nSearching for μ matches...")
    mu_matches = (mu_grid == classical_E_x_t).all(dim=1)
    mu_match_indices = torch.nonzero(mu_matches).squeeze()
    
    print(f"Found {mu_matches.sum()} μ matches at indices: {mu_match_indices}")
    
    if mu_matches.sum() > 0:
        if mu_match_indices.dim() == 0:  # Single match
            mu_match_indices = mu_match_indices.unsqueeze(0)
            
        for idx in mu_match_indices:
            print(f"μ match at index {idx}: {mu_grid[idx]}")
    
    # Look for exact ν match  
    print("\nSearching for ν matches...")
    nu_matches = (nu_grid == classical_E_phi_t).all(dim=1)
    nu_match_indices = torch.nonzero(nu_matches).squeeze()
    
    print(f"Found {nu_matches.sum()} ν matches at indices: {nu_match_indices}")
    
    if nu_matches.sum() > 0:
        if nu_match_indices.dim() == 0:  # Single match
            nu_match_indices = nu_match_indices.unsqueeze(0)
            
        for idx in nu_match_indices:
            print(f"ν match at index {idx}: {nu_grid[idx]}")
    
    # Check if we have both matches
    if mu_matches.sum() > 0 and nu_matches.sum() > 0:
        mu_idx = mu_match_indices[0] if mu_match_indices.dim() > 0 else mu_match_indices
        nu_idx = nu_match_indices[0] if nu_match_indices.dim() > 0 else nu_match_indices
        
        mu_candidate = mu_grid[mu_idx]
        nu_candidate = nu_grid[nu_idx]
        
        print(f"\nTesting K scaling:")
        print(f"μ candidate: {mu_candidate}")
        print(f"ν candidate: {nu_candidate}")
        
        K_x_approx = 0.1 * mu_candidate.float()
        K_phi_approx = 0.1 * nu_candidate.float()
        
        print(f"K_x_approx: {K_x_approx}")
        print(f"K_phi_approx: {K_phi_approx}")
        print(f"Target K_x: {classical_K_x_t}")
        print(f"Target K_phi: {classical_K_phi_t}")
        
        K_x_match = torch.allclose(K_x_approx, classical_K_x_t, atol=1e-6)
        K_phi_match = torch.allclose(K_phi_approx, classical_K_phi_t, atol=1e-6)
        
        print(f"K_x match: {K_x_match}")
        print(f"K_phi match: {K_phi_match}")
        
        if K_x_match and K_phi_match:
            print("✓ PERFECT MATCH FOUND!")
            return True
    
    print("❌ No perfect match found")
    return False

if __name__ == "__main__":
    debug_gpu_search()
