#!/usr/bin/env python3
"""
GPU-accelerated version of generate_p    classical_K_x_t   = torch.tensor(classical_K_x,   device=device, dtype=torch.float32)
    classical_K_phi_t = torch.tensor(classical_K_phi, device=device, dtype=torch.float32)fect_json.py using PyTorch.
"""

import numpy as np
import json
import os

def generate_perfect_match_json(
    n_sites: int,
    mu_max: int,
    nu_max: int,
    classical_E_x: np.ndarray,
    classical_E_phi: np.ndarray,
    classical_K_x: np.ndarray,
    classical_K_phi: np.ndarray,
    output_path: str
):
    """
    Finds the unique (μ_config, ν_config) that exactly equals the classical values
    (assuming classical_E_x, classical_E_phi are integers in [-mu_max..mu_max], etc.)
    by parallelizing the search on GPU.

    Writes a JSON file with keys: mu_config, nu_config, K_scale.
    """
    
    # Try to use PyTorch if available
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        use_gpu = True
    except ImportError:
        print("PyTorch not available, using NumPy CPU version")
        use_gpu = False

    if use_gpu:
        return _generate_gpu_version(n_sites, mu_max, nu_max, classical_E_x, classical_E_phi, 
                                   classical_K_x, classical_K_phi, output_path, device)
    else:
        return _generate_cpu_version(n_sites, mu_max, nu_max, classical_E_x, classical_E_phi,
                                   classical_K_x, classical_K_phi, output_path)

def _generate_gpu_version(n_sites, mu_max, nu_max, classical_E_x, classical_E_phi,
                         classical_K_x, classical_K_phi, output_path, device):
    """GPU-accelerated version using PyTorch."""
    import torch
    
    # Build all μ, ν possibilities as PyTorch tensors on the GPU
    mu_vals = torch.arange(-mu_max, mu_max + 1, device=device)    # e.g. [-2..2]
    nu_vals = torch.arange(-nu_max, nu_max + 1, device=device)

    # Create a grid of shape ( (2μ_max+1)^n_sites, n_sites )
    grids = [mu_vals] * n_sites
    mu_grid = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)  # shape: [5,5,5,5,5, 5]
    mu_grid = mu_grid.reshape(-1, n_sites)  # shape: [3125, 5]

    grids_nu = [nu_vals] * n_sites
    nu_grid = torch.stack(torch.meshgrid(*grids_nu, indexing="ij"), dim=-1)
    nu_grid = nu_grid.reshape(-1, n_sites)

    # Convert classical arrays to tensors
    total_mu = mu_grid.size(0)   # 5^5 = 3125
    classical_E_x_t = torch.tensor(classical_E_x, device=device, dtype=torch.int64)
    classical_E_phi_t = torch.tensor(classical_E_phi, device=device, dtype=torch.int64)
    classical_K_x_t = torch.tensor(classical_K_x, device=device)
    classical_K_phi_t = torch.tensor(classical_K_phi, device=device)

    print(f"Searching through {total_mu}×{nu_grid.size(0)} = {total_mu * nu_grid.size(0)} combinations...")

    # Search for exact matches
    found = False
    for i in range(total_mu):
        mu_candidate = mu_grid[i].unsqueeze(0)  # shape [1, 5]
        
        # Compare E_x first:
        if not torch.equal(mu_candidate.squeeze(), classical_E_x_t):
            continue

        # If E_x matches, iterate over all ν to find matching E_phi:
        phi_matches = (nu_grid == classical_E_phi_t).all(dim=1)  # shape [3125]
        idxs = torch.nonzero(phi_matches).squeeze()

        if idxs.numel() == 0:
            continue

        # Among those, check K scaling: (0.1 * μ_candidate == classical_K_x) and (0.1 * ν == classical_K_phi)
        mu_c = mu_candidate.squeeze().float()
        nu_cs = nu_grid[idxs].float()  # shape [n_matches, 5]
        
        # Compute Kx_approx = 0.1 * μ and Kφ_approx = 0.1 * ν
        Kx_approx = 0.1 * mu_c
        Kphi_approx = 0.1 * nu_cs

        # Compare to classical K values
        Kx_ok = torch.allclose(Kx_approx, classical_K_x_t, atol=1e-6)
        
        # Check each ν candidate
        for j in range(nu_cs.size(0)):
            Kphi_ok = torch.allclose(Kphi_approx[j], classical_K_phi_t, atol=1e-6)
            if Kx_ok and Kphi_ok:
                nu_candidate = nu_cs[j].cpu().long().numpy()
                mu_candidate_np = mu_candidate.squeeze().cpu().long().numpy()
                print("✓ Found perfect match:")
                print("  μ =", mu_candidate_np.tolist())
                print("  ν =", nu_candidate.tolist())
                found = True
                break
        
        if found:
            break

    if not found:
        raise RuntimeError("Target (μ,ν) not found among all combinations!")

    # Write out JSON
    out = {
        "mu_config": mu_candidate_np.tolist(),
        "nu_config": nu_candidate.tolist(),
        "K_scale": 0.1,  # we know K_approx = 0.1·μ
        "method": "gpu_accelerated"
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✓ Wrote perfect‐match JSON to {output_path}")
    return out

def _generate_cpu_version(n_sites, mu_max, nu_max, classical_E_x, classical_E_phi,
                         classical_K_x, classical_K_phi, output_path):
    """CPU fallback version using NumPy."""
    print("Using CPU fallback (may be slower for large searches)")
    
    # Generate all possible configurations
    mu_vals = np.arange(-mu_max, mu_max + 1)
    nu_vals = np.arange(-nu_max, nu_max + 1)
    
    # Simple brute force search
    for mu_config in np.ndindex(*([len(mu_vals)] * n_sites)):
        mu_candidate = np.array([mu_vals[i] for i in mu_config])
        
        # Check E_x match
        if not np.array_equal(mu_candidate, classical_E_x):
            continue
            
        for nu_config in np.ndindex(*([len(nu_vals)] * n_sites)):
            nu_candidate = np.array([nu_vals[i] for i in nu_config])
            
            # Check E_phi match
            if not np.array_equal(nu_candidate, classical_E_phi):
                continue
                
            # Check K scaling
            Kx_approx = 0.1 * mu_candidate
            Kphi_approx = 0.1 * nu_candidate
            
            if (np.allclose(Kx_approx, classical_K_x, atol=1e-6) and 
                np.allclose(Kphi_approx, classical_K_phi, atol=1e-6)):
                
                print("✓ Found perfect match:")
                print("  μ =", mu_candidate.tolist())
                print("  ν =", nu_candidate.tolist())
                
                # Write out JSON
                out = {
                    "mu_config": mu_candidate.tolist(),
                    "nu_config": nu_candidate.tolist(),
                    "K_scale": 0.1,
                    "method": "cpu_fallback"
                }
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(out, f, indent=2)
                    
                print(f"✓ Wrote perfect‐match JSON to {output_path}")
                return out
    
    raise RuntimeError("Target (μ,ν) not found among all combinations!")

def main():
    """Demo with 5-site integer example."""
    classical_E_x = np.array([2, 1, 0, -1, -2])
    classical_E_phi = np.array([1, 1, 0, -1, -1])
    classical_K_x = np.array([0.2, 0.1, 0.0, -0.1, -0.2])
    classical_K_phi = np.array([0.1, 0.1, 0.0, -0.1, -0.1])
    
    output_path = "examples/perfect_match_gpu.json"
    
    result = generate_perfect_match_json(
        n_sites=5,
        mu_max=2,
        nu_max=2,
        classical_E_x=classical_E_x,
        classical_E_phi=classical_E_phi,
        classical_K_x=classical_K_x,
        classical_K_phi=classical_K_phi,
        output_path=output_path
    )
    
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
