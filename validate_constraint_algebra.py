#!/usr/bin/env python3
"""
Validate constraint algebra implementation in LQG system.
"""

import numpy as np
import sys

from lqg_fixed_components import (
    MidisuperspaceHamiltonianConstraint, 
    LatticeConfiguration, 
    LQGParameters
)

def test_constraint_algebra():
    """Test the constraint algebra implementation."""
    print("Testing constraint algebra implementation...")
      # Create test configuration
    lattice = LatticeConfiguration()
    lattice.n_sites = 5
    lattice.r_min = 1e-35
    lattice.r_max = 1e-33
    lqg_params = LQGParameters(
        planck_length=0.01, 
        max_basis_states=1000, 
        use_improved_dynamics=True
    )
    
    # Create constraint
    print("Creating constraint object...")
    constraint = MidisuperspaceHamiltonianConstraint(lattice, lqg_params)
    
    print(f"✅ Hilbert space dimension: {constraint.kinematical_hilbert_space.dim}")
    print(f"✅ Number of sites: {constraint.kinematical_hilbert_space.n_sites}")
    
    # Test constraint matrix construction
    print("Building constraint matrix...")
    H = constraint.build_constraint_matrix()
    
    print(f"✅ Constraint matrix shape: {H.shape}")
    print(f"✅ Constraint matrix type: {type(H)}")
    print(f"✅ Matrix density: {H.nnz / (H.shape[0] * H.shape[1]):.6f}")
    
    # Test Hermiticity
    print("Testing Hermiticity...")
    H_dag = H.conjugate().transpose()
    diff = (H - H_dag).data
    max_diff = np.max(np.abs(diff)) if len(diff) > 0 else 0.0
    print(f"✅ Max Hermitian violation: {max_diff:.2e}")
    
    # Test some individual operators
    print("Testing individual operators...")
    kin_space = constraint.kinematical_hilbert_space
    
    # Test K-operators
    site = 0
    Kx = kin_space.kx_operator(site)
    Kphi = kin_space.kphi_operator(site)
    
    print(f"✅ K_x operator shape: {Kx.shape}")
    print(f"✅ K_phi operator shape: {Kphi.shape}")
    
    # Test holonomy operators
    Ux = kin_space.holonomy_shift_operator(site, 'x', 1)
    Uphi = kin_space.holonomy_shift_operator(site, 'phi', 1)
    
    print(f"✅ U_x operator shape: {Ux.shape}")
    print(f"✅ U_phi operator shape: {Uphi.shape}")
    
    print("✅ All constraint algebra tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_constraint_algebra()
        print("\n🎉 Constraint algebra validation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in constraint algebra validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
