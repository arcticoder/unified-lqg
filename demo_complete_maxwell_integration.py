#!/usr/bin/env python3
"""
Complete Maxwell Field Integration Demo

This script demonstrates the full integration of the Maxwell field into the LQG 
midisuperspace framework as requested. It shows:

1. Loading classical data with Maxwell fields (A_r, pi_r)
2. Creating Maxwell-extended kinematical Hilbert space
3. Computing combined phantom + Maxwell T00 stress-energy
4. Exporting quantum observables for the warp drive pipeline

This implements the complete Task requested: adding Maxwell field on top of 
the existing LQG midisuperspace setup with geometry + phantom scalar.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# Import Maxwell-extended framework
from kinematical_hilbert import MidisuperspaceHilbert, LatticeConfig, load_lattice_from_reduced_variables


def run_complete_maxwell_integration():
    """Demonstrate complete Maxwell field integration"""
    
    print("🔷 COMPLETE MAXWELL FIELD + LQG INTEGRATION")
    print("=" * 80)
    print("Integrating Maxwell field (A_r, π_r) on top of existing LQG midisuperspace")
    print("(geometry + phantom scalar) setup as requested.\n")
    
    # Ensure output directory exists
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load configuration with Maxwell field data
    print("🔄 Step 1: Loading Classical Configuration with Maxwell Fields")
    print("-" * 60)
    
    try:
        # Try to load from the updated example file
        config = load_lattice_from_reduced_variables("examples/example_reduced_variables.json")
        print("✓ Successfully loaded configuration with Maxwell field data")
    except Exception as e:
        print(f"⚠️  Creating demonstration configuration: {e}")
        # Create a demo configuration with Maxwell fields
        config = LatticeConfig(
            n_sites=5,
            mu_range=(-2, 2),
            nu_range=(-2, 2),
            gamma=0.2375,
            E_x_classical=[1.2, 1.25, 1.3, 1.25, 1.2],
            E_phi_classical=[0.8, 0.85, 0.9, 0.85, 0.8],
            A_r_classical=[0.0, 0.01, 0.02, 0.015, 0.005],    # Maxwell vector potential
            pi_r_classical=[0.0, 0.002, 0.004, 0.003, 0.001]  # Maxwell momentum
        )
    
    print(f"   Number of lattice sites: {config.n_sites}")
    print(f"   Classical geometry E^x: {config.E_x_classical}")
    print(f"   Classical geometry E^φ: {config.E_phi_classical}")
    print(f"   Classical Maxwell A_r:  {config.A_r_classical}")
    print(f"   Classical Maxwell π_r:  {config.pi_r_classical}")
    print(f"   Barbero-Immirzi γ: {config.gamma}")
    
    # Step 2: Create Maxwell-extended kinematical Hilbert space
    print("\n🔬 Step 2: Building Maxwell-Extended Kinematical Hilbert Space")
    print("-" * 60)
    
    maxwell_levels = 1  # Each site: Maxwell quantum numbers n_i ∈ {0, 1}
    hilbert = MidisuperspaceHilbert(config, maxwell_levels=maxwell_levels)
    
    print(f"   Flux (geometry) basis states: {len(hilbert.flux_states)}")
    print(f"   Maxwell basis states: {len(hilbert.maxwell_states)}")
    print(f"   Total kinematical dimension: {hilbert.hilbert_dim}")
    print(f"   Maxwell levels per site: 0 to {maxwell_levels}")
    
    # Verify Maxwell states structure
    print(f"   Example Maxwell states:")
    for i, max_state in enumerate(hilbert.maxwell_states[:min(8, len(hilbert.maxwell_states))]):
        print(f"     {i}: {max_state}")
    
    # Step 3: Demonstrate Maxwell operators
    print("\n⚙️  Step 3: Maxwell Field Operators")
    print("-" * 60)
    
    print("   Testing Maxwell operators at each site:")
    for site in range(hilbert.n_sites):
        # Build Maxwell operators
        pi_r_op = hilbert.maxwell_pi_operator(site)
        grad_A_op = hilbert.maxwell_gradient_operator(site)
        T00_mx_op = hilbert.maxwell_T00_operator(site)
        
        print(f"   Site {site}:")
        print(f"     π^r operator: {pi_r_op.shape}, {pi_r_op.nnz} non-zero elements")
        print(f"     ∇A operator: {grad_A_op.shape}, {grad_A_op.nnz} non-zero elements")
        print(f"     T00_Maxwell operator: {T00_mx_op.shape}, {T00_mx_op.nnz} non-zero elements")
    
    # Step 4: Create quantum coherent state
    print("\n🌊 Step 4: Creating Quantum Coherent State")
    print("-" * 60)
    
    psi = hilbert.create_coherent_state(
        np.array(config.E_x_classical),
        np.array(config.E_phi_classical),
        width=1.0
    )
    
    print(f"   Coherent state norm: {np.linalg.norm(psi):.6f}")
    print(f"   Non-zero amplitudes: {np.count_nonzero(np.abs(psi) > 1e-6)}")
    
    # Step 5: Compute combined expectation values (THE KEY RESULT)
    print("\n📊 Step 5: Computing Combined Phantom + Maxwell T^00")
    print("-" * 60)
    
    # This is the main result: 5-tuple return with Maxwell contributions
    Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = hilbert.compute_expectation_E_and_T00(psi)
    
    print("   Site-by-site quantum expectation values:")
    print("   Site |    E^x    |   E^φ     | T00_phantom | T00_Maxwell |  T00_total")
    print("   -----|-----------|-----------|-------------|-------------|------------")
    
    for site in range(hilbert.n_sites):
        print(f"   {site:4d} | {Ex_vals[site]:9.4f} | {Ephi_vals[site]:9.4f} | "
              f"{T00_ph_vals[site]:11.4e} | {T00_mx_vals[site]:11.4e} | {T00_tot_vals[site]:11.4e}")
    
    # Compute totals
    total_phantom = sum(T00_ph_vals)
    total_maxwell = sum(T00_mx_vals)
    total_T00 = sum(T00_tot_vals)
    
    print("\n   🎯 Integrated Stress-Energy Components:")
    print(f"     Total phantom scalar T00: {total_phantom:.6e}")
    print(f"     Total Maxwell field T00:  {total_maxwell:.6e}")
    print(f"     Combined total T00:       {total_T00:.6e}")
    
    if abs(total_phantom) > 1e-12:
        maxwell_ratio = abs(total_maxwell / total_phantom)
        print(f"     Maxwell/Phantom ratio:    {maxwell_ratio:.4f}")
    
    # Step 6: Export quantum observables
    print("\n📁 Step 6: Exporting for Warp Drive Pipeline Integration")
    print("-" * 60)
    
    # Export comprehensive quantum data
    quantum_file = os.path.join(output_dir, "complete_maxwell_quantum_data.json")
    quantum_data = hilbert.export_quantum_observables(psi, quantum_file)
    
    # Export T00 data in pipeline-compatible format
    T00_data = []
    r_grid = [1e-35, 2e-35, 3e-35, 4e-35, 5e-35]  # Example radial coordinates
    
    for i, r in enumerate(r_grid[:hilbert.n_sites]):
        record = {
            "r": r,
            "site": i,
            "T00_phantom": T00_ph_vals[i],
            "T00_maxwell": T00_mx_vals[i], 
            "T00_total": T00_tot_vals[i],
            "E_x_quantum": Ex_vals[i],
            "E_phi_quantum": Ephi_vals[i],
            "A_r_classical": config.A_r_classical[i],
            "pi_r_classical": config.pi_r_classical[i]
        }
        T00_data.append(record)
    
    # Export as JSON
    T00_json_file = os.path.join(output_dir, "T00_quantum_maxwell.json")
    with open(T00_json_file, 'w') as f:
        json.dump(T00_data, f, indent=2)
    
    # Export as NDJSON for pipeline
    T00_ndjson_file = os.path.join(output_dir, "T00_quantum_maxwell.ndjson")
    with open(T00_ndjson_file, 'w') as f:
        for record in T00_data:
            json.dump(record, f)
            f.write('\n')
    
    print(f"   ✓ Quantum observables → {quantum_file}")
    print(f"   ✓ T00 data (JSON) → {T00_json_file}")
    print(f"   ✓ T00 data (NDJSON) → {T00_ndjson_file}")
    
    # Summary and verification
    print("\n✅ MAXWELL INTEGRATION COMPLETE")
    print("-" * 60)
    print("Successfully integrated Maxwell field (A_r, π_r) into LQG midisuperspace:")
    print(f"  ✓ Extended Hilbert space dimension: {hilbert.hilbert_dim}")
    print(f"  ✓ Maxwell operators defined at all {hilbert.n_sites} sites")
    print(f"  ✓ Combined T00 = T00_phantom + T00_Maxwell computed")
    print(f"  ✓ Quantum backreaction ready for classical warp drive pipeline")
    
    if abs(total_maxwell) > 1e-12:
        print(f"  ✓ Maxwell field provides {abs(total_maxwell):.2e} contribution to stress-energy")
    else:
        print(f"  ℹ️  Maxwell field in vacuum state (no excitations)")
    
    return True


if __name__ == "__main__":
    try:
        success = run_complete_maxwell_integration()
        if success:
            print("\n🎉 Maxwell field integration demonstration completed successfully!")
        else:
            print("\n❌ Maxwell field integration failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during Maxwell integration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
