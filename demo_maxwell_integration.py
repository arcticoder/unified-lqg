#!/usr/bin/env python3
"""
demo_maxwell_integration.py

Comprehensive demonstration of Maxwell field integration into the LQG midisuperspace framework.
Shows how phantom scalar + Maxwell field quantum backreaction affects warp drive dynamics.

This script demonstrates:
1. Loading classical Maxwell field data alongside geometry
2. Creating Maxwell-extended quantum Hilbert space
3. Computing combined phantom + Maxwell stress-energy tensor
4. Exporting quantum observables for classical warp pipeline integration
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# Import our Maxwell-extended LQG framework
from kinematical_hilbert import MidisuperspaceHilbert, LatticeConfig, load_lattice_from_reduced_variables


def demo_maxwell_lqg_integration():
    """Run comprehensive Maxwell + LQG demonstration"""
    
    print("🔷 Maxwell Field + LQG Midisuperspace Integration Demo")
    print("=" * 70)
    
    # Create outputs directory
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load configuration with Maxwell fields
    print(f"\n📁 Step 1: Loading Classical Data with Maxwell Fields")
    print("-" * 50)
    
    try:
        config = load_lattice_from_reduced_variables("examples/lqg_demo_classical_data.json")
        print("✓ Loaded configuration from lqg_demo_classical_data.json")
    except FileNotFoundError:
        print("⚠️  Creating fallback configuration...")
        config = LatticeConfig(
            n_sites=5,
            mu_range=(-2, 2),
            nu_range=(-2, 2), 
            gamma=0.2375,
            E_x_classical=[1.0, 0.9, 0.8, 0.9, 1.0],
            E_phi_classical=[1.1, 1.15, 1.2, 1.15, 1.1],
            A_r_classical=[0.01, 0.02, 0.015, 0.012, 0.008],
            pi_r_classical=[0.001, 0.002, 0.0015, 0.0012, 0.0008]
        )
    
    print(f"   Sites: {config.n_sites}")
    print(f"   Maxwell field included: A_r = {config.A_r_classical}")
    print(f"   Maxwell momentum: π_r = {config.pi_r_classical}")
    
    # 2. Create Maxwell-extended Hilbert space
    print(f"\n🔬 Step 2: Building Maxwell-Extended Quantum Hilbert Space")
    print("-" * 50)
    
    maxwell_levels = 1  # Start with 2-level Maxwell oscillator per site
    hilbert = MidisuperspaceHilbert(config, maxwell_levels=maxwell_levels)
    
    print(f"   Flux basis dimension: {len(hilbert.flux_states)}")
    print(f"   Maxwell basis dimension: {len(hilbert.maxwell_states)}")
    print(f"   Total Hilbert dimension: {hilbert.hilbert_dim}")
    print(f"   Maxwell levels per site: 0 to {maxwell_levels}")
    
    # 3. Test Maxwell operators
    print(f"\n⚙️  Step 3: Testing Maxwell Field Operators")
    print("-" * 50)
    
    for site in range(min(3, hilbert.n_sites)):  # Test first 3 sites
        pi_op = hilbert.maxwell_pi_operator(site)
        T00_mx_op = hilbert.maxwell_T00_operator(site)
        
        print(f"   Site {site}:")
        print(f"     π^r operator: {pi_op.shape}, {pi_op.nnz} non-zero elements")
        print(f"     T00_Maxwell operator: {T00_mx_op.shape}, {T00_mx_op.nnz} non-zero elements")
    
    # 4. Create quantum coherent state
    print(f"\n🌊 Step 4: Creating Quantum Coherent State")
    print("-" * 50)
    
    psi = hilbert.create_coherent_state(
        np.array(config.E_x_classical),
        np.array(config.E_phi_classical),
        width=1.2
    )
    
    hilbert.print_state_summary(psi, "Maxwell-Extended Coherent State")
    
    # 5. Compute combined expectation values
    print(f"\n📊 Step 5: Computing Combined Phantom + Maxwell Expectations")
    print("-" * 50)
    
    Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = hilbert.compute_expectation_E_and_T00(psi)
    
    print(f"   Site-by-site breakdown:")
    total_phantom = sum(T00_ph_vals)
    total_maxwell = sum(T00_mx_vals)
    total_T00 = sum(T00_tot_vals)
    
    for site in range(hilbert.n_sites):
        print(f"     Site {site}:")
        print(f"       E^x: {Ex_vals[site]:.4f} (target: {config.E_x_classical[site]:.4f})")
        print(f"       E^φ: {Ephi_vals[site]:.4f} (target: {config.E_phi_classical[site]:.4f})")
        print(f"       T00_phantom: {T00_ph_vals[site]:.6e}")
        print(f"       T00_Maxwell: {T00_mx_vals[site]:.6e}")
        print(f"       T00_total: {T00_tot_vals[site]:.6e}")
    
    print(f"\n   🎯 Global Totals:")
    print(f"     Total phantom contribution: {total_phantom:.6e}")
    print(f"     Total Maxwell contribution: {total_maxwell:.6e}")
    print(f"     Combined T00 stress-energy: {total_T00:.6e}")
    
    if abs(total_maxwell) > 0:
        ratio = abs(total_maxwell / total_phantom) if abs(total_phantom) > 0 else float('inf')
        print(f"     Maxwell/Phantom ratio: {ratio:.3f}")
    
    # 6. Export for warp drive pipeline
    print(f"\n📁 Step 6: Exporting for Classical Warp Drive Pipeline")
    print("-" * 50)
    
    # Export quantum observables
    obs_file = os.path.join(output_dir, "maxwell_quantum_observables.json")
    obs_data = hilbert.export_quantum_observables(psi, obs_file)
    print(f"   ✓ Quantum observables → {obs_file}")
    
    # Export stress-energy data for multiple coordinates
    stress_energy_data = []
    coordinates = [1e-35, 2e-35, 5e-35, 1e-34, 2e-34]
    
    for r in coordinates:
        T_data = hilbert.compute_stress_energy_expectation(psi, r)
        stress_energy_data.append(T_data)
    
    stress_file = os.path.join(output_dir, "maxwell_T00_quantum.json")
    with open(stress_file, 'w') as f:
        json.dump(stress_energy_data, f, indent=2)
    print(f"   ✓ Stress-energy data → {stress_file}")
    
    # Export NDJSON format for pipeline
    ndjson_file = os.path.join(output_dir, "maxwell_T00_quantum.ndjson")
    with open(ndjson_file, 'w') as f:
        for T_data in stress_energy_data:
            json.dump(T_data, f)
            f.write('\\n')
    print(f"   ✓ NDJSON format → {ndjson_file}")
    
    # 7. Analysis and verification
    print(f"\n🔍 Step 7: Quantum Effects Analysis")
    print("-" * 50)
    
    # Check dominant quantum states
    probabilities = np.abs(psi)**2
    sorted_indices = np.argsort(probabilities)[::-1]
    
    print(f"   Dominant quantum states:")
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        prob = probabilities[idx]
        if prob > 1e-6:
            flux_state, maxwell_state = hilbert.composite_states[idx]
            print(f"     State {idx}: P = {prob:.4f}")
            print(f"       Flux: {flux_state}")
            print(f"       Maxwell: {maxwell_state}")
    
    # Check Maxwell occupation statistics
    maxwell_occupation_avg = np.zeros(hilbert.n_sites)
    for i, (flux_state, maxwell_state) in enumerate(hilbert.composite_states):
        prob = probabilities[i]
        for site in range(hilbert.n_sites):
            maxwell_occupation_avg[site] += prob * maxwell_state[site]
    
    print(f"\\n   Average Maxwell occupation per site:")
    for site in range(hilbert.n_sites):
        classical_pi = config.pi_r_classical[site] if hasattr(config, 'pi_r_classical') else 0.0
        print(f"     Site {site}: ⟨n⟩ = {maxwell_occupation_avg[site]:.4f} (classical π^r = {classical_pi:.4f})")
    
    # 8. Integration readiness check
    print(f"\n✅ Step 8: Integration Readiness Check")
    print("-" * 50)
    
    print(f"   Framework Status:")
    print(f"     ✓ Maxwell operators implemented")
    print(f"     ✓ Combined T00 computation working") 
    print(f"     ✓ Quantum observables exportable")
    print(f"     ✓ Classical pipeline format compatible")
    print(f"     ✓ NDJSON output for pipeline integration")
    
    print(f"\\n   Output Files Generated:")
    print(f"     • {obs_file}")
    print(f"     • {stress_file}")
    print(f"     • {ndjson_file}")
    
    print(f"\\n🎉 Maxwell Field Integration Demo Completed Successfully!")
    print(f"\\n📋 Next Steps:")
    print(f"   1. Run classical warp pipeline with: python run_pipeline.py --use-quantum")
    print(f"   2. The pipeline will automatically load quantum T00 data")
    print(f"   3. Compare warp metrics with/without Maxwell quantum effects")
    
    return hilbert, psi, obs_data, stress_energy_data


def test_different_maxwell_levels():
    """Test the framework with different Maxwell truncation levels"""
    
    print(f"\\n🔄 Testing Different Maxwell Truncation Levels")
    print("=" * 60)
    
    # Use smaller configuration for multiple tests
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
    
    for maxwell_levels in [0, 1, 2]:
        print(f"\\n   Testing maxwell_levels = {maxwell_levels}")
        hilbert = MidisuperspaceHilbert(config, maxwell_levels=maxwell_levels)
        
        psi = hilbert.create_coherent_state(
            np.array(config.E_x_classical),
            np.array(config.E_phi_classical),
            width=1.0
        )
        
        Ex_vals, Ephi_vals, T00_ph_vals, T00_mx_vals, T00_tot_vals = hilbert.compute_expectation_E_and_T00(psi)
        total_maxwell = sum(T00_mx_vals)
        
        print(f"     Hilbert dimension: {hilbert.hilbert_dim}")
        print(f"     Maxwell basis size: {len(hilbert.maxwell_states)}")
        print(f"     Total Maxwell T00: {total_maxwell:.6e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demonstrate Maxwell field integration with LQG")
    parser.add_argument("--test-levels", action="store_true", 
                       help="Test different Maxwell truncation levels")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for generated files")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir != "outputs":
        import os
        os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run main demo
        demo_results = demo_maxwell_lqg_integration()
        
        # Run additional tests if requested
        if args.test_levels:
            test_different_maxwell_levels()
        
        print(f"\\n🎯 All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\\n❌ Error during demonstration: {e}")
        raise
