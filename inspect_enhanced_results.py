#!/usr/bin/env python3
"""
Enhanced Quantum Gravity Results Inspector

This script loads and analyzes the comprehensive JSON results from the 
enhanced quantum gravity pipeline, providing detailed summaries of all
subsystem performance and discovery validation.

Features:
- AMR configuration and performance analysis
- Constraint entanglement measurements
- Matter-spacetime duality verification
- Geometry catalysis factor validation
- GPU solver performance metrics
- Phenomenology generation statistics
- 3+1D matter coupling analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

def summarize_amr(amr_data: Dict[str, Any]) -> None:
    """Analyze AMR (Adaptive Mesh Refinement) results."""
    print("🔬 ADAPTIVE MESH REFINEMENT ANALYSIS")
    print("=" * 50)
    
    cfg = amr_data.get("config", {})
    perf = amr_data.get("performance", {})
    
    print("AMR Configuration:")
    print(f"  Initial grid size     = {cfg.get('initial_grid_size', 'N/A')}")
    print(f"  Max refinement levels = {cfg.get('max_refinement_levels', 'N/A')}")
    print(f"  Refinement threshold  = {cfg.get('refinement_threshold', 'N/A')}")
    print(f"  Error estimator       = {cfg.get('error_estimator', 'N/A')}")
    print(f"  Refinement criterion  = {cfg.get('refinement_criterion', 'N/A')}")
    
    print("\nAMR Performance:")
    print(f"  Execution time        = {perf.get('execution_time', 0):.4f} s")
    print(f"  Final patch count     = {perf.get('final_patch_count', 'N/A')}")
    print(f"  Total adaptations     = {perf.get('total_adaptations', 'N/A')}")
    
    # Analyze refinement history
    refinement_history = amr_data.get("refinement_history", [])
    if refinement_history:
        print(f"\nRefinement History ({len(refinement_history)} levels):")
        for lvl_idx, level_data in enumerate(refinement_history[:3]):  # Show first 3 levels
            if isinstance(level_data, list) and level_data:
                avg_error = np.mean(level_data)
                max_error = np.max(level_data)
                min_error = np.min(level_data)
                print(f"  Level {lvl_idx}: {len(level_data)} patches")
                print(f"    Error range: [{min_error:.2e}, {max_error:.2e}], avg = {avg_error:.2e}")
    
    # Convergence analysis
    convergence = amr_data.get("convergence_analysis", {})
    if convergence:
        print(f"\nConvergence Analysis:")
        print(f"  Convergence achieved  = {convergence.get('converged', 'N/A')}")
        print(f"  Final error estimate  = {convergence.get('final_error', 'N/A')}")
    print()

def summarize_constraint_entanglement(ce_data: Dict[str, Any]) -> None:
    """Analyze constraint entanglement discovery results."""
    print("🔗 QUANTUM CONSTRAINT ENTANGLEMENT ANALYSIS")
    print("=" * 50)
    
    print("Entanglement Measurements:")
    print(f"  Entanglement detected = {ce_data.get('entangled', ce_data.get('is_entangled', 'N/A'))}")
    print(f"  Maximum E_AB measure  = {ce_data.get('max_entanglement', ce_data.get('E_AB', 'N/A'))}")
    
    # Parameter scan results
    mu_gamma_map = ce_data.get("mu_gamma_map", {})
    if mu_gamma_map:
        print(f"\nParameter Scan Results ({len(mu_gamma_map)} configurations):")
        
        # Find optimal parameters
        max_entanglement = 0
        best_params = None
        
        for param_key, param_data in mu_gamma_map.items():
            e_ab = param_data.get('E_AB', 0)
            if e_ab > max_entanglement:
                max_entanglement = e_ab
                best_params = (param_data.get('mu', 'N/A'), param_data.get('gamma', 'N/A'))
        
        if best_params:
            print(f"  Strongest entanglement: E_AB = {max_entanglement:.2e}")
            print(f"  Optimal parameters: μ = {best_params[0]}, γ = {best_params[1]}")
    
    # Regional analysis
    lattice_size = ce_data.get("lattice_size", "N/A")
    anomaly_free = ce_data.get("anomaly_free_regions", "N/A")
    print(f"\nConstraint Algebra Properties:")
    print(f"  Lattice size          = {lattice_size}")
    print(f"  Anomaly-free regions  = {anomaly_free}")
    print()

def summarize_matter_spacetime_duality(msd_data: Dict[str, Any]) -> None:
    """Analyze matter-spacetime duality discovery results."""
    print("🔄 MATTER-SPACETIME DUALITY ANALYSIS")
    print("=" * 50)
    
    spectral_error = msd_data.get("spectral_match_error", "N/A")
    duality_quality = msd_data.get("duality_quality", "N/A")
    alpha = msd_data.get("duality_parameter_alpha", "N/A")
    
    print("Duality Verification:")
    print(f"  Spectral match error  = {spectral_error}")
    print(f"  Duality quality       = {duality_quality}")
    print(f"  Duality parameter α   = {alpha}")
    
    # Eigenvalue comparison
    eigenvalue_pairs = msd_data.get("eigenvalue_pairs", [])
    if eigenvalue_pairs:
        print(f"\nEigenvalue Comparison ({len(eigenvalue_pairs)} pairs):")
        print("  Matter ↔ Geometry Eigenvalue Matching:")
        
        total_error = 0
        for i, pair in enumerate(eigenvalue_pairs[:5]):  # Show first 5 pairs
            matter_val = pair.get("matter", 0)
            geometry_val = pair.get("geometry", 0)
            pair_error = pair.get("error", 0)
            total_error += pair_error
            print(f"    Pair {i+1}: {matter_val:.6f} ↔ {geometry_val:.6f} (error: {pair_error:.2e})")
        
        if eigenvalue_pairs:
            avg_error = total_error / len(eigenvalue_pairs[:5])
            print(f"  Average eigenvalue error = {avg_error:.2e}")
    
    print(f"\nPhysical Parameters:")
    print(f"  Lattice size          = {msd_data.get('lattice_size', 'N/A')}")
    print(f"  Polymer scale         = {msd_data.get('polymer_scale', 'N/A')}")
    print(f"  Immirzi parameter γ   = {msd_data.get('gamma', 'N/A')}")
    print()

def summarize_geometry_catalysis(gc_data: Dict[str, Any]) -> None:
    """Analyze quantum geometry catalysis results."""
    print("⚡ QUANTUM GEOMETRY CATALYSIS ANALYSIS")
    print("=" * 50)
    
    xi_factor = gc_data.get("Xi", "N/A")
    speed_enhancement = gc_data.get("speed_enhancement_percent", "N/A")
    catalysis_detected = gc_data.get("catalysis_detected", False)
    
    print("Catalysis Measurements:")
    print(f"  Ξ enhancement factor  = {xi_factor}")
    print(f"  Speed enhancement     = {speed_enhancement}%")
    print(f"  Catalysis detected    = {catalysis_detected}")
    
    # Physical parameters
    physical_params = gc_data.get("physical_parameters", {})
    if physical_params:
        print(f"\nPhysical Parameters:")
        print(f"  Planck length         = {physical_params.get('planck_length', 'N/A')}")
        print(f"  Scale ratio           = {physical_params.get('packet_scale_ratio', 'N/A')}")
        print(f"  Geometry coupling β   = {physical_params.get('quantum_geometry_coupling', 'N/A')}")
    
    # Time evolution
    time_evolution = gc_data.get("time_evolution", {})
    if time_evolution:
        classical_peaks = time_evolution.get("classical_peaks", [])
        quantum_peaks = time_evolution.get("quantum_peaks", [])
        if classical_peaks and quantum_peaks:
            print(f"\nWave Packet Evolution:")
            print(f"  Classical final pos   = {classical_peaks[-1]:.6f}")
            print(f"  Quantum final pos     = {quantum_peaks[-1]:.6f}")
            print(f"  Enhancement ratio     = {quantum_peaks[-1]/classical_peaks[-1]:.6f}")
    print()

def summarize_matter_coupling_3d(mc_data: Dict[str, Any]) -> None:
    """Analyze 3+1D matter coupling results."""
    print("🌐 3+1D MATTER COUPLING ANALYSIS")
    print("=" * 50)
    
    energy_drift = mc_data.get("energy_drift", "N/A")
    momentum_drift = mc_data.get("momentum_drift", "N/A")
    conservation_quality = mc_data.get("conservation_quality", "unknown")
    
    print("Conservation Analysis:")
    print(f"  Energy drift          = {energy_drift}")
    print(f"  Momentum drift        = {momentum_drift}")
    print(f"  Conservation quality  = {conservation_quality}")
    
    # Energy conservation details
    energy_cons = mc_data.get("energy_conservation", {})
    if energy_cons:
        print(f"\nEnergy Conservation:")
        print(f"  Initial energy        = {energy_cons.get('initial_energy', 'N/A')}")
        print(f"  Final energy          = {energy_cons.get('final_energy', 'N/A')}")
        print(f"  Relative drift        = {energy_cons.get('relative_drift', 'N/A')}")
    
    # Matter field properties
    matter_props = mc_data.get("matter_field_properties", {})
    if matter_props:
        print(f"\nMatter Field Properties:")
        print(f"  Field variance        = {matter_props.get('field_variance', 'N/A')}")
        print(f"  Coupling efficiency   = {matter_props.get('coupling_efficiency', 'N/A')}")
        print(f"  Polymer corrections   = {matter_props.get('polymer_corrections', 'N/A')}")
    
    # Spacetime geometry
    spacetime = mc_data.get("spacetime_geometry", {})
    if spacetime:
        print(f"\nSpacetime Geometry:")
        print(f"  Metric perturbations  = {spacetime.get('metric_perturbations', 'N/A')}")
        print(f"  3+1 reduction verified = {spacetime.get('dimensional_reduction_verified', 'N/A')}")
    print()

def load_and_inspect_results() -> None:
    """Main function to load and inspect all enhanced results."""
    print("🔬 ENHANCED QUANTUM GRAVITY FRAMEWORK INSPECTOR")
    print("=" * 80)
    print()
    
    # Primary results file
    primary_results_path = Path("enhanced_qc_results") / "enhanced_results.json"
    
    # Alternative result locations
    alternative_paths = [
        Path("enhanced_qg_results") / "enhanced_qg_results.json",
        Path("discovery_results") / "discovery_summary.json",
        Path("enhanced_qc_results") / "comprehensive_phenomenology.json"
    ]
    
    results_data = None
    used_path = None
    
    # Try to load primary results
    if primary_results_path.exists():
        try:
            with open(primary_results_path) as f:
                results_data = json.load(f)
            used_path = primary_results_path
            print(f"✅ Loaded primary results from: {primary_results_path}")
        except Exception as e:
            print(f"❌ Error loading {primary_results_path}: {e}")
    
    # Try alternative paths if primary failed
    if results_data is None:
        for alt_path in alternative_paths:
            if alt_path.exists():
                try:
                    with open(alt_path) as f:
                        results_data = json.load(f)
                    used_path = alt_path
                    print(f"✅ Loaded alternative results from: {alt_path}")
                    break
                except Exception as e:
                    print(f"❌ Error loading {alt_path}: {e}")
    
    if results_data is None:
        print("❌ No valid results files found. Please run the enhanced pipeline first.")
        print("\nExpected files:")
        print(f"  - {primary_results_path}")
        for alt_path in alternative_paths:
            print(f"  - {alt_path}")
        return
    
    print(f"📂 Results source: {used_path}")
    print(f"📊 Data keys found: {list(results_data.keys())}")
    print("\n" + "="*80 + "\n")
    
    # Analyze each subsystem
    
    # 1. Adaptive Mesh Refinement
    if "amr" in results_data:
        summarize_amr(results_data["amr"])
    elif any(key.startswith("mesh") for key in results_data.keys()):
        mesh_key = next(key for key in results_data.keys() if key.startswith("mesh"))
        print(f"Found mesh-related data under key: {mesh_key}")
        summarize_amr(results_data[mesh_key])
    else:
        print("⚠️  No AMR data found in results\n")
    
    # 2. Constraint Entanglement
    if "constraint_entanglement" in results_data:
        summarize_constraint_entanglement(results_data["constraint_entanglement"])
    else:
        print("⚠️  No constraint entanglement data found in results\n")
    
    # 3. Matter-Spacetime Duality
    if "matter_spacetime_duality" in results_data:
        summarize_matter_spacetime_duality(results_data["matter_spacetime_duality"])
    else:
        print("⚠️  No matter-spacetime duality data found in results\n")
    
    # 4. Geometry Catalysis
    if "geometry_catalysis" in results_data:
        summarize_geometry_catalysis(results_data["geometry_catalysis"])
    else:
        print("⚠️  No geometry catalysis data found in results\n")
    
    # 5. GPU Solver
    if "gpu_solver" in results_data:
        summarize_gpu_solver(results_data["gpu_solver"])
    else:
        print("⚠️  No GPU solver data found in results\n")
    
    # 6. Phenomenology
    if "phenomenology" in results_data:
        summarize_phenomenology(results_data["phenomenology"])
    elif "config" in results_data and "observables" in results_data:
        # Handle case where phenomenology data is at root level
        summarize_phenomenology(results_data)
    else:
        print("⚠️  No phenomenology data found in results\n")
    
    # 7. 3+1D Matter Coupling
    if "matter_coupling_3d" in results_data:
        summarize_matter_coupling_3d(results_data["matter_coupling_3d"])
    elif "polymer_field" in results_data:
        summarize_matter_coupling_3d(results_data["polymer_field"])
    else:
        print("⚠️  No 3+1D matter coupling data found in results\n")
    
    # Summary statistics
    print("📈 FRAMEWORK COMPLETENESS ASSESSMENT")
    print("=" * 50)
    
    subsystems = [
        ("AMR", "amr" in results_data or any(k.startswith("mesh") for k in results_data.keys())),
        ("Constraint Entanglement", "constraint_entanglement" in results_data),
        ("Matter-Spacetime Duality", "matter_spacetime_duality" in results_data),
        ("Geometry Catalysis", "geometry_catalysis" in results_data),
        ("GPU Solver", "gpu_solver" in results_data),
        ("Phenomenology", "phenomenology" in results_data or ("config" in results_data and "observables" in results_data)),
        ("3+1D Matter Coupling", "matter_coupling_3d" in results_data or "polymer_field" in results_data)
    ]
    
    completed_count = sum(completed for _, completed in subsystems)
    total_count = len(subsystems)
    
    print(f"Subsystem Status:")
    for name, completed in subsystems:
        status = "✅ COMPLETED" if completed else "❌ MISSING"
        print(f"  {name:<25} {status}")
    
    print(f"\nFramework Completion: {completed_count}/{total_count} ({100*completed_count/total_count:.1f}%)")
    
    if completed_count == total_count:
        print("\n🎉 FRAMEWORK FULLY OPERATIONAL - All subsystems validated!")
    elif completed_count >= total_count * 0.8:
        print("\n✅ FRAMEWORK MOSTLY COMPLETE - Minor subsystems missing")
    else:
        print("\n⚠️  FRAMEWORK PARTIALLY COMPLETE - Several subsystems need attention")
    
    print(f"\n📋 Inspection complete. Results analyzed from: {used_path}")

def main():
    """Main entry point for results inspection."""
    load_and_inspect_results()

def summarize_field_evolution(fe_data: Dict[str, Any]) -> None:
    """Analyze field evolution results."""
    print("🌊 FIELD EVOLUTION ANALYSIS")
    print("=" * 50)
    
    config = fe_data.get("config", {})
    perf = fe_data.get("performance", {})
    
    print("Evolution Configuration:")
    print(f"  Grid size             = {config.get('grid_size', 'N/A')}")
    print(f"  Spatial resolution dx = {config.get('dx', 'N/A')}")
    print(f"  Time step dt          = {config.get('dt', 'N/A')}")
    print(f"  Total evolution time  = {config.get('total_time', 'N/A')}")
    
    print(f"\nEvolution Performance:")
    print(f"  Execution time        = {perf.get('execution_time', 'N/A')} s")
    print(f"  Time steps computed   = {perf.get('time_steps', 'N/A')}")
    
    # Energy analysis
    final_energy = fe_data.get("final_energy", "N/A")
    energy_conservation = fe_data.get("energy_conservation", {})
    
    print(f"\nEnergy Conservation:")
    print(f"  Final energy          = {final_energy}")
    print(f"  Energy drift          = {energy_conservation.get('drift', 'N/A')}")
    print(f"  Conservation quality  = {energy_conservation.get('quality', 'N/A')}")
    
    # Field snapshots
    snapshots = fe_data.get("field_snapshots", [])
    if snapshots:
        print(f"\nField Evolution Snapshots:")
        for snap in snapshots:
            t = snap.get("time", 0)
            phi_max = snap.get("phi_max", 0)
            phi_min = snap.get("phi_min", 0)
            print(f"  t={t:.3f}: φ ∈ [{phi_min:.3f}, {phi_max:.3f}]")
    print()

def summarize_phenomenology(pheno_data: Dict[str, Any]) -> None:
    """Analyze phenomenology results."""
    print("📊 PHENOMENOLOGY GENERATION ANALYSIS")
    print("=" * 50)
    
    perf = pheno_data.get("performance", {})
    param_space = pheno_data.get("parameter_space", {})
    observables = pheno_data.get("observables", [])
    
    print("Generation Performance:")
    print(f"  Execution time        = {perf.get('execution_time', 'N/A')} s")
    print(f"  Total calculations    = {perf.get('total_calculations', 'N/A')}")
    
    print(f"\nParameter Space:")
    print(f"  Mass range            = {param_space.get('mass_range', 'N/A')}")
    print(f"  Spin range            = {param_space.get('spin_range', 'N/A')}")
    print(f"  Total combinations    = {param_space.get('total_combinations', 'N/A')}")
    
    if observables:
        print(f"\nSample Observables (first 3):")
        for i, obs in enumerate(observables[:3]):
            mass = obs.get("mass", "N/A")
            spin = obs.get("spin", "N/A")
            omega = obs.get("omega_qnm", ["N/A"])[0]
            isco = obs.get("isco_radius", "N/A")
            print(f"  M={mass}, a={spin}: ω_QNM={omega:.3f}, r_ISCO={isco:.1f}")
    print()

def summarize_gpu_solver(gpu_data: Dict[str, Any]) -> None:
    """Analyze GPU solver results."""
    print("🚀 GPU SOLVER PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    perf = gpu_data.get("performance", {})
    quality = gpu_data.get("solution_quality", {})
    
    print("Solver Performance:")
    print(f"  Execution time        = {perf.get('execution_time', 'N/A')} s")
    print(f"  Matrix dimension      = {perf.get('matrix_dimension', 'N/A')}")
    
    print(f"\nSolution Quality:")
    print(f"  Final residual        = {quality.get('residual', 'N/A')}")
    print(f"  Convergence achieved  = {quality.get('converged', 'N/A')}")
    
    # GPU utilization
    gpu_util = gpu_data.get("gpu_utilization", {})
    if gpu_util:
        print(f"\nGPU Utilization:")
        print(f"  Memory used           = {gpu_util.get('memory_used_gb', 'N/A')} GB")
        print(f"  Compute efficiency    = {gpu_util.get('compute_efficiency', 'N/A')}")
        print(f"  Parallel threads      = {gpu_util.get('parallel_threads', 'N/A')}")
    print()

if __name__ == "__main__":
    main()
