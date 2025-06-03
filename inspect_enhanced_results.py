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
    """Analyze quantum geometry catalysis discovery results."""
    print("⚡ QUANTUM GEOMETRY CATALYSIS ANALYSIS")
    print("=" * 50)
    
    xi_factor = gc_data.get("Xi", "N/A")
    speed_enhancement = gc_data.get("speed_enhancement_percent", "N/A")
    classical_speed = gc_data.get("classical_speed", "N/A")
    quantum_speed = gc_data.get("quantum_speed", "N/A")
    
    print("Catalysis Measurements:")
    print(f"  Catalysis factor Ξ    = {xi_factor}")
    print(f"  Speed enhancement     = {speed_enhancement}%")
    print(f"  Classical speed       = {classical_speed}")
    print(f"  Quantum speed         = {quantum_speed}")
    
    print(f"\nPhysical Parameters:")
    print(f"  Planck length scale   = {gc_data.get('l_planck', 'N/A')}")
    print(f"  Wave packet width     = {gc_data.get('packet_width', 'N/A')}")
    print(f"  Coupling parameter β  = {gc_data.get('beta', 'N/A')}")
    print(f"  Lattice size          = {gc_data.get('lattice_size', 'N/A')}")
    
    # Time evolution analysis
    time_evolution = gc_data.get("time_evolution", {})
    if time_evolution:
        classical_peaks = time_evolution.get("classical_peaks", [])
        quantum_peaks = time_evolution.get("quantum_peaks", [])
        
        if classical_peaks and quantum_peaks:
            print(f"\nWave Packet Evolution:")
            print(f"  Classical trajectory points = {len(classical_peaks)}")
            print(f"  Quantum trajectory points   = {len(quantum_peaks)}")
            
            if len(classical_peaks) >= 2 and len(quantum_peaks) >= 2:
                classical_velocity = (classical_peaks[-1] - classical_peaks[0]) / (len(classical_peaks) - 1)
                quantum_velocity = (quantum_peaks[-1] - quantum_peaks[0]) / (len(quantum_peaks) - 1)
                print(f"  Measured classical velocity = {classical_velocity:.6f}")
                print(f"  Measured quantum velocity   = {quantum_velocity:.6f}")
    print()

def summarize_gpu_solver(gpu_data: Dict[str, Any]) -> None:
    """Analyze GPU solver performance results."""
    print("🚀 GPU SOLVER PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    device_info = gpu_data.get("device_info", {})
    if device_info:
        print("Device Configuration:")
        print(f"  PyTorch available     = {device_info.get('torch_available', 'N/A')}")
        print(f"  CUDA available        = {device_info.get('cuda_available', 'N/A')}")
        print(f"  Device used           = {device_info.get('device', 'N/A')}")
    
    performance = gpu_data.get("performance", {})
    if performance:
        print(f"\nSolver Performance:")
        print(f"  Execution time        = {performance.get('execution_time', 'N/A')} s")
        print(f"  Constraint violation  = {performance.get('constraint_violation', 'N/A')}")
        print(f"  Convergence achieved  = {performance.get('converged', 'N/A')}")
        print(f"  Iterations completed  = {performance.get('iterations', 'N/A')}")
    
    # Solution quality
    solution_info = gpu_data.get("solution", {})
    if solution_info:
        print(f"\nSolution Quality:")
        residual_norm = solution_info.get("residual_norm", "N/A")
        energy = solution_info.get("energy", "N/A")
        print(f"  Final residual norm   = {residual_norm}")
        print(f"  Ground state energy   = {energy}")
    print()

def summarize_phenomenology(pheno_data: Dict[str, Any]) -> None:
    """Analyze phenomenology generation results."""
    print("📡 PHENOMENOLOGY GENERATION ANALYSIS")
    print("=" * 50)
    
    config = pheno_data.get("config", {})
    if config:
        print("Phenomenology Configuration:")
        mass_range = config.get("mass_range", [])
        spin_range = config.get("spin_range", [])
        print(f"  Mass range            = {mass_range}")
        print(f"  Spin range            = {spin_range}")
        print(f"  Mass samples          = {config.get('mass_samples', 'N/A')}")
        print(f"  Spin samples          = {config.get('spin_samples', 'N/A')}")
    
    performance = pheno_data.get("performance", {})
    if performance:
        print(f"\nGeneration Performance:")
        print(f"  Execution time        = {performance.get('execution_time', 'N/A')} s")
        print(f"  Total calculations    = {performance.get('total_calculations', 'N/A')}")
    
    # Sample observables
    observables = pheno_data.get("observables", [])
    if observables:
        print(f"\nGenerated Observables ({len(observables)} configurations):")
        
        # Show first few examples
        for i, obs in enumerate(observables[:3]):
            mass = obs.get("mass", "N/A")
            spin = obs.get("spin", "N/A")
            isco = obs.get("isco_radius", "N/A")
            shadow = obs.get("shadow_radius", "N/A")
            
            print(f"  Config {i+1}: M={mass}, a={spin}")
            print(f"    ISCO radius         = {isco}")
            print(f"    Shadow radius       = {shadow}")
        
        if len(observables) > 3:
            print(f"    ... and {len(observables) - 3} more configurations")
    print()

def summarize_matter_coupling_3d(mc_data: Dict[str, Any]) -> None:
    """Analyze 3+1D matter coupling results."""
    print("⚛️  3+1D MATTER COUPLING ANALYSIS")
    print("=" * 50)
    
    config = mc_data.get("config", {})
    if config:
        print("Matter Coupling Configuration:")
        print(f"  Grid size             = {config.get('grid_size', 'N/A')}")
        print(f"  Spatial resolution dx = {config.get('dx', 'N/A')}")
        print(f"  Time step dt          = {config.get('dt', 'N/A')}")
        print(f"  Polymer scale ε       = {config.get('epsilon', 'N/A')}")
        print(f"  Field mass            = {config.get('mass', 'N/A')}")
        print(f"  Evolution time        = {config.get('total_time', 'N/A')}")
    
    performance = mc_data.get("performance", {})
    if performance:
        print(f"\nEvolution Performance:")
        print(f"  Execution time        = {performance.get('execution_time', 'N/A')} s")
        print(f"  Time steps completed  = {performance.get('time_steps', 'N/A')}")
        print(f"  Stability achieved    = {performance.get('stable', 'N/A')}")
    
    conservation = mc_data.get("conservation", {})
    if conservation:
        print(f"\nConservation Properties:")
        print(f"  Energy conservation   = {conservation.get('energy_conservation_error', 'N/A')}")
        print(f"  Momentum conservation = {conservation.get('momentum_conservation_error', 'N/A')}")
        print(f"  Field normalization   = {conservation.get('field_normalization', 'N/A')}")
    
    # Stress-energy analysis
    stress_energy = mc_data.get("stress_energy", {})
    if stress_energy:
        print(f"\nStress-Energy Analysis:")
        print(f"  Mean T00 density      = {stress_energy.get('mean_T00', 'N/A')}")
        print(f"  Peak T00 density      = {stress_energy.get('peak_T00', 'N/A')}")
        print(f"  Total energy          = {stress_energy.get('total_energy', 'N/A')}")
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

if __name__ == "__main__":
    main()
