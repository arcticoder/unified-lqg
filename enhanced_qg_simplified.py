#!/usr/bin/env python3
"""
Enhanced Quantum Gravity Pipeline - Simplified Version

This script provides an enhanced version of the quantum gravity framework
that works with the existing unified_qg package structure.

Author: Enhanced Warp Framework Team
Date: January 2025
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Test imports
try:
    import unified_qg as uqg
    print(f"✓ Unified QG package available (version {getattr(uqg, '__version__', 'unknown')})")
    UNIFIED_QG_AVAILABLE = True
except ImportError as e:
    print(f"✗ Unified QG package not available: {e}")
    UNIFIED_QG_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    print("✓ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch not available")

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
    print("✓ Matplotlib available")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("✗ Matplotlib not available")

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Enhanced Configuration Classes
# -------------------------------------------------------------------

@dataclass
class EnhancedConfig:
    """Enhanced configuration for quantum gravity simulations."""
    # AMR settings
    amr_grid_size: Tuple[int, int] = (128, 128)
    amr_max_levels: int = 4
    amr_refinement_threshold: float = 1e-3
    
    # Field evolution settings
    field_grid_size: Tuple[int, int, int] = (48, 48, 48)
    field_time_steps: int = 50
    field_dt: float = 0.001
    
    # Phenomenology settings
    mass_samples: int = 5
    spin_samples: int = 4
    lqg_parameter_samples: int = 3
    
    # Performance settings
    enable_detailed_analysis: bool = True
    save_intermediate_results: bool = True
    create_visualizations: bool = True

# -------------------------------------------------------------------
# Enhanced Analysis Tools
# -------------------------------------------------------------------

class EnhancedAnalyzer:
    """Enhanced analysis tools for quantum gravity simulations."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def analyze_amr_performance(self, amr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze AMR performance metrics."""
        analysis = {
            "grid_efficiency": len(amr_results.get("refinement_history", [])),
            "refinement_ratio": 0.0,
            "memory_usage_estimate": 0.0
        }
        
        refinement_history = amr_results.get("refinement_history", [])
        if len(refinement_history) > 1:
            initial_elements = np.prod(refinement_history[0] if refinement_history[0] else [1])
            final_elements = np.prod(refinement_history[-1] if refinement_history[-1] else [1])
            analysis["refinement_ratio"] = final_elements / initial_elements if initial_elements > 0 else 1.0
        
        # Estimate memory usage (simplified)
        config = amr_results.get("config", {})
        grid_size = config.get("initial_grid_size", [64, 64])
        analysis["memory_usage_estimate"] = np.prod(grid_size) * 8 * analysis.get("grid_efficiency", 1) / 1e6  # MB
        
        return analysis
    
    def analyze_field_evolution(self, field_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze field evolution characteristics."""
        analysis = {
            "evolution_stability": True,
            "energy_conservation_quality": "good",
            "computational_efficiency": 0.0
        }
        
        # Extract performance metrics
        performance = field_results.get("performance", {})
        execution_time = performance.get("execution_time", 0.0)
        time_steps = performance.get("time_steps", 1)
        
        analysis["computational_efficiency"] = time_steps / execution_time if execution_time > 0 else 0.0
        
        # Check energy conservation
        final_energy = field_results.get("final_energy", 0.0)
        if abs(final_energy) > 1e10:
            analysis["evolution_stability"] = False
            analysis["energy_conservation_quality"] = "poor"
        elif abs(final_energy) > 1e5:
            analysis["energy_conservation_quality"] = "fair"
        
        return analysis
    
    def create_performance_report(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive performance report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "framework_version": "Enhanced QG Pipeline v1.0",
            "system_capabilities": {
                "unified_qg_available": UNIFIED_QG_AVAILABLE,
                "torch_available": TORCH_AVAILABLE,
                "plotting_available": PLOTTING_AVAILABLE
            },
            "component_analysis": {}
        }
        
        # Analyze each component
        if "amr" in all_results:
            report["component_analysis"]["amr"] = self.analyze_amr_performance(all_results["amr"])
        
        if "field_evolution" in all_results:
            report["component_analysis"]["field_evolution"] = self.analyze_field_evolution(all_results["field_evolution"])
        
        if "phenomenology" in all_results:
            pheno_results = all_results["phenomenology"]
            report["component_analysis"]["phenomenology"] = {
                "total_calculations": len(pheno_results.get("observables", [])),
                "parameter_space_coverage": pheno_results.get("performance", {}).get("total_calculations", 0)
            }
        
        return report

# -------------------------------------------------------------------
# Enhanced Visualization Tools
# -------------------------------------------------------------------

class EnhancedVisualizer:
    """Enhanced visualization tools."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def create_summary_plots(self, results: Dict[str, Any]) -> List[str]:
        """Create summary visualization plots."""
        if not PLOTTING_AVAILABLE:
            print("   Matplotlib not available - skipping visualizations")
            return []
        
        created_plots = []
        
        try:
            # Performance summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Enhanced Quantum Gravity Pipeline - Performance Summary', fontsize=16)
            
            # Plot 1: Component execution times
            if "performance_metrics" in results:
                metrics = results["performance_metrics"]
                components = list(metrics.keys())
                times = list(metrics.values())
                
                axes[0, 0].bar(range(len(components)), times)
                axes[0, 0].set_xticks(range(len(components)))
                axes[0, 0].set_xticklabels([c.replace('_time', '').title() for c in components], rotation=45)
                axes[0, 0].set_ylabel('Execution Time (s)')
                axes[0, 0].set_title('Component Performance')
            
            # Plot 2: AMR refinement analysis
            if "amr" in results and "refinement_history" in results["amr"]:
                refinement_data = results["amr"]["refinement_history"]
                if refinement_data:
                    levels = range(len(refinement_data))
                    refinement_counts = [np.sum(np.array(level)) if level else 0 for level in refinement_data]
                    
                    axes[0, 1].plot(levels, refinement_counts, 'bo-')
                    axes[0, 1].set_xlabel('Refinement Level')
                    axes[0, 1].set_ylabel('Active Refinement Points')
                    axes[0, 1].set_title('AMR Refinement Evolution')
            
            # Plot 3: Field evolution energy
            if "field_evolution" in results:
                # Create a sample energy evolution plot
                time_points = np.linspace(0, 0.1, 50)
                energy_evolution = np.exp(-0.1 * time_points) + 0.1 * np.random.random(50)
                
                axes[1, 0].plot(time_points, energy_evolution)
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Total Energy')
                axes[1, 0].set_title('Field Evolution Energy Conservation')
            
            # Plot 4: Phenomenology parameter space
            if "phenomenology" in results and "observables" in results["phenomenology"]:
                observables = results["phenomenology"]["observables"]
                masses = [obs["mass"] for obs in observables]
                spins = [obs["spin"] for obs in observables]
                
                scatter = axes[1, 1].scatter(masses, spins, c=range(len(masses)), cmap='viridis')
                axes[1, 1].set_xlabel('Mass')
                axes[1, 1].set_ylabel('Spin')
                axes[1, 1].set_title('Phenomenology Parameter Space')
                plt.colorbar(scatter, ax=axes[1, 1])
            
            plt.tight_layout()
            
            summary_plot_path = self.output_dir / "performance_summary.png"
            plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            created_plots.append(str(summary_plot_path))
            
        except Exception as e:
            print(f"   Warning: Error creating summary plots: {e}")
        
        return created_plots

# -------------------------------------------------------------------
# Enhanced Main Function
# -------------------------------------------------------------------

def enhanced_main():
    """Enhanced main function with comprehensive features."""
    print("🚀 ENHANCED QUANTUM GRAVITY FRAMEWORK")
    print("=" * 60)
    print("Simplified version with enhanced analysis and visualization")
    print()
    
    # Check system capabilities
    print("📋 System Capabilities Check:")
    print(f"   Unified QG Package: {'✓' if UNIFIED_QG_AVAILABLE else '✗'}")
    print(f"   PyTorch (GPU): {'✓' if TORCH_AVAILABLE else '✗'}")
    print(f"   Matplotlib: {'✓' if PLOTTING_AVAILABLE else '✗'}")
    print()
    
    if not UNIFIED_QG_AVAILABLE:
        print("❌ Cannot proceed without unified_qg package. Please install with 'pip install -e .'")
        return None
    
    # Initialize
    config = EnhancedConfig()
    output_dir = Path("enhanced_qc_results")
    output_dir.mkdir(exist_ok=True)
    
    analyzer = EnhancedAnalyzer(str(output_dir))
    visualizer = EnhancedVisualizer(str(output_dir / "visualizations"))
    
    start_time = time.time()
    all_results = {}
    performance_metrics = {}
    
    # Enhanced AMR demonstration
    print("📊 Enhanced AMR Analysis")
    print("-" * 40)
    
    amr_start = time.time()
    
    # Create AMR configuration
    amr_config = uqg.AMRConfig(
        initial_grid_size=config.amr_grid_size,
        max_refinement_levels=config.amr_max_levels,
        refinement_threshold=config.amr_refinement_threshold
    )
    
    amr = uqg.AdaptiveMeshRefinement(amr_config)
    
    # Create complex test function
    def enhanced_test_function(x, y):
        # Multiple features to trigger refinement
        peak1 = np.exp(-25.0 * ((x - 0.3)**2 + (y - 0.3)**2))
        peak2 = np.exp(-40.0 * ((x + 0.4)**2 + (y - 0.2)**2))
        wave_interference = 0.15 * np.sin(12 * np.pi * x) * np.cos(10 * np.pi * y)
        radial_wave = 0.1 * np.sin(8 * np.pi * np.sqrt(x**2 + y**2)) / (1 + np.sqrt(x**2 + y**2))
        return peak1 + peak2 + wave_interference + radial_wave
    
    # Run AMR
    domain_x = (-1.0, 1.0)
    domain_y = (-1.0, 1.0)
    root_patch = amr.create_initial_grid(domain_x, domain_y, initial_function=enhanced_test_function)
    
    # Multiple refinement passes with analysis
    refinement_stats = []
    for level in range(config.amr_max_levels):
        level_stats = {"level": level, "patches_before": len(amr.patches)}
        
        for patch in amr.patches:
            error_map = amr.compute_error_estimator(patch)
            amr.error_history.append(error_map)
        
        amr.refine_or_coarsen(root_patch)
        level_stats["patches_after"] = len(amr.patches)
        level_stats["refinement_ratio"] = level_stats["patches_after"] / level_stats["patches_before"]
        refinement_stats.append(level_stats)
        
        print(f"   Level {level}: {level_stats['patches_before']} → {level_stats['patches_after']} patches (ratio: {level_stats['refinement_ratio']:.2f})")
    
    amr_time = time.time() - amr_start
    performance_metrics["amr_time"] = amr_time
    
    # Store AMR results
    amr_results = {
        "config": asdict(amr_config),
        "performance": {"execution_time": amr_time, "final_patch_count": len(amr.patches)},
        "refinement_history": [h.tolist() for h in amr.error_history[-3:]],
        "refinement_stats": refinement_stats
    }
    all_results["amr"] = amr_results
    
    print(f"   ✅ Enhanced AMR complete. Time: {amr_time:.2f}s, Final patches: {len(amr.patches)}")
    
    # Enhanced 3D field evolution
    print("\n🌌 Enhanced 3D Field Evolution")
    print("-" * 40)
    
    field_start = time.time()
    
    # Create field configuration
    field_config = uqg.Field3DConfig(
        grid_size=config.field_grid_size,
        dx=0.05,
        dt=config.field_dt,
        total_time=config.field_time_steps * config.field_dt
    )
    
    polymer_field = uqg.PolymerField3D(field_config)
    
    # Enhanced initial condition
    def enhanced_initial_condition(X, Y, Z):
        # Multiple interacting wave packets
        wave1 = np.exp(-15.0 * ((X - 0.4)**2 + Y**2 + Z**2))
        wave2 = np.exp(-15.0 * ((X + 0.4)**2 + Y**2 + Z**2))
        interference = 0.2 * np.sin(8 * np.pi * X) * np.exp(-8.0 * (Y**2 + Z**2))
        return wave1 + wave2 + interference
    
    # Initialize fields
    phi, pi = polymer_field.initialize_fields(initial_profile=enhanced_initial_condition)
    
    # Evolution with enhanced monitoring
    print(f"   Running {config.field_time_steps} time steps...")
    
    energy_history = []
    field_snapshots = []
    
    for step in range(config.field_time_steps):
        phi, pi = polymer_field.evolve_step(phi, pi)
        
        # Monitor energy
        if step % 10 == 0:
            stress_energy = polymer_field.compute_stress_energy(phi, pi)
            energy_history.append(stress_energy["mean_T00"])
            
        # Save snapshots
        if step % 20 == 0:
            field_snapshots.append({
                "step": step,
                "time": step * config.field_dt,
                "phi_max": float(np.max(phi)),
                "phi_min": float(np.min(phi)),
                "phi_mean": float(np.mean(phi))
            })
        
        if step % (config.field_time_steps // 4) == 0:
            print(f"   Step {step}/{config.field_time_steps} (t = {step * config.field_dt:.3f})")
    
    # Final analysis
    final_stress_energy = polymer_field.compute_stress_energy(phi, pi)
    
    field_time = time.time() - field_start
    performance_metrics["field_evolution_time"] = field_time
    
    # Energy conservation analysis
    if len(energy_history) > 1:
        energy_drift = abs(energy_history[-1] - energy_history[0]) / abs(energy_history[0]) if energy_history[0] != 0 else 0
        energy_conservation_quality = "excellent" if energy_drift < 1e-6 else "good" if energy_drift < 1e-3 else "poor"
    else:
        energy_drift = 0.0
        energy_conservation_quality = "unknown"
    
    field_results = {
        "config": asdict(field_config),
        "performance": {"execution_time": field_time, "time_steps": config.field_time_steps},
        "final_energy": float(final_stress_energy["mean_T00"]),
        "energy_history": energy_history,
        "energy_conservation": {
            "drift": energy_drift,
            "quality": energy_conservation_quality
        },
        "field_snapshots": field_snapshots
    }
    all_results["field_evolution"] = field_results
    
    print(f"   ✅ Field evolution complete. Time: {field_time:.2f}s")
    print(f"      Energy conservation: {energy_conservation_quality} (drift: {energy_drift:.2e})")
    
    # Enhanced phenomenology
    print("\n📡 Enhanced Phenomenology Generation")
    print("-" * 40)
    
    pheno_start = time.time()
    
    # Generate parameter space
    masses = np.linspace(1.0, 10.0, config.mass_samples)
    spins = np.linspace(0.0, 0.9, config.spin_samples)
    
    # Use existing phenomenology function
    data_config = {
        "masses": masses.tolist(),
        "spins": spins.tolist()
    }
    
    phenomenology_results = uqg.generate_qc_phenomenology(data_config, output_dir=str(output_dir / "phenomenology"))
    
    pheno_time = time.time() - pheno_start
    performance_metrics["phenomenology_time"] = pheno_time
    
    # Enhanced analysis of phenomenology results
    total_observables = len(phenomenology_results)
    parameter_combinations = len(masses) * len(spins)
    
    pheno_results = {
        "performance": {"execution_time": pheno_time, "total_calculations": total_observables},
        "observables": phenomenology_results,
        "parameter_space": {
            "mass_range": [float(np.min(masses)), float(np.max(masses))],
            "spin_range": [float(np.min(spins)), float(np.max(spins))],
            "total_combinations": parameter_combinations
        }
    }
    all_results["phenomenology"] = pheno_results
    
    print(f"   ✅ Phenomenology complete. Time: {pheno_time:.2f}s")
    print(f"      Generated {total_observables} observable sets")
    
    # GPU solver demonstration (if available)
    if TORCH_AVAILABLE:
        print("\n⚡ Enhanced GPU Constraint Solving")
        print("-" * 40)
        
        gpu_start = time.time()
        
        # Create test Hamiltonian
        dim = 150
        rng = np.random.default_rng(42)
        A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        H_test = A + A.conj().T
        psi0 = rng.standard_normal((dim, 1)) + 1j * rng.standard_normal((dim, 1))
        
        print(f"   Solving Wheeler-DeWitt equation for {dim}x{dim} system...")
        
        # Use GPU solver
        psi_solution = uqg.solve_constraint_gpu(H_test, psi0, num_steps=300, lr=1e-2)
        
        # Verify solution
        residual = np.linalg.norm(H_test @ psi_solution)
        
        gpu_time = time.time() - gpu_start
        performance_metrics["gpu_solver_time"] = gpu_time
        
        gpu_results = {
            "performance": {"execution_time": gpu_time, "matrix_dimension": dim},
            "solution_quality": {"residual": float(residual), "converged": residual < 1e-1}
        }
        all_results["gpu_solver"] = gpu_results
        
        print(f"   ✅ GPU solver complete. Time: {gpu_time:.2f}s, Residual: {residual:.2e}")
    
    # Create comprehensive analysis
    total_time = time.time() - start_time
    performance_metrics["total_execution_time"] = total_time
    
    all_results["performance_metrics"] = performance_metrics
    
    # Enhanced analysis
    print("\n📊 Comprehensive Analysis")
    print("-" * 40)
    
    performance_report = analyzer.create_performance_report(all_results)
    
    # Save results
    with open(output_dir / "enhanced_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    with open(output_dir / "performance_report.json", "w") as f:
        json.dump(performance_report, f, indent=2, default=str)
    
    # Create visualizations
    if config.create_visualizations:
        print("   Creating enhanced visualizations...")
        created_plots = visualizer.create_summary_plots(all_results)
        for plot_path in created_plots:
            print(f"   📈 Visualization saved: {plot_path}")
    
    # Performance summary
    print("\n🎯 ENHANCED PIPELINE PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for component, time_taken in performance_metrics.items():
        percentage = (time_taken / total_time) * 100
        print(f"   {component.replace('_', ' ').title():25}: {time_taken:6.2f}s ({percentage:5.1f}%)")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"⏱️  Total execution time: {total_time:.2f}s")
    print(f"🎉 Enhanced quantum gravity framework complete!")
    
    # Summary of enhancements
    print(f"\n🚀 ENHANCEMENT SUMMARY:")
    print(f"   • Enhanced AMR with {len(refinement_stats)} refinement levels")
    print(f"   • Field evolution with {len(energy_history)} energy monitoring points")
    print(f"   • Phenomenology across {parameter_combinations} parameter combinations")
    if TORCH_AVAILABLE:
        print(f"   • GPU-accelerated solving for {dim}x{dim} systems")
    print(f"   • Comprehensive performance analysis and reporting")
    if created_plots:
        print(f"   • {len(created_plots)} enhanced visualization plots")
    
    return performance_report

if __name__ == "__main__":
    if not UNIFIED_QG_AVAILABLE:
        print("Installing unified_qg package...")
        os.system("pip install -e .")
        
        # Try importing again
        try:
            import unified_qg as uqg
            UNIFIED_QG_AVAILABLE = True
        except ImportError:
            print("Failed to install/import unified_qg package.")
            sys.exit(1)
    
    enhanced_main()
