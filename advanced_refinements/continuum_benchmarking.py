#!/usr/bin/env python3
"""
Continuum-Limit Benchmarking Framework

This module provides comprehensive continuum limit benchmarking by running
LQG calculations across extended lattice sizes (N=3 to N=15) and performing
sophisticated extrapolation analysis.

Key features:
- Extended lattice size scanning (N=3,5,7,9,11,13,15)
- Richardson extrapolation to continuum limit
- Multi-field stress-energy analysis
- Convergence quality assessment
- Performance scaling analysis
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Optional
import warnings
from scipy.optimize import curve_fit
from scipy.stats import linregress
import sys

# Add parent directory to path for imports
sys.path.append('..')

try:
    from lqg_fixed_components import (
        LatticeConfiguration, 
        LQGParameters, 
        KinematicalHilbertSpace, 
        MidisuperspaceHamiltonianConstraint
    )
    from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField
    from AdditionalMatterFieldsDemo import AdditionalMatterFieldsDemo
except ImportError as e:
    print(f"Warning: Could not import LQG components: {e}")
    print("Creating mock implementations for testing...")


class ContinuumBenchmarkingFramework:
    """
    Advanced continuum limit benchmarking framework.
    
    Systematically tests lattice sizes from N=3 to N=15 and extrapolates
    key observables to the continuum limit using Richardson extrapolation.
    """
    
    def __init__(self, output_dir: str = "outputs/continuum_benchmarking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.lattice_results = {}
        self.extrapolation_results = {}
        self.performance_metrics = {}
        
        print(f"📏 Continuum Benchmarking Framework initialized")
        print(f"   Output directory: {self.output_dir}")
        
    def run_single_lattice(self, N: int, throat_radius: float = 1.0) -> Dict[str, Any]:
        """
        Run complete LQG calculation for a single lattice size.
        
        Computes multi-field stress-energy, constraint violation,
        and key quantum observables.
        """
        print(f"  Computing N={N}...")
        start_time = time.time()
        
        try:
            # 1. Setup lattice configuration
            lattice_config = LatticeConfiguration(
                n_sites=N, 
                throat_radius=throat_radius
            )
            
            lqg_params = LQGParameters(
                mu_max=2,
                nu_max=2,
                basis_truncation=min(300, N * 60),  # Scale with N
                regularization_epsilon=1e-10
            )
            
            # 2. Build kinematic Hilbert space
            kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
            kin_space.generate_flux_basis()
            hilbert_dim = kin_space.dim
            
            # 3. Setup matter fields with appropriate profiles
            r_coords = np.linspace(0.5, 2.0, N)
            
            # Multi-field stress-energy calculation
            demo = AdditionalMatterFieldsDemo(n_sites=N)
            demo.setup_multi_field_framework()
            
            # Phantom field (built into demo)
            T00_phantom_op = demo._build_phantom_stress_energy()
            
            # Maxwell field with classical profile
            A_r = np.sin(np.pi * r_coords) / (r_coords + 0.1)  # Avoid division by zero
            pi_EM = np.cos(np.pi * r_coords) * 0.1
            maxwell = MaxwellField(n_sites=N)
            maxwell.load_classical_data(A_r_data=A_r, pi_EM_data=pi_EM)
            T00_maxwell_op = maxwell.compute_stress_energy_operator(kin_space)
            
            # Dirac field with Gaussian profile
            psi1 = 0.01 * np.exp(-((r_coords - 1.0)/0.3)**2)
            psi2 = 0.005 * np.exp(-((r_coords - 1.0)/0.3)**2)
            dirac = DiracField(n_sites=N, mass=0.1)
            dirac.load_classical_data(psi1, psi2)
            T00_dirac_op = dirac.compute_stress_energy_operator(kin_space)
            
            # Total stress-energy operator
            T00_total_op = T00_phantom_op + T00_maxwell_op + T00_dirac_op
            
            # 4. Solve constraint eigenvalue problem
            constraint_solver = MidisuperspaceHamiltonianConstraint(kin_space, lqg_params)
            constraint_solver.build_constraint_matrix()
            
            # Get lowest eigenvalue and ground state
            eigenvals, eigenvecs = constraint_solver.solve_constraint_eigenvalue_problem(k=3)
            omega_min_squared = float(np.real(eigenvals[0]))
            ground_state = eigenvecs[:, 0]
            
            # 5. Compute observables on ground state
            # Total stress-energy expectation
            total_T00 = float(np.real(
                ground_state.conj() @ T00_total_op @ ground_state
            ))
            
            # Individual components
            phantom_T00 = float(np.real(
                ground_state.conj() @ T00_phantom_op @ ground_state
            ))
            maxwell_T00 = float(np.real(
                ground_state.conj() @ T00_maxwell_op @ ground_state
            ))
            dirac_T00 = float(np.real(
                ground_state.conj() @ T00_dirac_op @ ground_state
            ))
            
            # Energy density variance (quantum fluctuations)
            T00_squared = T00_total_op @ T00_total_op
            T00_var = float(np.real(
                ground_state.conj() @ T00_squared @ ground_state
            )) - total_T00**2
            
            # Constraint violation measure
            H_matrix = constraint_solver.H_matrix
            constraint_violation = float(np.linalg.norm(H_matrix @ ground_state))
            
            # Higher eigenvalue gap (energy scale)
            if len(eigenvals) > 1:
                energy_gap = float(np.real(eigenvals[1] - eigenvals[0]))
            else:
                energy_gap = 0.0
            
            computation_time = time.time() - start_time
            
            result = {
                "N": N,
                "lattice_spacing": 1.0 / (N - 1),
                "hilbert_dimension": hilbert_dim,
                "omega_min_squared": omega_min_squared,
                "total_stress_energy": total_T00,
                "phantom_stress_energy": phantom_T00,
                "maxwell_stress_energy": maxwell_T00,
                "dirac_stress_energy": dirac_T00,
                "stress_energy_variance": T00_var,
                "constraint_violation": constraint_violation,
                "energy_gap": energy_gap,
                "computation_time": computation_time,
                "convergence_indicators": {
                    "relative_constraint_error": constraint_violation / max(abs(omega_min_squared), 1e-12),
                    "quantum_fluctuation_ratio": np.sqrt(abs(T00_var)) / max(abs(total_T00), 1e-12)
                }
            }
            
            print(f"    ✅ N={N}: ω²={omega_min_squared:.3e}, |T⁰⁰|={abs(total_T00):.3e}, "
                  f"dim={hilbert_dim}, t={computation_time:.1f}s")
            
            return result
            
        except Exception as e:
            print(f"    ❌ N={N}: Failed - {e}")
            return {"N": N, "error": str(e)}
    
    def extended_lattice_sweep(self, 
                              N_values: List[int] = None,
                              throat_radius: float = 1.0) -> Dict[str, Any]:
        """
        Perform extended lattice sweep across multiple sizes.
        """
        if N_values is None:
            N_values = [3, 5, 7, 9, 11, 13, 15]
        
        print(f"\n🔍 Extended lattice sweep...")
        print(f"   Lattice sizes: {N_values}")
        print(f"   Total calculations: {len(N_values)}")
        
        sweep_results = {}
        successful_results = []
        
        for N in N_values:
            result = self.run_single_lattice(N, throat_radius)
            sweep_results[N] = result
            
            if "error" not in result:
                successful_results.append(result)
        
        print(f"\n📊 Sweep completed: {len(successful_results)}/{len(N_values)} successful")
        
        self.lattice_results = sweep_results
        return {
            "sweep_results": sweep_results,
            "successful_count": len(successful_results),
            "failed_count": len(N_values) - len(successful_results),
            "N_values": N_values
        }
    
    def richardson_extrapolation(self, 
                                observable_name: str,
                                order: int = 2) -> Dict[str, Any]:
        """
        Perform Richardson extrapolation to continuum limit.
        
        For observable O(h), fits:
        O(h) = O_continuum + A*h^p + B*h^(p+1) + ...
        where h = lattice spacing = 1/(N-1)
        """
        print(f"  🔬 Richardson extrapolation for {observable_name}")
        
        # Extract successful data
        successful_data = [r for r in self.lattice_results.values() 
                          if "error" not in r and observable_name in r]
        
        if len(successful_data) < 3:
            return {"error": f"Insufficient data for {observable_name}"}
        
        N_values = np.array([r["N"] for r in successful_data])
        h_values = np.array([r["lattice_spacing"] for r in successful_data])
        obs_values = np.array([r[observable_name] for r in successful_data])
        
        # Try different extrapolation orders
        best_fit = None
        best_error = float('inf')
        best_order = order
        
        for test_order in range(1, min(len(obs_values), 4)):
            try:
                # Fit polynomial: obs = a0 + a1*h^order + a2*h^(order+1) + ...
                max_degree = min(len(h_values) - 1, 3)
                
                # Create design matrix
                A_matrix = np.column_stack([
                    h_values**(test_order + i) for i in range(max_degree)
                ])
                A_matrix = np.column_stack([np.ones(len(h_values)), A_matrix])
                
                # Least squares fit
                coeffs, residuals, rank, s = np.linalg.lstsq(A_matrix, obs_values, rcond=None)
                
                # Compute fit error
                fitted_values = A_matrix @ coeffs
                fit_error = np.sqrt(np.mean((obs_values - fitted_values)**2))
                
                if fit_error < best_error:
                    best_fit = coeffs
                    best_error = fit_error
                    best_order = test_order
                    
            except Exception:
                continue
        
        if best_fit is None:
            # Fallback to linear extrapolation
            slope, intercept, r_value, p_value, std_err = linregress(h_values, obs_values)
            continuum_value = intercept
            extrapolation_error = std_err
            convergence_order = 1.0
            r_squared = r_value**2
        else:
            # Continuum value is the constant term
            continuum_value = best_fit[0]
            
            # Estimate error from leading correction term
            if len(best_fit) > 1:
                leading_correction = best_fit[1] * (h_values[-1]**best_order)
                extrapolation_error = abs(leading_correction)
            else:
                extrapolation_error = best_error
            
            convergence_order = float(best_order)
            
            # Compute R-squared
            fitted_values = best_fit[0] + sum(
                best_fit[i+1] * (h_values**(best_order + i)) 
                for i in range(len(best_fit)-1)
            )
            ss_res = np.sum((obs_values - fitted_values)**2)
            ss_tot = np.sum((obs_values - np.mean(obs_values))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        result = {
            "observable": observable_name,
            "continuum_value": float(continuum_value),
            "extrapolation_error": float(extrapolation_error),
            "convergence_order": convergence_order,
            "r_squared": r_squared,
            "lattice_values": obs_values.tolist(),
            "lattice_spacings": h_values.tolist(),
            "N_values": N_values.tolist(),
            "fit_quality": "excellent" if r_squared > 0.99 else 
                          "good" if r_squared > 0.95 else 
                          "fair" if r_squared > 0.8 else "poor"
        }
        
        print(f"    ✅ {observable_name}: continuum = {continuum_value:.6f} ± {extrapolation_error:.6f}")
        print(f"       Convergence order: {convergence_order:.2f}, R² = {r_squared:.4f}")
        
        return result
    
    def extrapolate_all_observables(self) -> Dict[str, Any]:
        """Extrapolate all key observables to continuum limit."""
        
        print(f"\n🎯 Extrapolating all observables to continuum limit...")
        
        key_observables = [
            "omega_min_squared",
            "total_stress_energy", 
            "phantom_stress_energy",
            "maxwell_stress_energy",
            "dirac_stress_energy",
            "stress_energy_variance",
            "constraint_violation",
            "energy_gap"
        ]
        
        extrapolations = {}
        
        for obs in key_observables:
            result = self.richardson_extrapolation(obs)
            if "error" not in result:
                extrapolations[obs] = result
            else:
                print(f"    ⚠️  {obs}: {result['error']}")
        
        self.extrapolation_results = extrapolations
        return extrapolations
    
    def analyze_performance_scaling(self) -> Dict[str, Any]:
        """Analyze computational performance scaling."""
        
        print(f"\n⚡ Analyzing performance scaling...")
        
        successful_data = [r for r in self.lattice_results.values() if "error" not in r]
        
        if len(successful_data) < 3:
            return {"error": "Insufficient data for scaling analysis"}
        
        N_values = np.array([r["N"] for r in successful_data])
        times = np.array([r["computation_time"] for r in successful_data])
        dims = np.array([r["hilbert_dimension"] for r in successful_data])
        
        # Fit scaling laws
        # Time scaling: t ~ N^alpha
        log_N = np.log(N_values)
        log_t = np.log(times)
        time_slope, time_intercept, time_r, _, _ = linregress(log_N, log_t)
        
        # Dimension scaling: dim ~ N^beta  
        log_dim = np.log(dims)
        dim_slope, dim_intercept, dim_r, _, _ = linregress(log_N, log_dim)
        
        # Efficiency metric: time per matrix element
        matrix_elements = dims**2
        efficiency = times / matrix_elements
        
        scaling_analysis = {
            "time_scaling_exponent": float(time_slope),
            "time_scaling_r_squared": float(time_r**2),
            "dimension_scaling_exponent": float(dim_slope), 
            "dimension_scaling_r_squared": float(dim_r**2),
            "average_efficiency": float(np.mean(efficiency)),
            "efficiency_trend": "improving" if efficiency[-1] < efficiency[0] else "degrading",
            "predicted_time_N20": float(np.exp(time_intercept) * (20**time_slope)),
            "predicted_dim_N20": int(np.exp(dim_intercept) * (20**dim_slope))
        }
        
        print(f"   Time scaling: t ~ N^{time_slope:.2f} (R² = {time_r**2:.3f})")
        print(f"   Dimension scaling: dim ~ N^{dim_slope:.2f} (R² = {dim_r**2:.3f})")
        print(f"   Predicted for N=20: t ≈ {scaling_analysis['predicted_time_N20']:.1f}s, "
              f"dim ≈ {scaling_analysis['predicted_dim_N20']}")
        
        self.performance_metrics = scaling_analysis
        return scaling_analysis
    
    def generate_convergence_plots(self):
        """Generate comprehensive convergence plots."""
        
        print(f"\n📊 Generating convergence plots...")
        
        if not self.extrapolation_results:
            print("   ⚠️  No extrapolation results to plot")
            return
        
        # Create subplot grid
        n_obs = len(self.extrapolation_results)
        cols = 3
        rows = (n_obs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        for obs_name, result in self.extrapolation_results.items():
            if plot_idx >= len(axes.flat):
                break
                
            ax = axes.flat[plot_idx]
            
            h_values = np.array(result["lattice_spacings"])
            obs_values = np.array(result["lattice_values"])
            continuum_value = result["continuum_value"]
            
            # Plot data points
            ax.scatter(h_values, obs_values, color='blue', s=50, label='Lattice data')
            
            # Plot extrapolation line
            h_ext = np.linspace(0, max(h_values)*1.1, 100)
            extrapolation_line = np.full_like(h_ext, continuum_value)
            ax.plot(h_ext, extrapolation_line, '--', color='red', 
                   label=f'Continuum limit: {continuum_value:.4f}')
            
            # Plot fit line if available
            if result["convergence_order"] > 0:
                # Approximate fit line using continuum value and leading correction
                correction = result["extrapolation_error"] * (h_ext / h_values[-1])**result["convergence_order"]
                fit_line = continuum_value + correction
                ax.plot(h_ext, fit_line, ':', color='green', alpha=0.7, label='Fit')
            
            ax.set_xlabel('Lattice Spacing h = 1/(N-1)')
            ax.set_ylabel(obs_name.replace('_', ' ').title())
            ax.set_title(f'{obs_name} Continuum Extrapolation\nR² = {result["r_squared"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes.flat)):
            axes.flat[idx].set_visible(False)
        
        plt.tight_layout()
        plot_file = self.output_dir / "continuum_extrapolation_plots.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Performance scaling plot
        if self.performance_metrics:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            successful_data = [r for r in self.lattice_results.values() if "error" not in r]
            N_values = [r["N"] for r in successful_data]
            times = [r["computation_time"] for r in successful_data]
            dims = [r["hilbert_dimension"] for r in successful_data]
            
            # Time scaling
            ax1.loglog(N_values, times, 'o-', label='Actual')
            N_fit = np.linspace(min(N_values), max(N_values), 50)
            time_fit = np.exp(self.performance_metrics["time_scaling_exponent"] * np.log(N_fit) + 
                             np.log(times[0]) - self.performance_metrics["time_scaling_exponent"] * np.log(N_values[0]))
            ax1.loglog(N_fit, time_fit, '--', label=f'N^{self.performance_metrics["time_scaling_exponent"]:.2f}')
            ax1.set_xlabel('Lattice Size N')
            ax1.set_ylabel('Computation Time (s)')
            ax1.set_title('Time Scaling')
            ax1.legend()
            ax1.grid(True)
            
            # Dimension scaling  
            ax2.loglog(N_values, dims, 'o-', label='Actual')
            dim_fit = np.exp(self.performance_metrics["dimension_scaling_exponent"] * np.log(N_fit) + 
                            np.log(dims[0]) - self.performance_metrics["dimension_scaling_exponent"] * np.log(N_values[0]))
            ax2.loglog(N_fit, dim_fit, '--', label=f'N^{self.performance_metrics["dimension_scaling_exponent"]:.2f}')
            ax2.set_xlabel('Lattice Size N')
            ax2.set_ylabel('Hilbert Dimension')
            ax2.set_title('Dimension Scaling')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            perf_plot_file = self.output_dir / "performance_scaling_plots.png"
            plt.savefig(perf_plot_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✅ Plots saved to {self.output_dir}")
    
    def export_results(self) -> str:
        """Export complete results to JSON."""
        
        results_data = {
            "metadata": {
                "framework_version": "1.0",
                "analysis_timestamp": str(np.datetime64('now')),
                "total_lattice_sizes": len(self.lattice_results)
            },
            "lattice_results": self.lattice_results,
            "extrapolation_results": self.extrapolation_results,
            "performance_metrics": self.performance_metrics
        }
        
        results_file = self.output_dir / "continuum_benchmarking_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"   📁 Results exported to {results_file}")
        return str(results_file)
    
    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        
        report_file = self.output_dir / "continuum_benchmarking_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("📏 CONTINUUM LIMIT BENCHMARKING SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            # System overview
            successful_results = [r for r in self.lattice_results.values() if "error" not in r]
            f.write(f"System Configuration:\n")
            f.write(f"  Lattice sizes tested: {len(self.lattice_results)}\n")
            f.write(f"  Successful calculations: {len(successful_results)}\n")
            f.write(f"  Analysis timestamp: {np.datetime64('now')}\n\n")
            
            # Continuum extrapolations
            if self.extrapolation_results:
                f.write("Continuum Limit Extrapolations:\n")
                for obs, result in self.extrapolation_results.items():
                    f.write(f"  {obs}:\n")
                    f.write(f"    Continuum value: {result['continuum_value']:.6e} ± {result['extrapolation_error']:.2e}\n")
                    f.write(f"    Convergence order: {result['convergence_order']:.2f}\n")
                    f.write(f"    Fit quality: {result['fit_quality']} (R² = {result['r_squared']:.4f})\n\n")
            
            # Performance analysis
            if self.performance_metrics:
                f.write("Performance Scaling Analysis:\n")
                f.write(f"  Time scaling: t ~ N^{self.performance_metrics['time_scaling_exponent']:.2f}\n")
                f.write(f"  Dimension scaling: dim ~ N^{self.performance_metrics['dimension_scaling_exponent']:.2f}\n")
                f.write(f"  Predicted time for N=20: {self.performance_metrics['predicted_time_N20']:.1f}s\n")
                f.write(f"  Efficiency trend: {self.performance_metrics['efficiency_trend']}\n\n")
            
            # Quality assessment
            if self.extrapolation_results:
                excellent_fits = sum(1 for r in self.extrapolation_results.values() 
                                   if r['fit_quality'] == 'excellent')
                total_fits = len(self.extrapolation_results)
                f.write(f"Quality Assessment:\n")
                f.write(f"  Excellent extrapolations: {excellent_fits}/{total_fits}\n")
                
                if excellent_fits / total_fits > 0.8:
                    f.write("  🟢 VERDICT: High-quality continuum extrapolation achieved!\n")
                elif excellent_fits / total_fits > 0.5:
                    f.write("  🟡 VERDICT: Good continuum extrapolation with some uncertainty.\n") 
                else:
                    f.write("  🔴 VERDICT: Continuum extrapolation needs improvement.\n")
        
        print(f"   📝 Summary report saved to {report_file}")
        return str(report_file)
    
    def run_complete_benchmarking(self, N_values: List[int] = None) -> Dict[str, Any]:
        """
        Run complete continuum limit benchmarking pipeline.
        
        Includes:
        1. Extended lattice sweep
        2. Richardson extrapolation
        3. Performance analysis
        4. Visualization
        5. Results export
        """
        print(f"🚀 Starting complete continuum limit benchmarking...")
        
        try:
            # Step 1: Extended lattice sweep
            print(f"\n📍 Step 1: Extended lattice sweep")
            sweep_results = self.extended_lattice_sweep(N_values)
            
            # Step 2: Continuum extrapolation
            print(f"\n📍 Step 2: Richardson extrapolation")
            extrapolation_results = self.extrapolate_all_observables()
            
            # Step 3: Performance analysis
            print(f"\n📍 Step 3: Performance scaling analysis")
            performance_results = self.analyze_performance_scaling()
            
            # Step 4: Generate plots
            print(f"\n📍 Step 4: Generate visualization")
            self.generate_convergence_plots()
            
            # Step 5: Export results
            print(f"\n📍 Step 5: Export results")
            results_file = self.export_results()
            summary_file = self.generate_summary_report()
            
            print(f"\n✅ Complete continuum benchmarking finished!")
            print(f"   Results: {results_file}")
            print(f"   Summary: {summary_file}")
            
            return {
                "sweep_results": sweep_results,
                "extrapolation_results": extrapolation_results,
                "performance_results": performance_results,
                "files_created": [results_file, summary_file]
            }
            
        except Exception as e:
            print(f"\n❌ Benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


def demo_continuum_benchmarking():
    """Demonstration of continuum limit benchmarking."""
    
    print("📏 CONTINUUM LIMIT BENCHMARKING DEMO")
    print("=" * 60)
    
    # Create benchmarking framework
    framework = ContinuumBenchmarkingFramework()
    
    # Run complete analysis (smaller N for demo)
    N_demo = [3, 5, 7, 9]
    results = framework.run_complete_benchmarking(N_demo)
    
    # Print key results
    if "extrapolation_results" in results:
        print(f"\n🏆 KEY CONTINUUM EXTRAPOLATIONS:")
        for obs, data in results["extrapolation_results"].items():
            if "continuum_value" in data:
                print(f"   {obs}: {data['continuum_value']:.6e} ± {data['extrapolation_error']:.2e}")
    
    if "performance_results" in results:
        perf = results["performance_results"]
        if "error" not in perf:
            print(f"\n⚡ PERFORMANCE SCALING:")
            print(f"   Time: t ~ N^{perf['time_scaling_exponent']:.2f}")
            print(f"   Dimension: dim ~ N^{perf['dimension_scaling_exponent']:.2f}")
    
    return results


if __name__ == "__main__":
    demo_continuum_benchmarking()
