#!/usr/bin/env python3
"""
Automated Lattice Refinement & Continuum Extrapolation for LQG

This module implements systematic lattice refinement studies:
1. Automated runs for increasing N = 3, 5, 7, 9, 11, ...
2. Convergence analysis of physical observables
3. Continuum extrapolation with uncertainty quantification
4. Publication-ready convergence plots

Author: LQG Framework Team  
Date: June 2025
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
from pathlib import Path

# Import LQG components
from lqg_fixed_components import (
    MidisuperspaceHamiltonianConstraint,
    LatticeConfiguration,
    LQGParameters
)
from kinematical_hilbert import KinematicalHilbertSpace

@dataclass
class RefinementObservables:
    """Container for observables computed at each lattice refinement."""
    N: int                              # Number of lattice sites
    omega_min_squared: float           # Minimum eigenvalue (stability)
    total_energy_density: float       # ∫|⟨T^00⟩| 4πr² dr
    throat_stress_energy: float       # ⟨T^00⟩ at throat
    constraint_violation: float       # ||H|ψ⟩|| for ground state
    computation_time: float           # Wall time for this N
    hilbert_dimension: int            # Actual Hilbert space size
    matrix_sparsity: float            # Fraction of non-zero elements
    
@dataclass 
class ConvergenceResults:
    """Results from convergence analysis."""
    extrapolated_values: Dict[str, float]
    extrapolation_errors: Dict[str, float]
    convergence_rates: Dict[str, float]
    fit_parameters: Dict[str, Dict[str, float]]
    quality_metrics: Dict[str, float]

class LatticeRefinementFramework:
    """
    Automated framework for LQG lattice refinement studies.
    
    Systematically increases lattice resolution and tracks convergence
    of physical observables toward continuum limit.
    """
    
    def __init__(self, 
                 base_config: Dict,
                 output_dir: str = "refinement_study"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.refinement_data: List[RefinementObservables] = []
        self.convergence_results: Optional[ConvergenceResults] = None
        
    def run_systematic_refinement(self, 
                                 N_values: List[int] = [3, 5, 7, 9, 11],
                                 max_basis_states: int = 1000) -> List[RefinementObservables]:
        """
        Run LQG computation for each lattice size N and collect observables.
        """
        print("🔬 Starting Systematic Lattice Refinement Study")
        print("=" * 50)
        
        for N in N_values:
            print(f"\n📊 Computing N = {N} lattice...")
            
            # Create configuration for this N
            lattice_config = LatticeConfiguration()
            lattice_config.n_sites = N
            lattice_config.r_min = self.base_config['r_min']
            lattice_config.r_max = self.base_config['r_max']
            
            lqg_params = LQGParameters(
                planck_length=self.base_config['planck_length'],
                max_basis_states=min(max_basis_states, 2**(2*N)),  # Scale with N
                mu_bar_scheme=self.base_config['mu_bar_scheme'],
                regularization_epsilon=self.base_config['regularization_epsilon']
            )
            
            # Time the computation
            start_time = time.time()
            
            observables = self._compute_observables_for_N(lattice_config, lqg_params)
            
            computation_time = time.time() - start_time
            observables.computation_time = computation_time
            
            self.refinement_data.append(observables)
            
            print(f"   ✓ N={N}: ω²_min = {observables.omega_min_squared:.6e}")
            print(f"   ✓ Total energy: {observables.total_energy_density:.6e}")
            print(f"   ✓ Computation time: {computation_time:.2f}s")
            print(f"   ✓ Hilbert dimension: {observables.hilbert_dimension}")
            
            # Save intermediate results
            self._save_intermediate_results()
        
        print(f"\n✅ Refinement study complete! Data saved to {self.output_dir}")
        return self.refinement_data
    
    def _compute_observables_for_N(self, 
                                  lattice_config: LatticeConfiguration,
                                  lqg_params: LQGParameters) -> RefinementObservables:
        """Compute all observables for a given lattice size N."""
        
        N = lattice_config.n_sites
        
        # Build kinematical Hilbert space
        kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
        kin_space.generate_flux_basis()
        
        # Setup classical background
        r_sites = np.linspace(lattice_config.r_min, lattice_config.r_max, N)
        classical_data = self._generate_classical_background(r_sites)
        
        # Build constraint operator
        constraint = MidisuperspaceHamiltonianConstraint(
            lattice_config, lqg_params, kin_space
        )
        
        H_matrix = constraint.construct_full_hamiltonian(**classical_data)
        
        # Solve for lowest eigenvalues
        try:
            eigenvals, eigenvecs = constraint.solve_constraint(num_eigs=3)
            omega_min_squared = eigenvals[0] if len(eigenvals) > 0 else np.inf
            ground_state = eigenvecs[:, 0] if len(eigenvals) > 0 else None
        except:
            omega_min_squared = np.inf
            ground_state = None
        
        # Compute stress-energy observables
        total_energy = self._compute_total_energy_density(
            r_sites, classical_data, ground_state
        )
        
        throat_stress = self._compute_throat_stress_energy(
            classical_data, ground_state
        )
        
        # Compute constraint violation
        if ground_state is not None:
            constraint_violation = np.linalg.norm(H_matrix @ ground_state)
        else:
            constraint_violation = np.inf
        
        # Matrix properties
        hilbert_dimension = kin_space.dim
        matrix_sparsity = H_matrix.nnz / (H_matrix.shape[0] * H_matrix.shape[1])
        
        return RefinementObservables(
            N=N,
            omega_min_squared=omega_min_squared,
            total_energy_density=total_energy,
            throat_stress_energy=throat_stress,
            constraint_violation=constraint_violation,
            computation_time=0.0,  # Set by caller
            hilbert_dimension=hilbert_dimension,
            matrix_sparsity=matrix_sparsity
        )
    
    def _generate_classical_background(self, r_sites: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate consistent classical background for all lattice sizes."""
        
        # Scale background to maintain physical throat radius
        throat_radius = self.base_config.get('throat_radius', 1e-34)
        
        # Normalized radial coordinate
        xi = r_sites / throat_radius
        
        # Smooth classical profiles (independent of lattice discretization)
        classical_data = {
            'classical_E_x': np.ones(len(r_sites)) * 1e-32,
            'classical_E_phi': np.ones(len(r_sites)) * 1e-32,
            'classical_K_x': np.tanh(xi) * 1e2,
            'classical_K_phi': np.sech(xi)**2 * 1e2,
            'scalar_field': -np.tanh(xi),  # Phantom field
            'scalar_momentum': np.zeros(len(r_sites))
        }
        
        return classical_data
    
    def _compute_total_energy_density(self, 
                                     r_sites: np.ndarray,
                                     classical_data: Dict[str, np.ndarray],
                                     quantum_state: Optional[np.ndarray]) -> float:
        """Compute ∫|⟨T^00⟩| 4πr² dr."""
        
        if quantum_state is None:
            return np.inf
        
        # Simplified T^00 computation (full version would use quantum expectation)
        phi = classical_data['scalar_field']
        pi = classical_data['scalar_momentum']
        
        # Classical stress-energy density
        T00_classical = 0.5 * (pi**2 + np.gradient(phi)**2)  # Simplified
        
        # Quantum corrections (placeholder - would compute ⟨ψ|T^00|ψ⟩)
        quantum_correction = 1.0 + 0.1 * np.sum(np.abs(quantum_state)**2)
        
        T00_quantum = T00_classical * quantum_correction
        
        # Integrate over 4πr² volume elements
        dr = np.gradient(r_sites)
        integrand = np.abs(T00_quantum) * 4 * np.pi * r_sites**2
        total_energy = np.trapz(integrand, dx=dr[0])
        
        return total_energy
    
    def _compute_throat_stress_energy(self,
                                     classical_data: Dict[str, np.ndarray],
                                     quantum_state: Optional[np.ndarray]) -> float:
        """Compute ⟨T^00⟩ at the throat."""
        
        if quantum_state is None:
            return np.inf
        
        # Stress-energy at first lattice point (near throat)
        phi_throat = classical_data['scalar_field'][0]
        pi_throat = classical_data['scalar_momentum'][0]
        
        T00_throat = 0.5 * (pi_throat**2 + phi_throat**2)  # Simplified
        
        return T00_throat
    
    def analyze_convergence(self, 
                          extrapolation_orders: List[int] = [1, 2]) -> ConvergenceResults:
        """
        Analyze convergence of observables and extrapolate to continuum limit.
        
        Fits observables O(N) vs 1/N with polynomials of different orders.
        """
        print("\n📈 Analyzing Convergence to Continuum Limit")
        print("=" * 40)
        
        if len(self.refinement_data) < 3:
            raise ValueError("Need at least 3 data points for convergence analysis")
        
        # Extract data arrays
        N_values = np.array([obs.N for obs in self.refinement_data])
        inv_N = 1.0 / N_values
        
        observables_data = {
            'omega_min_squared': np.array([obs.omega_min_squared for obs in self.refinement_data]),
            'total_energy_density': np.array([obs.total_energy_density for obs in self.refinement_data]),
            'throat_stress_energy': np.array([obs.throat_stress_energy for obs in self.refinement_data]),
            'constraint_violation': np.array([obs.constraint_violation for obs in self.refinement_data])
        }
        
        # Perform extrapolation for each observable
        extrapolated_values = {}
        extrapolation_errors = {}
        convergence_rates = {}
        fit_parameters = {}
        quality_metrics = {}
        
        for obs_name, obs_data in observables_data.items():
            
            # Skip infinite values
            finite_mask = np.isfinite(obs_data)
            if np.sum(finite_mask) < 3:
                print(f"   ⚠️ Skipping {obs_name}: insufficient finite data")
                continue
            
            inv_N_finite = inv_N[finite_mask]
            obs_finite = obs_data[finite_mask]
            
            print(f"\n   Analyzing {obs_name}:")
            
            best_fit = None
            best_extrapolation = None
            best_error = np.inf
            
            # Try different extrapolation orders
            for order in extrapolation_orders:
                
                # Fit polynomial: O(N) = O_∞ + a₁/N + a₂/N² + ...
                coeffs = np.polyfit(inv_N_finite, obs_finite, order)
                
                # Continuum limit (1/N → 0)
                O_continuum = coeffs[-1]  # Constant term
                
                # Compute fit quality
                fit_values = np.polyval(coeffs, inv_N_finite)
                residuals = obs_finite - fit_values
                rms_error = np.sqrt(np.mean(residuals**2))
                
                print(f"     Order {order}: O_∞ = {O_continuum:.6e}, RMS = {rms_error:.6e}")
                
                if rms_error < best_error:
                    best_error = rms_error
                    best_fit = coeffs
                    best_extrapolation = O_continuum
            
            # Store results
            extrapolated_values[obs_name] = best_extrapolation
            extrapolation_errors[obs_name] = best_error
            
            # Compute convergence rate (leading 1/N coefficient)
            if len(best_fit) > 1:
                convergence_rates[obs_name] = abs(best_fit[-2])  # |a₁| coefficient
            else:
                convergence_rates[obs_name] = 0.0
            
            fit_parameters[obs_name] = {f"coeff_{i}": float(best_fit[i]) for i in range(len(best_fit))}
            quality_metrics[obs_name] = {
                "rms_error": float(best_error),
                "relative_error": float(best_error / abs(best_extrapolation)) if best_extrapolation != 0 else np.inf
            }
            
            print(f"     ✓ Continuum limit: {best_extrapolation:.6e} ± {best_error:.6e}")
        
        # Build results
        self.convergence_results = ConvergenceResults(
            extrapolated_values=extrapolated_values,
            extrapolation_errors=extrapolation_errors,
            convergence_rates=convergence_rates,
            fit_parameters=fit_parameters,
            quality_metrics=quality_metrics
        )
        
        return self.convergence_results
    
    def generate_convergence_plots(self, save_plots: bool = True) -> None:
        """Generate publication-ready convergence plots."""
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib not available - skipping plots")
            return
        
        if not self.refinement_data or not self.convergence_results:
            print("⚠️ No data available for plotting")
            return
        
        print("\n📊 Generating Convergence Plots")
        
        # Extract data
        N_values = np.array([obs.N for obs in self.refinement_data])
        inv_N = 1.0 / N_values
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LQG Lattice Refinement: Convergence to Continuum Limit', fontsize=16)
        
        observable_labels = {
            'omega_min_squared': r'$\omega^2_{\min}$ (Stability)',
            'total_energy_density': r'$\int |T^{00}| 4\pi r^2 dr$ (Total Energy)',
            'throat_stress_energy': r'$T^{00}$(throat) (Throat Stress)',
            'constraint_violation': r'$||H|\psi\rangle||$ (Constraint Violation)'
        }
        
        for idx, (obs_name, label) in enumerate(observable_labels.items()):
            
            if obs_name not in self.convergence_results.extrapolated_values:
                continue
            
            ax = axes[idx // 2, idx % 2]
            
            # Get data
            obs_data = np.array([getattr(obs, obs_name) for obs in self.refinement_data])
            finite_mask = np.isfinite(obs_data)
            
            # Plot data points
            ax.scatter(inv_N[finite_mask], obs_data[finite_mask], 
                      color='blue', s=50, alpha=0.7, label='LQG Data')
            
            # Plot extrapolation
            inv_N_fine = np.linspace(0, max(inv_N), 100)
            fit_coeffs = list(self.convergence_results.fit_parameters[obs_name].values())
            extrapolation = np.polyval(fit_coeffs, inv_N_fine)
            
            ax.plot(inv_N_fine, extrapolation, 'r-', linewidth=2, label='Extrapolation')
            
            # Mark continuum limit
            continuum_value = self.convergence_results.extrapolated_values[obs_name]
            ax.axhline(y=continuum_value, color='green', linestyle='--', alpha=0.7, 
                      label=f'Continuum: {continuum_value:.3e}')
            
            ax.set_xlabel(r'$1/N$')
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add convergence rate text
            conv_rate = self.convergence_results.convergence_rates[obs_name]
            ax.text(0.05, 0.95, f'Rate: O(1/N), coeff={conv_rate:.2e}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "convergence_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Convergence plots saved to {plot_path}")
        
        plt.show()
    
    def _save_intermediate_results(self) -> None:
        """Save intermediate results to JSON."""
        
        results_data = {
            "refinement_study": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_config": self.base_config,
                "observables": [asdict(obs) for obs in self.refinement_data]
            }
        }
        
        if self.convergence_results:
            results_data["convergence_analysis"] = asdict(self.convergence_results)
        
        results_path = self.output_dir / "refinement_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        
        if not self.convergence_results:
            return "No convergence analysis available."
        
        report = []
        report.append("🎯 LQG Lattice Refinement Summary")
        report.append("=" * 40)
        report.append(f"Lattice sizes tested: {[obs.N for obs in self.refinement_data]}")
        report.append(f"Total computation time: {sum(obs.computation_time for obs in self.refinement_data):.1f}s")
        report.append("")
        
        report.append("📊 Continuum Extrapolations:")
        for obs_name, value in self.convergence_results.extrapolated_values.items():
            error = self.convergence_results.extrapolation_errors[obs_name]
            relative_error = self.convergence_results.quality_metrics[obs_name]["relative_error"]
            
            report.append(f"  {obs_name}:")
            report.append(f"    Continuum limit: {value:.6e} ± {error:.6e}")
            report.append(f"    Relative error: {relative_error:.2%}")
            report.append("")
        
        # Overall assessment
        max_rel_error = max(
            metrics["relative_error"] for metrics in self.convergence_results.quality_metrics.values()
            if np.isfinite(metrics["relative_error"])
        )
        
        if max_rel_error < 0.02:
            report.append("✅ Excellent convergence achieved (< 2% uncertainty)")
        elif max_rel_error < 0.05:
            report.append("✅ Good convergence achieved (< 5% uncertainty)")
        else:
            report.append("⚠️ Poor convergence - consider more lattice points or better regularization")
        
        return "\n".join(report)

def run_automated_refinement_study():
    """Main function to run automated lattice refinement study."""
    
    print("🚀 Automated LQG Lattice Refinement Study")
    print("=" * 50)
    
    # Base configuration
    base_config = {
        'r_min': 1e-35,
        'r_max': 1e-33, 
        'throat_radius': 1e-34,
        'planck_length': 0.01,
        'mu_bar_scheme': 'improved_dynamics',
        'regularization_epsilon': 1e-10
    }
    
    # Create framework
    framework = LatticeRefinementFramework(base_config)
    
    # Run refinement study
    N_values = [3, 5, 7]  # Start small for testing
    refinement_data = framework.run_systematic_refinement(N_values)
    
    # Analyze convergence
    convergence_results = framework.analyze_convergence()
    
    # Generate plots
    framework.generate_convergence_plots()
    
    # Print summary
    summary = framework.generate_summary_report()
    print(f"\n{summary}")
    
    return framework

if __name__ == "__main__":
    framework = run_automated_refinement_study()
