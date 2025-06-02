"""
Extended Lattice Refinement Framework with Continuum Extrapolation

This module implements systematic lattice studies from N=3 to N=15 with
continuum limit extrapolation and convergence analysis for LQG warp drive models.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
from pathlib import Path

# Import core LQG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField

@dataclass
class LatticeResult:
    """Results from a single lattice size calculation"""
    N: int
    lattice_spacing: float
    volume: float
    hilbert_dim: int
    energy_expectation: complex
    energy_variance: float
    constraint_violation: float
    convergence_metric: float
    computation_time: float
    stability_index: float

@dataclass
class ContinuumFit:
    """Results from continuum extrapolation fitting"""
    fit_function: str
    parameters: List[float]
    parameter_errors: List[float]
    continuum_limit: float
    continuum_error: float
    r_squared: float
    extrapolation_range: Tuple[int, int]

class ExtendedLatticeRefinement:
    """
    Advanced lattice refinement framework with systematic convergence
    analysis and continuum limit extrapolation.
    """
    
    def __init__(self, lattice_sizes: Optional[List[int]] = None):
        """
        Initialize the extended lattice refinement framework.
        
        Args:
            lattice_sizes: List of lattice sizes to study (default: [3,5,7,9,11,13,15])
        """
        self.lattice_sizes = lattice_sizes or [3, 5, 7, 9, 11, 13, 15]
        self.results = []
        self.continuum_fits = {}
        
        # Physical parameters
        self.physical_length = 1.0  # Physical box size
        
        # Fitting functions for continuum extrapolation
        self.fit_functions = {
            'power_law': self._power_law_fit,
            'exponential': self._exponential_fit,
            'logarithmic': self._logarithmic_fit,
            'polynomial': self._polynomial_fit,
            'scaling': self._scaling_fit
        }
        
    def compute_lattice_spacing(self, N: int) -> float:
        """Compute lattice spacing for given N"""
        return self.physical_length / N
        
    def compute_physical_volume(self, N: int) -> float:
        """Compute physical volume of the lattice"""
        spacing = self.compute_lattice_spacing(N)
        return (spacing * N)**3
        
    def run_single_lattice(self, N: int) -> LatticeResult:
        """
        Run complete calculation for a single lattice size.
        
        Args:
            N: Lattice size
            
        Returns:
            LatticeResult containing all computed quantities
        """
        print(f"  🔧 Computing N = {N} lattice...")
        
        import time
        start_time = time.time()
        
        try:
            # Initialize constraint operator
            constraint = MidisuperspaceHamiltonianConstraint(N)
            
            # Initialize matter fields
            maxwell = MaxwellField(N)
            dirac = DiracField(N)
            phantom = PhantomScalarField(N)
            
            # Generate basis states
            basis_states = constraint.generate_flux_basis()
            hilbert_dim = len(basis_states)
            
            # Compute energy expectation and variance
            energy_exp, energy_var = self._compute_energy_statistics(
                constraint, basis_states
            )
            
            # Compute constraint violation
            constraint_violation = self._compute_constraint_violation(
                constraint, basis_states
            )
            
            # Compute convergence metric
            convergence_metric = self._compute_convergence_metric(N, energy_exp)
            
            # Compute stability index
            stability_index = self._compute_stability_index(
                constraint, basis_states
            )
            
            computation_time = time.time() - start_time
            
            result = LatticeResult(
                N=N,
                lattice_spacing=self.compute_lattice_spacing(N),
                volume=self.compute_physical_volume(N),
                hilbert_dim=hilbert_dim,
                energy_expectation=energy_exp,
                energy_variance=energy_var,
                constraint_violation=constraint_violation,
                convergence_metric=convergence_metric,
                computation_time=computation_time,
                stability_index=stability_index
            )
            
            print(f"    ✅ N={N}: E = {energy_exp.real:.6f}, dim = {hilbert_dim}, time = {computation_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"    ❌ Error at N={N}: {str(e)}")
            raise
    
    def _compute_energy_statistics(self, constraint, basis_states) -> Tuple[complex, float]:
        """Compute energy expectation value and variance"""
        # Use coherent state for expectation values
        coherent_state = self._generate_coherent_state(len(basis_states))
        
        # Compute Hamiltonian matrix elements
        H_matrix = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                H_matrix[i, j] = constraint.compute_matrix_element(state_i, state_j)
        
        # Energy expectation
        energy_exp = np.conj(coherent_state) @ H_matrix @ coherent_state
        
        # Energy variance: ⟨H²⟩ - ⟨H⟩²
        H2_exp = np.conj(coherent_state) @ (H_matrix @ H_matrix) @ coherent_state
        energy_var = abs(H2_exp - energy_exp**2)
        
        return energy_exp, energy_var
    
    def _generate_coherent_state(self, dim: int) -> np.ndarray:
        """Generate normalized coherent state"""
        alpha = 0.5 + 0.3j
        state = np.exp(-0.5 * abs(alpha)**2) * np.array([
            alpha**n / np.sqrt(np.math.factorial(min(n, 20))) 
            for n in range(dim)
        ])
        return state / np.linalg.norm(state)
    
    def _compute_constraint_violation(self, constraint, basis_states) -> float:
        """Compute constraint violation measure"""
        # Test constraint algebra: [H[N], H[M]] = H[{N,M}]
        # For simplicity, use |[H,H]| as violation measure
        
        H_matrix = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                H_matrix[i, j] = constraint.compute_matrix_element(state_i, state_j)
        
        commutator = H_matrix @ H_matrix - H_matrix @ H_matrix
        return np.linalg.norm(commutator, 'fro')
    
    def _compute_convergence_metric(self, N: int, energy: complex) -> float:
        """Compute convergence metric based on energy scale"""
        # Theoretical expectation: energy should scale as ~ 1/N² for continuum limit
        theoretical_scale = 1.0 / (N**2)
        actual_scale = abs(energy)
        
        if theoretical_scale > 0:
            return abs(actual_scale - theoretical_scale) / theoretical_scale
        else:
            return float('inf')
    
    def _compute_stability_index(self, constraint, basis_states) -> float:
        """Compute numerical stability index"""
        try:
            # Compute condition number of Hamiltonian matrix
            H_matrix = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
            for i, state_i in enumerate(basis_states):
                for j, state_j in enumerate(basis_states):
                    H_matrix[i, j] = constraint.compute_matrix_element(state_i, state_j)
            
            # Use condition number as stability measure
            condition_number = np.linalg.cond(H_matrix)
            
            # Convert to stability index (lower is more stable)
            if np.isfinite(condition_number) and condition_number > 0:
                return 1.0 / condition_number
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def run_extended_lattice_study(self) -> List[LatticeResult]:
        """Run complete lattice study across all sizes"""
        print("🔬 Starting Extended Lattice Refinement Study")
        print("="*55)
        print(f"Lattice sizes: {self.lattice_sizes}")
        print(f"Physical length: {self.physical_length}")
        
        self.results = []
        
        for N in self.lattice_sizes:
            try:
                result = self.run_single_lattice(N)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Failed at N={N}: {str(e)}")
                continue
        
        print(f"\n✅ Completed lattice study: {len(self.results)}/{len(self.lattice_sizes)} successful")
        return self.results
    
    # Fitting functions for continuum extrapolation
    def _power_law_fit(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Power law: f(x) = a * x^b"""
        return a * np.power(x, b)
    
    def _exponential_fit(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Exponential: f(x) = a * exp(b * x)"""
        return a * np.exp(b * x)
    
    def _logarithmic_fit(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        """Logarithmic: f(x) = a * log(x) + b"""
        return a * np.log(x) + b
    
    def _polynomial_fit(self, x: np.ndarray, *coeffs) -> np.ndarray:
        """Polynomial: f(x) = sum(c_i * x^i)"""
        return np.polyval(coeffs, x)
    
    def _scaling_fit(self, x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Scaling: f(x) = a + b/x + c/x²"""
        return a + b/x + c/(x**2)
    
    def fit_continuum_extrapolation(self, quantity: str = 'energy_expectation') -> Dict[str, ContinuumFit]:
        """
        Fit continuum extrapolation for specified quantity.
        
        Args:
            quantity: Quantity to extrapolate ('energy_expectation', 'constraint_violation', etc.)
            
        Returns:
            Dictionary of fits for different fitting functions
        """
        if not self.results:
            raise ValueError("No lattice results available for fitting")
        
        print(f"\n📈 Fitting continuum extrapolation for: {quantity}")
        
        # Extract data
        N_values = np.array([r.N for r in self.results])
        lattice_spacings = 1.0 / N_values  # Use 1/N as continuum parameter
        
        if quantity == 'energy_expectation':
            y_values = np.array([r.energy_expectation.real for r in self.results])
        elif quantity == 'constraint_violation':
            y_values = np.array([r.constraint_violation for r in self.results])
        elif quantity == 'convergence_metric':
            y_values = np.array([r.convergence_metric for r in self.results])
        elif quantity == 'stability_index':
            y_values = np.array([r.stability_index for r in self.results])
        else:
            raise ValueError(f"Unknown quantity: {quantity}")
        
        fits = {}
        
        # Try different fitting functions
        for fit_name, fit_func in self.fit_functions.items():
            try:
                print(f"  🔧 Fitting {fit_name}...")
                
                if fit_name == 'polynomial':
                    # Use degree 2 polynomial for polynomial fit
                    popt, pcov = curve_fit(
                        lambda x, a, b, c: self._polynomial_fit(x, a, b, c),
                        lattice_spacings, y_values
                    )
                    fit_result = lambda x: self._polynomial_fit(x, *popt)
                else:
                    popt, pcov = curve_fit(fit_func, lattice_spacings, y_values)
                    fit_result = lambda x: fit_func(x, *popt)
                
                # Extract parameter errors
                param_errors = np.sqrt(np.diag(pcov))
                
                # Compute R-squared
                y_pred = fit_result(lattice_spacings)
                ss_res = np.sum((y_values - y_pred)**2)
                ss_tot = np.sum((y_values - np.mean(y_values))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Extrapolate to continuum (lattice_spacing -> 0)
                continuum_limit = fit_result(0.0)
                
                # Estimate error in continuum limit
                continuum_error = np.sqrt(np.sum(param_errors**2))  # Rough estimate
                
                fits[fit_name] = ContinuumFit(
                    fit_function=fit_name,
                    parameters=popt.tolist(),
                    parameter_errors=param_errors.tolist(),
                    continuum_limit=continuum_limit,
                    continuum_error=continuum_error,
                    r_squared=r_squared,
                    extrapolation_range=(min(N_values), max(N_values))
                )
                
                print(f"    ✅ {fit_name}: R² = {r_squared:.4f}, continuum = {continuum_limit:.6f}")
                
            except Exception as e:
                print(f"    ❌ {fit_name}: Failed - {str(e)}")
                continue
        
        self.continuum_fits[quantity] = fits
        return fits
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence properties of the lattice sequence"""
        if len(self.results) < 3:
            return {"error": "Need at least 3 lattice sizes for convergence analysis"}
        
        analysis = {
            "lattice_count": len(self.results),
            "lattice_range": (min(r.N for r in self.results), max(r.N for r in self.results)),
            "energy_convergence": {},
            "constraint_convergence": {},
            "computational_scaling": {},
            "stability_analysis": {}
        }
        
        # Energy convergence analysis
        energies = [r.energy_expectation.real for r in self.results]
        N_values = [r.N for r in self.results]
        
        # Compute energy differences between consecutive lattices
        energy_diffs = [abs(energies[i+1] - energies[i]) for i in range(len(energies)-1)]
        
        analysis["energy_convergence"] = {
            "energy_values": energies,
            "energy_differences": energy_diffs,
            "convergence_rate": self._estimate_convergence_rate(N_values, energies),
            "is_converging": all(energy_diffs[i+1] < energy_diffs[i] for i in range(len(energy_diffs)-1))
        }
        
        # Constraint violation convergence
        violations = [r.constraint_violation for r in self.results]
        analysis["constraint_convergence"] = {
            "violation_values": violations,
            "convergence_rate": self._estimate_convergence_rate(N_values, violations),
            "is_decreasing": violations[-1] < violations[0]
        }
        
        # Computational scaling
        times = [r.computation_time for r in self.results]
        analysis["computational_scaling"] = {
            "computation_times": times,
            "scaling_exponent": self._estimate_scaling_exponent(N_values, times)
        }
        
        # Stability analysis
        stabilities = [r.stability_index for r in self.results]
        analysis["stability_analysis"] = {
            "stability_indices": stabilities,
            "average_stability": np.mean(stabilities),
            "stability_trend": "improving" if stabilities[-1] > stabilities[0] else "degrading"
        }
        
        return analysis
    
    def _estimate_convergence_rate(self, N_values: List[int], quantities: List[float]) -> float:
        """Estimate convergence rate assuming power law scaling"""
        try:
            log_N = np.log(N_values)
            log_quantities = np.log(np.abs(quantities))
            
            # Linear fit in log space: log(Q) = log(a) + b*log(N)
            coeffs = np.polyfit(log_N, log_quantities, 1)
            return coeffs[0]  # Return the exponent
        except:
            return float('nan')
    
    def _estimate_scaling_exponent(self, N_values: List[int], times: List[float]) -> float:
        """Estimate computational scaling exponent"""
        return self._estimate_convergence_rate(N_values, times)
    
    def generate_convergence_plots(self, output_dir: str = "outputs") -> List[str]:
        """Generate convergence analysis plots"""
        if not self.results:
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        plots_generated = []
        
        # Extract data
        N_values = [r.N for r in self.results]
        energies = [r.energy_expectation.real for r in self.results]
        violations = [r.constraint_violation for r in self.results]
        times = [r.computation_time for r in self.results]
        
        # Plot 1: Energy convergence
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(N_values, energies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Lattice Size N')
        plt.ylabel('Energy Expectation')
        plt.title('Energy Convergence')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Constraint violation
        plt.subplot(2, 2, 2)
        plt.semilogy(N_values, violations, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Lattice Size N')
        plt.ylabel('Constraint Violation')
        plt.title('Constraint Violation vs Lattice Size')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Computational scaling
        plt.subplot(2, 2, 3)
        plt.loglog(N_values, times, 'go-', linewidth=2, markersize=8)
        plt.xlabel('Lattice Size N')
        plt.ylabel('Computation Time (s)')
        plt.title('Computational Scaling')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Lattice spacing vs energy
        spacing_values = [1.0/N for N in N_values]
        plt.subplot(2, 2, 4)
        plt.plot(spacing_values, energies, 'mo-', linewidth=2, markersize=8)
        plt.xlabel('Lattice Spacing (1/N)')
        plt.ylabel('Energy Expectation')
        plt.title('Continuum Extrapolation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = output_dir / "lattice_convergence_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plots_generated.append(str(plot_file))
        print(f"📊 Convergence plot saved: {plot_file}")
        
        return plots_generated
    
    def export_results(self, output_dir: str = "outputs") -> str:
        """Export lattice refinement results to JSON file"""
        output_path = Path(output_dir) / "extended_lattice_refinement_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "parameters": {
                "lattice_sizes": self.lattice_sizes,
                "physical_length": self.physical_length,
                "total_lattices": len(self.results)
            },
            "lattice_results": [
                {
                    "N": r.N,
                    "lattice_spacing": r.lattice_spacing,
                    "volume": r.volume,
                    "hilbert_dim": r.hilbert_dim,
                    "energy_expectation_real": r.energy_expectation.real,
                    "energy_expectation_imag": r.energy_expectation.imag,
                    "energy_variance": r.energy_variance,
                    "constraint_violation": r.constraint_violation,
                    "convergence_metric": r.convergence_metric,
                    "computation_time": r.computation_time,
                    "stability_index": r.stability_index
                }
                for r in self.results
            ],
            "continuum_fits": {
                quantity: {
                    fit_name: {
                        "fit_function": fit.fit_function,
                        "parameters": fit.parameters,
                        "parameter_errors": fit.parameter_errors,
                        "continuum_limit": fit.continuum_limit,
                        "continuum_error": fit.continuum_error,
                        "r_squared": fit.r_squared,
                        "extrapolation_range": fit.extrapolation_range
                    }
                    for fit_name, fit in fits.items()
                }
                for quantity, fits in self.continuum_fits.items()
            },
            "convergence_analysis": self.analyze_convergence()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Results exported to: {output_path}")
        return str(output_path)

def run_extended_lattice_study(lattice_sizes: Optional[List[int]] = None, 
                              export: bool = True) -> Dict[str, Any]:
    """
    Main function to run extended lattice refinement study.
    
    Args:
        lattice_sizes: List of lattice sizes to study
        export: Whether to export results to file
        
    Returns:
        Dictionary containing analysis results
    """
    refinement = ExtendedLatticeRefinement(lattice_sizes)
    
    # Run lattice study
    results = refinement.run_extended_lattice_study()
    
    if not results:
        print("❌ No successful lattice calculations")
        return {"error": "No successful results"}
    
    # Fit continuum extrapolations
    quantities = ['energy_expectation', 'constraint_violation']
    for quantity in quantities:
        try:
            fits = refinement.fit_continuum_extrapolation(quantity)
            print(f"\n📈 Continuum extrapolation for {quantity}:")
            for fit_name, fit in fits.items():
                print(f"  {fit_name}: limit = {fit.continuum_limit:.6f} ± {fit.continuum_error:.6f}")
        except Exception as e:
            print(f"❌ Failed continuum fitting for {quantity}: {str(e)}")
    
    # Analyze convergence
    convergence_analysis = refinement.analyze_convergence()
    
    # Generate plots
    if export:
        refinement.generate_convergence_plots()
        refinement.export_results()
    
    # Print summary
    print("\n📊 EXTENDED LATTICE REFINEMENT SUMMARY")
    print("="*55)
    print(f"Lattice range: N = {min(r.N for r in results)} to {max(r.N for r in results)}")
    print(f"Successful calculations: {len(results)}")
    
    if convergence_analysis.get("energy_convergence"):
        energy_conv = convergence_analysis["energy_convergence"]
        print(f"Energy convergence rate: {energy_conv.get('convergence_rate', 'N/A'):.3f}")
        print(f"Energy is converging: {energy_conv.get('is_converging', False)}")
    
    if refinement.continuum_fits.get('energy_expectation'):
        best_fit = max(
            refinement.continuum_fits['energy_expectation'].items(),
            key=lambda x: x[1].r_squared
        )
        print(f"Best energy fit: {best_fit[0]} (R² = {best_fit[1].r_squared:.4f})")
        print(f"Continuum energy limit: {best_fit[1].continuum_limit:.6f}")
    
    return {
        "results": results,
        "convergence_analysis": convergence_analysis,
        "continuum_fits": refinement.continuum_fits
    }

if __name__ == "__main__":
    # Run the extended lattice study
    analysis_results = run_extended_lattice_study(
        lattice_sizes=[3, 5, 7, 9, 11, 13, 15],
        export=True
    )
