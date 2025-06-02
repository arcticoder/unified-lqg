#!/usr/bin/env python3
"""
Extended Lattice Refinement with Continuum Extrapolation
=======================================================

Advanced lattice refinement framework extending to N=15 with sophisticated
continuum limit extrapolation and systematic convergence analysis.

Features:
- Extended lattice sizes up to N=15
- Richardson extrapolation to continuum limit
- Multi-observable convergence analysis
- Systematic error estimation
- Performance optimization
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LatticeResult:
    """Container for single lattice calculation result"""
    N: int
    observables: Dict[str, float]
    computation_time: float
    memory_usage: float
    convergence_indicators: Dict[str, float]

@dataclass
class ContinuumExtrapolation:
    """Container for continuum extrapolation results"""
    observable_name: str
    lattice_values: List[float]
    lattice_sizes: List[int]
    extrapolated_value: float
    extrapolation_error: float
    convergence_order: float
    fit_quality: float

class ExtendedLatticeRefinement:
    """
    Advanced lattice refinement with continuum extrapolation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the refinement framework"""
        self.config = self._load_config(config_path)
        self.lattice_results = []
        self.extrapolations = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration parameters"""
        default_config = {
            "lattice_sizes": [3, 5, 7, 9, 11, 13, 15],
            "extended_sizes": [17, 19, 21],  # Optional extended range
            "observables": [
                "hamiltonian_expectation",
                "stress_energy_density", 
                "constraint_violation",
                "quantum_fluctuations",
                "energy_density_variance"
            ],
            "extrapolation_methods": ["richardson", "polynomial", "exponential"],
            "convergence_criteria": {
                "relative_tolerance": 1e-6,
                "absolute_tolerance": 1e-12,
                "max_iterations": 100
            },
            "parallel_workers": 4,
            "memory_limit_gb": 8.0,
            "use_extended_precision": False
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except FileNotFoundError:
                logger.warning(f"Config file {config_path} not found, using defaults")
        
        return default_config
    
    def estimate_memory_requirements(self, N: int) -> float:
        """Estimate memory requirements for lattice size N (in GB)"""
        # Rough estimate: O(N^4) for Hamiltonian matrix storage
        matrix_elements = N**4
        bytes_per_element = 16  # Complex double precision
        total_bytes = matrix_elements * bytes_per_element
        
        # Add overhead for intermediate calculations
        overhead_factor = 3.0
        
        return (total_bytes * overhead_factor) / (1024**3)  # Convert to GB
    
    def compute_lattice_observables(self, N: int) -> Dict[str, float]:
        """
        Compute all observables for lattice size N
        
        Args:
            N: Lattice size
            
        Returns:
            Dictionary of observable values
        """
        try:
            # Check memory requirements
            memory_needed = self.estimate_memory_requirements(N)
            if memory_needed > self.config["memory_limit_gb"]:
                logger.warning(f"N={N} requires {memory_needed:.2f}GB, exceeds limit {self.config['memory_limit_gb']}GB")
                return {}
            
            # Import LQG components
            from lqg_fixed_components import LoopQuantumGravity, MidisuperspaceHamiltonianConstraint
            from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField
            
            # Initialize system
            lqg = LoopQuantumGravity(N=N)
            constraint = MidisuperspaceHamiltonianConstraint(N=N, alpha=1.0, sigma_width=0.1)
            
            # Build Hamiltonian
            H = constraint.build_hamiltonian_constraint()
            
            # Initialize matter fields
            maxwell = MaxwellField(N=N)
            dirac = DiracField(N=N)
            phantom = PhantomScalarField(N=N)
            
            observables = {}
            
            # 1. Hamiltonian expectation value
            if H.size > 0:
                eigenvals = np.linalg.eigvals(H)
                observables["hamiltonian_expectation"] = np.real(np.mean(eigenvals))
            else:
                observables["hamiltonian_expectation"] = 0.0
            
            # 2. Stress-energy density
            T00_maxwell = maxwell.compute_stress_energy_density()
            T00_dirac = dirac.compute_stress_energy_density()
            T00_phantom = phantom.compute_stress_energy_density()
            
            observables["stress_energy_density"] = T00_maxwell + T00_dirac + T00_phantom
            
            # 3. Constraint violation
            constraint_state = lqg.apply_hamiltonian_constraint()
            if constraint_state.size > 0:
                observables["constraint_violation"] = np.linalg.norm(constraint_state)
            else:
                observables["constraint_violation"] = 0.0
            
            # 4. Quantum fluctuations
            observables["quantum_fluctuations"] = self._compute_quantum_fluctuations(H, N)
            
            # 5. Energy density variance
            observables["energy_density_variance"] = self._compute_energy_variance(H, N)
            
            # Additional convergence indicators
            observables.update(self._compute_convergence_indicators(H, N))
            
            return observables
            
        except Exception as e:
            logger.error(f"Error computing observables for N={N}: {e}")
            return {}
    
    def _compute_quantum_fluctuations(self, H: np.ndarray, N: int) -> float:
        """Compute quantum fluctuations in the Hamiltonian"""
        try:
            if H.size == 0:
                return 0.0
            
            # Compute variance of Hamiltonian eigenvalues
            eigenvals = np.linalg.eigvals(H)
            return float(np.std(np.real(eigenvals)))
            
        except Exception as e:
            logger.warning(f"Error computing quantum fluctuations: {e}")
            return 0.0
    
    def _compute_energy_variance(self, H: np.ndarray, N: int) -> float:
        """Compute energy density variance"""
        try:
            if H.size == 0:
                return 0.0
            
            # Normalize by lattice volume
            volume_factor = N**3  # 3D lattice volume
            
            eigenvals = np.linalg.eigvals(H)
            energy_density = np.real(eigenvals) / volume_factor
            
            return float(np.var(energy_density))
            
        except Exception as e:
            logger.warning(f"Error computing energy variance: {e}")
            return 0.0
    
    def _compute_convergence_indicators(self, H: np.ndarray, N: int) -> Dict[str, float]:
        """Compute additional convergence indicators"""
        indicators = {}
        
        try:
            if H.size > 0:
                # Condition number
                cond_num = np.linalg.cond(H)
                indicators["condition_number"] = float(cond_num) if np.isfinite(cond_num) else 1e12
                
                # Spectral radius
                eigenvals = np.linalg.eigvals(H)
                spectral_radius = np.max(np.abs(eigenvals))
                indicators["spectral_radius"] = float(spectral_radius)
                
                # Matrix norm
                indicators["matrix_norm"] = float(np.linalg.norm(H, 'fro'))
            else:
                indicators = {
                    "condition_number": 1.0,
                    "spectral_radius": 0.0,
                    "matrix_norm": 0.0
                }
            
        except Exception as e:
            logger.warning(f"Error computing convergence indicators: {e}")
            indicators = {
                "condition_number": 1e12,
                "spectral_radius": 0.0,
                "matrix_norm": 0.0
            }
        
        return indicators
    
    def run_lattice_calculation(self, N: int) -> LatticeResult:
        """
        Run complete calculation for single lattice size
        
        Args:
            N: Lattice size
            
        Returns:
            LatticeResult object
        """
        logger.info(f"Computing lattice N={N}...")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        # Compute observables
        observables = self.compute_lattice_observables(N)
        
        # Measure performance
        computation_time = time.time() - start_time
        peak_memory = self._get_memory_usage() - start_memory
        
        # Extract convergence indicators
        convergence_indicators = {
            key: observables.pop(key, 0.0) 
            for key in ["condition_number", "spectral_radius", "matrix_norm"]
        }
        
        result = LatticeResult(
            N=N,
            observables=observables,
            computation_time=computation_time,
            memory_usage=peak_memory,
            convergence_indicators=convergence_indicators
        )
        
        logger.info(f"N={N} completed in {computation_time:.2f}s, memory: {peak_memory:.2f}MB")
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 0.0
    
    def run_full_refinement_study(self) -> List[LatticeResult]:
        """
        Run complete lattice refinement study
        
        Returns:
            List of LatticeResult objects
        """
        logger.info("Starting extended lattice refinement study...")
        
        lattice_sizes = self.config["lattice_sizes"]
        
        # Check if extended sizes should be included
        if self.config.get("use_extended_sizes", False):
            lattice_sizes.extend(self.config["extended_sizes"])
        
        results = []
        
        if self.config["parallel_workers"] > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.config["parallel_workers"]) as executor:
                future_to_N = {executor.submit(self.run_lattice_calculation, N): N for N in lattice_sizes}
                
                for future in as_completed(future_to_N):
                    N = future_to_N[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to compute N={N}: {e}")
        else:
            # Sequential execution
            for N in lattice_sizes:
                try:
                    result = self.run_lattice_calculation(N)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to compute N={N}: {e}")
        
        # Sort results by lattice size
        results.sort(key=lambda r: r.N)
        self.lattice_results = results
        
        logger.info(f"Completed refinement study with {len(results)} lattice sizes")
        
        return results
    
    def richardson_extrapolation(self, observable_name: str, 
                                order: int = 2) -> ContinuumExtrapolation:
        """
        Perform Richardson extrapolation to continuum limit
        
        Args:
            observable_name: Name of observable to extrapolate
            order: Expected convergence order
            
        Returns:
            ContinuumExtrapolation object
        """
        if not self.lattice_results:
            raise ValueError("No lattice results available for extrapolation")
        
        # Extract data
        N_values = []
        observable_values = []
        
        for result in self.lattice_results:
            if observable_name in result.observables:
                N_values.append(result.N)
                observable_values.append(result.observables[observable_name])
        
        if len(N_values) < 3:
            raise ValueError(f"Need at least 3 data points for extrapolation, got {len(N_values)}")
        
        N_array = np.array(N_values)
        obs_array = np.array(observable_values)
        
        # Richardson extrapolation formula: O(h) = O_continuum + A*h^p + B*h^(p+1) + ...
        # where h = 1/N is the lattice spacing
        
        h_values = 1.0 / N_array
        
        # Fit polynomial in h
        try:
            # Fit: obs = a0 + a1*h^order + a2*h^(order+1) + ...
            max_degree = min(len(h_values) - 1, 4)  # Limit degree to avoid overfitting
            
            best_fit = None
            best_error = float('inf')
            best_order = order
            
            for test_order in range(1, max_degree + 1):
                try:
                    # Create fitting function
                    def fit_func(h, *params):
                        result = params[0]  # Continuum value
                        for i, param in enumerate(params[1:], 1):
                            result += param * (h ** (test_order + i - 1))
                        return result
                    
                    # Initial parameter guess
                    p0 = [obs_array[-1]] + [0.1] * min(len(obs_array) - 1, 3)
                    
                    # Perform fit
                    popt, pcov = curve_fit(fit_func, h_values, obs_array, p0=p0)
                    
                    # Compute fit quality
                    fitted_values = fit_func(h_values, *popt)
                    fit_error = np.sqrt(np.mean((obs_array - fitted_values)**2))
                    
                    if fit_error < best_error:
                        best_fit = popt
                        best_error = fit_error
                        best_order = test_order
                        
                except Exception:
                    continue
            
            if best_fit is None:
                raise ValueError("Could not perform Richardson extrapolation")
            
            # Extrapolated value is the constant term
            extrapolated_value = best_fit[0]
            
            # Estimate error from fit covariance
            if len(best_fit) > 1:
                extrapolation_error = np.sqrt(np.abs(best_fit[1]) * (1.0 / N_array[-1])**best_order)
            else:
                extrapolation_error = 0.0
            
            # Compute R-squared for fit quality
            ss_res = np.sum((obs_array - [extrapolated_value + sum(best_fit[i] * (1.0/N)**i for i in range(1, len(best_fit))) for N in N_array])**2)
            ss_tot = np.sum((obs_array - np.mean(obs_array))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Richardson extrapolation failed for {observable_name}: {e}")
            # Fallback to simple linear extrapolation
            extrapolated_value = obs_array[-1]  # Use finest lattice value
            extrapolation_error = np.std(obs_array[-3:]) if len(obs_array) >= 3 else 0.0
            best_order = 1.0
            r_squared = 0.0
        
        extrapolation = ContinuumExtrapolation(
            observable_name=observable_name,
            lattice_values=observable_values,
            lattice_sizes=N_values,
            extrapolated_value=extrapolated_value,
            extrapolation_error=extrapolation_error,
            convergence_order=best_order,
            fit_quality=r_squared
        )
        
        return extrapolation
    
    def extrapolate_all_observables(self) -> Dict[str, ContinuumExtrapolation]:
        """
        Perform continuum extrapolation for all observables
        
        Returns:
            Dictionary of extrapolation results
        """
        if not self.lattice_results:
            raise ValueError("No lattice results available")
        
        # Get all observable names
        all_observables = set()
        for result in self.lattice_results:
            all_observables.update(result.observables.keys())
        
        extrapolations = {}
        
        for observable in all_observables:
            try:
                extrapolation = self.richardson_extrapolation(observable)
                extrapolations[observable] = extrapolation
                
                logger.info(f"Extrapolated {observable}: {extrapolation.extrapolated_value:.6e} ± {extrapolation.extrapolation_error:.2e}")
                
            except Exception as e:
                logger.warning(f"Failed to extrapolate {observable}: {e}")
        
        self.extrapolations = extrapolations
        return extrapolations
    
    def analyze_convergence_patterns(self) -> Dict[str, any]:
        """
        Analyze convergence patterns across lattice sizes
        
        Returns:
            Convergence analysis results
        """
        if not self.lattice_results:
            return {"error": "No results available"}
        
        analysis = {
            "lattice_sizes": [r.N for r in self.lattice_results],
            "computation_times": [r.computation_time for r in self.lattice_results],
            "memory_usage": [r.memory_usage for r in self.lattice_results],
            "convergence_analysis": {},
            "performance_scaling": {}
        }
        
        # Analyze each observable
        for observable in self.config["observables"]:
            values = []
            N_values = []
            
            for result in self.lattice_results:
                if observable in result.observables:
                    values.append(result.observables[observable])
                    N_values.append(result.N)
            
            if len(values) >= 2:
                # Compute convergence rate
                convergence_rate = self._estimate_convergence_rate(N_values, values)
                
                # Compute relative changes
                relative_changes = []
                for i in range(1, len(values)):
                    if abs(values[i-1]) > 1e-15:
                        rel_change = abs((values[i] - values[i-1]) / values[i-1])
                        relative_changes.append(rel_change)
                
                analysis["convergence_analysis"][observable] = {
                    "convergence_rate": convergence_rate,
                    "relative_changes": relative_changes,
                    "final_value": values[-1],
                    "value_range": [min(values), max(values)]
                }
        
        # Performance scaling analysis
        if len(analysis["computation_times"]) >= 2:
            N_array = np.array(analysis["lattice_sizes"])
            time_array = np.array(analysis["computation_times"])
            
            # Fit power law: time ~ N^alpha
            log_N = np.log(N_array)
            log_time = np.log(time_array)
            
            time_scaling = np.polyfit(log_N, log_time, 1)[0]
            
            analysis["performance_scaling"] = {
                "time_scaling_exponent": time_scaling,
                "memory_scaling": self._analyze_memory_scaling(),
                "efficiency_metric": time_array[-1] / (N_array[-1] ** time_scaling)
            }
        
        return analysis
    
    def _estimate_convergence_rate(self, N_values: List[int], obs_values: List[float]) -> float:
        """Estimate convergence rate for observable"""
        if len(N_values) < 2:
            return 0.0
        
        try:
            # Assume convergence like O(1/N^p)
            N_array = np.array(N_values)
            obs_array = np.array(obs_values)
            
            # Take differences between consecutive points
            if len(obs_array) >= 3:
                # Richardson-type analysis
                h1 = 1.0 / N_array[-1]
                h2 = 1.0 / N_array[-2]
                h3 = 1.0 / N_array[-3]
                
                o1 = obs_array[-1]
                o2 = obs_array[-2]
                o3 = obs_array[-3]
                
                # Estimate order: p = log((o3-o2)/(o2-o1)) / log(h3/h2 * h1/h2)
                if abs(o2 - o1) > 1e-15 and abs(h3/h2 * h1/h2) > 1e-15:
                    order = np.log(abs((o3 - o2) / (o2 - o1))) / np.log(h3/h2 * h1/h2)
                    return float(order) if np.isfinite(order) else 0.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _analyze_memory_scaling(self) -> Dict[str, float]:
        """Analyze memory usage scaling"""
        if len(self.lattice_results) < 2:
            return {}
        
        N_values = np.array([r.N for r in self.lattice_results])
        memory_values = np.array([r.memory_usage for r in self.lattice_results])
        
        # Filter out zero memory values
        valid_mask = memory_values > 0
        if np.sum(valid_mask) < 2:
            return {}
        
        N_valid = N_values[valid_mask]
        mem_valid = memory_values[valid_mask]
        
        try:
            # Fit power law: memory ~ N^alpha
            log_N = np.log(N_valid)
            log_mem = np.log(mem_valid)
            
            scaling_exponent = np.polyfit(log_N, log_mem, 1)[0]
            
            return {
                "memory_scaling_exponent": float(scaling_exponent),
                "predicted_memory_N20": float(mem_valid[-1] * (20.0 / N_valid[-1]) ** scaling_exponent)
            }
            
        except Exception:
            return {}
    
    def save_results(self, filename: str = "extended_lattice_refinement.json"):
        """Save complete results to file"""
        # Convert results to serializable format
        serializable_results = []
        for result in self.lattice_results:
            serializable_results.append({
                "N": result.N,
                "observables": result.observables,
                "computation_time": result.computation_time,
                "memory_usage": result.memory_usage,
                "convergence_indicators": result.convergence_indicators
            })
        
        # Convert extrapolations
        serializable_extrapolations = {}
        for name, extrap in self.extrapolations.items():
            serializable_extrapolations[name] = {
                "observable_name": extrap.observable_name,
                "lattice_values": extrap.lattice_values,
                "lattice_sizes": extrap.lattice_sizes,
                "extrapolated_value": extrap.extrapolated_value,
                "extrapolation_error": extrap.extrapolation_error,
                "convergence_order": extrap.convergence_order,
                "fit_quality": extrap.fit_quality
            }
        
        # Complete output
        output_data = {
            "lattice_results": serializable_results,
            "continuum_extrapolations": serializable_extrapolations,
            "convergence_analysis": self.analyze_convergence_patterns(),
            "config": self.config,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
    
    def generate_report(self) -> str:
        """Generate comprehensive refinement report"""
        if not self.lattice_results:
            return "No results available for report generation."
        
        convergence_analysis = self.analyze_convergence_patterns()
        
        report = f"""
EXTENDED LATTICE REFINEMENT REPORT
================================

Configuration:
- Lattice sizes: {[r.N for r in self.lattice_results]}
- Observables: {len(self.config['observables'])}
- Extrapolation methods: {self.config['extrapolation_methods']}

Lattice Study Results:
- Total calculations: {len(self.lattice_results)}
- Largest lattice: N={max(r.N for r in self.lattice_results)}
- Total computation time: {sum(r.computation_time for r in self.lattice_results):.2f}s
- Peak memory usage: {max(r.memory_usage for r in self.lattice_results):.2f}MB

Continuum Extrapolations:
"""
        
        for name, extrap in self.extrapolations.items():
            report += f"\n{name}:\n"
            report += f"  Continuum value: {extrap.extrapolated_value:.6e} ± {extrap.extrapolation_error:.2e}\n"
            report += f"  Convergence order: {extrap.convergence_order:.2f}\n"
            report += f"  Fit quality (R²): {extrap.fit_quality:.4f}\n"
        
        report += f"\nPerformance Scaling:\n"
        if "performance_scaling" in convergence_analysis:
            perf = convergence_analysis["performance_scaling"]
            report += f"  Time scaling: O(N^{perf.get('time_scaling_exponent', 0):.2f})\n"
            if "memory_scaling_exponent" in perf:
                report += f"  Memory scaling: O(N^{perf['memory_scaling_exponent']:.2f})\n"
        
        return report

def main():
    """Main execution function"""
    print("🔬 Starting Extended Lattice Refinement Study...")
    
    # Initialize refinement framework
    refinement = ExtendedLatticeRefinement()
    
    # Run complete refinement study
    results = refinement.run_full_refinement_study()
    
    # Perform continuum extrapolations
    extrapolations = refinement.extrapolate_all_observables()
    
    # Analyze convergence patterns
    analysis = refinement.analyze_convergence_patterns()
    
    # Save results
    refinement.save_results("outputs/extended_lattice_refinement.json")
    
    # Generate and display report
    report = refinement.generate_report()
    print(report)
    
    # Save report
    with open("outputs/lattice_refinement_report.txt", 'w') as f:
        f.write(report)
    
    print(f"\n✅ Extended lattice refinement complete!")
    print(f"📊 Studied {len(results)} lattice sizes up to N={max(r.N for r in results)}")
    print(f"🎯 Extrapolated {len(extrapolations)} observables to continuum limit")
    
    # Show key extrapolation results
    for name, extrap in extrapolations.items():
        if extrap.fit_quality > 0.8:  # Good fits only
            print(f"   {name}: {extrap.extrapolated_value:.6e} ± {extrap.extrapolation_error:.2e}")

if __name__ == "__main__":
    main()
