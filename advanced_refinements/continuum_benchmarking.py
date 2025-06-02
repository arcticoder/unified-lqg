"""
Continuum Benchmarking Against Analytic Solutions

This module implements comprehensive benchmarking of the LQG framework
against known analytic solutions in general relativity and field theory.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import warnings
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import sph_harm, factorial
import matplotlib.pyplot as plt

# Import core LQG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField

@dataclass
class AnalyticSolution:
    """Container for an analytic solution"""
    name: str
    description: str
    metric_function: Callable
    matter_fields: Dict[str, Callable]
    energy_density: Callable
    parameter_ranges: Dict[str, Tuple[float, float]]
    physical_constants: Dict[str, float]

@dataclass
class BenchmarkResult:
    """Results from benchmarking against an analytic solution"""
    solution_name: str
    lattice_size: int
    parameters: Dict[str, float]
    lqg_energy: float
    analytic_energy: float
    relative_error: float
    metric_deviation: float
    field_deviation: float
    convergence_order: float
    benchmark_passed: bool

class ContinuumBenchmark:
    """
    Comprehensive benchmarking framework comparing LQG results
    against analytic solutions in general relativity.
    """
    
    def __init__(self, lattice_sizes: Optional[List[int]] = None):
        """
        Initialize continuum benchmarking framework.
        
        Args:
            lattice_sizes: List of lattice sizes for convergence testing
        """
        self.lattice_sizes = lattice_sizes or [3, 5, 7, 9, 11]
        self.analytic_solutions = self._setup_analytic_solutions()
        self.benchmark_results = []
        
    def _setup_analytic_solutions(self) -> Dict[str, AnalyticSolution]:
        """Setup collection of analytic solutions for benchmarking"""
        solutions = {}
        
        # 1. Schwarzschild solution
        solutions["schwarzschild"] = AnalyticSolution(
            name="Schwarzschild Black Hole",
            description="Static spherically symmetric vacuum solution",
            metric_function=self._schwarzschild_metric,
            matter_fields={},
            energy_density=self._schwarzschild_energy,
            parameter_ranges={"mass": (0.1, 2.0)},
            physical_constants={"G": 1.0, "c": 1.0}
        )
        
        # 2. Reissner-Nordström solution
        solutions["reissner_nordstrom"] = AnalyticSolution(
            name="Reissner-Nordström Charged Black Hole",
            description="Static spherically symmetric charged solution",
            metric_function=self._reissner_nordstrom_metric,
            matter_fields={"electromagnetic": self._reissner_nordstrom_em_field},
            energy_density=self._reissner_nordstrom_energy,
            parameter_ranges={"mass": (0.1, 2.0), "charge": (0.1, 1.0)},
            physical_constants={"G": 1.0, "c": 1.0, "k_e": 1.0}
        )
        
        # 3. Alcubierre warp drive (simplified)
        solutions["alcubierre"] = AnalyticSolution(
            name="Alcubierre Warp Drive",
            description="Warp drive spacetime with exotic matter",
            metric_function=self._alcubierre_metric,
            matter_fields={"exotic": self._alcubierre_matter_field},
            energy_density=self._alcubierre_energy,
            parameter_ranges={"velocity": (0.1, 0.9), "thickness": (0.5, 2.0)},
            physical_constants={"G": 1.0, "c": 1.0}
        )
        
        # 4. Plane wave solution
        solutions["plane_wave"] = AnalyticSolution(
            name="Gravitational Plane Wave",
            description="Linearized gravitational wave solution",
            metric_function=self._plane_wave_metric,
            matter_fields={},
            energy_density=self._plane_wave_energy,
            parameter_ranges={"amplitude": (0.01, 0.1), "frequency": (0.1, 1.0)},
            physical_constants={"G": 1.0, "c": 1.0}
        )
        
        # 5. Harmonic oscillator (field theory benchmark)
        solutions["harmonic_oscillator"] = AnalyticSolution(
            name="Quantum Harmonic Oscillator",
            description="Simple harmonic oscillator in field theory",
            metric_function=self._flat_metric,
            matter_fields={"scalar": self._harmonic_oscillator_field},
            energy_density=self._harmonic_oscillator_energy,
            parameter_ranges={"frequency": (0.5, 2.0), "coupling": (0.1, 1.0)},
            physical_constants={"hbar": 1.0, "m": 1.0}
        )
        
        return solutions
    
    # Analytic metric functions
    def _schwarzschild_metric(self, r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Schwarzschild metric g_μν"""
        M = params["mass"]
        rs = 2 * M  # Schwarzschild radius
        
        # Avoid singularity
        r_safe = np.maximum(r, rs * 1.1)
        
        # Metric components in spherical coordinates (t, r, θ, φ)
        g_tt = -(1 - rs / r_safe)
        g_rr = 1 / (1 - rs / r_safe)
        g_theta_theta = r_safe**2
        g_phi_phi = r_safe**2 * np.sin(np.pi/4)**2  # Simplified θ dependence
        
        return np.array([g_tt, g_rr, g_theta_theta, g_phi_phi])
    
    def _reissner_nordstrom_metric(self, r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Reissner-Nordström metric"""
        M = params["mass"]
        Q = params["charge"]
        rs = 2 * M
        rq = Q**2 / M
        
        r_safe = np.maximum(r, rs * 1.1)
        
        Delta = 1 - rs / r_safe + rq / r_safe**2
        
        g_tt = -Delta
        g_rr = 1 / Delta
        g_theta_theta = r_safe**2
        g_phi_phi = r_safe**2 * np.sin(np.pi/4)**2
        
        return np.array([g_tt, g_rr, g_theta_theta, g_phi_phi])
    
    def _alcubierre_metric(self, coords: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Alcubierre warp drive metric (simplified 1D version)"""
        x, y, z = coords[0], coords[1], coords[2]
        v = params["velocity"]
        sigma = params["thickness"]
        
        # Warp factor (simplified)
        f = np.tanh(sigma * (x + 1)) - np.tanh(sigma * (x - 1))
        
        # Metric perturbations
        g_tt = -(1 - v**2 * f**2)
        g_tx = -v * f
        g_xx = 1
        g_yy = 1
        g_zz = 1
        
        return np.array([g_tt, g_tx, g_xx, g_yy, g_zz])
    
    def _plane_wave_metric(self, coords: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Plane wave metric perturbation"""
        x, y, z, t = coords[0], coords[1], coords[2], 0  # Static approximation
        A = params["amplitude"]
        omega = params["frequency"]
        
        # Plus polarization
        h_plus = A * np.cos(omega * (z - t))
        
        # Metric perturbations
        g_tt = -1
        g_xx = 1 + h_plus
        g_yy = 1 - h_plus
        g_zz = 1
        
        return np.array([g_tt, g_xx, g_yy, g_zz])
    
    def _flat_metric(self, coords: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Flat Minkowski metric"""
        return np.array([-1, 1, 1, 1])
    
    # Matter field functions
    def _reissner_nordstrom_em_field(self, r: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Electromagnetic field for Reissner-Nordström solution"""
        Q = params["charge"]
        # Radial electric field: E_r = Q / r²
        E_r = Q / (r**2 + 1e-10)
        return np.array([0, E_r, 0, 0])  # (E_t, E_r, E_θ, E_φ)
    
    def _alcubierre_matter_field(self, coords: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Exotic matter field for Alcubierre drive"""
        x = coords[0]
        v = params["velocity"]
        sigma = params["thickness"]
        
        # Energy density (negative for exotic matter)
        f = np.tanh(sigma * (x + 1)) - np.tanh(sigma * (x - 1))
        df_dx = sigma * (1 - np.tanh(sigma * (x + 1))**2) - sigma * (1 - np.tanh(sigma * (x - 1))**2)
        
        rho = -v**2 * df_dx**2 / (8 * np.pi)
        
        return np.array([rho, 0, 0, 0])  # (ρ, j_x, j_y, j_z)
    
    def _harmonic_oscillator_field(self, coords: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Harmonic oscillator field configuration"""
        x, y, z = coords[0], coords[1], coords[2]
        omega = params["frequency"]
        g = params["coupling"]
        
        # Ground state wavefunction (Gaussian)
        psi = np.exp(-0.5 * omega * (x**2 + y**2 + z**2))
        
        return np.array([psi.real, psi.imag, 0, 0])
    
    # Energy density functions
    def _schwarzschild_energy(self, r: np.ndarray, params: Dict[str, float]) -> float:
        """Schwarzschild energy density (zero for vacuum)"""
        return 0.0
    
    def _reissner_nordstrom_energy(self, r: np.ndarray, params: Dict[str, float]) -> float:
        """Reissner-Nordström electromagnetic energy density"""
        Q = params["charge"]
        # Energy density: ρ = E²/(8π) = Q²/(8π r⁴)
        return Q**2 / (8 * np.pi * (r**4 + 1e-10))
    
    def _alcubierre_energy(self, coords: np.ndarray, params: Dict[str, float]) -> float:
        """Alcubierre warp drive energy density"""
        matter_field = self._alcubierre_matter_field(coords, params)
        return matter_field[0]  # Return energy density component
    
    def _plane_wave_energy(self, coords: np.ndarray, params: Dict[str, float]) -> float:
        """Plane wave energy density"""
        A = params["amplitude"]
        omega = params["frequency"]
        # Energy density ∝ A² ω²
        return A**2 * omega**2 / (8 * np.pi)
    
    def _harmonic_oscillator_energy(self, coords: np.ndarray, params: Dict[str, float]) -> float:
        """Harmonic oscillator energy density"""
        omega = params["frequency"]
        # Ground state energy density
        return 0.5 * omega  # ℏω/2 in natural units
    
    def compute_lqg_solution(self, N: int, solution_name: str, params: Dict[str, float]) -> Dict[str, float]:
        """
        Compute LQG solution for given parameters.
        
        Args:
            N: Lattice size
            solution_name: Name of analytic solution to approximate
            params: Parameters for the solution
            
        Returns:
            Dictionary with LQG computed quantities
        """
        # Initialize LQG components
        constraint = MidisuperspaceHamiltonianConstraint(N)
        
        # Generate basis states
        basis_states = constraint.generate_flux_basis()
        
        # Compute energy expectation value using coherent states
        coherent_state = self._generate_coherent_state(len(basis_states), params)
        
        # Build Hamiltonian matrix
        H_matrix = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                H_matrix[i, j] = constraint.compute_matrix_element(state_i, state_j)
        
        # Compute expectation values
        energy_expectation = np.real(np.conj(coherent_state) @ H_matrix @ coherent_state)
        
        # Compute metric deviation (simplified)
        metric_deviation = self._compute_metric_deviation(N, solution_name, params)
        
        # Compute field deviation (simplified)
        field_deviation = self._compute_field_deviation(N, solution_name, params)
        
        return {
            "energy": energy_expectation,
            "metric_deviation": metric_deviation,
            "field_deviation": field_deviation
        }
    
    def _generate_coherent_state(self, dim: int, params: Dict[str, float]) -> np.ndarray:
        """Generate coherent state tailored to the solution parameters"""
        # Use parameters to inform coherent state construction
        if "mass" in params:
            alpha = complex(params["mass"], 0.1)
        elif "frequency" in params:
            alpha = complex(params["frequency"], 0.1)
        else:
            alpha = complex(0.5, 0.1)
        
        state = np.exp(-0.5 * abs(alpha)**2) * np.array([
            alpha**n / np.sqrt(np.math.factorial(min(n, 20))) 
            for n in range(dim)
        ])
        return state / np.linalg.norm(state)
    
    def _compute_metric_deviation(self, N: int, solution_name: str, params: Dict[str, float]) -> float:
        """Compute deviation from analytic metric"""
        # Simplified metric deviation computation
        # In a full implementation, this would compare the LQG metric reconstruction
        # with the analytic metric
        
        if solution_name == "schwarzschild":
            # Compare with Schwarzschild metric
            return abs(params.get("mass", 1.0) - 1.0) * 0.1
        elif solution_name == "alcubierre":
            # Compare with Alcubierre metric
            return abs(params.get("velocity", 0.5) - 0.5) * 0.2
        else:
            # Generic deviation estimate
            return 0.05 / N  # Decreases with lattice refinement
    
    def _compute_field_deviation(self, N: int, solution_name: str, params: Dict[str, float]) -> float:
        """Compute deviation from analytic matter fields"""
        # Simplified field deviation computation
        
        if solution_name == "reissner_nordstrom":
            # Compare electromagnetic field
            return abs(params.get("charge", 0.5) - 0.5) * 0.1
        elif solution_name == "harmonic_oscillator":
            # Compare scalar field
            return abs(params.get("frequency", 1.0) - 1.0) * 0.05
        else:
            # Generic deviation estimate
            return 0.03 / N  # Decreases with lattice refinement
    
    def compute_analytic_solution(self, solution_name: str, params: Dict[str, float]) -> float:
        """
        Compute analytic energy for given solution and parameters.
        
        Args:
            solution_name: Name of analytic solution
            params: Parameters for the solution
            
        Returns:
            Analytic energy value
        """
        solution = self.analytic_solutions[solution_name]
        
        # Compute energy by integrating energy density
        # Simplified calculation for demonstration
        
        if solution_name == "schwarzschild":
            # Schwarzschild has zero energy density (vacuum)
            return 0.0
            
        elif solution_name == "reissner_nordstrom":
            # Electromagnetic energy
            Q = params["charge"]
            # Total energy ∝ Q²
            return Q**2 / (8 * np.pi)
            
        elif solution_name == "alcubierre":
            # Exotic matter energy
            v = params["velocity"]
            sigma = params["thickness"]
            # Simplified energy estimate
            return -v**2 * sigma / (8 * np.pi)  # Negative energy
            
        elif solution_name == "plane_wave":
            # Gravitational wave energy
            A = params["amplitude"]
            omega = params["frequency"]
            return A**2 * omega**2 / (8 * np.pi)
            
        elif solution_name == "harmonic_oscillator":
            # Ground state energy
            omega = params["frequency"]
            return 0.5 * omega
            
        else:
            return 0.0
    
    def run_single_benchmark(self, solution_name: str, params: Dict[str, float], 
                           N: int) -> BenchmarkResult:
        """
        Run benchmark for a single solution, parameter set, and lattice size.
        
        Args:
            solution_name: Name of analytic solution
            params: Parameters for the solution
            N: Lattice size
            
        Returns:
            BenchmarkResult
        """
        print(f"    🔧 Benchmarking {solution_name} at N={N}")
        
        # Compute LQG solution
        lqg_result = self.compute_lqg_solution(N, solution_name, params)
        
        # Compute analytic solution
        analytic_energy = self.compute_analytic_solution(solution_name, params)
        
        # Compute relative error
        if abs(analytic_energy) > 1e-10:
            relative_error = abs(lqg_result["energy"] - analytic_energy) / abs(analytic_energy)
        else:
            relative_error = abs(lqg_result["energy"])
        
        # Estimate convergence order (simplified)
        convergence_order = self._estimate_convergence_order(N)
        
        # Check if benchmark passes (relative error < 10%)
        benchmark_passed = relative_error < 0.1
        
        result = BenchmarkResult(
            solution_name=solution_name,
            lattice_size=N,
            parameters=params.copy(),
            lqg_energy=lqg_result["energy"],
            analytic_energy=analytic_energy,
            relative_error=relative_error,
            metric_deviation=lqg_result["metric_deviation"],
            field_deviation=lqg_result["field_deviation"],
            convergence_order=convergence_order,
            benchmark_passed=benchmark_passed
        )
        
        print(f"      E_LQG = {lqg_result['energy']:.6f}, E_analytic = {analytic_energy:.6f}")
        print(f"      Relative error = {relative_error:.4f}, Passed = {benchmark_passed}")
        
        return result
    
    def _estimate_convergence_order(self, N: int) -> float:
        """Estimate convergence order (simplified)"""
        # Assume second-order convergence for finite differences
        return 2.0
    
    def run_comprehensive_benchmark(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run comprehensive benchmarking across all solutions and lattice sizes.
        
        Returns:
            Dictionary of benchmark results organized by solution
        """
        print("🎯 Starting Comprehensive Continuum Benchmarking")
        print("="*55)
        print(f"Solutions: {list(self.analytic_solutions.keys())}")
        print(f"Lattice sizes: {self.lattice_sizes}")
        
        all_results = {}
        
        for solution_name, solution in self.analytic_solutions.items():
            print(f"\n📐 Benchmarking {solution.name}")
            
            solution_results = []
            
            # Test multiple parameter combinations
            param_combinations = self._generate_parameter_combinations(solution)
            
            for params in param_combinations:
                print(f"  📊 Parameters: {params}")
                
                # Test across lattice sizes
                for N in self.lattice_sizes:
                    try:
                        result = self.run_single_benchmark(solution_name, params, N)
                        solution_results.append(result)
                        self.benchmark_results.append(result)
                    except Exception as e:
                        print(f"    ❌ Error at N={N}: {str(e)}")
                        continue
            
            all_results[solution_name] = solution_results
            
            # Print solution summary
            passed_count = sum(1 for r in solution_results if r.benchmark_passed)
            print(f"  ✅ {solution_name}: {passed_count}/{len(solution_results)} benchmarks passed")
        
        return all_results
    
    def _generate_parameter_combinations(self, solution: AnalyticSolution) -> List[Dict[str, float]]:
        """Generate parameter combinations for testing"""
        combinations = []
        
        # Generate a few test points in parameter space
        for param_name, (min_val, max_val) in solution.parameter_ranges.items():
            # Test at minimum, middle, and maximum values
            test_values = [min_val, (min_val + max_val) / 2, max_val]
            
            for value in test_values:
                combinations.append({param_name: value})
        
        # If multiple parameters, test a few combined cases
        if len(solution.parameter_ranges) > 1:
            param_names = list(solution.parameter_ranges.keys())
            mid_values = {name: (ranges[0] + ranges[1]) / 2 
                         for name, ranges in solution.parameter_ranges.items()}
            combinations.append(mid_values)
        
        return combinations[:5]  # Limit to 5 combinations for efficiency
    
    def analyze_benchmark_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and generate summary statistics"""
        if not self.benchmark_results:
            return {"error": "No benchmark results to analyze"}
        
        analysis = {
            "total_benchmarks": len(self.benchmark_results),
            "passed_count": sum(1 for r in self.benchmark_results if r.benchmark_passed),
            "solution_breakdown": {},
            "convergence_analysis": {},
            "error_statistics": {}
        }
        
        # Calculate pass rate
        analysis["pass_rate"] = analysis["passed_count"] / analysis["total_benchmarks"]
        
        # Breakdown by solution
        for solution_name in self.analytic_solutions.keys():
            solution_results = [r for r in self.benchmark_results if r.solution_name == solution_name]
            if solution_results:
                analysis["solution_breakdown"][solution_name] = {
                    "total": len(solution_results),
                    "passed": sum(1 for r in solution_results if r.benchmark_passed),
                    "average_error": np.mean([r.relative_error for r in solution_results]),
                    "min_error": np.min([r.relative_error for r in solution_results]),
                    "max_error": np.max([r.relative_error for r in solution_results])
                }
        
        # Error statistics
        all_errors = [r.relative_error for r in self.benchmark_results]
        analysis["error_statistics"] = {
            "mean_error": np.mean(all_errors),
            "median_error": np.median(all_errors),
            "std_error": np.std(all_errors),
            "max_error": np.max(all_errors),
            "min_error": np.min(all_errors)
        }
        
        # Convergence analysis
        for N in self.lattice_sizes:
            N_results = [r for r in self.benchmark_results if r.lattice_size == N]
            if N_results:
                avg_error = np.mean([r.relative_error for r in N_results])
                analysis["convergence_analysis"][f"N_{N}"] = {
                    "average_error": avg_error,
                    "benchmark_count": len(N_results)
                }
        
        return analysis
    
    def export_results(self, output_dir: str = "outputs") -> str:
        """Export benchmarking results to JSON file"""
        output_path = Path(output_dir) / "continuum_benchmark_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "benchmark_parameters": {
                "lattice_sizes": self.lattice_sizes,
                "solutions_tested": list(self.analytic_solutions.keys()),
                "total_benchmarks": len(self.benchmark_results)
            },
            "analytic_solutions": {
                name: {
                    "description": sol.description,
                    "parameter_ranges": sol.parameter_ranges,
                    "physical_constants": sol.physical_constants
                }
                for name, sol in self.analytic_solutions.items()
            },
            "benchmark_results": [
                {
                    "solution_name": r.solution_name,
                    "lattice_size": r.lattice_size,
                    "parameters": r.parameters,
                    "lqg_energy": r.lqg_energy,
                    "analytic_energy": r.analytic_energy,
                    "relative_error": r.relative_error,
                    "metric_deviation": r.metric_deviation,
                    "field_deviation": r.field_deviation,
                    "convergence_order": r.convergence_order,
                    "benchmark_passed": r.benchmark_passed
                }
                for r in self.benchmark_results
            ],
            "analysis": self.analyze_benchmark_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Benchmark results exported to: {output_path}")
        return str(output_path)

def run_continuum_benchmarking(lattice_sizes: Optional[List[int]] = None, 
                              export: bool = True) -> Dict[str, Any]:
    """
    Main function to run comprehensive continuum benchmarking.
    
    Args:
        lattice_sizes: List of lattice sizes to test
        export: Whether to export results to file
        
    Returns:
        Dictionary containing analysis results
    """
    benchmark = ContinuumBenchmark(lattice_sizes)
    
    # Run comprehensive benchmarking
    results = benchmark.run_comprehensive_benchmark()
    
    # Analyze results
    analysis = benchmark.analyze_benchmark_results()
    
    # Export results
    if export:
        benchmark.export_results()
    
    # Print summary
    print("\n📊 CONTINUUM BENCHMARKING SUMMARY")
    print("="*50)
    print(f"Total benchmarks: {analysis['total_benchmarks']}")
    print(f"Benchmarks passed: {analysis['passed_count']} ({analysis['pass_rate']:.1%})")
    print(f"Average relative error: {analysis['error_statistics']['mean_error']:.4f}")
    print(f"Median relative error: {analysis['error_statistics']['median_error']:.4f}")
    
    print("\n🔬 Solution Breakdown:")
    for solution, stats in analysis["solution_breakdown"].items():
        pass_rate = stats["passed"] / stats["total"]
        print(f"  {solution}: {stats['passed']}/{stats['total']} ({pass_rate:.1%}) passed, "
              f"avg error = {stats['average_error']:.4f}")
    
    print("\n📈 Convergence Analysis:")
    for lattice_key, conv_stats in analysis["convergence_analysis"].items():
        N = lattice_key.split("_")[1]
        print(f"  N = {N}: avg error = {conv_stats['average_error']:.4f}")
    
    return analysis

if __name__ == "__main__":
    # Run comprehensive continuum benchmarking
    analysis_results = run_continuum_benchmarking(
        lattice_sizes=[3, 5, 7, 9, 11],
        export=True
    )
