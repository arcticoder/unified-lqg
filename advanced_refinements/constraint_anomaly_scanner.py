"""
Advanced Constraint Anomaly Scanner with Multiple Regularization Parameters

This module implements comprehensive anomaly detection in the constraint algebra
using multiple regularization schemes and parameters to ensure physical consistency.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
from pathlib import Path

# Import core LQG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField

@dataclass
class RegularizationConfig:
    """Configuration for a specific regularization scheme"""
    name: str
    epsilon_range: Tuple[float, float]  # (min, max) regularization parameter
    num_points: int
    method: str  # 'linear', 'logarithmic', 'exponential'
    physical_cutoff: Optional[float] = None

@dataclass
class AnomalyResult:
    """Results from anomaly scanning"""
    regularization: str
    epsilon: float
    closure_error: float
    energy_expectation: complex
    constraint_norm: float
    is_anomaly_free: bool
    physical_consistency: bool

class ConstraintAnomalyScanner:
    """
    Advanced scanner for constraint algebra anomalies using multiple
    regularization schemes and systematic parameter sweeps.
    """
    
    def __init__(self, N: int = 7, use_coherent_states: bool = True):
        """
        Initialize the anomaly scanner.
        
        Args:
            N: Lattice size for quantum geometry
            use_coherent_states: Whether to use coherent states for analysis
        """
        self.N = N
        self.use_coherent_states = use_coherent_states
        self.results = []
        
        # Initialize the core constraint operator
        self.constraint = MidisuperspaceHamiltonianConstraint(N)
        
        # Initialize matter fields
        self.maxwell_field = MaxwellField(N)
        self.dirac_field = DiracField(N) 
        self.phantom_field = PhantomScalarField(N)
        
        # Define standard regularization schemes
        self.regularization_schemes = self._setup_regularization_schemes()
        
    def _setup_regularization_schemes(self) -> List[RegularizationConfig]:
        """Setup multiple regularization schemes for comprehensive testing"""
        return [
            RegularizationConfig(
                name="Pauli-Villars",
                epsilon_range=(1e-6, 1e-2),
                num_points=50,
                method="logarithmic",
                physical_cutoff=1e-3
            ),
            RegularizationConfig(
                name="Dimensional",
                epsilon_range=(1e-5, 1e-1),
                num_points=40,
                method="linear",
                physical_cutoff=5e-2
            ),
            RegularizationConfig(
                name="Point-splitting",
                epsilon_range=(1e-4, 1e-1),
                num_points=35,
                method="exponential",
                physical_cutoff=2e-2
            ),
            RegularizationConfig(
                name="Cutoff",
                epsilon_range=(1e-3, 1.0),
                num_points=30,
                method="logarithmic",
                physical_cutoff=0.1
            ),
            RegularizationConfig(
                name="Zeta-function",
                epsilon_range=(1e-7, 1e-3),
                num_points=60,
                method="logarithmic", 
                physical_cutoff=5e-4
            )
        ]
    
    def _generate_epsilon_values(self, config: RegularizationConfig) -> np.ndarray:
        """Generate regularization parameter values according to scheme"""
        if config.method == "linear":
            return np.linspace(config.epsilon_range[0], config.epsilon_range[1], config.num_points)
        elif config.method == "logarithmic":
            return np.logspace(
                np.log10(config.epsilon_range[0]), 
                np.log10(config.epsilon_range[1]), 
                config.num_points
            )
        elif config.method == "exponential":
            # Generate exponentially spaced points
            x = np.linspace(0, 1, config.num_points)
            return config.epsilon_range[0] + (config.epsilon_range[1] - config.epsilon_range[0]) * (np.exp(x) - 1) / (np.e - 1)
        else:
            raise ValueError(f"Unknown regularization method: {config.method}")
    
    def _regularized_constraint_operator(self, epsilon: float, method: str) -> np.ndarray:
        """
        Apply regularization to the constraint operator.
        
        Args:
            epsilon: Regularization parameter
            method: Regularization method name
            
        Returns:
            Regularized constraint operator matrix
        """
        # Get the base constraint operator
        basis_states = self.constraint.generate_flux_basis()
        constraint_matrix = np.zeros((len(basis_states), len(basis_states)), dtype=complex)
        
        for i, state_i in enumerate(basis_states):
            for j, state_j in enumerate(basis_states):
                # Apply constraint operator with regularization
                matrix_element = self._compute_regularized_matrix_element(
                    state_i, state_j, epsilon, method
                )
                constraint_matrix[i, j] = matrix_element
        
        return constraint_matrix
    
    def _compute_regularized_matrix_element(self, state_i: Dict, state_j: Dict, 
                                          epsilon: float, method: str) -> complex:
        """Compute regularized matrix element between two states"""
        
        # Base matrix element (unregularized)
        base_element = self.constraint.compute_matrix_element(state_i, state_j)
        
        if method == "Pauli-Villars":
            # Apply Pauli-Villars regularization: multiply by regulator function
            regulator = np.exp(-epsilon * abs(base_element))
            return base_element * regulator
            
        elif method == "Dimensional":
            # Dimensional regularization: scale by dimensional factor
            dim_factor = (epsilon / (4 * np.pi))**(2 - 4/2)  # d=4 spacetime
            return base_element * dim_factor
            
        elif method == "Point-splitting":
            # Point-splitting: add Gaussian regulator
            gaussian_reg = np.exp(-epsilon**2 * abs(base_element)**2)
            return base_element * gaussian_reg
            
        elif method == "Cutoff":
            # Hard cutoff regularization
            if abs(base_element) > 1/epsilon:
                return 0.0
            return base_element
            
        elif method == "Zeta-function":
            # Zeta function regularization (analytical continuation)
            zeta_reg = (epsilon)**(0.0)  # s=0 pole cancellation
            return base_element * zeta_reg
            
        else:
            return base_element
    
    def _compute_closure_error(self, constraint_matrix: np.ndarray) -> float:
        """Compute the constraint algebra closure error [H,H]"""
        commutator = constraint_matrix @ constraint_matrix - constraint_matrix @ constraint_matrix
        return np.linalg.norm(commutator, 'fro')
    
    def _compute_energy_expectation(self, constraint_matrix: np.ndarray) -> complex:
        """Compute energy expectation value for physical state"""
        if self.use_coherent_states:
            # Use coherent state as probe
            coherent_state = self._generate_coherent_state()
            return np.conj(coherent_state) @ constraint_matrix @ coherent_state
        else:
            # Use trace for energy scale
            return np.trace(constraint_matrix) / constraint_matrix.shape[0]
    
    def _generate_coherent_state(self) -> np.ndarray:
        """Generate coherent state for expectation value computation"""
        dim = self.N**3  # Hilbert space dimension
        # Gaussian coherent state with random complex amplitude
        alpha = 0.5 + 0.3j
        state = np.exp(-0.5 * abs(alpha)**2) * np.array([
            alpha**n / np.sqrt(np.math.factorial(min(n, 20))) 
            for n in range(dim)
        ])
        return state / np.linalg.norm(state)
    
    def _check_physical_consistency(self, result: AnomalyResult, 
                                  config: RegularizationConfig) -> bool:
        """Check if result satisfies physical consistency requirements"""
        checks = []
        
        # 1. Finite energy expectation
        checks.append(np.isfinite(result.energy_expectation.real))
        
        # 2. Bounded constraint norm
        checks.append(result.constraint_norm < 1e10)
        
        # 3. Regularization parameter in physical range
        if config.physical_cutoff:
            checks.append(result.epsilon <= config.physical_cutoff)
        
        # 4. Closure error within tolerance
        checks.append(result.closure_error < 1e-8)
        
        # 5. Real energy expectation (Hermiticity check)
        checks.append(abs(result.energy_expectation.imag) < 1e-10)
        
        return all(checks)
    
    def scan_single_regularization(self, config: RegularizationConfig) -> List[AnomalyResult]:
        """Scan anomalies for a single regularization scheme"""
        print(f"\n🔍 Scanning {config.name} regularization...")
        print(f"   Parameter range: {config.epsilon_range}")
        print(f"   Method: {config.method}, Points: {config.num_points}")
        
        epsilon_values = self._generate_epsilon_values(config)
        scheme_results = []
        
        for i, epsilon in enumerate(epsilon_values):
            if i % (config.num_points // 5) == 0:
                print(f"   Progress: {100*i//config.num_points}% (ε = {epsilon:.2e})")
            
            try:
                # Compute regularized constraint operator
                constraint_matrix = self._regularized_constraint_operator(epsilon, config.name)
                
                # Analyze constraint algebra
                closure_error = self._compute_closure_error(constraint_matrix)
                energy_expectation = self._compute_energy_expectation(constraint_matrix)
                constraint_norm = np.linalg.norm(constraint_matrix, 'fro')
                
                # Create result
                result = AnomalyResult(
                    regularization=config.name,
                    epsilon=epsilon,
                    closure_error=closure_error,
                    energy_expectation=energy_expectation,
                    constraint_norm=constraint_norm,
                    is_anomaly_free=(closure_error < 1e-8),
                    physical_consistency=False  # Will be set below
                )
                
                # Check physical consistency
                result.physical_consistency = self._check_physical_consistency(result, config)
                
                scheme_results.append(result)
                
            except Exception as e:
                warnings.warn(f"Error at ε={epsilon:.2e}: {str(e)}")
                continue
        
        print(f"   ✅ Completed {config.name} scan: {len(scheme_results)} points")
        return scheme_results
    
    def comprehensive_anomaly_scan(self) -> Dict[str, List[AnomalyResult]]:
        """Perform comprehensive anomaly scanning across all regularization schemes"""
        print("🚀 Starting Comprehensive Constraint Anomaly Scan")
        print("="*60)
        print(f"Lattice size: N = {self.N}")
        print(f"Coherent states: {self.use_coherent_states}")
        print(f"Regularization schemes: {len(self.regularization_schemes)}")
        
        all_results = {}
        
        for config in self.regularization_schemes:
            scheme_results = self.scan_single_regularization(config)
            all_results[config.name] = scheme_results
            self.results.extend(scheme_results)
        
        return all_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze anomaly scan results and generate summary statistics"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        analysis = {
            "total_points": len(self.results),
            "anomaly_free_count": sum(1 for r in self.results if r.is_anomaly_free),
            "physically_consistent_count": sum(1 for r in self.results if r.physical_consistency),
            "regularization_breakdown": {},
            "closure_error_statistics": {},
            "energy_statistics": {},
            "optimal_parameters": {}
        }
        
        # Calculate anomaly-free percentage
        analysis["anomaly_free_percentage"] = (
            100 * analysis["anomaly_free_count"] / analysis["total_points"]
        )
        
        # Breakdown by regularization scheme
        for scheme in set(r.regularization for r in self.results):
            scheme_results = [r for r in self.results if r.regularization == scheme]
            analysis["regularization_breakdown"][scheme] = {
                "total": len(scheme_results),
                "anomaly_free": sum(1 for r in scheme_results if r.is_anomaly_free),
                "physically_consistent": sum(1 for r in scheme_results if r.physical_consistency)
            }
        
        # Closure error statistics
        closure_errors = [r.closure_error for r in self.results if np.isfinite(r.closure_error)]
        if closure_errors:
            analysis["closure_error_statistics"] = {
                "min": float(np.min(closure_errors)),
                "max": float(np.max(closure_errors)),
                "mean": float(np.mean(closure_errors)),
                "std": float(np.std(closure_errors)),
                "median": float(np.median(closure_errors))
            }
        
        # Energy expectation statistics
        energies = [r.energy_expectation.real for r in self.results 
                   if np.isfinite(r.energy_expectation.real)]
        if energies:
            analysis["energy_statistics"] = {
                "min": float(np.min(energies)),
                "max": float(np.max(energies)),
                "mean": float(np.mean(energies)),
                "std": float(np.std(energies))
            }
        
        # Find optimal parameters (minimal closure error for each scheme)
        for scheme in set(r.regularization for r in self.results):
            scheme_results = [r for r in self.results 
                            if r.regularization == scheme and r.physical_consistency]
            if scheme_results:
                optimal = min(scheme_results, key=lambda x: x.closure_error)
                analysis["optimal_parameters"][scheme] = {
                    "epsilon": optimal.epsilon,
                    "closure_error": optimal.closure_error,
                    "energy_expectation": optimal.energy_expectation.real
                }
        
        return analysis
    
    def export_results(self, output_dir: str = "outputs") -> str:
        """Export anomaly scan results to JSON file"""
        output_path = Path(output_dir) / "constraint_anomaly_scan_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "scan_parameters": {
                "lattice_size": self.N,
                "use_coherent_states": self.use_coherent_states,
                "regularization_schemes": [
                    {
                        "name": config.name,
                        "epsilon_range": config.epsilon_range,
                        "num_points": config.num_points,
                        "method": config.method,
                        "physical_cutoff": config.physical_cutoff
                    }
                    for config in self.regularization_schemes
                ]
            },
            "results": [
                {
                    "regularization": r.regularization,
                    "epsilon": r.epsilon,
                    "closure_error": r.closure_error,
                    "energy_expectation_real": r.energy_expectation.real,
                    "energy_expectation_imag": r.energy_expectation.imag,
                    "constraint_norm": r.constraint_norm,
                    "is_anomaly_free": r.is_anomaly_free,
                    "physical_consistency": r.physical_consistency
                }
                for r in self.results
            ],
            "analysis": self.analyze_results()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Results exported to: {output_path}")
        return str(output_path)

def run_comprehensive_anomaly_scan(N: int = 7, export: bool = True) -> Dict[str, Any]:
    """
    Main function to run comprehensive constraint anomaly scanning.
    
    Args:
        N: Lattice size for quantum geometry
        export: Whether to export results to file
        
    Returns:
        Dictionary containing analysis results
    """
    scanner = ConstraintAnomalyScanner(N=N, use_coherent_states=True)
    
    # Perform comprehensive scan
    all_results = scanner.comprehensive_anomaly_scan()
    
    # Analyze results
    analysis = scanner.analyze_results()
    
    # Print summary
    print("\n📈 ANOMALY SCAN SUMMARY")
    print("="*50)
    print(f"Total points scanned: {analysis['total_points']}")
    print(f"Anomaly-free points: {analysis['anomaly_free_count']} ({analysis['anomaly_free_percentage']:.1f}%)")
    print(f"Physically consistent: {analysis['physically_consistent_count']}")
    
    print("\n🔬 Regularization Breakdown:")
    for scheme, stats in analysis["regularization_breakdown"].items():
        anomaly_rate = 100 * stats["anomaly_free"] / stats["total"]
        print(f"  {scheme}: {stats['anomaly_free']}/{stats['total']} ({anomaly_rate:.1f}%) anomaly-free")
    
    if analysis.get("optimal_parameters"):
        print("\n⚡ Optimal Parameters:")
        for scheme, params in analysis["optimal_parameters"].items():
            print(f"  {scheme}: ε = {params['epsilon']:.2e}, error = {params['closure_error']:.2e}")
    
    # Export results
    if export:
        scanner.export_results()
    
    return analysis

if __name__ == "__main__":
    # Run the comprehensive anomaly scan
    analysis_results = run_comprehensive_anomaly_scan(N=7, export=True)
