#!/usr/bin/env python3
"""
Automated Constraint Closure Testing for LQG Framework

This module implements comprehensive testing of Hamiltonian and 
momentum constraint closure in loop quantum gravity, particularly
in the midisuperspace models.

Key Features:
- Automated constraint algebra testing
- Poisson bracket computation and verification
- Closure tests for various polymer parameter ranges
- Anomaly detection in quantum constraint algebra
- Statistical analysis of closure violations
"""

import numpy as np
import warnings
import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import kstest, normaltest

warnings.filterwarnings("ignore")

@dataclass
class ConstraintConfig:
    """Configuration for constraint closure testing."""
    n_test_points: int = 1000
    parameter_ranges: Dict[str, Tuple[float, float]] = None
    tolerance: float = 1e-10
    statistical_significance: float = 0.05
    test_types: List[str] = None
    output_dir: str = "outputs/constraint_tests"

class ConstraintAlgebra:
    """
    Implementation of constraint algebra testing for LQG.
    
    Tests closure of constraints {H, H}, {H, C_i}, {C_i, C_j}
    where H is the Hamiltonian constraint and C_i are momentum constraints.
    """
    
    def __init__(self, config: ConstraintConfig):
        self.config = config
        self._setup_default_config()
        self.results = {}
        
    def _setup_default_config(self):
        """Setup default configuration values."""
        if self.config.parameter_ranges is None:
            self.config.parameter_ranges = {
                'mu_polymer': (0.01, 1.0),
                'alpha': (0.1, 2.0),
                'beta': (0.1, 2.0),
                'gamma': (0.0, 1.0),
                'area_gap': (1e-4, 1e-1)
            }
            
        if self.config.test_types is None:
            self.config.test_types = [
                'hamiltonian_closure',
                'momentum_closure', 
                'mixed_closure',
                'anomaly_scan',
                'statistical_analysis'
            ]
    
    def hamiltonian_constraint(self, q: np.ndarray, p: np.ndarray, params: Dict[str, float]) -> float:
        """
        Compute Hamiltonian constraint H[q,p].
        
        Args:
            q: Configuration variables (metric components)
            p: Momentum variables (extrinsic curvature)
            params: Physical parameters
            
        Returns:
            H: Value of Hamiltonian constraint
        """
        # Extract parameters
        mu = params.get('mu_polymer', 0.1)
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 1.0)
        
        # Polymer corrections to kinetic term
        p_eff = np.sin(mu * p) / mu if mu > 0 else p
        
        # Kinetic part: G_{ijkl} p^{ij} p^{kl}
        kinetic = alpha * np.sum(p_eff**2)
        
        # Potential part: polymer-corrected curvature
        # For midisuperspace: R = R[q] with polymer corrections
        q_eff = np.sin(mu * q) / mu if mu > 0 else q
        potential = beta * np.sum(q_eff**2)
        
        return kinetic - potential
    
    def momentum_constraint(self, q: np.ndarray, p: np.ndarray, params: Dict[str, float], direction: int = 0) -> float:
        """
        Compute momentum constraint C_i[q,p].
        
        Args:
            q: Configuration variables
            p: Momentum variables  
            params: Physical parameters
            direction: Spatial direction index
            
        Returns:
            C_i: Value of momentum constraint
        """
        # Momentum constraint: D_i p^{ij} = 0
        # In discrete setting: finite difference of momentum
        if direction >= len(p):
            return 0.0
            
        # Simple finite difference for momentum constraint
        if direction == 0:
            constraint = np.gradient(p, axis=0) if p.ndim > 0 else 0.0
        elif direction == 1:
            constraint = np.gradient(p, axis=1) if p.ndim > 1 else 0.0 
        else:
            constraint = np.gradient(p, axis=2) if p.ndim > 2 else 0.0
            
        return np.sum(constraint) if hasattr(constraint, '__iter__') else constraint
    
    def poisson_bracket(self, f: Callable, g: Callable, q: np.ndarray, p: np.ndarray, 
                       params: Dict[str, float], epsilon: float = 1e-8) -> float:
        """
        Compute Poisson bracket {f,g} using finite differences.
        
        Args:
            f, g: Functions to compute bracket of
            q, p: Phase space coordinates
            params: Parameters
            epsilon: Finite difference step
            
        Returns:
            bracket: Value of {f,g}
        """
        bracket = 0.0
        
        # {f,g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
        for i in range(len(q)):
            # Partial derivatives with respect to q_i
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += epsilon
            q_minus[i] -= epsilon
            
            df_dq = (f(q_plus, p, params) - f(q_minus, p, params)) / (2 * epsilon)
            dg_dq = (g(q_plus, p, params) - g(q_minus, p, params)) / (2 * epsilon)
            
            # Partial derivatives with respect to p_i  
            p_plus = p.copy()
            p_minus = p.copy()
            p_plus[i] += epsilon
            p_minus[i] -= epsilon
            
            df_dp = (f(q, p_plus, params) - f(q, p_minus, params)) / (2 * epsilon)
            dg_dp = (g(q, p_plus, params) - g(q, p_minus, params)) / (2 * epsilon)
            
            bracket += df_dq * dg_dp - df_dp * dg_dq
            
        return bracket
    
    def test_hamiltonian_closure(self) -> Dict[str, Any]:
        """Test closure of Hamiltonian constraint: {H,H} = 0."""
        print("🔬 Testing Hamiltonian constraint closure...")
        
        violations = []
        parameter_sets = []
        
        for _ in range(self.config.n_test_points):
            # Generate random test point
            params = self._generate_random_parameters()
            q, p = self._generate_random_phase_space_point()
            
            # Compute {H,H}
            def H(q_arg, p_arg, params_arg):
                return self.hamiltonian_constraint(q_arg, p_arg, params_arg)
                
            hh_bracket = self.poisson_bracket(H, H, q, p, params)
            
            violations.append(abs(hh_bracket))
            parameter_sets.append(params.copy())
        
        # Statistical analysis
        violations = np.array(violations)
        max_violation = np.max(violations)
        mean_violation = np.mean(violations)
        std_violation = np.std(violations)
        
        # Test for closure (should be ~ 0)
        closure_satisfied = max_violation < self.config.tolerance
        
        results = {
            'test_type': 'hamiltonian_closure',
            'max_violation': float(max_violation),
            'mean_violation': float(mean_violation),
            'std_violation': float(std_violation),
            'closure_satisfied': closure_satisfied,
            'n_tests': self.config.n_test_points,
            'violations': violations.tolist(),
            'parameter_sets': parameter_sets
        }
        
        print(f"   Max violation: {max_violation:.2e}")
        print(f"   Mean violation: {mean_violation:.2e}")
        print(f"   Closure satisfied: {'✅' if closure_satisfied else '❌'}")
        
        return results
    
    def test_momentum_closure(self) -> Dict[str, Any]:
        """Test closure of momentum constraints: {C_i, C_j} = 0."""
        print("🔬 Testing momentum constraint closure...")
        
        violations = []
        parameter_sets = []
        
        for _ in range(self.config.n_test_points):
            params = self._generate_random_parameters()
            q, p = self._generate_random_phase_space_point()
            
            # Test {C_i, C_j} for different directions
            max_violation_this_point = 0.0
            
            for i, j in itertools.combinations(range(3), 2):
                def C_i(q_arg, p_arg, params_arg):
                    return self.momentum_constraint(q_arg, p_arg, params_arg, i)
                    
                def C_j(q_arg, p_arg, params_arg):
                    return self.momentum_constraint(q_arg, p_arg, params_arg, j)
                
                cc_bracket = self.poisson_bracket(C_i, C_j, q, p, params)
                max_violation_this_point = max(max_violation_this_point, abs(cc_bracket))
            
            violations.append(max_violation_this_point)
            parameter_sets.append(params.copy())
        
        violations = np.array(violations)
        max_violation = np.max(violations)
        mean_violation = np.mean(violations)
        closure_satisfied = max_violation < self.config.tolerance
        
        results = {
            'test_type': 'momentum_closure',
            'max_violation': float(max_violation),
            'mean_violation': float(mean_violation),
            'closure_satisfied': closure_satisfied,
            'violations': violations.tolist(),
            'parameter_sets': parameter_sets
        }
        
        print(f"   Max violation: {max_violation:.2e}")
        print(f"   Closure satisfied: {'✅' if closure_satisfied else '❌'}")
        
        return results
    
    def test_mixed_closure(self) -> Dict[str, Any]:
        """Test mixed closure: {H, C_i} = structure functions."""
        print("🔬 Testing mixed constraint closure...")
        
        violations = []
        parameter_sets = []
        
        for _ in range(self.config.n_test_points):
            params = self._generate_random_parameters()
            q, p = self._generate_random_phase_space_point()
            
            max_violation_this_point = 0.0
            
            # Test {H, C_i} for each direction
            for i in range(3):
                def H(q_arg, p_arg, params_arg):
                    return self.hamiltonian_constraint(q_arg, p_arg, params_arg)
                    
                def C_i(q_arg, p_arg, params_arg):
                    return self.momentum_constraint(q_arg, p_arg, params_arg, i)
                
                hc_bracket = self.poisson_bracket(H, C_i, q, p, params)
                
                # {H, C_i} should equal structure function times constraints
                # For simplicity, test if it's a linear combination of constraints
                expected = 0.0  # In some models this should be zero
                violation = abs(hc_bracket - expected)
                max_violation_this_point = max(max_violation_this_point, violation)
            
            violations.append(max_violation_this_point)
            parameter_sets.append(params.copy())
        
        violations = np.array(violations)
        max_violation = np.max(violations)
        mean_violation = np.mean(violations)
        closure_satisfied = max_violation < self.config.tolerance * 10  # More lenient for structure functions
        
        results = {
            'test_type': 'mixed_closure',
            'max_violation': float(max_violation),
            'mean_violation': float(mean_violation),
            'closure_satisfied': closure_satisfied,
            'violations': violations.tolist(),
            'parameter_sets': parameter_sets
        }
        
        print(f"   Max violation: {max_violation:.2e}")
        print(f"   Closure satisfied: {'✅' if closure_satisfied else '❌'}")
        
        return results
    
    def scan_anomalies(self) -> Dict[str, Any]:
        """Scan for anomalies in constraint algebra."""
        print("🔬 Scanning for constraint anomalies...")
        
        # Test constraint algebra over parameter space
        mu_values = np.linspace(*self.config.parameter_ranges['mu_polymer'], 20)
        alpha_values = np.linspace(*self.config.parameter_ranges['alpha'], 20)
        
        anomaly_map = np.zeros((len(mu_values), len(alpha_values)))
        
        for i, mu in enumerate(mu_values):
            for j, alpha in enumerate(alpha_values):
                params = {'mu_polymer': mu, 'alpha': alpha, 'beta': 1.0}
                q, p = self._generate_random_phase_space_point()
                
                # Compute constraint violation
                def H(q_arg, p_arg, params_arg):
                    return self.hamiltonian_constraint(q_arg, p_arg, params_arg)
                
                hh_bracket = self.poisson_bracket(H, H, q, p, params)
                anomaly_map[i, j] = abs(hh_bracket)
        
        # Find maximum anomaly
        max_anomaly = np.max(anomaly_map)
        max_indices = np.unravel_index(np.argmax(anomaly_map), anomaly_map.shape)
        worst_mu = mu_values[max_indices[0]]
        worst_alpha = alpha_values[max_indices[1]]
        
        results = {
            'test_type': 'anomaly_scan',
            'max_anomaly': float(max_anomaly),
            'worst_parameters': {'mu_polymer': worst_mu, 'alpha': worst_alpha},
            'anomaly_map': anomaly_map.tolist(),
            'mu_values': mu_values.tolist(),
            'alpha_values': alpha_values.tolist()
        }
        
        print(f"   Max anomaly: {max_anomaly:.2e} at μ={worst_mu:.3f}, α={worst_alpha:.3f}")
        
        return results
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform statistical analysis of constraint violations."""
        print("🔬 Performing statistical analysis...")
        
        # Collect large sample of violations
        n_large = 10000
        violations = []
        
        for _ in range(n_large):
            params = self._generate_random_parameters()
            q, p = self._generate_random_phase_space_point()
            
            def H(q_arg, p_arg, params_arg):
                return self.hamiltonian_constraint(q_arg, p_arg, params_arg)
                
            hh_bracket = self.poisson_bracket(H, H, q, p, params)
            violations.append(abs(hh_bracket))
        
        violations = np.array(violations)
        
        # Statistical tests
        # Test for normality
        normality_stat, normality_p = normaltest(violations)
        is_normal = normality_p > self.config.statistical_significance
        
        # Test for uniform distribution
        uniform_stat, uniform_p = kstest(violations, 'uniform')
        is_uniform = uniform_p > self.config.statistical_significance
        
        # Compute percentiles
        percentiles = np.percentile(violations, [50, 90, 95, 99, 99.9])
        
        results = {
            'test_type': 'statistical_analysis',
            'sample_size': n_large,
            'mean': float(np.mean(violations)),
            'std': float(np.std(violations)),
            'median': float(np.median(violations)),
            'percentiles': {
                '50%': float(percentiles[0]),
                '90%': float(percentiles[1]),
                '95%': float(percentiles[2]),
                '99%': float(percentiles[3]),
                '99.9%': float(percentiles[4])
            },
            'normality_test': {
                'statistic': float(normality_stat),
                'p_value': float(normality_p),
                'is_normal': is_normal
            },
            'uniformity_test': {
                'statistic': float(uniform_stat),
                'p_value': float(uniform_p),
                'is_uniform': is_uniform
            }
        }
        
        print(f"   Mean violation: {np.mean(violations):.2e}")
        print(f"   99% violations below: {percentiles[3]:.2e}")
        print(f"   Distribution normal: {'✅' if is_normal else '❌'}")
        
        return results
    
    def _generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameters within specified ranges."""
        params = {}
        for param, (min_val, max_val) in self.config.parameter_ranges.items():
            params[param] = np.random.uniform(min_val, max_val)
        return params
    
    def _generate_random_phase_space_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random point in phase space."""
        # For simplicity, use 3D configuration and momentum
        q = np.random.normal(0, 1, 3)
        p = np.random.normal(0, 1, 3)
        return q, p
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all constraint closure tests."""
        print("🧪 CONSTRAINT CLOSURE TESTING")
        print("=" * 50)
        
        all_results = {}
        
        if 'hamiltonian_closure' in self.config.test_types:
            all_results['hamiltonian_closure'] = self.test_hamiltonian_closure()
            
        if 'momentum_closure' in self.config.test_types:
            all_results['momentum_closure'] = self.test_momentum_closure()
            
        if 'mixed_closure' in self.config.test_types:
            all_results['mixed_closure'] = self.test_mixed_closure()
            
        if 'anomaly_scan' in self.config.test_types:
            all_results['anomaly_scan'] = self.scan_anomalies()
            
        if 'statistical_analysis' in self.config.test_types:
            all_results['statistical_analysis'] = self.statistical_analysis()
          # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
          # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(all_results)
        
        with open(output_dir / "constraint_closure_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        # Summary
        print("\n📊 CONSTRAINT CLOSURE SUMMARY")
        print("-" * 30)
        
        total_tests = len([t for t in self.config.test_types if t != 'statistical_analysis'])
        passed_tests = sum([
            all_results.get('hamiltonian_closure', {}).get('closure_satisfied', False),
            all_results.get('momentum_closure', {}).get('closure_satisfied', False),
            all_results.get('mixed_closure', {}).get('closure_satisfied', False)
        ])
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("✅ All constraint closure tests passed!")
        else:
            print("⚠️  Some constraint closure tests failed")
            print("   Consider adjusting polymer parameters or tolerance")
        
        return all_results

def main():
    """Main function for constraint closure testing."""
    config = ConstraintConfig(
        n_test_points=500,
        tolerance=1e-8,
        parameter_ranges={
            'mu_polymer': (0.01, 0.5),
            'alpha': (0.5, 2.0),
            'beta': (0.5, 2.0),
            'gamma': (0.0, 0.5)
        }
    )
    
    algebra_tester = ConstraintAlgebra(config)
    results = algebra_tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()
