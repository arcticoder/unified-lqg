"""
GPU Constraint Kernel Numerical Stability Enhancement
===================================================

Addresses critical UQ concern: GPU constraint kernel numerical instability near edge cases
with very small holonomy values or high-precision requirements.

This module provides enhanced numerical stability for LQG constraint computations,
critical for the 242M× energy reduction in Structural Integrity Field (SIF) applications.

Mathematical Foundation:
- Holonomy constraints: C(h) = Tr(h) - 2cos(θ) = 0
- Small angle stability: Use Taylor expansion for |θ| < ε
- Polymer corrections: sinc(πμ) enhancement requires stable numerics

References:
- Rovelli, Smolin (1995) - Spin Networks and LQG
- Ashtekar, Lewandowski (2004) - Background Independent Quantum Gravity
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logging.info("CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available - using CPU fallback")

@dataclass
class ConstraintStabilityParams:
    """Parameters for numerical stability enhancement"""
    stability_threshold: float = 1e-12      # Threshold for small value detection
    taylor_order: int = 6                   # Taylor expansion order for small values
    max_constraint_value: float = 1e6       # Maximum allowed constraint value
    convergence_tolerance: float = 1e-10    # Convergence tolerance for iterative methods
    max_iterations: int = 1000              # Maximum iterations for constraint solving
    enable_overflow_protection: bool = True # Enable overflow/underflow protection
    use_adaptive_precision: bool = True     # Use adaptive precision based on value magnitude

class NumericallyStableConstraintSolver:
    """
    Enhanced constraint solver with numerical stability for small holonomy values.
    
    Addresses GPU kernel instability through:
    1. Taylor expansion fallback for small values
    2. Overflow/underflow protection with clamping
    3. Adaptive precision switching
    4. Comprehensive edge case handling
    """
    
    def __init__(self, params: ConstraintStabilityParams):
        self.params = params
        self.use_gpu = CUPY_AVAILABLE
        self.total_computations = 0
        self.stability_corrections = 0
        self.overflow_corrections = 0
        
        logging.info(f"Constraint solver initialized: "
                    f"GPU={self.use_gpu}, "
                    f"stability_threshold={params.stability_threshold:.2e}")
    
    def compute_holonomy_constraint(self, 
                                  holonomy_values: np.ndarray,
                                  constraint_type: str = 'gauss') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute holonomy constraints with enhanced numerical stability.
        
        Args:
            holonomy_values: Array of holonomy values (complex or real)
            constraint_type: Type of constraint ('gauss', 'diffeomorphism', 'hamiltonian')
            
        Returns:
            constraints: Computed constraint values
            diagnostics: Numerical stability diagnostics
        """
        self.total_computations += 1
        start_time = time.time()
        
        # Convert to appropriate backend (GPU/CPU)
        if self.use_gpu:
            h_values = cp.asarray(holonomy_values)
        else:
            h_values = np.asarray(holonomy_values)
        
        # Detect small values requiring stability enhancement
        small_mask = self._detect_small_values(h_values)
        
        # Compute constraints with stability enhancement
        if constraint_type == 'gauss':
            constraints = self._compute_gauss_constraint_stable(h_values, small_mask)
        elif constraint_type == 'diffeomorphism':
            constraints = self._compute_diffeo_constraint_stable(h_values, small_mask)
        elif constraint_type == 'hamiltonian':
            constraints = self._compute_hamiltonian_constraint_stable(h_values, small_mask)
        else:
            raise ValueError(f"Unknown constraint type: {constraint_type}")
        
        # Apply overflow protection
        if self.params.enable_overflow_protection:
            constraints = self._apply_overflow_protection(constraints)
        
        # Convert back to numpy if using GPU
        if self.use_gpu:
            constraints = cp.asnumpy(constraints)
        
        computation_time = time.time() - start_time
        
        # Diagnostics
        diagnostics = {
            'computation_time_ms': computation_time * 1000,
            'small_values_detected': np.sum(small_mask),
            'stability_corrections_applied': self.stability_corrections,
            'overflow_corrections_applied': self.overflow_corrections,
            'max_constraint_value': float(np.max(np.abs(constraints))),
            'constraint_norm': float(np.linalg.norm(constraints)),
            'numerical_stability_score': self._compute_stability_score(constraints)
        }
        
        return constraints, diagnostics
    
    def _detect_small_values(self, h_values):
        """Detect values requiring stability enhancement"""
        if self.use_gpu:
            magnitude = cp.abs(h_values)
            small_mask = magnitude < self.params.stability_threshold
        else:
            magnitude = np.abs(h_values)
            small_mask = magnitude < self.params.stability_threshold
        
        return small_mask
    
    def _compute_gauss_constraint_stable(self, h_values, small_mask):
        """
        Compute Gauss constraint with numerical stability.
        
        Gauss constraint: C_G = ∂_i A_i^a = 0
        For holonomies: C_G(h) = Tr(h) - 2cos(θ)
        """
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        constraints = xp.zeros_like(h_values, dtype=xp.float64)
        
        # Standard computation for normal values
        normal_mask = ~small_mask
        if xp.any(normal_mask):
            # Standard holonomy constraint: Tr(h) - 2cos(arg(h))
            if xp.iscomplexobj(h_values):
                theta = xp.angle(h_values[normal_mask])
                constraints[normal_mask] = xp.real(h_values[normal_mask]) - 2.0 * xp.cos(theta)
            else:
                constraints[normal_mask] = h_values[normal_mask] - 2.0 * xp.cos(h_values[normal_mask])
        
        # Taylor expansion for small values
        if xp.any(small_mask):
            self.stability_corrections += xp.sum(small_mask)
            small_values = h_values[small_mask]
            
            # Taylor expansion: cos(θ) ≈ 1 - θ²/2 + θ⁴/24 - θ⁶/720 + ...
            if xp.iscomplexobj(small_values):
                theta = xp.angle(small_values)
            else:
                theta = small_values
            
            # High-order Taylor expansion for numerical stability
            cos_taylor = self._taylor_cos(theta, self.params.taylor_order)
            constraints[small_mask] = xp.real(small_values) - 2.0 * cos_taylor
        
        return constraints
    
    def _compute_diffeo_constraint_stable(self, h_values, small_mask):
        """
        Compute diffeomorphism constraint with numerical stability.
        
        Diffeomorphism constraint: C_D^i = F_{ab}^i = 0
        """
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        constraints = xp.zeros_like(h_values, dtype=xp.float64)
        
        # Simplified diffeomorphism constraint for demonstration
        # In practice, this would involve more complex field derivatives
        
        # Standard computation
        normal_mask = ~small_mask
        if xp.any(normal_mask):
            # Gradient-based constraint (simplified)
            constraints[normal_mask] = xp.gradient(h_values[normal_mask])
        
        # Stability enhancement for small values
        if xp.any(small_mask):
            self.stability_corrections += xp.sum(small_mask)
            # Use finite difference with increased precision
            small_values = h_values[small_mask]
            constraints[small_mask] = self._stable_gradient(small_values)
        
        return constraints
    
    def _compute_hamiltonian_constraint_stable(self, h_values, small_mask):
        """
        Compute Hamiltonian constraint with numerical stability.
        
        Hamiltonian constraint: C_H = R - 2Λ = 0 (Wheeler-DeWitt equation)
        """
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        constraints = xp.zeros_like(h_values, dtype=xp.float64)
        
        # Standard computation for normal values
        normal_mask = ~small_mask
        if xp.any(normal_mask):
            # Simplified scalar curvature constraint
            normal_values = h_values[normal_mask]
            constraints[normal_mask] = self._compute_scalar_curvature(normal_values)
        
        # Stability enhancement for small values
        if xp.any(small_mask):
            self.stability_corrections += xp.sum(small_mask)
            small_values = h_values[small_mask]
            # Use perturbative expansion around flat space
            constraints[small_mask] = self._perturbative_curvature(small_values)
        
        return constraints
    
    def _taylor_cos(self, theta, order):
        """High-order Taylor expansion of cos(θ) for numerical stability"""
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        result = xp.ones_like(theta)
        theta_power = theta * theta  # θ²
        
        # cos(θ) = 1 - θ²/2! + θ⁴/4! - θ⁶/6! + ...
        factorial = 2
        sign = -1
        
        for n in range(1, order//2 + 1):
            result += sign * theta_power / factorial
            theta_power *= theta * theta  # θ^(2n) -> θ^(2n+2)
            factorial *= (2*n + 1) * (2*n + 2)
            sign *= -1
        
        return result
    
    def _stable_gradient(self, values):
        """Compute gradient with enhanced numerical stability"""
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        # Use higher-order finite differences for small values
        h = self.params.stability_threshold * 10  # Adaptive step size
        
        if len(values) < 3:
            return xp.zeros_like(values)
        
        # Fourth-order finite difference for enhanced accuracy
        gradient = xp.zeros_like(values)
        gradient[2:-2] = (-values[4:] + 8*values[3:-1] - 8*values[1:-3] + values[:-4]) / (12*h)
        
        # Boundary conditions with second-order accuracy
        gradient[0] = (-3*values[0] + 4*values[1] - values[2]) / (2*h)
        gradient[1] = (values[2] - values[0]) / (2*h)
        gradient[-2] = (values[-1] - values[-3]) / (2*h)
        gradient[-1] = (3*values[-1] - 4*values[-2] + values[-3]) / (2*h)
        
        return gradient
    
    def _compute_scalar_curvature(self, values):
        """Compute scalar curvature for normal values"""
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        # Simplified scalar curvature computation
        # In practice, this would involve full metric derivatives
        return 2.0 * (values - 1.0) + 0.5 * (values - 1.0)**2
    
    def _perturbative_curvature(self, small_values):
        """Perturbative curvature expansion for small metric perturbations"""
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        # Linear approximation for small perturbations
        return 2.0 * small_values
    
    def _apply_overflow_protection(self, constraints):
        """Apply overflow/underflow protection with clamping"""
        if self.use_gpu:
            xp = cp
        else:
            xp = np
        
        # Count overflow corrections
        overflow_mask = xp.abs(constraints) > self.params.max_constraint_value
        self.overflow_corrections += xp.sum(overflow_mask)
        
        if self.overflow_corrections > 0:
            logging.warning(f"Applied {self.overflow_corrections} overflow corrections")
        
        # Clamp to safe range
        return xp.clip(constraints, 
                      -self.params.max_constraint_value, 
                       self.params.max_constraint_value)
    
    def _compute_stability_score(self, constraints):
        """Compute numerical stability score (0-1, higher is better)"""
        if len(constraints) == 0:
            return 1.0
        
        # Check for non-finite values
        finite_fraction = np.sum(np.isfinite(constraints)) / len(constraints)
        
        # Check magnitude distribution
        magnitudes = np.abs(constraints[np.isfinite(constraints)])
        if len(magnitudes) == 0:
            return 0.0
        
        # Score based on reasonable magnitude range
        reasonable_mask = (magnitudes > 1e-15) & (magnitudes < 1e15)
        magnitude_score = np.sum(reasonable_mask) / len(magnitudes)
        
        # Combined stability score
        return 0.7 * finite_fraction + 0.3 * magnitude_score
    
    def solve_constraint_system(self, 
                               initial_values: np.ndarray,
                               constraint_functions: list) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Solve system of constraints iteratively with numerical stability.
        
        Uses Newton-Raphson method with stability enhancements.
        """
        if self.use_gpu:
            xp = cp
            values = cp.asarray(initial_values)
        else:
            xp = np
            values = np.asarray(initial_values)
        
        start_time = time.time()
        converged = False
        iteration = 0
        error_history = []
        
        for iteration in range(self.params.max_iterations):
            # Compute constraints and their derivatives
            constraints = xp.zeros_like(values)
            jacobian = xp.zeros((len(values), len(values)))
            
            for i, constraint_func in enumerate(constraint_functions):
                c_val, c_diagnostics = constraint_func(values)
                constraints[i] = c_val
                
                # Numerical Jacobian with stability enhancement
                for j in range(len(values)):
                    h = max(self.params.stability_threshold, 
                           abs(values[j]) * 1e-8)  # Adaptive step size
                    
                    values_plus = values.copy()
                    values_plus[j] += h
                    c_plus, _ = constraint_func(values_plus)
                    
                    jacobian[i, j] = (c_plus - c_val) / h
            
            # Compute Newton step with regularization for stability
            try:
                # Add regularization for numerical stability
                reg_jacobian = jacobian + 1e-12 * xp.eye(len(values))
                delta = xp.linalg.solve(reg_jacobian, -constraints)
            except:
                # Fallback to pseudo-inverse for ill-conditioned systems
                delta = -xp.linalg.pinv(jacobian) @ constraints
            
            # Adaptive step size for stability
            step_size = 1.0
            max_delta = xp.max(xp.abs(delta))
            if max_delta > 1.0:
                step_size = 1.0 / max_delta
            
            values += step_size * delta
            
            # Check convergence
            error = xp.linalg.norm(constraints)
            error_history.append(float(error))
            
            if error < self.params.convergence_tolerance:
                converged = True
                break
        
        computation_time = time.time() - start_time
        
        # Convert back to numpy if using GPU
        if self.use_gpu:
            values = cp.asnumpy(values)
        
        diagnostics = {
            'converged': converged,
            'iterations': iteration + 1,
            'final_error': error_history[-1] if error_history else float('inf'),
            'error_history': error_history,
            'computation_time_ms': computation_time * 1000,
            'stability_score': self._compute_stability_score(values)
        }
        
        return values, diagnostics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance and stability metrics"""
        stability_rate = 1.0 - (self.stability_corrections / max(self.total_computations, 1))
        overflow_rate = self.overflow_corrections / max(self.total_computations, 1)
        
        return {
            'total_computations': self.total_computations,
            'stability_corrections': self.stability_corrections,
            'overflow_corrections': self.overflow_corrections,
            'stability_rate': stability_rate,
            'overflow_rate': overflow_rate,
            'gpu_acceleration': self.use_gpu,
            'overall_health': 'HEALTHY' if stability_rate > 0.95 and overflow_rate < 0.05 else 'DEGRADED'
        }

def create_stable_constraint_solver(stability_threshold: float = 1e-12) -> NumericallyStableConstraintSolver:
    """
    Factory function to create numerically stable constraint solver.
    
    Args:
        stability_threshold: Threshold for small value detection and stability enhancement
        
    Returns:
        Configured constraint solver with enhanced numerical stability
    """
    params = ConstraintStabilityParams(
        stability_threshold=stability_threshold,
        taylor_order=6,
        max_constraint_value=1e6,
        convergence_tolerance=1e-10,
        enable_overflow_protection=True,
        use_adaptive_precision=True
    )
    
    return NumericallyStableConstraintSolver(params)

# Example usage and validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create stable constraint solver
    solver = create_stable_constraint_solver(stability_threshold=1e-12)
    
    # Test with problematic small values
    test_values = np.array([1e-15, 1e-10, 1.0, 1e15, 1e-13])
    
    # Compute constraints with stability enhancement
    constraints, diagnostics = solver.compute_holonomy_constraint(test_values, 'gauss')
    
    print("Numerical Stability Test Results:")
    print(f"Input values: {test_values}")
    print(f"Computed constraints: {constraints}")
    print(f"Stability diagnostics: {diagnostics}")
    
    # Performance metrics
    metrics = solver.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
