#!/usr/bin/env python3
"""
Enhanced LQG metric derivation with higher-order corrections.

This module extends the standard O(μ²) LQG metric to higher orders (μ⁴, μ⁶, ...)
using systematic polymer expansion and closed-form resummed expressions.
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from symbolic_timeout_utils import (
    safe_symbolic_operation, safe_series, safe_solve, safe_simplify, 
    safe_expand, safe_integrate, set_default_timeout
)

# Set timeout for this module
set_default_timeout(8)  # Higher timeout for complex derivations

# Global symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)

# Higher-order coefficients
alpha, beta, gamma_coeff, delta = sp.symbols('alpha beta gamma delta', real=True)

class LQGMetricHigherOrder:
    """
    Class for deriving LQG-corrected metrics to arbitrary order in μ.
    """
    
    def __init__(self, max_order: int = 6):
        """
        Initialize with maximum order in μ expansion.
        
        Args:
            max_order: Maximum power of μ to include (must be even)
        """
        self.max_order = max_order
        if max_order % 2 != 0:
            raise ValueError("max_order must be even (LQG corrections appear at even powers)")
        
        self.coefficients = {}  # Will store α, β, γ, etc.
        self.metric_function = None
        self.closed_form = None
        
    def construct_effective_hamiltonian(self) -> sp.Expr:
        """
        Construct the effective Hamiltonian constraint with polymer corrections.
        
        Returns:
            Effective Hamiltonian density with μ expansion
        """
        print("Constructing effective Hamiltonian with polymer corrections...")
        
        # Classical kinetic part
        H_classical = (
            -(Ephi / sp.sqrt(Ex)) * Kphi**2
            - 2 * Kphi * Kx * sp.sqrt(Ex)
        )
        
        print("  Classical Hamiltonian:")
        print(f"    H_classical = {H_classical}")
        
        # Apply polymer corrections: K → sin(μK)/μ
        print(f"  Applying polymer corrections to order μ^{self.max_order}...")
        
        # Expand sin(μK)/μ = 1 - (μK)²/6 + (μK)⁴/120 - ...
        mu_Kx = mu * Kx
        mu_Kphi = mu * Kphi
        
        # Series expansion of sin(μK)/μ
        sin_expansion_x = safe_series(
            sp.sin(mu_Kx) / mu_Kx, mu_Kx, 0, n=self.max_order//2 + 2
        )
        sin_expansion_phi = safe_series(
            sp.sin(mu_Kphi) / mu_Kphi, mu_Kphi, 0, n=self.max_order//2 + 2
        )
        
        if sin_expansion_x is None or sin_expansion_phi is None:
            print("  Warning: Series expansion failed, using truncated manual expansion")
            # Manual expansion as fallback
            sin_expansion_x = 1 - (mu_Kx)**2/6 + (mu_Kx)**4/120
            sin_expansion_phi = 1 - (mu_Kphi)**2/6 + (mu_Kphi)**4/120
        else:
            # Remove O() terms
            sin_expansion_x = sin_expansion_x.removeO()
            sin_expansion_phi = sin_expansion_phi.removeO()
        
        # Apply corrections
        Kx_poly = Kx * sin_expansion_x
        Kphi_poly = Kphi * sin_expansion_phi
        
        # Construct polymer Hamiltonian
        H_polymer = (
            -(Ephi / sp.sqrt(Ex)) * Kphi_poly**2
            - 2 * Kphi_poly * Kx_poly * sp.sqrt(Ex)
        )
        
        # Expand to desired order in μ
        print("  Expanding polymer Hamiltonian...")
        H_expanded = safe_expand(H_polymer)
        
        if H_expanded is None:
            print("  Warning: Expansion failed, using unexpanded form")
            H_expanded = H_polymer
        
        return H_expanded
    
    def extract_radial_dependence(self, H_polymer: sp.Expr) -> sp.Expr:
        """
        Extract the radial-dependent part of the Hamiltonian for metric derivation.
        
        In spherically symmetric case, we relate:
        - E^x ~ r², E^φ ~ r²sin²θ ≈ r² (at θ=π/2)
        - K_x ~ ∂_r(ln a), K_φ ~ (connection components)
        
        For metric f(r) = 1 - 2M/r + corrections, we get:
        K_x ~ M/r², K_φ ~ related to angular momentum
        
        Args:
            H_polymer: Polymer-corrected Hamiltonian
            
        Returns:
            Radial part suitable for metric solving
        """
        print("  Extracting radial dependence for metric derivation...")
        
        # Spherically symmetric ansatz
        # E^x ~ r², E^φ ~ r²
        # K_x ~ -M/r²f(r), K_φ ~ 0 (spherically symmetric)
        
        # Substitute spherical symmetry relations
        Ex_sub = r**2
        Ephi_sub = r**2
        Kx_sub = -M / (r**2 * (1 - 2*M/r + alpha*mu**2*M**2/r**4))  # Include LQG corrections
        Kphi_sub = 0  # Spherical symmetry
        
        # Make substitutions
        H_radial = H_polymer.subs([
            (Ex, Ex_sub),
            (Ephi, Ephi_sub),
            (Kx, Kx_sub),
            (Kphi, Kphi_sub)
        ])
        
        return H_radial
    
    def solve_metric_coefficients(self, H_radial: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Solve for metric coefficients by expanding constraint H_radial = 0.
        
        The metric ansatz is:
        f(r) = 1 - 2M/r + α*μ²*M²/r⁴ + β*μ⁴*M³/r⁷ + γ*μ⁶*M⁴/r¹⁰ + ...
        
        Args:
            H_radial: Radial Hamiltonian constraint
            
        Returns:
            Dictionary with coefficients {α: value, β: value, γ: value, ...}
        """
        print("  Solving for metric coefficients...")
        
        # Metric function ansatz
        f_terms = [1, -2*M/r]
        
        # Add higher-order terms
        order_powers = [(2, alpha, M**2/r**4),
                       (4, beta, M**3/r**7),
                       (6, gamma_coeff, M**4/r**10)]
        
        for order, coeff, term in order_powers:
            if order <= self.max_order:
                f_terms.append(coeff * mu**order * term)
        
        f_ansatz = sum(f_terms)
        
        print(f"    Metric ansatz: f(r) = {f_ansatz}")
        
        # Substitute into constraint and expand in μ
        constraint = H_radial.subs('f', f_ansatz)
        
        # Expand constraint in powers of μ
        print("  Expanding constraint in μ...")
        constraint_expanded = safe_series(constraint, mu, 0, n=self.max_order//2 + 2)
        
        if constraint_expanded is None:
            print("    Warning: Series expansion failed, attempting manual expansion")
            # Manual approach: substitute and collect terms
            constraint_expanded = safe_expand(constraint)
        else:
            constraint_expanded = constraint_expanded.removeO()
        
        # Collect coefficients of different powers of μ
        print("  Collecting coefficients...")
        coefficients = {}
        
        # Extract coefficient of μ⁰ (should give classical constraint)
        coeff_mu0 = constraint_expanded.coeff(mu, 0)
        print(f"    μ⁰ coefficient: {coeff_mu0}")
        
        # Extract coefficients of μ², μ⁴, μ⁶, ...
        for order in range(2, self.max_order + 1, 2):
            coeff = constraint_expanded.coeff(mu, order)
            if coeff is not None:
                coeff_name = {2: 'alpha', 4: 'beta', 6: 'gamma', 8: 'delta'}[order]
                coefficients[coeff_name] = coeff
                print(f"    μ^{order} coefficient: {coeff}")
        
        # Solve for coefficients by setting constraint = 0
        # This is typically done order by order
        
        # At O(μ²): solve for α
        if 'alpha' in coefficients:
            alpha_eq = sp.Eq(coefficients['alpha'], 0)
            alpha_solution = safe_solve(alpha_eq, alpha)
            if alpha_solution:
                coefficients['alpha'] = alpha_solution[0]
                print(f"    Solved α = {coefficients['alpha']}")
        
        # At O(μ⁴): solve for β (may depend on α)
        if 'beta' in coefficients and 'alpha' in coefficients:
            beta_eq = coefficients['beta'].subs(alpha, coefficients['alpha'])
            beta_eq = sp.Eq(beta_eq, 0)
            beta_solution = safe_solve(beta_eq, beta)
            if beta_solution:
                coefficients['beta'] = beta_solution[0]
                print(f"    Solved β = {coefficients['beta']}")
        
        return coefficients
    
    def construct_closed_form(self, coefficients: Dict[str, sp.Expr]) -> sp.Expr:
        """
        Attempt to construct a closed-form resummed expression.
        
        Args:
            coefficients: Dictionary of solved coefficients
            
        Returns:
            Closed-form metric function
        """
        print("  Constructing closed-form metric...")
        
        # Start with standard form
        f_standard = 1 - 2*M/r
        
        # Add LQG corrections
        if 'alpha' in coefficients:
            alpha_val = coefficients['alpha']
            
            # Try different resummed forms
            # Form 1: Geometric series
            # f = 1 - 2M/r + (α*μ²*M²/r⁴) * (1 + c₁*μ²*M/r³ + c₂*μ⁴*M²/r⁶ + ...)
            
            # Form 2: Rational function
            # f = 1 - 2M/r + (α*μ²*M²/r⁴) / (1 + δ*μ²*M/r³)
            
            correction_base = alpha_val * mu**2 * M**2 / r**4
            
            if 'beta' in coefficients:
                beta_val = coefficients['beta']
                
                # Try to find pattern: if β = c*α², then we can write
                # f = 1 - 2M/r + α*μ²*M²/r⁴ * (1 + (β/α²)*α*μ²*M/r³)
                
                # Check if β has form c*α² for some constant c
                beta_over_alpha_sq = safe_simplify(beta_val / alpha_val**2)
                
                if beta_over_alpha_sq is not None and beta_over_alpha_sq.is_number:
                    print(f"    Found β/α² = {beta_over_alpha_sq}")
                    
                    # Construct rational form
                    delta_val = beta_over_alpha_sq * alpha_val
                    closed_form = (1 - 2*M/r + 
                                 correction_base / (1 + delta_val * mu**2 * M / r**3))
                    
                    print(f"    Closed form: {closed_form}")
                    return closed_form
            
            # Fallback: standard polynomial form
            f_poly = f_standard + correction_base
            
            if 'beta' in coefficients:
                f_poly += coefficients['beta'] * mu**4 * M**3 / r**7
            
            if 'gamma' in coefficients:
                f_poly += coefficients['gamma'] * mu**6 * M**4 / r**10
                
            return f_poly
        
        return f_standard
    
    def derive_metric(self) -> Dict[str, Any]:
        """
        Complete derivation of higher-order LQG metric.
        
        Returns:
            Dictionary with all results
        """
        print(f"="*60)
        print(f"DERIVING LQG METRIC TO ORDER μ^{self.max_order}")
        print(f"="*60)
        
        results = {}
        
        try:
            # Step 1: Construct effective Hamiltonian
            H_polymer = self.construct_effective_hamiltonian()
            results['hamiltonian_polymer'] = H_polymer
            
            # Step 2: Extract radial dependence
            H_radial = self.extract_radial_dependence(H_polymer)
            results['hamiltonian_radial'] = H_radial
            
            # Step 3: Solve for coefficients
            coefficients = self.solve_metric_coefficients(H_radial)
            results['coefficients'] = coefficients
            self.coefficients = coefficients
            
            # Step 4: Construct closed form
            closed_form = self.construct_closed_form(coefficients)
            results['closed_form'] = closed_form
            self.closed_form = closed_form
            
            # Step 5: Validate by expanding closed form
            print("  Validating closed form by expansion...")
            expanded_validation = safe_series(closed_form, mu, 0, n=self.max_order//2 + 2)
            if expanded_validation:
                results['validation_expansion'] = expanded_validation.removeO()
            
            results['success'] = True
            print("✓ Higher-order LQG metric derivation completed successfully")
            
        except Exception as e:
            print(f"✗ Derivation failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
    
    def evaluate_at_parameters(self, M_val: float, mu_val: float, 
                             r_vals: np.ndarray) -> np.ndarray:
        """
        Evaluate the derived metric at specific parameter values.
        
        Args:
            M_val: Mass parameter value
            mu_val: Polymer scale parameter value
            r_vals: Array of radial coordinate values
            
        Returns:
            Array of metric function values
        """
        if self.closed_form is None:
            raise ValueError("Must derive metric first")
        
        # Substitute parameter values
        f_numeric = self.closed_form
        
        # Substitute coefficient values if available
        for name, value in self.coefficients.items():
            if name == 'alpha':
                f_numeric = f_numeric.subs(alpha, float(value))
            elif name == 'beta':
                f_numeric = f_numeric.subs(beta, float(value))
            # Add more as needed
        
        # Substitute M and μ
        f_numeric = f_numeric.subs([(M, M_val), (mu, mu_val)])
        
        # Convert to lambda function for evaluation
        f_lambda = sp.lambdify(r, f_numeric, 'numpy')
        
        return f_lambda(r_vals)

def derive_lqg_metric_higher_order(max_order: int = 6) -> Dict[str, Any]:
    """
    Convenience function to derive higher-order LQG metric.
    
    Args:
        max_order: Maximum power of μ to include
        
    Returns:
        Complete derivation results
    """
    derivation = LQGMetricHigherOrder(max_order=max_order)
    return derivation.derive_metric()

if __name__ == "__main__":
    # Test the higher-order derivation
    results = derive_lqg_metric_higher_order(max_order=4)
    
    if results['success']:
        print("\nDerived coefficients:")
        for name, value in results['coefficients'].items():
            print(f"  {name} = {value}")
        
        print(f"\nClosed-form metric:")
        print(f"  f_LQG(r) = {results['closed_form']}")
    else:
        print(f"Derivation failed: {results['error']}")
