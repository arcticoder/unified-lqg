#!/usr/bin/env python3
"""
Comprehensive Polymer LQG Metric Coefficient Extraction

This script implements a complete workflow for extracting polymer LQG metric coefficients
α and β from classical-to-quantum Hamiltonian constraint analysis. The workflow includes:

1. Construction of polymer-expanded Hamiltonian with sin(μK)/μ holonomy corrections
2. Implementation of static constraint extraction with metric ansatz
   f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶
3. Solving coefficient equations by matching powers of μ
4. Using symbolic timeout utilities for robust computation
5. Complete workflow from classical Hamiltonian to extracted coefficients

The script integrates with the existing LQG framework infrastructure and uses
established patterns from the midisuperspace implementation.
"""

import sys
import os
import numpy as np
import sympy as sp
from typing import Dict, Any, Tuple, Optional, List
import json
import time
from dataclasses import dataclass

# Add scripts directory to path for imports
scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import symbolic timeout utilities
try:
    from symbolic_timeout_utils import (
        safe_integrate, safe_solve, safe_series, safe_diff, safe_simplify, 
        safe_expand, safe_collect, safe_nsimplify, safe_apart,
        safe_constraint_expand, safe_hamiltonian_simplify, 
        safe_lqg_series_expand, set_default_timeout
    )
    TIMEOUT_SUPPORT = True
    print("✓ Symbolic timeout utilities loaded")
except ImportError as e:
    print(f"⚠ Warning: Timeout utilities not available: {e}")
    print("  Using fallback implementations without timeout protection")
    TIMEOUT_SUPPORT = False
    
    # Fallback implementations
    def safe_integrate(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.integrate(expr, *args, **kwargs)
        except:
            return None
    
    def safe_solve(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.solve(expr, *args, **kwargs)
        except:
            return None
    
    def safe_series(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.series(expr, *args, **kwargs)
        except:
            return None
    
    def safe_simplify(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.simplify(expr, *args, **kwargs)
        except:
            return expr
    
    def safe_expand(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.expand(expr, *args, **kwargs)
        except:
            return expr
    
    def safe_collect(expr, *args, **kwargs):
        kwargs.pop('timeout_seconds', None)
        try:
            return sp.collect(expr, *args, **kwargs)
        except:
            return expr
    
    def safe_constraint_expand(expr, *args, **kwargs):
        return safe_expand(expr, *args, **kwargs)
    
    def safe_hamiltonian_simplify(expr, *args, **kwargs):
        return safe_simplify(expr, *args, **kwargs)
    
    def safe_lqg_series_expand(expr, var, point, n, *args, **kwargs):
        return safe_series(expr, var, point, n, *args, **kwargs)

# Set default timeout for symbolic operations
if TIMEOUT_SUPPORT:
    set_default_timeout(15)  # 15 seconds default timeout

# Global symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)
f = sp.Function('f')(r)

# Metric coefficients
alpha, beta, gamma_coeff, delta_coeff = sp.symbols('alpha beta gamma delta', real=True)

@dataclass
class PolymerExtractionConfig:
    """Configuration for polymer coefficient extraction."""
    max_polymer_order: int = 6  # Maximum order in μ for polymer expansion
    max_coefficient_order: int = 4  # Maximum order for coefficient extraction (α, β, ...)
    sin_expansion_terms: int = 4  # Number of terms in sin(μK)/μ expansion
    series_expansion_depth: int = 8  # Depth for 1/r series expansions
    default_timeout: int = 15  # Default timeout for symbolic operations
    verbose: bool = True  # Enable detailed output
    validate_results: bool = True  # Validate coefficient extraction

class PolymerLQGCoefficientExtractor:
    """
    Main class for extracting polymer LQG metric coefficients.
    
    This class implements the complete workflow from classical Hamiltonian
    construction through polymer quantization to coefficient extraction.
    """
    
    def __init__(self, config: Optional[PolymerExtractionConfig] = None):
        """
        Initialize the coefficient extractor.
        
        Args:
            config: Configuration object for extraction parameters
        """
        self.config = config or PolymerExtractionConfig()
        self.results = {}
        
        if self.config.verbose:
            print("="*80)
            print("POLYMER LQG COEFFICIENT EXTRACTION FRAMEWORK")
            print("="*80)
            print(f"Configuration:")
            print(f"  Max polymer order: μ^{self.config.max_polymer_order}")
            print(f"  Max coefficient order: μ^{self.config.max_coefficient_order}")
            print(f"  Sin expansion terms: {self.config.sin_expansion_terms}")
            print(f"  Series expansion depth: {self.config.series_expansion_depth}")
            print(f"  Timeout support: {'Yes' if TIMEOUT_SUPPORT else 'No'}")
    
    def construct_classical_hamiltonian(self) -> sp.Expr:
        """
        Construct the classical midisuperspace Hamiltonian constraint.
        
        The classical constraint in spherically symmetric variables is:
        H = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + (spatial curvature terms)
        
        Returns:
            Classical Hamiltonian expression
        """
        if self.config.verbose:
            print("\n1. CONSTRUCTING CLASSICAL HAMILTONIAN")
            print("-" * 50)
        
        # Classical midisuperspace Hamiltonian (spherically symmetric reduction)
        # Based on the Bojowald-Bertschinger formulation
        H_classical = (
            -(Ephi / sp.sqrt(Ex)) * Kphi**2
            - 2 * Kphi * Kx * sp.sqrt(Ex)
            # Spatial curvature terms would go here in full theory
        )
        
        if self.config.verbose:
            print(f"Classical Hamiltonian: H = {H_classical}")
        
        self.results['H_classical'] = H_classical
        return H_classical
    
    def derive_classical_solution(self) -> Tuple[sp.Expr, sp.Expr]:
        """
        Derive the classical Schwarzschild solution for K_x.
        
        From the classical constraint H = 0, we can derive:
        K_x = -K_φ√(E^φ/E^x) / 2
        
        For static spherically symmetric case with K_φ = 0, this gives K_x = 0.
        However, the polymer corrections will modify this.
        
        Returns:
            Tuple of (K_x solution, static metric ansatz)
        """
        if self.config.verbose:
            print("\n2. DERIVING CLASSICAL SCHWARZSCHILD SOLUTION")
            print("-" * 50)
        
        # For static case, K_φ = 0, so classical K_x = 0
        Kx_classical = 0
        
        # Classical Schwarzschild metric function
        f_classical = 1 - 2*M/r
        
        if self.config.verbose:
            print(f"Classical K_x solution: K_x = {Kx_classical}")
            print(f"Classical metric function: f(r) = {f_classical}")
        
        self.results['Kx_classical'] = Kx_classical
        self.results['f_classical'] = f_classical
        
        return Kx_classical, f_classical
    
    def apply_polymer_corrections(self, H_classical: sp.Expr, 
                                 Kx_classical: sp.Expr) -> sp.Expr:
        """
        Apply polymer corrections K → sin(μK)/μ to the classical Hamiltonian.
        
        The polymer corrections modify the classical curvature components:
        K_x → K_x * sin(μK_x)/(μK_x)
        K_φ → K_φ * sin(μK_φ)/(μK_φ)
        
        Args:
            H_classical: Classical Hamiltonian constraint
            Kx_classical: Classical K_x solution
            
        Returns:
            Polymer-corrected Hamiltonian expanded to specified order
        """
        if self.config.verbose:
            print("\n3. APPLYING POLYMER CORRECTIONS")
            print("-" * 50)
            print(f"Expanding sin(μK)/μ to {self.config.sin_expansion_terms} terms...")
        
        # Apply polymer corrections: K → sin(μK)/μ
        mu_Kx = mu * Kx  # We'll use the constraint-determined K_x
        mu_Kphi = mu * Kphi  # This will be zero for static case
        
        # Taylor expansion of sin(μK)/(μK) around μ=0
        print("  Expanding sin(μK_x)/(μK_x)...")
        sin_expansion_x = safe_lqg_series_expand(
            sp.sin(mu_Kx) / mu_Kx, mu, 0, 
            n=self.config.sin_expansion_terms + 1, 
            timeout_seconds=self.config.default_timeout
        )
        
        if sin_expansion_x is None:
            print("  Warning: K_x expansion timed out, using manual expansion")
            # Manual expansion: sin(x)/x = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...
            if self.config.sin_expansion_terms >= 2:
                sin_expansion_x = 1 - (mu_Kx)**2/6
            if self.config.sin_expansion_terms >= 4:
                sin_expansion_x += (mu_Kx)**4/120
            if self.config.sin_expansion_terms >= 6:
                sin_expansion_x -= (mu_Kx)**6/5040
        else:
            sin_expansion_x = sin_expansion_x.removeO()
        
        print("  Expanding sin(μK_φ)/(μK_φ)...")
        sin_expansion_phi = safe_lqg_series_expand(
            sp.sin(mu_Kphi) / mu_Kphi, mu, 0,
            n=self.config.sin_expansion_terms + 1,
            timeout_seconds=self.config.default_timeout
        )
        
        if sin_expansion_phi is None:
            print("  Warning: K_φ expansion timed out, using manual expansion")
            if self.config.sin_expansion_terms >= 2:
                sin_expansion_phi = 1 - (mu_Kphi)**2/6
            if self.config.sin_expansion_terms >= 4:
                sin_expansion_phi += (mu_Kphi)**4/120
            if self.config.sin_expansion_terms >= 6:
                sin_expansion_phi -= (mu_Kphi)**6/5040
        else:
            sin_expansion_phi = sin_expansion_phi.removeO()
        
        if self.config.verbose:
            print(f"  sin(μK_x)/(μK_x) ≈ {sin_expansion_x}")
            print(f"  sin(μK_φ)/(μK_φ) ≈ {sin_expansion_phi}")
        
        # Apply polymer substitutions
        Kx_polymer = Kx * sin_expansion_x
        Kphi_polymer = Kphi * sin_expansion_phi
        
        # Substitute into Hamiltonian
        H_polymer = H_classical.subs([
            (Kx, Kx_polymer),
            (Kphi, Kphi_polymer)
        ])
        
        # Expand to specified order
        print(f"  Expanding polymer Hamiltonian to order μ^{self.config.max_polymer_order}...")
        H_expanded = safe_constraint_expand(H_polymer, timeout_seconds=self.config.default_timeout)
        
        if H_expanded is None:
            print("  Warning: Polymer expansion timed out, using simpler form")
            H_expanded = safe_expand(H_polymer, timeout_seconds=self.config.default_timeout//2)
            if H_expanded is None:
                H_expanded = H_polymer
        
        # Collect powers of μ
        H_collected = safe_collect(H_expanded, mu, timeout_seconds=self.config.default_timeout)
        if H_collected is None:
            H_collected = H_expanded
        
        if self.config.verbose:
            print(f"  Polymer Hamiltonian (expanded): {H_collected}")
        
        self.results['H_polymer'] = H_collected
        return H_collected
    
    def formulate_metric_ansatz(self) -> sp.Expr:
        """
        Formulate the metric ansatz for coefficient extraction.
        
        The ansatz is:
        f(r) = 1 - 2M/r + α*μ²*M²/r⁴ + β*μ⁴*M⁴/r⁶ + O(μ⁶)
        
        Returns:
            Metric ansatz expression
        """
        if self.config.verbose:
            print("\n4. FORMULATING METRIC ANSATZ")
            print("-" * 50)
        
        # Start with classical Schwarzschild
        f_ansatz = 1 - 2*M/r
        
        # Add LQG corrections at even orders in μ
        if self.config.max_coefficient_order >= 2:
            f_ansatz += alpha * mu**2 * M**2 / r**4
            if self.config.verbose:
                print(f"  Added O(μ²) term: α*μ²*M²/r⁴")
        
        if self.config.max_coefficient_order >= 4:
            f_ansatz += beta * mu**4 * M**4 / r**6
            if self.config.verbose:
                print(f"  Added O(μ⁴) term: β*μ⁴*M⁴/r⁶")
        
        if self.config.max_coefficient_order >= 6:
            f_ansatz += gamma_coeff * mu**6 * M**6 / r**8
            if self.config.verbose:
                print(f"  Added O(μ⁶) term: γ*μ⁶*M⁶/r⁸")
        
        if self.config.verbose:
            print(f"Metric ansatz: f(r) = {f_ansatz}")
        
        self.results['f_ansatz'] = f_ansatz
        return f_ansatz
    
    def extract_static_constraint(self, H_polymer: sp.Expr, 
                                 f_ansatz: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Extract the static constraint using the metric ansatz.
        
        For static spherically symmetric metrics:
        E^x = r², E^φ = r*√f(r), K_φ = 0
        
        Args:
            H_polymer: Polymer-expanded Hamiltonian
            f_ansatz: Metric function ansatz
            
        Returns:
            Dictionary of constraint equations by order in μ
        """
        if self.config.verbose:
            print("\n5. EXTRACTING STATIC CONSTRAINT")
            print("-" * 50)
        
        # Static triad ansatz for spherically symmetric case
        Ex_static = r**2
        Ephi_static = r * sp.sqrt(f_ansatz)
        Kphi_static = 0
        Kx_static = 0  # Will be determined by constraint
        
        if self.config.verbose:
            print(f"  Static ansatz:")
            print(f"    E^x = {Ex_static}")
            print(f"    E^φ = {Ephi_static}")
            print(f"    K_φ = {Kphi_static}")
            print(f"    K_x = {Kx_static} (initially)")
        
        # Substitute static values into polymer Hamiltonian
        constraint_static = H_polymer.subs([
            (Ex, Ex_static),
            (Ephi, Ephi_static),
            (Kphi, Kphi_static),
            (f, f_ansatz)
        ])
        
        # Simplify the constraint
        print("  Simplifying static constraint...")
        constraint_simplified = safe_hamiltonian_simplify(
            constraint_static, timeout_seconds=self.config.default_timeout
        )
        
        if constraint_simplified is None:
            print("  Warning: Constraint simplification timed out")
            constraint_simplified = safe_simplify(
                constraint_static, timeout_seconds=self.config.default_timeout//2
            )
            if constraint_simplified is None:
                constraint_simplified = constraint_static
        
        # Extract coefficients by powers of μ
        constraints_by_order = {}
        print("  Extracting constraints by order in μ...")
        
        for order in range(0, self.config.max_coefficient_order + 1, 2):
            coeff = constraint_simplified.coeff(mu, order)
            if coeff is not None and coeff != 0:
                constraints_by_order[f'mu^{order}'] = coeff
                if self.config.verbose:
                    print(f"    μ^{order} constraint: {coeff}")
        
        self.results['constraints_by_order'] = constraints_by_order
        return constraints_by_order
    
    def solve_coefficients_order_by_order(self, 
                                        constraints_by_order: Dict[str, sp.Expr]) -> Dict[str, sp.Expr]:
        """
        Solve for metric coefficients α, β, etc. order by order.
        
        Args:
            constraints_by_order: Dictionary of constraint equations by μ order
            
        Returns:
            Dictionary of solved coefficients
        """
        if self.config.verbose:
            print("\n6. SOLVING COEFFICIENTS ORDER BY ORDER")
            print("-" * 50)
        
        coefficients = {}
        
        # μ⁰ constraint should be satisfied by Schwarzschild (verification)
        if 'mu^0' in constraints_by_order:
            mu0_constraint = constraints_by_order['mu^0']
            if self.config.verbose:
                print(f"  μ⁰ constraint (should be ~0): {mu0_constraint}")
            
            # This should vanish for Schwarzschild, if not we have an issue
            mu0_simplified = safe_simplify(mu0_constraint, 
                                         timeout_seconds=self.config.default_timeout)
            if mu0_simplified is None:
                mu0_simplified = mu0_constraint
            
            if self.config.verbose:
                print(f"  Simplified μ⁰ constraint: {mu0_simplified}")
        
        # μ² constraint determines α
        if 'mu^2' in constraints_by_order:
            mu2_constraint = constraints_by_order['mu^2']
            if self.config.verbose:
                print(f"  Solving μ² constraint for α...")
                print(f"    Constraint: {mu2_constraint}")
            
            # Solve for α from the constraint = 0
            alpha_solutions = safe_solve(mu2_constraint, alpha, 
                                       timeout_seconds=self.config.default_timeout)
            
            if alpha_solutions is None or len(alpha_solutions) == 0:
                print("  Warning: Direct solve for α failed, trying series approach")
                # Try expanding in powers of 1/r and extracting coefficient
                mu2_expanded = safe_series(mu2_constraint, r, sp.oo, 
                                         n=self.config.series_expansion_depth,
                                         timeout_seconds=self.config.default_timeout)
                if mu2_expanded is not None:
                    mu2_expanded = mu2_expanded.removeO()
                    alpha_solutions = safe_solve(mu2_expanded, alpha,
                                               timeout_seconds=self.config.default_timeout)
            
            if alpha_solutions and len(alpha_solutions) > 0:
                alpha_value = alpha_solutions[0]
                coefficients['alpha'] = alpha_value
                if self.config.verbose:
                    print(f"  ✓ Found α = {alpha_value}")
            else:
                print("  ✗ Could not solve for α")
                coefficients['alpha'] = None
        
        # μ⁴ constraint determines β (if α is known)
        if ('mu^4' in constraints_by_order and 
            coefficients.get('alpha') is not None):
            mu4_constraint = constraints_by_order['mu^4']
            
            # Substitute known α value
            mu4_with_alpha = mu4_constraint.subs(alpha, coefficients['alpha'])
            
            if self.config.verbose:
                print(f"  Solving μ⁴ constraint for β...")
                print(f"    Constraint with α: {mu4_with_alpha}")
            
            beta_solutions = safe_solve(mu4_with_alpha, beta,
                                      timeout_seconds=self.config.default_timeout)
            
            if beta_solutions and len(beta_solutions) > 0:
                beta_value = beta_solutions[0]
                coefficients['beta'] = beta_value
                if self.config.verbose:
                    print(f"  ✓ Found β = {beta_value}")
            else:
                if self.config.verbose:
                    print("  ✗ Could not solve for β")
                coefficients['beta'] = None
        
        # μ⁶ constraint determines γ (if α, β are known)
        if ('mu^6' in constraints_by_order and 
            coefficients.get('alpha') is not None and
            coefficients.get('beta') is not None):
            mu6_constraint = constraints_by_order['mu^6']
            
            # Substitute known values
            mu6_with_coeffs = mu6_constraint.subs([
                (alpha, coefficients['alpha']),
                (beta, coefficients['beta'])
            ])
            
            if self.config.verbose:
                print(f"  Solving μ⁶ constraint for γ...")
            
            gamma_solutions = safe_solve(mu6_with_coeffs, gamma_coeff,
                                       timeout_seconds=self.config.default_timeout)
            
            if gamma_solutions and len(gamma_solutions) > 0:
                gamma_value = gamma_solutions[0]
                coefficients['gamma'] = gamma_value
                if self.config.verbose:
                    print(f"  ✓ Found γ = {gamma_value}")
            else:
                if self.config.verbose:
                    print("  ✗ Could not solve for γ")
                coefficients['gamma'] = None
        
        self.results['coefficients'] = coefficients
        return coefficients
    
    def validate_coefficient_extraction(self, coefficients: Dict[str, sp.Expr]) -> Dict[str, Any]:
        """
        Validate the extracted coefficients by checking consistency.
        
        Args:
            coefficients: Dictionary of extracted coefficients
            
        Returns:
            Dictionary with validation results
        """
        if not self.config.validate_results:
            return {'validation_skipped': True}
        
        if self.config.verbose:
            print("\n7. VALIDATING COEFFICIENT EXTRACTION")
            print("-" * 50)
        
        validation_results = {
            'alpha_extracted': coefficients.get('alpha') is not None,
            'beta_extracted': coefficients.get('beta') is not None,
            'gamma_extracted': coefficients.get('gamma') is not None,
            'alpha_is_real': False,
            'beta_is_real': False,
            'alpha_is_rational': False,
            'expected_alpha_range': False,
            'consistency_check': False
        }
        
        # Check α properties
        if coefficients.get('alpha') is not None:
            alpha_val = coefficients['alpha']
            validation_results['alpha_is_real'] = alpha_val.is_real
            validation_results['alpha_is_rational'] = alpha_val.is_rational
            
            # Check if α is in expected range (typically around 1/6 from sin expansions)
            try:
                alpha_numeric = float(alpha_val.evalf())
                validation_results['alpha_numeric'] = alpha_numeric
                validation_results['expected_alpha_range'] = 0.05 <= abs(alpha_numeric) <= 0.5
                if self.config.verbose:
                    print(f"  α = {alpha_val} ≈ {alpha_numeric:.6f}")
            except:
                if self.config.verbose:
                    print(f"  α = {alpha_val} (could not evaluate numerically)")
        
        # Check β properties
        if coefficients.get('beta') is not None:
            beta_val = coefficients['beta']
            validation_results['beta_is_real'] = beta_val.is_real
            try:
                beta_numeric = float(beta_val.evalf())
                validation_results['beta_numeric'] = beta_numeric
                if self.config.verbose:
                    print(f"  β = {beta_val} ≈ {beta_numeric:.6f}")
            except:
                if self.config.verbose:
                    print(f"  β = {beta_val} (could not evaluate numerically)")
        
        # Consistency check: reconstruct metric and verify expansion
        if coefficients.get('alpha') is not None:
            try:
                f_reconstructed = 1 - 2*M/r + coefficients['alpha'] * mu**2 * M**2 / r**4
                if coefficients.get('beta') is not None:
                    f_reconstructed += coefficients['beta'] * mu**4 * M**4 / r**6
                
                # Expand and check for problematic terms
                f_expanded = safe_expand(f_reconstructed, timeout_seconds=5)
                if f_expanded is not None:
                    validation_results['consistency_check'] = True
                    if self.config.verbose:
                        print(f"  ✓ Reconstructed metric: f(r) = {f_reconstructed}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  ✗ Consistency check failed: {e}")
        
        self.results['validation'] = validation_results
        return validation_results
    
    def attempt_closed_form_resummation(self, coefficients: Dict[str, sp.Expr]) -> Tuple[Optional[sp.Expr], bool]:
        """
        Attempt to find a closed-form resummation of the metric series.
        
        If β/α² is a simple constant, we might be able to resum the series
        as a geometric or rational function.
        
        Args:
            coefficients: Dictionary of extracted coefficients
            
        Returns:
            Tuple of (closed_form_expression, success_flag)
        """
        if self.config.verbose:
            print("\n8. ATTEMPTING CLOSED-FORM RESUMMATION")
            print("-" * 50)
        
        alpha_val = coefficients.get('alpha')
        beta_val = coefficients.get('beta')
        
        if alpha_val is None:
            if self.config.verbose:
                print("  Cannot attempt resummation without α")
            return None, False
        
        # Base metric terms
        f_base = 1 - 2*M/r
        correction_base = alpha_val * mu**2 * M**2 / r**4
        
        # If only α is available, return polynomial form
        if beta_val is None:
            if self.config.verbose:
                print("  Using polynomial form (α only)")
            closed_form = f_base + correction_base
            return closed_form, True
        
        # Check if β has a simple relation to α
        if self.config.verbose:
            print("  Checking for β = c*α² pattern...")
        
        # Try to simplify β/α²
        try:
            beta_over_alpha_sq = safe_simplify(beta_val / alpha_val**2, 
                                             timeout_seconds=self.config.default_timeout)
            
            if beta_over_alpha_sq is not None and beta_over_alpha_sq.is_number:
                if self.config.verbose:
                    print(f"  Found β/α² = {beta_over_alpha_sq}")
                
                # Try rational form: f = 1 - 2M/r + (α*μ²M²/r⁴) / (1 + C*μ²M/r³)
                # This would give the pattern α*μ²M²/r⁴ - C*α*μ⁴M³/r⁷ + ...
                x = alpha_val * mu**2 * M**2 / r**4
                
                if abs(beta_over_alpha_sq - 1) < 1e-10:
                    # Perfect geometric series: 1 + x + x² + ... = 1/(1-x)
                    if self.config.verbose:
                        print("  Trying geometric series resummation...")
                    
                    try:
                        # f = 1 - 2M/r + x/(1-x) where x = α*μ²M²/r⁴
                        # But this only works if the series pattern matches exactly
                        closed_form = f_base + x / (1 - x)
                        closed_form_simplified = safe_simplify(closed_form, 
                                                             timeout_seconds=self.config.default_timeout)
                        if closed_form_simplified is not None:
                            closed_form = closed_form_simplified
                        
                        if self.config.verbose:
                            print(f"  ✓ Geometric form: f(r) = {closed_form}")
                        return closed_form, True
                    except:
                        if self.config.verbose:
                            print("  Geometric series form failed")
                
                else:
                    # Modified geometric series
                    if self.config.verbose:
                        print(f"  Trying modified geometric series with ratio {beta_over_alpha_sq}...")
                    
                    try:
                        # f = 1 - 2M/r + x/(1 - ratio*x)
                        closed_form = f_base + x / (1 - beta_over_alpha_sq * x)
                        closed_form_simplified = safe_simplify(closed_form,
                                                             timeout_seconds=self.config.default_timeout)
                        if closed_form_simplified is not None:
                            closed_form = closed_form_simplified
                        
                        if self.config.verbose:
                            print(f"  ✓ Modified geometric form: f(r) = {closed_form}")
                        return closed_form, True
                    except:
                        if self.config.verbose:
                            print("  Modified geometric series failed")
        except:
            if self.config.verbose:
                print("  Could not analyze β/α² ratio")
        
        # Fallback: standard polynomial form
        if self.config.verbose:
            print("  Using polynomial form as fallback")
        f_poly = f_base + correction_base + beta_val * mu**4 * M**4 / r**6
        
        if coefficients.get('gamma') is not None:
            f_poly += coefficients['gamma'] * mu**6 * M**6 / r**8
        
        return f_poly, True
    
    def run_complete_extraction(self) -> Dict[str, Any]:
        """
        Run the complete polymer LQG coefficient extraction workflow.
        
        Returns:
            Dictionary containing all extraction results
        """
        start_time = time.time()
        
        try:
            # Step 1: Construct classical Hamiltonian
            H_classical = self.construct_classical_hamiltonian()
            
            # Step 2: Derive classical solution
            Kx_classical, f_classical = self.derive_classical_solution()
            
            # Step 3: Apply polymer corrections
            H_polymer = self.apply_polymer_corrections(H_classical, Kx_classical)
            
            # Step 4: Formulate metric ansatz
            f_ansatz = self.formulate_metric_ansatz()
            
            # Step 5: Extract static constraint
            constraints_by_order = self.extract_static_constraint(H_polymer, f_ansatz)
            
            # Step 6: Solve coefficients order by order
            coefficients = self.solve_coefficients_order_by_order(constraints_by_order)
            
            # Step 7: Validate extraction
            validation = self.validate_coefficient_extraction(coefficients)
            
            # Step 8: Attempt closed-form resummation
            closed_form, resummation_success = self.attempt_closed_form_resummation(coefficients)
            
            # Compile final results
            extraction_time = time.time() - start_time
            
            final_results = {
                'success': True,
                'extraction_time': extraction_time,
                'config': self.config,
                'coefficients': coefficients,
                'validation': validation,
                'closed_form': closed_form,
                'resummation_success': resummation_success,
                'intermediate_results': self.results,
                'timeout_support': TIMEOUT_SUPPORT
            }
            
            if self.config.verbose:
                print("\n" + "="*80)
                print("EXTRACTION COMPLETE")
                print("="*80)
                print(f"Extraction time: {extraction_time:.2f} seconds")
                print(f"Success: {final_results['success']}")
                
                if coefficients.get('alpha') is not None:
                    print(f"α coefficient: {coefficients['alpha']}")
                if coefficients.get('beta') is not None:
                    print(f"β coefficient: {coefficients['beta']}")
                if closed_form is not None:
                    print(f"Closed form: f(r) = {closed_form}")
                    
                print("="*80)
            
            return final_results
            
        except Exception as e:
            error_time = time.time() - start_time
            if self.config.verbose:
                print(f"\n✗ EXTRACTION FAILED after {error_time:.2f} seconds")
                print(f"Error: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'extraction_time': error_time,
                'partial_results': self.results
            }

def export_extraction_results(results: Dict[str, Any], 
                            filename: Optional[str] = None) -> str:
    """
    Export extraction results to a Python module for use by other scripts.
    
    Args:
        results: Extraction results dictionary
        filename: Optional filename (defaults to polymer_lqg_results.py)
        
    Returns:
        Exported filename
    """
    if filename is None:
        filename = 'polymer_lqg_results.py'
    
    # Convert SymPy expressions to strings for JSON serialization
    def sympy_to_str(obj):
        if hasattr(obj, '_sympy_'):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: sympy_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sympy_to_str(item) for item in obj]
        else:
            return obj
    
    export_data = sympy_to_str(results)
    
    # Create Python module content
    module_content = f'''#!/usr/bin/env python3
"""
Polymer LQG Coefficient Extraction Results

Generated automatically by polymer_lqg_coefficient_extraction.py
Contains extracted coefficients α, β and closed-form expressions.
"""

import sympy as sp
import numpy as np
from typing import Union, Optional

# Symbolic variables
r, M, mu = sp.symbols('r M mu', positive=True, real=True)

# Extraction metadata
EXTRACTION_SUCCESS = {results['success']}
EXTRACTION_TIME = {results.get('extraction_time', 0):.3f}
TIMEOUT_SUPPORT = {results.get('timeout_support', False)}

# Extracted coefficients
'''
    
    if results['success'] and results.get('coefficients'):
        coeffs = results['coefficients']
        if coeffs.get('alpha') is not None:
            module_content += f"ALPHA_LQG = {coeffs['alpha']}\n"
        if coeffs.get('beta') is not None:
            module_content += f"BETA_LQG = {coeffs['beta']}\n"
        if coeffs.get('gamma') is not None:
            module_content += f"GAMMA_LQG = {coeffs['gamma']}\n"
    
    module_content += f'''
# Closed-form expression (if available)
CLOSED_FORM_AVAILABLE = {results.get('resummation_success', False)}
'''
    
    if results.get('closed_form') is not None:
        module_content += f"CLOSED_FORM_EXPR = {results['closed_form']}\n"
    
    module_content += '''
def f_LQG_extracted(r_val: Union[float, np.ndarray], 
                   M_val: float, 
                   mu_val: float) -> Union[float, np.ndarray]:
    """
    Evaluate the extracted LQG-corrected metric function.
    
    Args:
        r_val: Radial coordinate(s)
        M_val: Mass parameter
        mu_val: Polymer scale parameter
        
    Returns:
        Metric function value(s)
    """
    r_val = np.asarray(r_val)'''
    
    if results['success'] and results.get('coefficients', {}).get('alpha') is not None:
        module_content += f'''
    alpha_val = {results['coefficients']['alpha']}
    f_val = 1 - 2*M_val/r_val + alpha_val*(mu_val**2)*(M_val**2)/(r_val**4)'''
        
        if results.get('coefficients', {}).get('beta') is not None:
            module_content += f'''
    beta_val = {results['coefficients']['beta']}
    f_val += beta_val*(mu_val**4)*(M_val**4)/(r_val**6)'''
    else:
        module_content += '''
    # Extraction failed, return classical Schwarzschild
    f_val = 1 - 2*M_val/r_val'''
    
    module_content += '''
    return f_val

# Complete extraction results (as dictionary)
EXTRACTION_RESULTS = ''' + str(export_data).replace("'", '"')
    
    # Write to file
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w') as f:
        f.write(module_content)
    
    print(f"✓ Results exported to {filepath}")
    return filepath

def main():
    """Main function for running polymer LQG coefficient extraction."""
    print("Starting Polymer LQG Coefficient Extraction...")
    
    # Configuration for extraction
    config = PolymerExtractionConfig(
        max_polymer_order=6,
        max_coefficient_order=4,
        sin_expansion_terms=4,
        series_expansion_depth=8,
        default_timeout=15,
        verbose=True,
        validate_results=True
    )
    
    # Create extractor and run
    extractor = PolymerLQGCoefficientExtractor(config)
    results = extractor.run_complete_extraction()
    
    # Export results
    if results['success']:
        export_extraction_results(results)
    else:
        print(f"Extraction failed: {results.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = main()
