#!/usr/bin/env python3
"""
Enhanced Alpha Extraction Script for LQG Polymer Corrections

This script implements a systematic approach to extract alpha (α) and beta (β) 
coefficients from LQG polymer corrections by:

1. Solving the classical Hamiltonian constraint H_classical = 0 for K_x(r)
2. Applying polymer corrections K → sin(μK)/μ to the classical solution
3. Expanding to higher orders (up to μ^6) 
4. Extracting α and β coefficients from constraint equations by order
5. Implementing robust symbolic operations with timeout handling
6. Providing comprehensive analysis and validation

Mathematical Framework:
- Metric ansatz: f(r) = 1 - 2M/r + α*μ²*M²/r⁴ + β*μ⁴*M⁴/r⁶ + ...
- Triad ansatz: E^x = r², E^φ = r√f(r), K_φ = 0
- Classical constraint: H_classical = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + R^(2) = 0
- Polymer correction: K → sin(μK)/μ

Author: Enhanced LQG Framework
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
import time
import warnings

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import symbolic timeout utilities
try:
    from scripts.symbolic_timeout_utils import (
        safe_symbolic_operation, safe_series, safe_solve, safe_simplify, 
        safe_expand, safe_integrate, safe_diff, safe_collect, safe_factor,
        safe_cancel, safe_together, safe_apart, safe_trigsimp, safe_ratsimp,
        SymbolicTimeoutError, set_default_timeout
    )
    print("✓ Imported timeout utilities successfully")
except ImportError:
    print("⚠ Timeout utilities not found, using basic implementations")
    # Fallback implementations
    def safe_symbolic_operation(op, *args, **kwargs):
        return op(*args, **kwargs)
    def safe_series(expr, var, point, n): return expr.series(var, point, n)
    def safe_solve(eq, var): return sp.solve(eq, var)
    def safe_simplify(expr): return sp.simplify(expr)
    def safe_expand(expr): return sp.expand(expr)
    def safe_diff(expr, var): return sp.diff(expr, var)
    def safe_collect(expr, var): return sp.collect(expr, var)
    def safe_factor(expr): return sp.factor(expr)

# Set default timeout for symbolic operations
try:
    set_default_timeout(10)  # 10 seconds timeout
except:
    pass

# Global symbolic variables
print("Setting up symbolic variables...")
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
Ex, Ephi = sp.symbols('Ex Ephi', positive=True, real=True)
Kx, Kphi = sp.symbols('Kx Kphi', real=True)

# Metric function and correction coefficients
f = sp.Function('f')(r)
alpha, beta, gamma, delta = sp.symbols('alpha beta gamma delta', real=True)

# Higher-order μ expansion terms
mu2, mu4, mu6 = mu**2, mu**4, mu**6

print("✓ Symbolic setup complete")

class EnhancedAlphaExtractor:
    """
    Enhanced class for systematic extraction of LQG polymer correction coefficients.
    """
    
    def __init__(self):
        """Initialize the alpha extractor with symbolic variables."""
        self.results = {}
        self.classical_solution = None
        self.polymer_expansion = None
        self.constraint_orders = {}
        
        print("Enhanced Alpha Extractor initialized")
    
    def construct_classical_hamiltonian(self) -> sp.Expr:
        """
        Construct the classical Hamiltonian constraint for spherically symmetric LQG.
        
        For spherical symmetry in radial-triad variables:
        H_classical = -(E^φ/√E^x) K_φ² - 2 K_φ K_x √E^x + R^(2)
        
        Returns:
            Classical Hamiltonian constraint expression
        """
        print("\n" + "="*60)
        print("STEP 1: CONSTRUCTING CLASSICAL HAMILTONIAN CONSTRAINT")
        print("="*60)
        
        # Classical kinetic terms
        kinetic_term = -(Ephi / sp.sqrt(Ex)) * Kphi**2 - 2 * Kphi * Kx * sp.sqrt(Ex)
        
        # Spatial curvature term R^(2) for spherical symmetry
        # In radial-triad variables: R^(2) = -2/(r²√f) ∂_r(r²∂_r√f)
        sqrt_f = sp.sqrt(f)
        spatial_curvature = -2 / (r**2 * sqrt_f) * safe_diff(r**2 * safe_diff(sqrt_f, r), r)
        
        # Total classical Hamiltonian
        H_classical = kinetic_term + spatial_curvature
        
        print(f"Kinetic terms: {kinetic_term}")
        print(f"Spatial curvature: {spatial_curvature}")
        print(f"Classical Hamiltonian: H_classical = {H_classical}")
        
        return H_classical
    
    def derive_static_ansatz(self) -> Dict[str, sp.Expr]:
        """
        Derive the static spherically symmetric ansatz.
        
        For static metric ds² = -f(r)dt² + dr²/f(r) + r²dΩ²:
        - E^x = r²
        - E^φ = r√f(r)  
        - K_φ = 0 (static condition)
        
        Returns:
            Dictionary of ansatz expressions
        """
        print("\n" + "="*60)
        print("STEP 2: DERIVING STATIC ANSATZ")
        print("="*60)
        
        ansatz = {
            'Ex': r**2,
            'Ephi': r * sp.sqrt(f),
            'Kphi': 0
        }
        
        print("Static spherically symmetric ansatz:")
        for key, value in ansatz.items():
            print(f"  {key} = {value}")
        
        return ansatz
    
    def solve_classical_constraint_for_kx(self, H_classical: sp.Expr, ansatz: Dict[str, sp.Expr]) -> sp.Expr:
        """
        Solve the classical constraint H_classical = 0 for K_x(r).
        
        Args:
            H_classical: Classical Hamiltonian constraint
            ansatz: Static ansatz dictionary
            
        Returns:
            Classical K_x as function of metric f(r) and its derivatives
        """
        print("\n" + "="*60)
        print("STEP 3: SOLVING CLASSICAL CONSTRAINT FOR K_x")
        print("="*60)
        
        # Substitute static ansatz
        H_substituted = H_classical.subs([
            (Ex, ansatz['Ex']),
            (Ephi, ansatz['Ephi']),
            (Kphi, ansatz['Kphi'])
        ])
        
        print(f"Constraint after ansatz substitution:")
        print(f"H = {H_substituted}")
        
        # Since K_φ = 0, the kinetic term vanishes, leaving only spatial curvature
        # For Schwarzschild solution, we need K_x to satisfy the constraint
        
        # For spherical symmetry with static metric, K_x is related to ∂_r(ln√f)
        # From Einstein equations: K_x = ∂_r(ln√f) = (1/2f) ∂_r f
        classical_Kx = safe_diff(sp.log(sp.sqrt(f)), r)
        
        print(f"Classical solution: K_x = ∂_r(ln√f) = {classical_Kx}")
        
        # Simplify using f' notation
        fprime = sp.symbols("f'")
        classical_Kx_simplified = fprime / (2 * f)
        
        print(f"Simplified: K_x = f'/(2f)")
        
        self.classical_solution = classical_Kx
        return classical_Kx
    
    def construct_metric_ansatz(self) -> sp.Expr:
        """
        Construct the metric ansatz with polymer correction coefficients.
        
        f(r) = 1 - 2M/r + α*μ²*M²/r⁴ + β*μ⁴*M⁴/r⁶ + γ*μ⁶*M⁶/r⁸ + ...
        
        Returns:
            Metric function with correction terms
        """
        print("\n" + "="*60)
        print("STEP 4: CONSTRUCTING METRIC ANSATZ WITH CORRECTIONS")
        print("="*60)
        
        # Base Schwarzschild solution
        f_schwarzschild = 1 - 2*M/r
        
        # Polymer correction terms
        alpha_term = alpha * mu2 * M**2 / r**4
        beta_term = beta * mu4 * M**4 / r**6
        gamma_term = gamma * mu6 * M**6 / r**8
        
        # Complete metric ansatz
        f_ansatz = f_schwarzschild + alpha_term + beta_term + gamma_term
        
        print("Metric ansatz breakdown:")
        print(f"  Schwarzschild: {f_schwarzschild}")
        print(f"  α correction: {alpha_term}")
        print(f"  β correction: {beta_term}")
        print(f"  γ correction: {gamma_term}")
        print(f"  Complete: f(r) = {f_ansatz}")
        
        return f_ansatz
    
    def apply_polymer_corrections(self, classical_Kx: sp.Expr, f_ansatz: sp.Expr) -> sp.Expr:
        """
        Apply polymer corrections K → sin(μK)/μ to the classical solution.
        
        Args:
            classical_Kx: Classical K_x solution
            f_ansatz: Metric ansatz with corrections
            
        Returns:
            Polymerized K_x expression
        """
        print("\n" + "="*60)
        print("STEP 5: APPLYING POLYMER CORRECTIONS")
        print("="*60)
        
        # Substitute metric ansatz into classical K_x
        classical_Kx_with_ansatz = classical_Kx.subs(f, f_ansatz)
        
        print(f"Classical K_x with metric ansatz:")
        print(f"K_x_classical = {classical_Kx_with_ansatz}")
        
        # Apply polymer correction: K → sin(μK)/μ
        print("\nApplying polymer correction: K → sin(μK)/μ")
        
        # For small μK, expand sin(μK)/μ = 1 - μ²K²/6 + μ⁴K⁴/120 - μ⁶K⁶/5040 + ...
        mu_Kx = mu * classical_Kx_with_ansatz
        
        # Series expansion of sin(μK)/μ
        polymer_factor = safe_series(sp.sin(mu_Kx)/mu, mu, 0, 7).removeO()
        
        print(f"Polymer factor sin(μK_x)/μ ≈ {polymer_factor}")
        
        # Polymerized K_x
        Kx_polymer = polymer_factor
        
        print(f"Polymerized K_x = {Kx_polymer}")
        
        self.polymer_expansion = Kx_polymer
        return Kx_polymer
    
    def expand_constraint_by_orders(self, Kx_polymer: sp.Expr, f_ansatz: sp.Expr) -> Dict[int, sp.Expr]:
        """
        Expand the constraint equation by orders of μ to extract coefficients.
        
        Args:
            Kx_polymer: Polymerized K_x expression
            f_ansatz: Metric ansatz with corrections
            
        Returns:
            Dictionary of constraint equations by μ order
        """
        print("\n" + "="*60)
        print("STEP 6: EXPANDING CONSTRAINT BY μ ORDERS")
        print("="*60)
        
        # Construct full constraint with polymerized K_x
        ansatz = self.derive_static_ansatz()
        
        # Substitute everything into the constraint
        # For static case, constraint becomes spatial curvature term only
        sqrt_f_ansatz = sp.sqrt(f_ansatz)
        spatial_curvature = -2 / (r**2 * sqrt_f_ansatz) * safe_diff(r**2 * safe_diff(sqrt_f_ansatz, r), r)
        
        print(f"Spatial curvature constraint: {spatial_curvature}")
        
        # Expand in powers of μ
        print("\nExpanding constraint in powers of μ...")
        
        constraint_expanded = safe_series(spatial_curvature, mu, 0, 7).removeO()
        constraint_expanded = safe_expand(constraint_expanded)
        
        print(f"Expanded constraint: {constraint_expanded}")
        
        # Collect coefficients by powers of μ
        orders = {}
        for order in [0, 2, 4, 6]:
            coeff = constraint_expanded.coeff(mu, order)
            if coeff is not None:
                orders[order] = safe_simplify(coeff)
                print(f"μ^{order} coefficient: {orders[order]}")
        
        self.constraint_orders = orders
        return orders
    
    def extract_alpha_beta_coefficients(self, constraint_orders: Dict[int, sp.Expr]) -> Dict[str, sp.Expr]:
        """
        Extract α and β coefficients by solving constraint equations order by order.
        
        Args:
            constraint_orders: Constraint equations by μ order
            
        Returns:
            Dictionary of extracted coefficients
        """
        print("\n" + "="*60)
        print("STEP 7: EXTRACTING α AND β COEFFICIENTS")
        print("="*60)
        
        coefficients = {}
        
        # Order μ^0: Classical constraint (should be satisfied)
        if 0 in constraint_orders:
            print(f"μ^0 constraint: {constraint_orders[0]} = 0")
            print("This should be automatically satisfied by Schwarzschild solution")
        
        # Order μ^2: Extract α
        if 2 in constraint_orders:
            print(f"\nμ^2 constraint: {constraint_orders[2]} = 0")
            print("Solving for α...")
            
            alpha_solutions = safe_solve(constraint_orders[2], alpha)
            if alpha_solutions:
                alpha_value = alpha_solutions[0]
                coefficients['alpha'] = safe_simplify(alpha_value)
                print(f"α = {coefficients['alpha']}")
            else:
                print("Could not solve for α analytically")
                coefficients['alpha'] = None
        
        # Order μ^4: Extract β (may depend on α)
        if 4 in constraint_orders:
            print(f"\nμ^4 constraint: {constraint_orders[4]} = 0")
            print("Solving for β...")
            
            # Substitute known α value if available
            eq_for_beta = constraint_orders[4]
            if 'alpha' in coefficients and coefficients['alpha'] is not None:
                eq_for_beta = eq_for_beta.subs(alpha, coefficients['alpha'])
            
            beta_solutions = safe_solve(eq_for_beta, beta)
            if beta_solutions:
                beta_value = beta_solutions[0]
                coefficients['beta'] = safe_simplify(beta_value)
                print(f"β = {coefficients['beta']}")
            else:
                print("Could not solve for β analytically")
                coefficients['beta'] = None
        
        # Order μ^6: Extract γ (may depend on α, β)
        if 6 in constraint_orders:
            print(f"\nμ^6 constraint: {constraint_orders[6]} = 0")
            print("Solving for γ...")
            
            # Substitute known coefficient values
            eq_for_gamma = constraint_orders[6]
            for coeff_name, coeff_value in coefficients.items():
                if coeff_value is not None:
                    eq_for_gamma = eq_for_gamma.subs(eval(coeff_name), coeff_value)
            
            gamma_solutions = safe_solve(eq_for_gamma, gamma)
            if gamma_solutions:
                gamma_value = gamma_solutions[0]
                coefficients['gamma'] = safe_simplify(gamma_value)
                print(f"γ = {coefficients['gamma']}")
            else:
                print("Could not solve for γ analytically")
                coefficients['gamma'] = None
        
        self.results = coefficients
        return coefficients
    
    def validate_and_analyze_results(self, coefficients: Dict[str, sp.Expr]) -> None:
        """
        Validate and analyze the extracted coefficients.
        
        Args:
            coefficients: Dictionary of extracted coefficients
        """
        print("\n" + "="*60)
        print("STEP 8: VALIDATION AND ANALYSIS")
        print("="*60)
        
        print("Extracted LQG polymer correction coefficients:")
        print("-" * 50)
        
        for name, value in coefficients.items():
            if value is not None:
                print(f"{name} = {value}")
                
                # Analyze the structure
                if value.has(M):
                    print(f"  → Contains mass M: Yes")
                if value.has(r):
                    print(f"  → Contains radius r: Yes")
                if value.has(mu):
                    print(f"  → Contains μ dependence: Yes")
            else:
                print(f"{name} = [Could not determine analytically]")
            print()
        
        # Check dimensional analysis
        print("Dimensional analysis:")
        print("- α should be dimensionless")
        print("- β should be dimensionless")
        print("- γ should be dimensionless")
        print()
        
        # Physical interpretation
        print("Physical interpretation:")
        print("- α: Leading quantum correction to Schwarzschild metric")
        print("- β: Next-order quantum correction")
        print("- γ: Higher-order quantum correction")
        print()
        
        # Verify classical limit
        print("Classical limit check (μ → 0):")
        print("Metric should reduce to f(r) = 1 - 2M/r")
        print("✓ This is ensured by construction")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete alpha extraction analysis pipeline.
        
        Returns:
            Dictionary containing all results
        """
        print("🚀 STARTING ENHANCED ALPHA EXTRACTION ANALYSIS")
        print("=" * 80)
        
        try:
            # Step 1: Classical Hamiltonian
            H_classical = self.construct_classical_hamiltonian()
            
            # Step 2: Static ansatz
            ansatz = self.derive_static_ansatz()
            
            # Step 3: Solve for classical K_x
            classical_Kx = self.solve_classical_constraint_for_kx(H_classical, ansatz)
            
            # Step 4: Metric ansatz with corrections
            f_ansatz = self.construct_metric_ansatz()
            
            # Step 5: Apply polymer corrections
            Kx_polymer = self.apply_polymer_corrections(classical_Kx, f_ansatz)
            
            # Step 6: Expand by μ orders
            constraint_orders = self.expand_constraint_by_orders(Kx_polymer, f_ansatz)
            
            # Step 7: Extract coefficients
            coefficients = self.extract_alpha_beta_coefficients(constraint_orders)
            
            # Step 8: Validation and analysis
            self.validate_and_analyze_results(coefficients)
            
            # Compile final results
            final_results = {
                'coefficients': coefficients,
                'classical_solution': self.classical_solution,
                'polymer_expansion': self.polymer_expansion,
                'constraint_orders': self.constraint_orders,
                'metric_ansatz': f_ansatz,
                'success': True
            }
            
            print("\n🎉 ALPHA EXTRACTION ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            return final_results
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")
            print("Returning partial results...")
            
            return {
                'coefficients': getattr(self, 'results', {}),
                'classical_solution': getattr(self, 'classical_solution', None),
                'polymer_expansion': getattr(self, 'polymer_expansion', None),
                'constraint_orders': getattr(self, 'constraint_orders', {}),
                'error': str(e),
                'success': False
            }


def main():
    """Main function to run the enhanced alpha extraction analysis."""
    print("Enhanced Alpha Extraction for LQG Polymer Corrections")
    print("=" * 60)
    print("This script systematically extracts α and β coefficients")
    print("from LQG polymer corrections using constraint equations.")
    print()
    
    # Create extractor instance
    extractor = EnhancedAlphaExtractor()
    
    # Run complete analysis
    results = extractor.run_complete_analysis()
    
    # Save results summary
    if results['success']:
        print("\n📊 FINAL RESULTS SUMMARY:")
        print("=" * 40)
        
        for name, value in results['coefficients'].items():
            if value is not None:
                print(f"{name} = {value}")
        
        print("\n✅ Analysis completed successfully!")
        print("Results saved in extractor.results")
    else:
        print(f"\n⚠ Analysis encountered issues: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    results = main()
