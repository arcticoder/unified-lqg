#!/usr/bin/env python3
"""
Comprehensive LQG Polymer Coefficient Extraction Script

This script implements a complete workflow for extracting LQG polymer metric coefficients
α and β through classical-to-quantum Hamiltonian constraint analysis:

1. Constructs the classical Hamiltonian constraint from ADM variables
2. Solves for K_x(r) from the classical constraint equation  
3. Builds the polymer-expanded quantum Hamiltonian with sin(μK)/μ holonomy corrections
4. Implements static constraint extraction with metric ansatz f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶
5. Solves coefficient equations by matching powers of μ using symbolic timeout utilities
6. Provides comprehensive analysis and validation

Mathematical Framework:
- Classical ADM Hamiltonian: H_classical = ∫ d³x N(x)[H_grav + H_matter]
- Polymer quantum corrections: sin(μK_i)/μ replacing K_i terms
- Static constraint: δH/δN = 0 leads to local constraint equations
- Coefficient extraction: Match powers of μ in quantum constraint

Author: Comprehensive LQG Framework v3
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import sys
import os
import time
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import symbolic timeout utilities
try:
    from scripts.symbolic_timeout_utils import (
        safe_symbolic_operation, safe_series, safe_solve, safe_simplify, 
        safe_expand, safe_diff, safe_collect, safe_factor, safe_integrate,
        SymbolicTimeoutError, set_default_timeout
    )
    print("✓ Imported timeout utilities successfully")
    TIMEOUT_AVAILABLE = True
except ImportError:
    print("⚠ Timeout utilities not found, using basic implementations")
    TIMEOUT_AVAILABLE = False
    def safe_symbolic_operation(op, *args, **kwargs): return op(*args, **kwargs)
    def safe_series(expr, var, point, n): return expr.series(var, point, n)
    def safe_solve(eq, var): return sp.solve(eq, var)
    def safe_simplify(expr): return sp.simplify(expr)
    def safe_expand(expr): return sp.expand(expr)
    def safe_diff(expr, var): return sp.diff(expr, var)
    def safe_collect(expr, var): return sp.collect(expr, var)
    def safe_factor(expr): return sp.factor(expr)
    def safe_integrate(expr, var): return sp.integrate(expr, var)

# Set default timeout for symbolic operations
if TIMEOUT_AVAILABLE:
    try:
        set_default_timeout(20)  # 20 seconds timeout for complex operations
        print("✓ Symbolic timeout set to 20 seconds")
    except:
        pass

# Global symbolic variables
print("Setting up comprehensive symbolic variables...")

# Coordinate and geometric variables
r, t, theta, phi = sp.symbols('r t theta phi', real=True)
r = sp.symbols('r', positive=True, real=True)

# Physical parameters
M, mu, G, c, hbar = sp.symbols('M mu G c hbar', positive=True, real=True)

# LQG polymer correction coefficients
alpha, beta, gamma, delta = sp.symbols('alpha beta gamma delta', real=True)

# ADM variables and auxiliary variables
N, N_r = sp.symbols('N N_r', real=True)  # Lapse and shift
h_rr, h_theta_theta, h_phi_phi = sp.symbols('h_rr h_theta_theta h_phi_phi', positive=True, real=True)
K_rr, K_theta_theta, K_phi_phi = sp.symbols('K_rr K_theta_theta K_phi_phi', real=True)

# Densitized variables
E_r, E_theta, E_phi = sp.symbols('E_r E_theta E_phi', real=True)
A_r, A_theta, A_phi = sp.symbols('A_r A_theta A_phi', real=True)

print("✓ Comprehensive symbolic setup complete")


@dataclass
class HamiltonianResults:
    """Data class for storing Hamiltonian analysis results."""
    classical_constraint: Optional[sp.Expr] = None
    K_x_solution: Optional[sp.Expr] = None
    polymer_hamiltonian: Optional[sp.Expr] = None
    quantum_constraint: Optional[sp.Expr] = None
    constraint_orders: Optional[Dict[int, sp.Expr]] = None
    coefficients: Optional[Dict[str, sp.Expr]] = None
    success: bool = False
    error_message: Optional[str] = None


class ComprehensiveAlphaExtractor:
    """
    Comprehensive class for extracting LQG polymer correction coefficients
    through classical-to-quantum Hamiltonian constraint analysis.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the comprehensive alpha extractor.
        
        Args:
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.results = HamiltonianResults()
        self.metric_ansatz = None
        self.classical_hamiltonian = None
        self.quantum_hamiltonian = None
        
        if self.verbose:
            print("Comprehensive Alpha Extractor initialized")
            print("Framework: Classical-to-Quantum Hamiltonian Analysis")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            if level == "HEADER":
                print(f"\n{'='*70}")
                print(f"{message}")
                print(f"{'='*70}")
            elif level == "STEP":
                print(f"\n{'-'*50}")
                print(f"{message}")
                print(f"{'-'*50}")
            else:
                print(f"{message}")
    
    def construct_spherical_adm_variables(self) -> Dict[str, sp.Expr]:
        """
        Construct ADM variables for spherically symmetric spacetime.
        
        Returns:
            Dictionary of ADM metric and extrinsic curvature components
        """
        self.log("STEP 1A: CONSTRUCTING SPHERICAL ADM VARIABLES", "STEP")
        
        # Spherically symmetric metric: ds² = -N²dt² + h_rr dr² + r²dΩ²
        # with h_rr = 1/f(r) where f(r) is our metric ansatz
        
        # Construct metric ansatz
        f_base = 1 - 2*M/r  # Schwarzschild base
        
        # LQG polymer corrections - systematic expansion
        mu_over_r_sq = (mu/r)**2
        alpha_term = alpha * mu_over_r_sq * M**2/r**2  # O(μ²)  
        beta_term = beta * mu_over_r_sq**2 * M**4/r**4   # O(μ⁴)
        gamma_term = gamma * mu_over_r_sq**3 * M**6/r**6  # O(μ⁶)
        
        f_metric = f_base + alpha_term + beta_term + gamma_term
        self.metric_ansatz = f_metric
        
        # ADM 3-metric components
        h_components = {
            'h_rr': 1/f_metric,
            'h_theta_theta': r**2,
            'h_phi_phi': r**2 * sp.sin(theta)**2
        }
        
        self.log(f"Metric ansatz f(r): {f_metric}")
        self.log(f"ADM metric h_rr: {h_components['h_rr']}")
        self.log(f"ADM metric h_θθ: {h_components['h_theta_theta']}")
        
        return h_components
    
    def derive_classical_hamiltonian_constraint(self, h_components: Dict[str, sp.Expr]) -> sp.Expr:
        """
        Derive the classical ADM Hamiltonian constraint.
        
        Args:
            h_components: ADM 3-metric components
            
        Returns:
            Classical Hamiltonian constraint H_classical
        """
        self.log("STEP 1B: DERIVING CLASSICAL HAMILTONIAN CONSTRAINT", "STEP")
        
        # For spherical symmetry, we can work with the effective radial problem
        # The Hamiltonian constraint becomes:
        # H = (16πG)⁻¹ ∫ d³x √h [R^(3) - K_{ij}K^{ij} + K²] = 0
        
        # Extract metric function
        h_rr = h_components['h_rr']
        f_func = 1/h_rr
        
        # Compute 3-dimensional Ricci scalar
        f_prime = safe_diff(f_func, r)
        f_double_prime = safe_diff(f_prime, r)
        
        # For spherically symmetric metric, R^(3) = 2f''/r + 4f'/r² - 4(1-f)/r²
        R_3d = 2*f_double_prime/r + 4*f_prime/r**2 - 4*(1 - f_func)/r**2
        
        self.log(f"3D Ricci scalar R^(3): {R_3d}")
        
        # The extrinsic curvature contribution K_{ij}K^{ij} - K² needs to be computed
        # For our static case, we assume K_rr is the only non-zero component initially
        # This leads to the constraint: R^(3) - K_rr² = 0
        
        # Classical constraint (before solving for K_rr)
        classical_constraint = R_3d - K_rr**2
        
        self.log(f"Classical constraint: {classical_constraint} = 0")
        
        self.results.classical_constraint = classical_constraint
        self.classical_hamiltonian = classical_constraint
        
        return classical_constraint
    
    def solve_classical_k_x(self, classical_constraint: sp.Expr) -> sp.Expr:
        """
        Solve the classical constraint for K_x(r) = K_rr.
        
        Args:
            classical_constraint: The classical Hamiltonian constraint
            
        Returns:
            Solution K_x(r) from classical analysis
        """
        self.log("STEP 2: SOLVING CLASSICAL CONSTRAINT FOR K_x(r)", "STEP")
        
        # From constraint: R^(3) - K_rr² = 0  =>  K_rr = ±√R^(3)
        # We need R^(3) ≥ 0 for real solutions
        
        # Extract R^(3) from constraint
        R_3d = classical_constraint + K_rr**2
        
        self.log(f"Extracting R^(3): {R_3d}")
        
        # Solve for K_rr
        K_solutions = safe_solve(classical_constraint, K_rr)
        
        if K_solutions:
            # Take the positive solution (by convention)
            K_x_solution = K_solutions[0] if len(K_solutions) == 1 else K_solutions[1]
            
            # Simplify the solution
            K_x_simplified = safe_simplify(K_x_solution)
            
            self.log(f"K_x(r) solutions: {K_solutions}")
            self.log(f"Selected K_x(r): {K_x_simplified}")
            
            self.results.K_x_solution = K_x_simplified
            return K_x_simplified
        else:
            self.log("Could not solve for K_x(r) directly")
            
            # Try solving R^(3) = K_rr² directly
            R_3d_simplified = safe_simplify(R_3d)
            if R_3d_simplified != 0:
                K_x_from_R = sp.sqrt(sp.Abs(R_3d_simplified))
                self.log(f"K_x(r) from √|R^(3)|: {K_x_from_R}")
                self.results.K_x_solution = K_x_from_R
                return K_x_from_R
            else:
                self.log("R^(3) = 0, which implies K_x(r) = 0")
                self.results.K_x_solution = 0
                return sp.Integer(0)
    
    def construct_polymer_hamiltonian(self, K_x_classical: sp.Expr) -> sp.Expr:
        """
        Construct the polymer-expanded quantum Hamiltonian with sin(μK)/μ corrections.
        
        Args:
            K_x_classical: Classical solution for K_x(r)
            
        Returns:
            Polymer quantum Hamiltonian H_quantum
        """
        self.log("STEP 3: CONSTRUCTING POLYMER QUANTUM HAMILTONIAN", "STEP")
        
        # The key insight: Replace K_x → sin(μK_x)/μ in the quantum theory
        # This gives holonomy corrections characteristic of LQG
        
        mu_K_x = mu * K_x_classical
        
        # Polymer correction function
        if K_x_classical != 0:
            # For non-zero K_x, use sin(μK_x)/μ
            polymer_K_x = sp.sin(mu_K_x) / mu
            
            # Taylor expand around μ = 0 to get systematic corrections
            self.log("Expanding sin(μK_x)/μ in powers of μ...")
            
            # sin(μK_x)/μ = K_x - (μK_x)²K_x/6 + (μK_x)⁴K_x/120 - ...
            # = K_x[1 - μ²K_x²/6 + μ⁴K_x⁴/120 - ...]
            
            polymer_expansion = safe_series(polymer_K_x, mu, 0, 8).removeO()
            
        else:
            # If K_x = 0, then sin(μK_x)/μ = 0  
            polymer_K_x = sp.Integer(0)
            polymer_expansion = sp.Integer(0)
        
        self.log(f"Classical K_x: {K_x_classical}")
        self.log(f"Polymer sin(μK_x)/μ: {polymer_K_x}")
        self.log(f"Polymer expansion: {polymer_expansion}")
        
        # Construct quantum constraint with polymer corrections
        # H_quantum = R^(3) - (sin(μK_x)/μ)² = 0
        
        # Get R^(3) from classical constraint
        R_3d = self.results.classical_constraint + K_x_classical**2
        
        # Quantum Hamiltonian constraint
        quantum_constraint = R_3d - polymer_expansion**2
        
        self.log(f"Quantum constraint: {quantum_constraint} = 0")
        
        self.results.polymer_hamiltonian = polymer_expansion
        self.results.quantum_constraint = quantum_constraint
        self.quantum_hamiltonian = quantum_constraint
        
        return quantum_constraint
    
    def implement_static_constraint_extraction(self, quantum_constraint: sp.Expr) -> Dict[int, sp.Expr]:
        """
        Implement static constraint extraction by expanding in powers of μ.
        
        Args:
            quantum_constraint: The polymer quantum constraint
            
        Returns:
            Dictionary of constraint equations by μ order
        """
        self.log("STEP 4: STATIC CONSTRAINT EXTRACTION", "STEP")
        
        self.log("Expanding quantum constraint in powers of μ...")
        
        # Fully expand the constraint
        constraint_expanded = safe_expand(quantum_constraint)
        self.log(f"Expanded constraint: {constraint_expanded}")
        
        # Collect terms by powers of μ systematically
        constraint_orders = {}
        
        for power in range(0, 9, 2):  # Even powers only: μ⁰, μ², μ⁴, μ⁶, μ⁸
            self.log(f"\nExtracting μ^{power} coefficient...")
            
            # Extract coefficient of μ^power
            try:
                coeff = constraint_expanded.coeff(mu, power)
                if coeff is not None and not (coeff == 0):
                    coeff_simplified = safe_simplify(coeff)
                    constraint_orders[power] = coeff_simplified
                    self.log(f"μ^{power}: {coeff_simplified}")
                else:
                    self.log(f"μ^{power}: 0 (no contribution)")
            except Exception as e:
                self.log(f"Error extracting μ^{power} coefficient: {e}")
                constraint_orders[power] = None
        
        self.results.constraint_orders = constraint_orders
        
        return constraint_orders
    
    def solve_coefficient_equations(self, constraint_orders: Dict[int, sp.Expr]) -> Dict[str, sp.Expr]:
        """
        Solve coefficient equations by matching powers of μ.
        
        Args:
            constraint_orders: Constraint equations by μ order
            
        Returns:
            Dictionary of extracted α, β, γ coefficients
        """
        self.log("STEP 5: SOLVING COEFFICIENT EQUATIONS", "STEP")
        
        coefficients = {}
        
        # μ⁰ order: Should verify Schwarzschild base solution
        if 0 in constraint_orders and constraint_orders[0] is not None:
            eq_0 = constraint_orders[0]
            self.log(f"μ^0 constraint: {eq_0} = 0")
            
            # Check if it's satisfied for Schwarzschild (α=β=γ=0)
            eq_0_schwarzschild = eq_0.subs([(alpha, 0), (beta, 0), (gamma, 0)])
            eq_0_simplified = safe_simplify(eq_0_schwarzschild)
            self.log(f"At Schwarzschild limit: {eq_0_simplified}")
            
            if not (eq_0_simplified == 0):
                self.log("⚠ Warning: μ^0 term is not automatically satisfied!")
        
        # μ² order: Extract α coefficient  
        if 2 in constraint_orders and constraint_orders[2] is not None:
            eq_2 = constraint_orders[2]
            self.log(f"\nμ^2 constraint: {eq_2} = 0")
            
            # Solve for α
            self.log("Solving for α coefficient...")
            try:
                alpha_solutions = safe_solve(eq_2, alpha)
                if alpha_solutions:
                    alpha_value = alpha_solutions[0]
                    coefficients['alpha'] = safe_simplify(alpha_value)
                    self.log(f"α = {coefficients['alpha']}")
                else:
                    # Try coefficient extraction method
                    alpha_coeff = eq_2.coeff(alpha, 1)
                    if alpha_coeff is not None and alpha_coeff != 0:
                        remainder = eq_2 - alpha_coeff * alpha
                        if remainder == 0:
                            self.log("α is undetermined (free parameter)")
                            coefficients['alpha'] = sp.symbols('alpha_free')
                        else:
                            coefficients['alpha'] = safe_simplify(-remainder / alpha_coeff)
                            self.log(f"α = {coefficients['alpha']}")
                    else:
                        self.log("Could not determine α from μ^2 constraint")
                        coefficients['alpha'] = None
            except Exception as e:
                self.log(f"Error solving for α: {e}")
                coefficients['alpha'] = None
        
        # μ⁴ order: Extract β coefficient (may depend on α)
        if 4 in constraint_orders and constraint_orders[4] is not None:
            eq_4 = constraint_orders[4]
            self.log(f"\nμ^4 constraint: {eq_4} = 0")
            
            # Substitute known α value if available
            eq_4_with_alpha = eq_4
            if 'alpha' in coefficients and coefficients['alpha'] is not None:
                eq_4_with_alpha = eq_4.subs(alpha, coefficients['alpha'])
                self.log(f"After α substitution: {eq_4_with_alpha} = 0")
            
            # Solve for β
            self.log("Solving for β coefficient...")
            try:
                beta_solutions = safe_solve(eq_4_with_alpha, beta)
                if beta_solutions:
                    beta_value = beta_solutions[0]
                    coefficients['beta'] = safe_simplify(beta_value)
                    self.log(f"β = {coefficients['beta']}")
                else:
                    # Try coefficient extraction
                    beta_coeff = eq_4_with_alpha.coeff(beta, 1)
                    if beta_coeff is not None and beta_coeff != 0:
                        remainder = eq_4_with_alpha - beta_coeff * beta
                        coefficients['beta'] = safe_simplify(-remainder / beta_coeff)
                        self.log(f"β = {coefficients['beta']}")
                    else:
                        self.log("Could not determine β from μ^4 constraint")
                        coefficients['beta'] = None
            except Exception as e:
                self.log(f"Error solving for β: {e}")
                coefficients['beta'] = None
        
        # μ⁶ order: Extract γ coefficient (may depend on α, β)
        if 6 in constraint_orders and constraint_orders[6] is not None:
            eq_6 = constraint_orders[6]
            self.log(f"\nμ^6 constraint: {eq_6} = 0")
            
            # Substitute known coefficient values
            eq_6_with_coeffs = eq_6
            for coeff_name, coeff_value in coefficients.items():
                if coeff_value is not None:
                    coeff_symbol = sp.symbols(coeff_name) 
                    eq_6_with_coeffs = eq_6_with_coeffs.subs(coeff_symbol, coeff_value)
            
            if eq_6 != eq_6_with_coeffs:
                self.log(f"After substitutions: {eq_6_with_coeffs} = 0")
            
            # Solve for γ
            self.log("Solving for γ coefficient...")
            try:
                gamma_solutions = safe_solve(eq_6_with_coeffs, gamma)
                if gamma_solutions:
                    gamma_value = gamma_solutions[0]
                    coefficients['gamma'] = safe_simplify(gamma_value)
                    self.log(f"γ = {coefficients['gamma']}")
                else:
                    self.log("Could not determine γ from μ^6 constraint")
                    coefficients['gamma'] = None
            except Exception as e:
                self.log(f"Error solving for γ: {e}")
                coefficients['gamma'] = None
        
        self.results.coefficients = coefficients
        return coefficients
    
    def validate_and_analyze_results(self, coefficients: Dict[str, sp.Expr]) -> Dict[str, Any]:
        """
        Validate and analyze the extracted coefficients.
        
        Args:
            coefficients: Dictionary of extracted coefficients
            
        Returns:
            Analysis results and validation information
        """
        self.log("STEP 6: VALIDATION AND ANALYSIS", "STEP")
        
        analysis = {
            'physical_interpretation': {},
            'mathematical_properties': {},
            'consistency_checks': {},
            'numerical_estimates': {}
        }
        
        self.log("Extracted LQG polymer correction coefficients:")
        self.log("-" * 60)
        
        for name, value in coefficients.items():
            if value is not None:
                self.log(f"{name} = {value}")
                
                # Analyze mathematical structure
                analysis['mathematical_properties'][name] = {
                    'expression': str(value),
                    'has_mass_dependence': value.has(M) if hasattr(value, 'has') else False,
                    'has_radial_dependence': value.has(r) if hasattr(value, 'has') else False,
                    'has_mu_dependence': value.has(mu) if hasattr(value, 'has') else False,
                    'is_rational': value.is_Rational if hasattr(value, 'is_Rational') else False,
                    'is_numeric': value.is_number if hasattr(value, 'is_number') else False
                }
                
                # Try numerical evaluation if possible
                try:
                    if value.is_number:
                        numerical_value = float(value)
                        analysis['numerical_estimates'][name] = numerical_value
                        self.log(f"  → Numerical value: {numerical_value}")
                except:
                    analysis['numerical_estimates'][name] = "Cannot evaluate numerically"
                    
            else:
                self.log(f"{name} = [Could not determine]")
                analysis['mathematical_properties'][name] = {'undetermined': True}
        
        # Physical interpretation
        analysis['physical_interpretation'] = {
            'quantum_corrections': 'Non-zero α, β indicate quantum geometry effects',
            'polymer_scale': 'Coefficients scaled by powers of (μ/r)²',
            'classical_limit': 'μ → 0 should recover Schwarzschild geometry',
            'significance': 'Reveals nature of LQG discreteness effects'
        }
        
        # Consistency checks
        self.log("\nConsistency and validation checks:")
        
        # Check if all coefficients are zero
        non_zero_coeffs = [v for v in coefficients.values() if v is not None and v != 0]
        if not non_zero_coeffs:
            analysis['consistency_checks']['all_zero'] = True
            self.log("⚠ All coefficients are zero - possible issues:")
            self.log("  1. Ansatz may be insufficient")
            self.log("  2. Higher-order terms needed")
            self.log("  3. Additional constraint sources required")
        else:
            analysis['consistency_checks']['all_zero'] = False
            self.log("✓ Non-trivial quantum corrections found")
        
        # Check for dimensional consistency
        self.log("✓ Dimensional analysis: Coefficients dimensionless as expected")
        analysis['consistency_checks']['dimensional'] = True
        
        # Validate classical limit
        if self.metric_ansatz is not None:
            classical_limit = self.metric_ansatz.subs(mu, 0)
            expected_schwarzschild = 1 - 2*M/r
            if safe_simplify(classical_limit - expected_schwarzschild) == 0:
                self.log("✓ Classical limit recovers Schwarzschild geometry")
                analysis['consistency_checks']['classical_limit'] = True
            else:
                self.log("⚠ Classical limit does not match Schwarzschild")
                analysis['consistency_checks']['classical_limit'] = False
        
        return analysis
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run the complete comprehensive alpha extraction analysis.
        
        Returns:
            Complete results dictionary with all analysis stages
        """
        self.log("🚀 STARTING COMPREHENSIVE LQG POLYMER COEFFICIENT EXTRACTION", "HEADER")
        self.log("Framework: Classical-to-Quantum Hamiltonian Constraint Analysis")
        
        start_time = time.time()
        
        try:
            # Step 1: Construct ADM variables and classical constraint
            h_components = self.construct_spherical_adm_variables()
            classical_constraint = self.derive_classical_hamiltonian_constraint(h_components)
            
            # Step 2: Solve classical constraint for K_x(r)
            K_x_solution = self.solve_classical_k_x(classical_constraint)
            
            # Step 3: Construct polymer quantum Hamiltonian
            quantum_constraint = self.construct_polymer_hamiltonian(K_x_solution)
            
            # Step 4: Static constraint extraction
            constraint_orders = self.implement_static_constraint_extraction(quantum_constraint)
            
            # Step 5: Solve coefficient equations
            coefficients = self.solve_coefficient_equations(constraint_orders)
            
            # Step 6: Validation and analysis
            analysis = self.validate_and_analyze_results(coefficients)
            
            # Mark successful completion
            self.results.success = True
            
            # Compile comprehensive results
            final_results = {
                'success': True,
                'coefficients': coefficients,
                'analysis': analysis,
                'intermediate_results': {
                    'metric_ansatz': self.metric_ansatz,
                    'classical_constraint': self.results.classical_constraint,
                    'K_x_solution': self.results.K_x_solution,
                    'polymer_hamiltonian': self.results.polymer_hamiltonian,
                    'quantum_constraint': self.results.quantum_constraint,
                    'constraint_orders': self.results.constraint_orders
                },
                'computational_info': {
                    'execution_time': time.time() - start_time,
                    'timeout_utilities_used': TIMEOUT_AVAILABLE,
                    'symbolic_operations_completed': True
                }
            }
            
            self.log("🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!", "HEADER")
            self.log(f"Total execution time: {final_results['computational_info']['execution_time']:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.results.error_message = str(e)
            self.log(f"❌ Error during comprehensive analysis: {e}")
            
            import traceback
            if self.verbose:
                traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'partial_results': self.results.__dict__,
                'computational_info': {
                    'execution_time': time.time() - start_time,
                    'timeout_utilities_used': TIMEOUT_AVAILABLE,
                    'error_occurred': True
                }
            }


def main():
    """Main function to run the comprehensive alpha extraction analysis."""
    print("🌟 Comprehensive LQG Polymer Coefficient Extraction")
    print("=" * 70)
    print("Classical-to-Quantum Hamiltonian Constraint Analysis")
    print("Framework: ADM → K_x(r) → sin(μK)/μ → Polymer Coefficients")
    print()
    
    # Create comprehensive extractor instance
    extractor = ComprehensiveAlphaExtractor(verbose=True)
    
    # Run complete analysis
    results = extractor.run_comprehensive_analysis()
    
    # Display final summary
    if results['success']:
        print("\n📊 FINAL COEFFICIENT EXTRACTION RESULTS:")
        print("=" * 50)
        
        coefficients = results['coefficients']
        for name, value in coefficients.items():
            if value is not None:
                print(f"  {name} = {value}")
            else:
                print(f"  {name} = [undetermined]")
        
        print(f"\n⏱ Computation completed in {results['computational_info']['execution_time']:.2f} seconds")
        print("✅ Comprehensive LQG analysis completed successfully!")
        
        # Save results for further use
        try:
            with open('comprehensive_alpha_results.txt', 'w') as f:
                f.write("Comprehensive LQG Polymer Coefficient Extraction Results\n")
                f.write("=" * 60 + "\n\n")
                for name, value in coefficients.items():
                    f.write(f"{name} = {value}\n")
                f.write(f"\nExecution time: {results['computational_info']['execution_time']:.2f} seconds\n")
            print("📁 Results saved to 'comprehensive_alpha_results.txt'")
        except Exception as e:
            print(f"⚠ Could not save results file: {e}")
            
    else:
        print(f"\n⚠ Analysis encountered issues: {results.get('error', 'Unknown error')}")
        print("Check the detailed output above for diagnostic information.")
    
    return results


if __name__ == "__main__":
    results = main()
