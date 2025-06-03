#!/usr/bin/env python3
"""
Targeted LQG α,β Coefficient Extraction Script

This script implements the focused extraction of polymer LQG metric coefficients 
α and β from classical-to-quantum Hamiltonian constraint analysis as requested.

Specifically extracts coefficients for the metric ansatz:
f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶

Workflow:
1. Classical ADM Hamiltonian constraint construction  
2. Solve for K_x(r) from R^(3) - K_x² = 0
3. Polymer quantum Hamiltonian with sin(μK_x)/μ holonomy corrections
4. Static constraint extraction by μ-expansion 
5. Extract α from μ² term, β from μ⁴ term
6. Validate results with symbolic timeout utilities

Author: Targeted LQG Framework
"""

import sympy as sp
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
import os
import time

# Add current directory to path for imports  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import symbolic timeout utilities
try:
    from scripts.symbolic_timeout_utils import (
        safe_symbolic_operation, safe_series, safe_solve, safe_simplify, 
        safe_expand, safe_diff, safe_collect, safe_factor,
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

# Set timeout for complex operations
if TIMEOUT_AVAILABLE:
    try:
        set_default_timeout(15)
        print("✓ Symbolic timeout set to 15 seconds")
    except:
        pass

# Define symbolic variables for targeted extraction
print("Setting up targeted symbolic variables...")
r, M, mu = sp.symbols('r M mu', positive=True, real=True)
alpha, beta = sp.symbols('alpha beta', real=True)  # Target coefficients only
K_rr = sp.symbols('K_rr', real=True)

print("✓ Symbolic setup complete")


class TargetedAlphaExtractor:
    """
    Targeted class for extracting specifically α and β polymer coefficients.
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the targeted extractor."""
        self.verbose = verbose
        self.metric_ansatz = None
        self.results = {'alpha': None, 'beta': None}
        
        if self.verbose:
            print("Targeted Alpha/Beta Extractor initialized")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log message if verbose."""
        if self.verbose:
            if level == "HEADER":
                print(f"\n{'='*60}")
                print(f"{message}")  
                print(f"{'='*60}")
            elif level == "STEP":
                print(f"\n{'-'*40}")
                print(f"{message}")
                print(f"{'-'*40}")
            else:
                print(f"{message}")
    
    def construct_targeted_metric_ansatz(self) -> sp.Expr:
        """
        Construct the specific metric ansatz for α, β extraction.
        
        Returns:
            f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶
        """
        self.log("STEP 1: CONSTRUCTING TARGETED METRIC ANSATZ", "STEP")
        
        # Base Schwarzschild
        f_base = 1 - 2*M/r
        
        # Target LQG corrections as specified
        alpha_correction = alpha * mu**2 * M**2 / r**4
        beta_correction = beta * mu**4 * M**4 / r**6
        
        # Complete targeted ansatz
        f_metric = f_base + alpha_correction + beta_correction
        
        self.log(f"Base Schwarzschild: {f_base}")
        self.log(f"α correction: {alpha_correction}")  
        self.log(f"β correction: {beta_correction}")
        self.log(f"Complete ansatz: f(r) = {f_metric}")
        
        self.metric_ansatz = f_metric
        return f_metric
    
    def derive_classical_constraint(self, f_metric: sp.Expr) -> sp.Expr:
        """
        Derive classical Hamiltonian constraint R^(3) - K_x² = 0.
        
        Args:
            f_metric: The metric function f(r)
            
        Returns:
            Classical constraint equation
        """
        self.log("STEP 2: DERIVING CLASSICAL CONSTRAINT", "STEP")
        
        # Compute derivatives of f(r)
        f_prime = safe_diff(f_metric, r)
        f_double_prime = safe_diff(f_prime, r)
        
        # Ricci scalar for spherically symmetric metric
        # R^(3) = 2f''/r + 4f'/r² - 4(1-f)/r²
        R_3d = 2*f_double_prime/r + 4*f_prime/r**2 - 4*(1 - f_metric)/r**2
        
        # Classical constraint: R^(3) - K_x² = 0
        classical_constraint = R_3d - K_rr**2
        
        self.log(f"f'(r): {f_prime}")
        self.log(f"f''(r): {f_double_prime}")
        self.log(f"R^(3): {R_3d}")
        self.log(f"Classical constraint: {classical_constraint} = 0")
        
        return classical_constraint
    
    def solve_for_K_x(self, classical_constraint: sp.Expr) -> sp.Expr:
        """
        Solve classical constraint for K_x(r).
        
        Args:
            classical_constraint: R^(3) - K_x² = 0
            
        Returns:
            K_x(r) solution
        """
        self.log("STEP 3: SOLVING FOR K_x(r)", "STEP")
        
        # Extract R^(3) = K_x²
        R_3d = classical_constraint + K_rr**2
        
        # Solve for K_x = ±√R^(3)
        K_x_solutions = safe_solve(classical_constraint, K_rr)
        
        if K_x_solutions:
            # Take positive solution
            K_x = K_x_solutions[1] if len(K_x_solutions) > 1 else K_x_solutions[0]
            K_x_simplified = safe_simplify(K_x)
            
            self.log(f"K_x solutions: {K_x_solutions}")
            self.log(f"Selected K_x(r): {K_x_simplified}")
            
            return K_x_simplified
        else:
            # Direct approach: K_x = √R^(3)
            R_3d_simplified = safe_simplify(R_3d)
            K_x_direct = sp.sqrt(sp.Abs(R_3d_simplified))
            
            self.log(f"Direct K_x(r): {K_x_direct}")
            return K_x_direct
    
    def construct_polymer_quantum_constraint(self, K_x_classical: sp.Expr) -> sp.Expr:
        """
        Construct polymer quantum constraint with sin(μK_x)/μ.
        
        Args:
            K_x_classical: Classical K_x(r) solution
            
        Returns:
            Quantum constraint with polymer corrections
        """
        self.log("STEP 4: CONSTRUCTING POLYMER QUANTUM CONSTRAINT", "STEP")
        
        # Polymer correction: K_x → sin(μK_x)/μ
        mu_K_x = mu * K_x_classical
        
        if K_x_classical != 0:
            polymer_K_x = sp.sin(mu_K_x) / mu
            
            # Expand in powers of μ up to μ⁴ (sufficient for β)
            self.log("Expanding sin(μK_x)/μ to μ⁴ order...")
            polymer_expansion = safe_series(polymer_K_x, mu, 0, 6).removeO()
            
        else:
            polymer_expansion = sp.Integer(0)
        
        # Quantum constraint: R^(3) - (sin(μK_x)/μ)² = 0
        R_3d = self.derive_classical_constraint(self.metric_ansatz) + K_x_classical**2
        quantum_constraint = R_3d - polymer_expansion**2
        
        self.log(f"Classical K_x: {K_x_classical}")
        self.log(f"Polymer expansion: {polymer_expansion}")
        self.log(f"Quantum constraint: {quantum_constraint} = 0")
        
        return quantum_constraint
    
    def extract_alpha_beta_coefficients(self, quantum_constraint: sp.Expr) -> Dict[str, sp.Expr]:
        """
        Extract α and β coefficients by μ-expansion.
        
        Args:
            quantum_constraint: Polymer quantum constraint
            
        Returns:
            Dictionary with α and β values
        """
        self.log("STEP 5: EXTRACTING α AND β COEFFICIENTS", "STEP")
        
        # Expand constraint in powers of μ
        constraint_expanded = safe_expand(quantum_constraint)
        self.log("Expanding quantum constraint...")
        
        coefficients = {}
        
        # Extract μ² coefficient for α
        self.log("\nExtracting α from μ² term...")
        mu2_coeff = constraint_expanded.coeff(mu, 2)
        
        if mu2_coeff is not None and mu2_coeff != 0:
            self.log(f"μ² constraint: {mu2_coeff} = 0")
            
            # Solve for α
            alpha_solutions = safe_solve(mu2_coeff, alpha)
            if alpha_solutions:
                alpha_value = safe_simplify(alpha_solutions[0])
                coefficients['alpha'] = alpha_value
                self.log(f"α = {alpha_value}")
            else:
                self.log("Could not solve for α directly")
                coefficients['alpha'] = None
        else:
            self.log("No μ² term found")
            coefficients['alpha'] = sp.Integer(0)
        
        # Extract μ⁴ coefficient for β (substitute α if found)
        self.log("\nExtracting β from μ⁴ term...")
        mu4_coeff = constraint_expanded.coeff(mu, 4)
        
        if mu4_coeff is not None and mu4_coeff != 0:
            # Substitute known α value
            if coefficients['alpha'] is not None:
                mu4_coeff_with_alpha = mu4_coeff.subs(alpha, coefficients['alpha'])
                self.log(f"μ⁴ constraint (with α): {mu4_coeff_with_alpha} = 0")
            else:
                mu4_coeff_with_alpha = mu4_coeff
                self.log(f"μ⁴ constraint: {mu4_coeff} = 0")
            
            # Solve for β
            beta_solutions = safe_solve(mu4_coeff_with_alpha, beta)
            if beta_solutions:
                beta_value = safe_simplify(beta_solutions[0])
                coefficients['beta'] = beta_value
                self.log(f"β = {beta_value}")
            else:
                self.log("Could not solve for β directly")
                coefficients['beta'] = None
        else:
            self.log("No μ⁴ term found")
            coefficients['beta'] = sp.Integer(0)
        
        self.results = coefficients
        return coefficients
    
    def validate_results(self, coefficients: Dict[str, sp.Expr]) -> Dict[str, Any]:
        """
        Validate the extracted α, β coefficients.
        
        Args:
            coefficients: Extracted α, β values
            
        Returns:
            Validation analysis
        """
        self.log("STEP 6: VALIDATION AND ANALYSIS", "STEP")
        
        validation = {
            'coefficients_found': {},
            'dimensional_analysis': {},
            'classical_limit': False,
            'physical_interpretation': {}
        }
        
        # Check each coefficient
        for name, value in coefficients.items():
            if value is not None:
                self.log(f"{name} = {value}")
                
                validation['coefficients_found'][name] = {
                    'value': str(value),
                    'is_zero': (value == 0),
                    'has_mass_dependence': value.has(M) if hasattr(value, 'has') else False,
                    'has_radial_dependence': value.has(r) if hasattr(value, 'has') else False,
                    'is_rational': value.is_Rational if hasattr(value, 'is_Rational') else False
                }
                
                # Try numerical evaluation at sample values
                try:
                    if value.has(r) and value.has(M):
                        # Evaluate at r = 10M (outside horizon)
                        sample_value = value.subs([(r, 10*M), (M, 1)])
                        if sample_value.is_number:
                            numerical_val = float(sample_value)
                            self.log(f"  At r=10M, M=1: {name} = {numerical_val:.6f}")
                except:
                    pass
            else:
                self.log(f"{name} = [Could not determine]")
                validation['coefficients_found'][name] = {'undetermined': True}
        
        # Classical limit check
        if self.metric_ansatz is not None:
            classical_limit = self.metric_ansatz.subs(mu, 0)
            expected_schwarzschild = 1 - 2*M/r
            if safe_simplify(classical_limit - expected_schwarzschild) == 0:
                self.log("✓ Classical limit (μ→0) recovers Schwarzschild")
                validation['classical_limit'] = True
            else:
                self.log("⚠ Classical limit does not match Schwarzschild")
        
        # Physical interpretation
        non_zero_coeffs = [v for v in coefficients.values() if v is not None and v != 0]
        if non_zero_coeffs:
            self.log("✓ Non-trivial LQG quantum corrections found")
            validation['physical_interpretation']['quantum_corrections'] = True
        else:
            self.log("⚠ All coefficients are zero - may need higher order terms")
            validation['physical_interpretation']['quantum_corrections'] = False
        
        return validation
    
    def run_targeted_extraction(self) -> Dict[str, Any]:
        """
        Run the complete targeted α, β extraction analysis.
        
        Returns:
            Complete results dictionary
        """
        self.log("🎯 STARTING TARGETED α,β COEFFICIENT EXTRACTION", "HEADER")
        self.log("Target: f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶")
        
        start_time = time.time()
        
        try:
            # Step 1: Construct targeted metric ansatz
            f_metric = self.construct_targeted_metric_ansatz()
            
            # Step 2: Derive classical constraint  
            classical_constraint = self.derive_classical_constraint(f_metric)
            
            # Step 3: Solve for K_x(r)
            K_x_solution = self.solve_for_K_x(classical_constraint)
            
            # Step 4: Construct polymer quantum constraint
            quantum_constraint = self.construct_polymer_quantum_constraint(K_x_solution)
            
            # Step 5: Extract α, β coefficients
            coefficients = self.extract_alpha_beta_coefficients(quantum_constraint)
            
            # Step 6: Validate results
            validation = self.validate_results(coefficients)
            
            # Compile results
            final_results = {
                'success': True,
                'coefficients': coefficients,
                'validation': validation,
                'intermediate_results': {
                    'metric_ansatz': f_metric,
                    'classical_constraint': classical_constraint,
                    'K_x_solution': K_x_solution,
                    'quantum_constraint': quantum_constraint
                },
                'computation_time': time.time() - start_time,
                'timeout_utilities_used': TIMEOUT_AVAILABLE
            }
            
            self.log("🎉 TARGETED EXTRACTION COMPLETED SUCCESSFULLY!", "HEADER")
            self.log(f"Computation time: {final_results['computation_time']:.2f} seconds")
            
            return final_results
            
        except Exception as e:
            self.log(f"❌ Error during targeted extraction: {e}")
            
            import traceback
            if self.verbose:
                traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'partial_results': self.results,
                'computation_time': time.time() - start_time
            }


def main():
    """Main function for targeted α, β extraction."""
    print("🎯 Targeted LQG α, β Coefficient Extraction")
    print("=" * 50)
    print("Extracting polymer coefficients from:")
    print("f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶")
    print()
    
    # Create targeted extractor
    extractor = TargetedAlphaExtractor(verbose=True)
    
    # Run extraction analysis
    results = extractor.run_targeted_extraction()
    
    # Display final results
    if results['success']:
        print("\n📋 EXTRACTED COEFFICIENTS:")
        print("=" * 30)
        
        coefficients = results['coefficients']
        for name, value in coefficients.items():
            if value is not None:
                print(f"  {name} = {value}")
            else:
                print(f"  {name} = [undetermined]")
        
        print(f"\n⏱ Completed in {results['computation_time']:.2f} seconds")
        print("✅ Targeted α, β extraction successful!")
        
        # Save targeted results
        try:
            with open('targeted_alpha_beta_results.txt', 'w') as f:
                f.write("Targeted LQG α, β Coefficient Extraction Results\n")
                f.write("=" * 50 + "\n\n")
                f.write("Metric ansatz: f(r) = 1 - 2M/r + αμ²M²/r⁴ + βμ⁴M⁴/r⁶\n\n")
                f.write("Extracted coefficients:\n")
                for name, value in coefficients.items():
                    f.write(f"  {name} = {value}\n")
                f.write(f"\nComputation time: {results['computation_time']:.2f} seconds\n")
            print("📁 Results saved to 'targeted_alpha_beta_results.txt'")
        except Exception as e:
            print(f"⚠ Could not save results: {e}")
            
    else:
        print(f"\n❌ Extraction failed: {results.get('error', 'Unknown error')}")
    
    return results


if __name__ == "__main__":
    results = main()
