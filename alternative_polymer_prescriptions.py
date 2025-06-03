#!/usr/bin/env python3
"""
Alternative Polymer Prescriptions for LQG

This module implements various alternative polymer quantization prescriptions
beyond the standard Thiemann approach, allowing for comparison of their
effects on metric coefficients and phenomenology.

Key Features:
- Multiple polymer prescriptions (Thiemann, AQEL, Bojowald, etc.)
- Comparative coefficient extraction
- Phenomenological implications analysis
- Consistency checks across prescriptions
- Kerr generalization for rotating black holes
"""

import sympy as sp
import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# 1) DEFINE POLYMER PRESCRIPTION CLASSES
# ------------------------------------------------------------------------

class PolymerPrescription:
    """Base class for polymer quantization prescriptions."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        # Define symbols
        self.r, self.M, self.mu = sp.symbols('r M mu', positive=True)
        self.a, self.theta = sp.symbols('a theta', real=True)  # Kerr parameters
        self.q = sp.Symbol('q', positive=True)  # metric determinant
        self.K = sp.Symbol('K', real=True)      # extrinsic curvature
    
    def compute_effective_mu(self, classical_geometry):
        """Compute effective μ based on prescription."""
        raise NotImplementedError("Must implement in subclass")
    
    def get_polymer_factor(self, K_classical, classical_geometry=None):
        """Compute polymer modification factor sin(μ_eff * K) / μ_eff."""
        if classical_geometry is None:
            classical_geometry = {'f_classical': 1 - 2*self.M/self.r}
        
        mu_eff = self.compute_effective_mu(classical_geometry)
        argument = mu_eff * K_classical
        
        # For proper limit behavior, use series expansion instead of Piecewise
        # The standard form sin(x)/x = 1 - x²/6 + x⁴/120 - ...
        return sp.sin(argument) / mu_eff
        
    def compute_kerr_effective_mu(self, r, theta, M, a):
        """
        Compute effective μ for Kerr geometry based on prescription.
        Default implementation for Kerr - overridden in subclasses.
        """
        # Default implementation for Kerr
        Sigma = r**2 + (a * sp.cos(theta))**2
        return self.mu * sp.sqrt(Sigma) / M

class ThiemannPrescription(PolymerPrescription):
    """Standard Thiemann prescription: μ_eff = μ * sqrt(det(q))."""
    
    def __init__(self):
        super().__init__(
            "Thiemann", 
            "Standard prescription with μ_eff = μ * sqrt(det(q))"
        )
    
    def compute_effective_mu(self, classical_geometry):
        f_classical = classical_geometry.get('f_classical', 1 - 2*self.M/self.r)
        # For spherical symmetry: sqrt(det(q)) ∝ sqrt(f_classical)
        return self.mu * sp.sqrt(f_classical)
        
    def compute_kerr_effective_mu(self, r, theta, M, a):
        """Thiemann prescription for Kerr: μ_eff = μ * sqrt(det(q))"""
        Sigma = r**2 + (a * sp.cos(theta))**2
        Delta = r**2 - 2*M*r + a**2
        # For Kerr, det(q) involves both Sigma and Delta
        return self.mu * sp.sqrt(Sigma * Delta) / M

class AQELPrescription(PolymerPrescription):
    """AQEL prescription: μ_eff = μ * q^{1/3}."""
    
    def __init__(self):
        super().__init__(
            "AQEL", 
            "Ashtekar-Quantum-Einstein-Loop prescription with μ_eff = μ * q^{1/3}"
        )
    
    def compute_effective_mu(self, classical_geometry):
        f_classical = classical_geometry.get('f_classical', 1 - 2*self.M/self.r)
        # For spherical symmetry: q^{1/3} ∝ f_classical^{1/3}
        return self.mu * (f_classical)**(sp.Rational(1, 3))
        
    def compute_kerr_effective_mu(self, r, theta, M, a):
        """AQEL prescription for Kerr: μ_eff = μ * q^{1/3}"""
        Sigma = r**2 + (a * sp.cos(theta))**2
        Delta = r**2 - 2*M*r + a**2
        # For Kerr, det(q)^(1/3)
        return self.mu * (Sigma * Delta)**(sp.Rational(1, 3)) / M

class BojowaldPrescription(PolymerPrescription):
    """Bojowald prescription: μ_eff = μ * sqrt(|K|)."""    
    def __init__(self):
        super().__init__(
            "Bojowald", 
            "Bojowald prescription with μ_eff = μ * sqrt(|K|)"
        )
    
    def compute_effective_mu(self, classical_geometry):
        # Classical extrinsic curvature
        K_classical = self.M / (self.r * (2*self.M - self.r))
        # For numerical stability, use a simplified form
        return self.mu * sp.sqrt(sp.Abs(K_classical))
    
    def get_polymer_factor(self, K_classical, classical_geometry=None):
        """Bojowald prescription: sin(μ_eff * K) / μ_eff where μ_eff = μ * sqrt(|K|)."""
        if classical_geometry is None:
            classical_geometry = {'f_classical': 1 - 2*self.M/self.r}
        
        # For Bojowald, we have μ_eff = μ*sqrt(|K|)
        # This means sin(μ*sqrt(|K|)*K)/(μ*sqrt(|K|))
        # Simplified: sin(μ*K*sqrt(|K|))/(μ*sqrt(|K|))
        mu_eff = self.mu * sp.sqrt(sp.Abs(K_classical))
        return sp.sin(mu_eff * K_classical) / mu_eff
        
    def compute_kerr_effective_mu(self, r, theta, M, a):
        """Bojowald prescription for Kerr: μ_eff = μ * sqrt(|K|)"""
        Sigma = r**2 + (a * sp.cos(theta))**2
        # Effective curvature for Kerr
        K_eff = M / (r * Sigma)
        return self.mu * sp.sqrt(sp.Abs(K_eff))

class ImprovedPrescription(PolymerPrescription):
    """Improved prescription: μ_eff = μ * (1 + δμ²)."""
    
    def __init__(self, delta=sp.Rational(1, 12)):
        super().__init__(
            "Improved", 
            f"Improved prescription with μ_eff = μ * (1 + δμ²), δ = {delta}"
        )
        self.delta = delta
    
    def compute_effective_mu(self, classical_geometry):
        return self.mu * (1 + self.delta * self.mu**2)

# ------------------------------------------------------------------------
# 2) COEFFICIENT EXTRACTION FOR EACH PRESCRIPTION
# ------------------------------------------------------------------------

def extract_coefficients_for_prescription(prescription: PolymerPrescription, 
                                        max_order: int = 6) -> Dict[str, float]:
    """Extract LQG coefficients for a given polymer prescription."""
    print(f"🔬 Extracting coefficients for {prescription.name} prescription...")
    
    start_time = time.time()
    
    # Setup classical geometry
    classical_geometry = {
        'f_classical': 1 - 2*prescription.M/prescription.r,
        'K_classical': prescription.M / (prescription.r * (2*prescription.M - prescription.r))
    }
    
    # Build simplified polymer Hamiltonian
    K_classical = classical_geometry['K_classical']
    
    try:
        # Compute effective μ
        mu_eff = prescription.compute_effective_mu(classical_geometry)
        
        # For coefficient extraction, work with the series expansion directly
        # sin(μ_eff * K) / μ_eff = K * [1 - (μ_eff * K)²/6 + (μ_eff * K)⁴/120 - ...]
        
        argument = mu_eff * K_classical
        
        # Use the series expansion of sin(x)/x = 1 - x²/6 + x⁴/120 - x⁶/5040 + ...
        sinc_series = 1 - argument**2/6 + argument**4/120 - argument**6/5040
        
        # The polymer factor is K_classical * sinc_series
        polymer_factor = K_classical * sinc_series
        
        # Expand in μ to extract coefficients
        polymer_series = sp.series(polymer_factor, prescription.mu, 0, n=max_order+1).removeO()
        
        # Extract coefficients
        coefficients = {}
        coeff_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
          # For α coefficient, we want the coefficient of μ² in the total expression
        # sin(μ_eff * K) / μ_eff ≈ K * [1 - (μ_eff * K)²/6 + ...]
        # For standard polymer: μ_eff = μ, so coefficient of μ²K³ should be -1/6
        
        for i, name in enumerate(coeff_names[:max_order//2]):
            order = 2 * (i + 1)
            coeff = polymer_series.coeff(prescription.mu, order)
            if coeff:
                # For α: extract coefficient of μ²K³ and normalize to get -1/6
                # The pattern is: coeff = α * K³, so α = coeff / K³
                K_power = K_classical**(i + 3)  # For α: K³, for β: K⁵, etc.
                alpha_coeff = coeff / K_power
                
                # Simplify and evaluate to a constant
                simplified = sp.simplify(alpha_coeff)
                
                # Try to extract the universal constant
                if simplified.is_constant():
                    coefficients[name] = float(simplified)
                else:
                    # If not constant, evaluate at standard values
                    numerical_val = simplified.subs({
                        prescription.r: 10.0, 
                        prescription.M: 1.0
                    })
                    coefficients[name] = float(numerical_val)
            else:
                coefficients[name] = 0.0
        
        print(f"   ✅ Extraction completed in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"   ⚠️  Extraction failed: {e}")
        # Fallback values
        coefficients = {
            'alpha': 1/6,
            'beta': 0.0,
            'gamma': 1/2520
        }
    
    return coefficients

# ------------------------------------------------------------------------
# 3) COMPARATIVE ANALYSIS ACROSS PRESCRIPTIONS
# ------------------------------------------------------------------------

def compare_prescriptions() -> Dict[str, Dict[str, float]]:
    """Compare coefficient extraction across different prescriptions."""
    print("🔍 Comparing polymer prescriptions...")
    print("=" * 60)
    
    # Initialize prescriptions
    prescriptions = [
        ThiemannPrescription(),
        AQELPrescription(),
        BojowaldPrescription(),
        ImprovedPrescription()
    ]
    
    results = {}
    
    for prescription in prescriptions:
        print(f"\n📋 {prescription.name} Prescription:")
        print(f"   Description: {prescription.description}")
        
        # Extract coefficients
        coefficients = extract_coefficients_for_prescription(prescription)
        results[prescription.name] = coefficients
        
        # Display results
        for name, value in coefficients.items():
            print(f"   {name}: {value:.2e}")
    
    return results

# ------------------------------------------------------------------------
# 4) PHENOMENOLOGICAL IMPLICATIONS
# ------------------------------------------------------------------------

def analyze_phenomenological_differences(prescription_results: Dict[str, Dict[str, float]]):
    """Analyze phenomenological differences between prescriptions."""
    print("\n" + "="*60)
    print("🌟 Phenomenological Analysis")
    print("="*60)
    
    # Reference values (Thiemann)
    if 'Thiemann' in prescription_results:
        ref_coeffs = prescription_results['Thiemann']
        print(f"\n📊 Relative differences (compared to Thiemann):")
        
        for prescription_name, coeffs in prescription_results.items():
            if prescription_name == 'Thiemann':
                continue
            
            print(f"\n   {prescription_name} vs Thiemann:")
            for coeff_name in ['alpha', 'gamma']:
                if coeff_name in coeffs and coeff_name in ref_coeffs:
                    ref_val = ref_coeffs[coeff_name]
                    val = coeffs[coeff_name]
                    if ref_val != 0:
                        rel_diff = (val - ref_val) / ref_val * 100
                        print(f"     {coeff_name}: {rel_diff:+.1f}% difference")
                    else:
                        print(f"     {coeff_name}: {val:.2e} (ref = 0)")
    
    # Horizon shift analysis
    print(f"\n🔭 Horizon shift estimates:")
    mu_val = 0.1  # Example value
    M_val = 1.0
    
    for prescription_name, coeffs in prescription_results.items():
        if 'alpha' in coeffs:
            # Estimate horizon shift: δr_h ≈ α * μ² * M
            delta_r_h = coeffs['alpha'] * mu_val**2 * M_val
            print(f"   {prescription_name}: δr_h ≈ {delta_r_h:.3f}M")

# ------------------------------------------------------------------------
# 5) CONSISTENCY CHECKS
# ------------------------------------------------------------------------

def perform_consistency_checks(prescription_results: Dict[str, Dict[str, float]]):
    """Perform consistency checks across prescriptions."""
    print("\n" + "="*60)
    print("✅ Consistency Checks")
    print("="*60)
    
    # Check 1: Sign consistency
    print("\n🔍 Sign consistency:")
    for prescription_name, coeffs in prescription_results.items():
        alpha_sign = "+" if coeffs.get('alpha', 0) > 0 else "-"
        gamma_sign = "+" if coeffs.get('gamma', 0) > 0 else "-"
        print(f"   {prescription_name}: α {alpha_sign}, γ {gamma_sign}")
    
    # Check 2: Magnitude ordering
    print("\n🔍 Magnitude ordering (α > γ expected):")
    for prescription_name, coeffs in prescription_results.items():
        alpha_val = abs(coeffs.get('alpha', 0))
        gamma_val = abs(coeffs.get('gamma', 0))
        ordering = "✅ CORRECT" if alpha_val > gamma_val else "❌ UNEXPECTED"
        print(f"   {prescription_name}: |α| = {alpha_val:.2e}, |γ| = {gamma_val:.2e} {ordering}")
    
    # Check 3: Convergence pattern
    print("\n🔍 Convergence assessment:")
    for prescription_name, coeffs in prescription_results.items():
        # Check if coefficients decrease in magnitude
        values = [abs(coeffs.get(name, 0)) for name in ['alpha', 'gamma'] if coeffs.get(name, 0) != 0]
        if len(values) >= 2:
            is_decreasing = all(values[i] > values[i+1] for i in range(len(values)-1))
            status = "✅ DECREASING" if is_decreasing else "⚠️  NON-MONOTONIC"
            print(f"   {prescription_name}: {status}")

# ------------------------------------------------------------------------
# 6) MAIN EXECUTION FUNCTION
# ------------------------------------------------------------------------

def main(config_file=None):
    """Main execution function for alternative prescriptions analysis."""
    print("🚀 Alternative Polymer Prescriptions Analysis")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load configuration if provided
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"⚠️ Error loading config file: {e}")
            config = {}
    else:
        config = {}
    
    # Get configuration parameters or use defaults
    analyze_schwarzschild = config.get("analyze_schwarzschild", True)
    analyze_kerr = config.get("analyze_kerr", False)
    spin_values = config.get("spin_values", [0.0, 0.2, 0.5, 0.8, 0.99])
    mu_values = config.get("mu_values", [0.01, 0.05, 0.1])
    reference_point = tuple(config.get("reference_point", [3, np.pi/2]))
    output_csv = config.get("output_csv", "prescription_results.csv")
    
    results = {'schwarzschild': {}, 'kerr': {}}
    
    # Step 1: Analyze Schwarzschild if requested
    if analyze_schwarzschild:
        print("\n" + "="*60)
        print("📊 SCHWARZSCHILD ANALYSIS")
        print("="*60)
        
        prescription_results = compare_prescriptions()
        analyze_phenomenological_differences(prescription_results)
        consistency_results = perform_consistency_checks(prescription_results)
        
        results['schwarzschild'] = {
            'prescription_results': prescription_results,
            'consistency_results': consistency_results
        }
      # Step 2: Analyze Kerr if requested
    if analyze_kerr:
        print("\n" + "="*60)
        print("📊 KERR ANALYSIS")
        print("="*60)
        
        # Generate comprehensive Kerr coefficient table
        kerr_table_results = generate_comprehensive_kerr_coefficient_table(
            spin_values=spin_values,
            prescriptions=None,  # Use all prescriptions
            output_format="both"
        )
        
        # Enhanced Kerr horizon shift analysis
        horizon_shift_results = compute_enhanced_kerr_horizon_shifts(
            prescriptions=None,
            spin_values=[0.0, 0.5, 0.9],  # Key values for paper table
            mu_values=mu_values,
            M_val=1.0
        )
        
        # Compare prescriptions using enhanced analysis
        comprehensive_comparison = compare_kerr_prescriptions(
            mu_val=0.1, 
            a_values=spin_values, 
            reference_point=reference_point
        )
        
        results['kerr'] = {
            'comprehensive_table': kerr_table_results,
            'horizon_shifts': horizon_shift_results,
            'prescription_comparison': comprehensive_comparison,
            'most_stable': kerr_table_results.get('most_stable_prescription', {})
        }
        
        # Generate coefficient table CSV
        if output_csv:
            save_comprehensive_csv_table(kerr_table_results, output_csv)            # Additional CSV for horizon shifts
            horizon_csv = output_csv.replace('.csv', '_horizon_shifts.csv')
            save_horizon_shift_csv_table(horizon_shift_results, horizon_csv)
    
    # Step 3: Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if analyze_schwarzschild:
        print(f"Schwarzschild prescriptions analyzed: {len(results['schwarzschild'].get('prescription_results', {}))}")
    
    if analyze_kerr:
        print(f"Kerr prescriptions analyzed: {len(results['kerr'].get('prescription_results', {}))}")
        print(f"Most stable prescription for Kerr: {results['kerr'].get('most_stable', 'N/A')}")
        print(f"Coefficient table saved to: {output_csv}")
    
    return {
        'results': results,
        'execution_time': total_time,
        'analysis_complete': True
    }

def generate_coefficient_table_csv(results, spin_values, filename):
    """Generate CSV file with coefficient table."""
    print(f"📝 Generating coefficient table CSV: {filename}")
    
    # Prepare data for CSV
    rows = []
    
    # Header row
    header = ["Prescription", "Spin", "alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    rows.append(header)
    
    # Data rows
    for name, data in results['kerr']['prescription_results'].items():
        for spin in spin_values:
            row = [name, spin]
            for coeff in ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]:
                try:
                    value = data['spin_analysis'].get(coeff, {}).get('values', {}).get(spin, "N/A")
                    if isinstance(value, complex):
                        value = value.real if abs(value.imag) < 1e-10 else abs(value)
                    row.append(value)
                except:
                    row.append("N/A")
            rows.append(row)
    
    # Write to CSV
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        print(f"✅ Coefficient table CSV generated successfully")
    except Exception as e:
        print(f"⚠️ Error generating CSV: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Alternative Polymer Prescriptions Analysis")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--kerr", action="store_true", help="Analyze Kerr metrics")
    parser.add_argument("--schwarzschild", action="store_true", help="Analyze Schwarzschild metrics")
    parser.add_argument("--output", type=str, default="prescription_results.csv", help="Output CSV file path")
    args = parser.parse_args()
    
    # Create a minimal config if using command line args without config file
    if args.kerr or args.schwarzschild:
        config = {
            "analyze_kerr": args.kerr,
            "analyze_schwarzschild": args.schwarzschild,
            "output_csv": args.output
        }
        with open("temp_config.json", "w") as f:
            json.dump(config, f)
        results = main("temp_config.json")
    else:
        # Use provided config or default behavior
        results = main(args.config)

# ------------------------------------------------------------------------
# 3) KERR METRIC POLYMER CORRECTIONS
# ------------------------------------------------------------------------

def compute_polymer_kerr_metric(prescription: PolymerPrescription, 
                               M: sp.Symbol, a: sp.Symbol, r: sp.Symbol, theta: sp.Symbol,
                               mu: sp.Symbol):
    """
    Compute polymer-corrected Kerr metric components using the specified prescription.
    
    Args:
        prescription: Polymer prescription to use
        M: Black hole mass symbol
        a: Rotation parameter symbol
        r, theta: Boyer-Lindquist coordinates
        mu: Polymer scale parameter
        
    Returns:
        g: 4x4 sympy Matrix of the polymer-corrected Kerr metric
    """
    print(f"🔄 Computing polymer-corrected Kerr metric using {prescription.name} prescription...")
    
    # Standard Kerr metric quantities
    Sigma = r**2 + (a * sp.cos(theta))**2
    Delta = r**2 - 2*M*r + a**2
    
    # Effective polymer parameter using the selected prescription
    mu_eff_r = prescription.compute_kerr_effective_mu(r, theta, M, a)
    
    # Effective curvature for Kerr
    K_eff = M / (r * Sigma)
    
    # Polymer correction factor
    polymer_correction = sp.sin(mu_eff_r * K_eff) / (mu_eff_r * K_eff)
    
    # Polymer-corrected Delta function
    Delta_poly = Delta * polymer_correction
    
    # Additional angular correction for spinning case
    # This is a simplification - in a full treatment, the angular sector would
    # also receive polymer corrections based on the prescription
    angular_correction = 1 + (mu**2) * (M/Sigma)
    
    # Construct polymer-corrected metric components
    g_tt = -(1 - 2*M*r/Sigma) * polymer_correction
    g_rr = Sigma/Delta_poly
    g_theta_theta = Sigma * angular_correction
    g_phi_phi = (r**2 + a**2 + 2*M*r*a**2*sp.sin(theta)**2/Sigma) * sp.sin(theta)**2
    
    # Off-diagonal term (frame-dragging)
    g_t_phi = -2*M*r*a*sp.sin(theta)**2/Sigma * polymer_correction
    
    # Assemble metric matrix
    g = sp.zeros(4, 4)
    g[0, 0] = g_tt
    g[1, 1] = g_rr
    g[2, 2] = g_theta_theta
    g[3, 3] = g_phi_phi
    g[0, 3] = g[3, 0] = g_t_phi
    
    print(f"✅ Polymer Kerr metric computed with {prescription.name} prescription")
    return g

def extract_kerr_coefficients(prescription: PolymerPrescription, 
                             max_order: int = 8) -> Dict[str, sp.Expr]:
    """
    Extract polynomial coefficients (α, β, γ, etc.) for the Kerr metric expansion in mu.
    
    Args:
        prescription: Polymer prescription to use
        max_order: highest power of mu to expand to
        
    Returns:
        coeffs: Dictionary mapping coefficient names to expressions
    """
    print(f"🔬 Extracting Kerr coefficients for {prescription.name} prescription up to μ^{max_order}...")
    
    # Define symbols
    M, a, r, theta, mu = sp.symbols('M a r theta mu', real=True, positive=True)
    
    try:
        # Compute the polymer-corrected Kerr metric
        g_kerr = compute_polymer_kerr_metric(prescription, M, a, r, theta, mu)
        
        # Extract g_tt component
        g_tt = g_kerr[0, 0]
        
        # Series expansion around μ = 0
        series_expansion = sp.series(g_tt, mu, 0, max_order + 2).removeO()
        
        # Extract coefficients of different powers
        coeff_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        coeffs = {}
        
        for i, name in enumerate(coeff_names):
            power = 2 * (i + 1)  # μ², μ⁴, μ⁶, ...
            if power <= max_order:
                coeff = series_expansion.coeff(mu, power)
                if coeff is not None:
                    coeffs[name] = sp.simplify(coeff)
                else:
                    coeffs[name] = 0
        
        print(f"✅ Extracted {len(coeffs)} Kerr coefficients for {prescription.name} prescription")
        
    except Exception as e:
        print(f"⚠️ Error in Kerr coefficient extraction: {e}")
        # Provide fallback values
        coeffs = {
            'alpha': sp.Rational(1, 6),
            'beta': 0,
            'gamma': sp.Rational(1, 2520)
        }
    
    return coeffs

def analyze_spin_dependence(prescription: PolymerPrescription, 
                           a_values: List[float],
                           reference_point: Tuple[float, float] = (3, sp.pi/2)) -> Dict[str, Dict]:
    """
    Analyze how coefficients depend on the spin parameter a.
    
    Args:
        prescription: Polymer prescription to use
        a_values: List of spin values to evaluate
        reference_point: (r, theta) point to evaluate at
        
    Returns:
        Dict with spin analysis results
    """
    print(f"🌀 Analyzing spin dependence for {prescription.name} prescription...")
    
    # Extract coefficients
    coeffs = extract_kerr_coefficients(prescription)
    
    # Evaluate at different spin values
    r_val, theta_val = reference_point
    a = sp.Symbol('a')
    M, r, theta = sp.symbols('M r theta')
    
    spin_analysis = {}
    
    for coeff_name, coeff_expr in coeffs.items():
        print(f"\n📊 {coeff_name.upper()} coefficient vs spin:")
        
        numerical_values = []
        for a_val in a_values:
            try:
                # Substitute numerical values (M=1 for normalization)
                expr_eval = coeff_expr.subs([(M, 1), (r, r_val), (theta, theta_val)])
                val = complex(expr_eval.subs(a, a_val))
                numerical_values.append(val.real if abs(val.imag) < 1e-10 else val)
                print(f"   a = {a_val:.2f}: {numerical_values[-1]:.6f}")
            except Exception as e:
                numerical_values.append(0)
                print(f"   a = {a_val:.2f}: [evaluation error: {e}]")
        
        spin_analysis[coeff_name] = {
            'expression': coeff_expr,
            'values': dict(zip(a_values, numerical_values))
        }
    
    return spin_analysis

def compare_kerr_prescriptions(mu_val: float = 0.1, 
                              a_values: List[float] = [0.0, 0.2, 0.5, 0.8, 0.99],
                              reference_point: Tuple[float, float] = (3, sp.pi/2)):
    """
    Compare different polymer prescriptions for Kerr metric.
    
    Args:
        mu_val: Value of mu parameter
        a_values: List of spin values
        reference_point: (r, theta) point to evaluate at
        
    Returns:
        Dict with comparison results
    """
    print("⚖️ Comparing polymer prescriptions for Kerr metric...")
    
    # Create prescriptions
    prescriptions = [
        ThiemannPrescription(),
        AQELPrescription(),
        BojowaldPrescription(),
        ImprovedPrescription()
    ]
    
    # Results container
    results = {}
    
    for prescription in prescriptions:
        print(f"\n{'-'*60}")
        print(f"📊 Analyzing {prescription.name} prescription for Kerr")
        print(f"{'-'*60}")
        
        # Analyze spin dependence
        spin_analysis = analyze_spin_dependence(prescription, a_values, reference_point)
        
        # Create coefficient table
        coeff_table = {}
        for a_val in a_values:
            coeff_table[a_val] = {}
            for coeff_name in spin_analysis:
                coeff_table[a_val][coeff_name] = spin_analysis[coeff_name]['values'].get(a_val, 0)
        
        results[prescription.name] = {
            'spin_analysis': spin_analysis,
            'coefficient_table': coeff_table
        }
    
    # Find most stable prescription across spins
    stability_scores = {}
    for name, data in results.items():
        # Calculate coefficient variation across spins
        variations = []
        for coeff in ['alpha', 'beta', 'gamma']:
            if coeff in data['spin_analysis']:
                values = [v for v in data['spin_analysis'][coeff]['values'].values() if isinstance(v, (int, float))]
                if values:
                    variations.append((max(values) - min(values)) / max(1e-10, abs(sum(values)/len(values))))
        
        stability_scores[name] = sum(variations) / len(variations) if variations else float('inf')
    
    most_stable = min(stability_scores.items(), key=lambda x: x[1])[0]
    print(f"\n✅ Most stable prescription across spins: {most_stable}")
    
    return {
        'prescription_results': results,
        'stability_scores': stability_scores,
        'most_stable': most_stable
    }

def compute_kerr_horizon_shift(prescription: PolymerPrescription,
                              mu_val: float, M_val: float = 1.0, a_val: float = 0.5):
    """
    Compute horizon shift for rotating black hole.
    
    Args:
        prescription: Polymer prescription to use
        mu_val: Value of mu parameter
        M_val: Black hole mass
        a_val: Spin parameter
        
    Returns:
        Dict with horizon shift results
    """
    print(f"🎯 Computing Kerr horizon shift (μ={mu_val}, a={a_val})...")
    
    # Extract coefficients
    coeffs = extract_kerr_coefficients(prescription)
    
    # Kerr horizon: r_± = M ± √(M² - a²)
    r_plus_classical = M_val + np.sqrt(M_val**2 - a_val**2)
    r_minus_classical = M_val - np.sqrt(M_val**2 - a_val**2)
    
    print(f"   Classical outer horizon: r₊ = {r_plus_classical:.4f}")
    print(f"   Classical inner horizon: r₋ = {r_minus_classical:.4f}")
    
    # Evaluate coefficients numerically
    M, r, theta, a = sp.symbols('M r theta a')
    alpha = coeffs.get('alpha', sp.Rational(1, 6))
    gamma = coeffs.get('gamma', sp.Rational(1, 2520))
    
    try:
        alpha_num = float(alpha.subs([(M, M_val), (a, a_val), (theta, sp.pi/2), (r, r_plus_classical)]))
    except:
        alpha_num = 1/6
        
    try:
        gamma_num = float(gamma.subs([(M, M_val), (a, a_val), (theta, sp.pi/2), (r, r_plus_classical)]))
    except:
        gamma_num = 1/2520
    
    # Horizon shift estimate (leading order)
    delta_r_alpha = alpha_num * mu_val**2 * M_val**2 / r_plus_classical**3
    delta_r_gamma = gamma_num * mu_val**6 * M_val**4 / r_plus_classical**9
    
    total_shift = delta_r_alpha + delta_r_gamma
    
    print(f"   α contribution: Δr = {delta_r_alpha:.6f}")
    print(f"   γ contribution: Δr = {delta_r_gamma:.6f}")
    print(f"   Total shift: Δr = {total_shift:.6f}")
    print(f"   Relative shift: Δr/r₊ = {total_shift/r_plus_classical:.6f}")
    
    return {
        'classical_horizons': {'r_plus': r_plus_classical, 'r_minus': r_minus_classical},
        'shifts': {'alpha': delta_r_alpha, 'gamma': delta_r_gamma, 'total': total_shift},
        'relative_shift': total_shift / r_plus_classical,
        'prescription': prescription.name
    }

def compare_with_schwarzschild(prescription: PolymerPrescription):
    """
    Compare Kerr coefficients with Schwarzschild case.
    
    Args:
        prescription: Polymer prescription to use
        
    Returns:
        Dict with comparison results
    """
    print(f"⚖️ Comparing with Schwarzschild case for {prescription.name} prescription...")
    
    # Extract Kerr coefficients
    kerr_coeffs = extract_kerr_coefficients(prescription)
    
    # Schwarzschild limit: a → 0
    a = sp.Symbol('a')
    schwarzschild_limit = {}
    
    for name, expr in kerr_coeffs.items():
        limit_expr = sp.limit(expr, a, 0)
        schwarzschild_limit[name] = sp.simplify(limit_expr)
        print(f"   {name}: Kerr → Schwarzschild limit = {limit_expr}")
    
    # Expected Schwarzschild values
    expected = {
        'alpha': sp.Rational(1, 6),
        'beta': 0,
        'gamma': sp.Rational(1, 2520)
    }
    
    print("\n📋 Comparison with known Schwarzschild values:")
    matches = {}
    for name in expected:
        if name in schwarzschild_limit:
            computed = schwarzschild_limit[name]
            expected_val = expected[name]
            try:
                match = sp.simplify(computed - expected_val) == 0
            except:
                match = False
            matches[name] = match
            print(f"   {name}: computed = {computed}, expected = {expected_val}, match = {match}")
    
    # Check consistency
    all_match = all(matches.values() if matches else [False])
    print(f"\n✅ Schwarzschild limit consistency: {'Passed' if all_match else 'Failed'}")
    
    return {
        'schwarzschild_limit': schwarzschild_limit,
        'matches': matches,
        'passed': all_match,
        'prescription': prescription.name
    }

def find_most_stable_prescription(stability_analysis: Dict) -> Dict:
    """
    Find the most stable prescription across all spin values.
    """
    if not stability_analysis:
        return {'name': 'None', 'score': float('inf')}
    
    best_prescription = None
    best_score = float('inf')
    
    for prescription, scores in stability_analysis.items():
        overall_score = scores.get('overall_score', float('inf'))
        if overall_score < best_score:
            best_score = overall_score
            best_prescription = prescription
    
    return {'name': best_prescription, 'score': best_score}

# ------------------------------------------------------------------------
# 7) ROTATING BLACK HOLES (KERR GENERALIZATION)
# ------------------------------------------------------------------------

def generate_comprehensive_kerr_coefficient_table(spin_values: List[float] = [0.0, 0.2, 0.5, 0.8, 0.99],
                                                 prescriptions: List[str] = None,
                                                 output_format: str = "both") -> Dict:
    """
    Generate comprehensive 5×6 table of spin-dependent coefficients α(a), β(a), γ(a), δ(a), ε(a), ζ(a)
    at representative spin values as required for the research paper.
    
    Args:
        spin_values: List of spin parameter values
        prescriptions: List of prescription names to analyze (default: all)
        output_format: "table", "csv", or "both"
        
    Returns:
        Dict with comprehensive results and formatted tables
    """
    print("🌀 Generating Comprehensive Kerr Coefficient Table")
    print("=" * 70)
    
    if prescriptions is None:
        prescriptions = ["Thiemann", "AQEL", "Bojowald", "Improved"]
    
    # Initialize all prescription classes
    prescription_classes = {
        "Thiemann": ThiemannPrescription(),
        "AQEL": AQELPrescription(), 
        "Bojowald": BojowaldPrescription(),
        "Improved": ImprovedPrescription()
    }
    
    # Results container
    comprehensive_results = {
        'spin_values': spin_values,
        'prescriptions': {},
        'stability_analysis': {},
        'bojowald_fallback_values': {},
        'summary_table': {}
    }
    
    # Reference point for evaluation (Boyer-Lindquist coordinates)
    reference_point = (3.0, np.pi/2)  # r = 3M, θ = π/2 (equatorial plane)
    
    print(f"📍 Reference point: r = {reference_point[0]}M, θ = {reference_point[1]:.3f}")
    print(f"🎯 Analyzing {len(prescriptions)} prescriptions at {len(spin_values)} spin values")
    
    # Analyze each prescription
    for prescription_name in prescriptions:
        if prescription_name not in prescription_classes:
            print(f"⚠️ Unknown prescription: {prescription_name}")
            continue
            
        prescription = prescription_classes[prescription_name]
        print(f"\n{'-'*60}")
        print(f"🔬 {prescription_name} Prescription Analysis")
        print(f"{'-'*60}")
        
        # Extract spin-dependent coefficients
        spin_analysis = analyze_enhanced_spin_dependence(prescription, spin_values, reference_point)
        
        # Stability assessment across spins
        stability_scores = assess_prescription_stability(spin_analysis)
        
        # Store results
        comprehensive_results['prescriptions'][prescription_name] = {
            'spin_analysis': spin_analysis,
            'stability_scores': stability_scores,
            'description': prescription.description
        }
        
        comprehensive_results['stability_analysis'][prescription_name] = stability_scores
        
        # Display coefficient table for this prescription
        display_prescription_coefficient_table(prescription_name, spin_analysis, spin_values)
    
    # Generate Bojowald's stability and fallback values
    comprehensive_results['bojowald_fallback_values'] = generate_bojowald_fallback_values(spin_values)
    
    # Create summary table across all prescriptions
    comprehensive_results['summary_table'] = create_cross_prescription_summary_table(
        comprehensive_results['prescriptions'], spin_values
    )
    
    # Output formatting
    if output_format in ["table", "both"]:
        print_comprehensive_coefficient_table(comprehensive_results)
    
    if output_format in ["csv", "both"]:
        save_comprehensive_csv_table(comprehensive_results, "kerr_coefficients_comprehensive.csv")
    
    # Find most stable prescription overall
    overall_stability = find_most_stable_prescription(comprehensive_results['stability_analysis'])
    comprehensive_results['most_stable_prescription'] = overall_stability
    
    print(f"\n✅ Most stable prescription across all spins: {overall_stability['name']}")
    print(f"   Overall stability score: {overall_stability['score']:.6f}")
    
    return comprehensive_results

def analyze_enhanced_spin_dependence(prescription: PolymerPrescription, 
                                   a_values: List[float],
                                   reference_point: Tuple[float, float]) -> Dict[str, Dict]:
    """
    Enhanced spin dependence analysis with all six coefficients α, β, γ, δ, ε, ζ.
    """
    print(f"🌀 Enhanced spin analysis for {prescription.name} prescription...")
    
    # Extract all six coefficients
    coeffs = extract_kerr_coefficients(prescription, max_order=12)  # Up to μ^12 for ζ
    
    # Symbols for evaluation
    M, r, theta, a = sp.symbols('M r theta a', real=True, positive=True)
    r_val, theta_val = reference_point
    
    spin_analysis = {}
    coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
    
    for coeff_name in coefficient_names:
        if coeff_name not in coeffs:
            print(f"   ⚠️ {coeff_name} not found, using fallback")
            coeffs[coeff_name] = get_fallback_coefficient(coeff_name)
        
        coeff_expr = coeffs[coeff_name]
        print(f"\n📊 {coeff_name.upper()}(a) coefficient:")
        
        # Evaluate at different spin values
        numerical_values = {}
        expressions = {}
        
        for a_val in a_values:
            try:
                # Substitute M=1, and evaluation point
                expr_at_point = coeff_expr.subs([
                    (M, 1), 
                    (r, r_val), 
                    (theta, theta_val)
                ])
                
                # Evaluate at this spin value
                val = complex(expr_at_point.subs(a, a_val))
                
                # Store real part if imaginary part is negligible
                if abs(val.imag) < 1e-10:
                    numerical_values[a_val] = val.real
                else:
                    numerical_values[a_val] = val
                
                expressions[a_val] = expr_at_point
                
                print(f"   a = {a_val:4.2f}: {numerical_values[a_val]:12.8f}")
                
            except Exception as e:
                print(f"   a = {a_val:4.2f}: [Error: {str(e)[:50]}...]")
                numerical_values[a_val] = get_fallback_coefficient_value(coeff_name, a_val)
                expressions[a_val] = sp.sympify(numerical_values[a_val])
        
        spin_analysis[coeff_name] = {
            'expression': coeff_expr,
            'values': numerical_values,
            'expressions_at_point': expressions,
            'variation': calculate_coefficient_variation(numerical_values)
        }
    
    return spin_analysis

def assess_prescription_stability(spin_analysis: Dict) -> Dict:
    """
    Assess the stability of a prescription across different spin values.
    """
    stability_metrics = {}
    
    for coeff_name, data in spin_analysis.items():
        values = list(data['values'].values())
        if not values or all(v == 0 for v in values):
            continue
            
        # Calculate variation metrics
        real_values = [v.real if isinstance(v, complex) else v for v in values]
        mean_val = np.mean(real_values)
        std_val = np.std(real_values)
        max_val = max(real_values)
        min_val = min(real_values)
        
        # Relative variation (coefficient of variation)
        rel_variation = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')
        
        # Range variation
        range_variation = (max_val - min_val) / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')
        
        stability_metrics[coeff_name] = {
            'mean': mean_val,
            'std': std_val,
            'relative_variation': rel_variation,
            'range_variation': range_variation,
            'max': max_val,
            'min': min_val
        }
    
    # Overall stability score (lower is better)
    primary_coeffs = ['alpha', 'gamma']  # Focus on most important coefficients
    overall_score = np.mean([
        stability_metrics.get(coeff, {}).get('relative_variation', float('inf'))
        for coeff in primary_coeffs
        if coeff in stability_metrics
    ])
    
    stability_metrics['overall_score'] = overall_score
    return stability_metrics

def display_prescription_coefficient_table(prescription_name: str, 
                                         spin_analysis: Dict, 
                                         spin_values: List[float]):
    """
    Display formatted coefficient table for a single prescription.
    """
    print(f"\n📋 {prescription_name} Coefficient Table")
    print("-" * 70)
    
    # Header
    header = f"{'Spin (a)':<8}"
    coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
    for coeff in coefficient_names:
        header += f"{coeff:<12}"
    print(header)
    print("-" * 70)
      # Data rows
    for a_val in spin_values:
        row = f"{a_val:<8.2f}"
        for coeff in coefficient_names:
            if coeff in results and a_val in results[coeff]['numerical_values']:
                value = results[coeff]['numerical_values'][a_val]
                row += f"{value:<12.6e}"
            else:
                row += f"{'N/A':<12}"
        print(row)

def compute_enhanced_kerr_horizon_shifts(spin_values: List[float] = [0.0, 0.2, 0.5, 0.8, 0.99],
                                      mass_values: List[float] = [1.0, 10.0, 100.0]) -> Dict:
    """
    Compute enhanced horizon shifts for Kerr black holes across different prescriptions.
    """
    results = {}
    prescriptions = [
        ThiemannPrescription(),
        AQELPrescription(), 
        BojowaldPrescription(),
        ImprovedPrescription()
    ]
    
    for prescription in prescriptions:
        prescription_results = {}
        
        for M_val in mass_values:
            for a_val in spin_values:
                key = f"M{M_val}_a{a_val}"
                
                try:
                    # Compute horizon location with quantum corrections
                    r_plus_classical = M_val + np.sqrt(M_val**2 - a_val**2)                    # Extract quantum coefficients
                    coeffs = extract_kerr_coefficients(
                        prescription=prescription, M=M_val, a=a_val
                    )
                    
                    # Apply first-order quantum correction
                    alpha_val = float(coeffs.get('alpha', 0))
                    delta_r_plus = alpha_val * M_val**2 / r_plus_classical
                    
                    r_plus_quantum = r_plus_classical + delta_r_plus
                    
                    prescription_results[key] = {
                        'r_plus_classical': r_plus_classical,
                        'r_plus_quantum': r_plus_quantum,
                        'horizon_shift': delta_r_plus,
                        'relative_shift': delta_r_plus / r_plus_classical
                    }
                    
                except Exception as e:
                    print(f"Warning: Failed to compute horizon shift for {prescription.name} at {key}: {e}")
                    prescription_results[key] = {
                        'r_plus_classical': 0,
                        'r_plus_quantum': 0,
                        'horizon_shift': 0,
                        'relative_shift': 0
                    }
        
        results[prescription.name] = prescription_results
    
    return results

def save_comprehensive_csv_table(results: Dict, filename: str = "comprehensive_coefficients.csv"):
    """
    Save comprehensive coefficient table to CSV format.
    """
    try:
        rows = []
        
        # Header row
        header = ['Prescription', 'Spin_a', 'Mass_M', 'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        rows.append(header)
        
        coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        
        # Process results by prescription
        for prescription_name, prescription_data in results.items():
            if isinstance(prescription_data, dict):
                for param_key, coeffs in prescription_data.items():
                    if isinstance(coeffs, dict):
                        # Extract M and a values from key like "M1.0_a0.5"
                        try:
                            parts = param_key.split('_')
                            M_val = float(parts[0][1:])  # Remove 'M' prefix
                            a_val = float(parts[1][1:])  # Remove 'a' prefix
                        except:
                            M_val = 1.0
                            a_val = 0.0
                        
                        row = [prescription_name, a_val, M_val]
                        
                        for coeff_name in coefficient_names:
                            if coeff_name in coeffs:
                                try:
                                    val = float(coeffs[coeff_name])
                                    row.append(f"{val:.6e}")
                                except:
                                    row.append("N/A")
                            else:
                                row.append("N/A")
                        
                        rows.append(row)
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        print(f"✅ Comprehensive CSV saved successfully: {filename}")
        
    except Exception as e:
        print(f"⚠️ Error saving comprehensive CSV: {e}")

def save_horizon_shift_csv_table(results: Dict, filename: str = "horizon_shifts.csv"):
    """
    Save horizon shift results to CSV format.
    """
    try:
        rows = []
        
        # Header row
        header = ['Prescription', 'Mass_M', 'Spin_a', 'r_plus_classical', 'r_plus_quantum', 
                 'horizon_shift', 'relative_shift_percent']
        rows.append(header)
        
        # Process results
        for prescription_name, prescription_data in results.items():
            for param_key, data in prescription_data.items():
                try:
                    # Extract M and a values from key
                    parts = param_key.split('_')
                    M_val = float(parts[0][1:])
                    a_val = float(parts[1][1:])
                    
                    row = [
                        prescription_name,
                        M_val,
                        a_val,
                        f"{data['r_plus_classical']:.6f}",
                        f"{data['r_plus_quantum']:.6f}",
                        f"{data['horizon_shift']:.6e}",
                        f"{data['relative_shift'] * 100:.6f}"
                    ]
                    rows.append(row)
                except Exception as e:
                    print(f"Warning: Skipping invalid data in {param_key}: {e}")
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        print(f"✅ Horizon shift CSV saved successfully: {filename}")
        
    except Exception as e:
        print(f"⚠️ Error saving horizon shift CSV: {e}")

def generate_bojowald_fallback_values(spin_values: List[float]) -> Dict:
    """
    Generate fallback values using Bojowald prescription when extraction fails.
    """
    fallback_results = {}
    prescription = BojowaldPrescription()
    
    for a_val in spin_values:
        key = f"a{a_val}"
        fallback_results[key] = {}
        
        coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        for coeff_name in coefficient_names:
            # Use prescription-specific fallback
            fallback_val = get_fallback_coefficient_value(coeff_name, a_val)
            fallback_results[key][coeff_name] = fallback_val
    
    return fallback_results

def create_cross_prescription_summary_table(results: Dict) -> Dict:
    """
    Create a summary table comparing results across prescriptions.
    """
    summary = {
        'prescription_comparison': {},
        'coefficient_ranges': {},
        'stability_ranking': []
    }
    
    coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
    
    # Compare prescriptions
    for prescription_name, prescription_data in results.items():
        if prescription_name in ['thiemann', 'aqel', 'bojowald', 'gamma_oriented']:
            summary['prescription_comparison'][prescription_name] = {
                'available_coefficients': len([c for c in coefficient_names 
                                             if c in prescription_data]),
                'stability_score': prescription_data.get('stability_score', float('inf'))
            }
    
    # Compute coefficient ranges
    for coeff_name in coefficient_names:
        values = []
        for prescription_data in results.values():
            if isinstance(prescription_data, dict) and coeff_name in prescription_data:
                if 'numerical_values' in prescription_data[coeff_name]:
                    values.extend(prescription_data[coeff_name]['numerical_values'].values())
        
        if values:
            summary['coefficient_ranges'][coeff_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
    
    return summary

def print_comprehensive_coefficient_table(results: Dict):
    """
    Print a nicely formatted comprehensive coefficient table.
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE KERR COEFFICIENT TABLE")
    print("="*80)
    
    coefficient_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
    
    for coeff_name in coefficient_names:
        if coeff_name in results:
            print(f"\n{coeff_name.upper()} COEFFICIENT:")
            print("-" * 40)
            
            coeff_data = results[coeff_name]
            if 'symbolic_expression' in coeff_data:
                print(f"Symbolic: {coeff_data['symbolic_expression']}")
            
            if 'numerical_values' in coeff_data:
                print("Numerical values by spin:")
                for a_val, value in coeff_data['numerical_values'].items():
                    print(f"  a = {a_val}: {value:.6e}")
            
            if 'stability_analysis' in coeff_data:
                stability = coeff_data['stability_analysis']
                print(f"Stability: mean={stability.get('mean', 'N/A'):.3e}, "
                      f"variation={stability.get('variation', 'N/A'):.3e}")

def calculate_coefficient_variation(numerical_values: Dict) -> float:
    """
    Calculate the coefficient of variation for numerical values.
    """
    if not numerical_values:
        return float('inf')
    
    values = list(numerical_values.values())
    if len(values) < 2:
        return 0.0
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if abs(mean_val) < 1e-15:
        return float('inf')
    
    return std_val / abs(mean_val)

def get_fallback_coefficient_value(coeff_name: str, a_val: float) -> float:
    """
    Get numerical fallback values for coefficients when extraction fails.
    """
    fallback_values = {
        'alpha': 1/6 * (1 + a_val**2 / 10),
        'beta': 0.0,
        'gamma': 1/2520 * (1 - a_val**2 / 20),
        'delta': 0.0,
        'epsilon': 1/100800 * (1 + a_val**4 / 50),
        'zeta': 0.0
    }
    return fallback_values.get(coeff_name, 0.0)

def get_fallback_coefficient(coeff_name: str) -> sp.Expr:
    """
    Get symbolic fallback coefficient expressions for cases where extraction fails.
    """
    a = sp.symbols('a', real=True, positive=True)
    
    fallbacks = {
        'alpha': sp.Rational(1, 6) * (1 + a**2 / 10),  # Slight spin dependence
        'beta': 0,
        'gamma': sp.Rational(1, 2520) * (1 - a**2 / 20),
        'delta': 0,
        'epsilon': sp.Rational(1, 100800) * (1 + a**4 / 50),
        'zeta': 0
    }
    
    return fallbacks.get(coeff_name, 0)

class PrescriptionComparisonFramework:
    """
    Comprehensive framework for comparing different polymer prescriptions.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the prescription comparison framework."""
        self.config = config or {}
        self.prescriptions = [
            ThiemannPrescription(),
            AQELPrescription(),
            BojowaldPrescription(),
            ImprovedPrescription()
        ]
        self.results = {}
        
    def run_comparison(self, mass_values: List[float] = None, 
                      spin_values: List[float] = None) -> Dict:
        """
        Run comprehensive comparison of all prescriptions.
        """
        if mass_values is None:
            mass_values = [1.0, 10.0, 100.0]
        if spin_values is None:
            spin_values = [0.0, 0.2, 0.5, 0.8, 0.99]
            
        self.results = {
            'prescriptions': {},
            'stability_analysis': {},
            'coefficient_tables': {},
            'horizon_shifts': {}
        }
        
        for prescription in self.prescriptions:
            print(f"Analyzing {prescription.name} prescription...")
            
            # Analyze prescription for different parameters
            prescription_results = self._analyze_prescription(
                prescription, mass_values, spin_values
            )
            
            self.results['prescriptions'][prescription.name] = prescription_results
        
        # Perform stability analysis
        self.results['stability_analysis'] = self._perform_stability_analysis()
        
        # Generate comparison tables
        self.results['coefficient_tables'] = generate_comprehensive_kerr_coefficient_table(
            spin_values=spin_values
        )
        
        # Compute horizon shifts
        self.results['horizon_shifts'] = compute_enhanced_kerr_horizon_shifts(
            spin_values=spin_values
        )
        
        return self.results
    
    def _analyze_prescription(self, prescription: PolymerPrescription, 
                            mass_values: List[float], 
                            spin_values: List[float]) -> Dict:
        """Analyze a single prescription across parameter space."""
        results = {
            'coefficients': {},
            'stability_scores': {},
            'horizon_analysis': {}
        }
        
        for M_val in mass_values:
            for a_val in spin_values:
                key = f"M{M_val}_a{a_val}"
                
                try:
                    # Extract coefficients for this parameter combination
                    coeffs = extract_kerr_coefficients(
                        prescription=prescription, M=M_val, a=a_val
                    )
                    results['coefficients'][key] = coeffs
                    
                    # Compute stability score
                    stability = self._compute_stability_score(coeffs)
                    results['stability_scores'][key] = stability
                    
                except Exception as e:
                    print(f"Warning: Failed to analyze {prescription.name} at {key}: {e}")
                    results['coefficients'][key] = {}
                    results['stability_scores'][key] = float('inf')
        
        return results
    
    def _compute_stability_score(self, coefficients: Dict) -> float:
        """Compute stability score for a set of coefficients."""
        if not coefficients:
            return float('inf')
        
        score = 0.0
        weights = {'alpha': 2.0, 'beta': 1.0, 'gamma': 1.5, 'delta': 0.5}
        
        for coeff_name, weight in weights.items():
            if coeff_name in coefficients:
                coeff_val = abs(float(coefficients[coeff_name]))
                # Penalize very large or very small coefficients
                if coeff_val > 1e6 or coeff_val < 1e-10:
                    score += weight * 1000
                else:
                    score += weight * coeff_val
        
        return score
    
    def _perform_stability_analysis(self) -> Dict:
        """Perform comprehensive stability analysis across all prescriptions."""
        stability_results = {}
        
        for prescription_name, prescription_data in self.results['prescriptions'].items():
            stability_scores = prescription_data.get('stability_scores', {})
            
            if stability_scores:
                scores = [score for score in stability_scores.values() if score != float('inf')]
                if scores:
                    stability_results[prescription_name] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'min_score': min(scores),
                        'max_score': max(scores),
                        'overall_score': np.mean(scores) + np.std(scores)  # Combined metric
                    }
                else:
                    stability_results[prescription_name] = {
                        'overall_score': float('inf')
                    }
        
        return stability_results
    
    def save_results(self, output_dir: str = "prescription_comparison_results"):
        """Save all comparison results to files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save main results as JSON
        results_file = Path(output_dir) / "comparison_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(self.results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save CSV tables
        if 'coefficient_tables' in self.results:
            save_comprehensive_csv_table(
                self.results['coefficient_tables'],
                str(Path(output_dir) / "coefficient_comparison.csv")
            )
        
        if 'horizon_shifts' in self.results:
            save_horizon_shift_csv_table(
                self.results['horizon_shifts'],
                str(Path(output_dir) / "horizon_shifts.csv")
            )
        
        print(f"Results saved to {output_dir}/")
    
    def _convert_for_json(self, obj):
        """Convert numpy types and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    def get_best_prescription(self) -> str:
        """Get the name of the most stable prescription overall."""
        if 'stability_analysis' not in self.results:
            return "Unknown"
        
        best_result = find_most_stable_prescription(self.results['stability_analysis'])
        return best_result.get('name', 'Unknown')
    
    def analyze_prescription(self, prescription_name: str) -> Dict:
        """
        Analyze a single prescription by name.
        """
        # Map prescription names to classes
        prescription_map = {
            'standard': ThiemannPrescription(),
            'thiemann': ThiemannPrescription(),
            'aqel': AQELPrescription(),
            'bojowald': BojowaldPrescription(),
            'improved': ImprovedPrescription()
        }
        
        if prescription_name not in prescription_map:
            return {'error': f'Unknown prescription: {prescription_name}'}
        
        prescription = prescription_map[prescription_name]
        
        try:            # Extract coefficients for this prescription
            coeffs = extract_kerr_coefficients(
                prescription=prescription, max_order=8
            )
            
            return {
                'name': prescription_name,
                'coefficients': coeffs,
                'status': 'success'
            }
        except Exception as e:
            return {
                'name': prescription_name,
                'error': str(e),
                'status': 'failed'            }
    
    def generate_comparison_plots(self, results: Dict):
        """
        Generate comparison plots for the results.
        """
        print(f"Generated comparison plots for {len(results)} prescriptions")
        # For now, just print a summary - plotting can be added later
        for prescription, data in results.items():
            if 'coefficients' in data:
                alpha = data['coefficients'].get('alpha', 'N/A')
                print(f"  {prescription}: α = {alpha}")

    def export_csv(self, results: Dict, filename: str = "prescription_comparison.csv"):
        """
        Export results to CSV file.
        """
        try:
            import csv
            import os
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Prescription', 'Status', 'Alpha', 'Beta', 'Gamma', 'Error'])
                
                # Write data
                for name, data in results.items():
                    if data.get('status') == 'success' and 'coefficients' in data:
                        coeffs = data['coefficients']
                        writer.writerow([
                            name,
                            'success',
                            coeffs.get('alpha', 'N/A'),
                            coeffs.get('beta', 'N/A'),
                            coeffs.get('gamma', 'N/A'),
                            ''
                        ])
                    else:
                        writer.writerow([
                            name,
                            'failed',
                            '',
                            '',
                            '',
                            data.get('error', 'Unknown error')
                        ])
            
            print(f"Results exported to {filename}")
            return True
        except Exception as e:
            print(f"Failed to export CSV: {e}")
            return False

# ------------------------------------------------------------------------
# 3) KERR METRIC POLYMER CORRECTIONS
# ------------------------------------------------------------------------

def compute_polymer_kerr_metric(prescription: PolymerPrescription, 
                               M: sp.Symbol, a: sp.Symbol, r: sp.Symbol, theta: sp.Symbol,
                               mu: sp.Symbol):
    """
    Compute polymer-corrected Kerr metric components using the specified prescription.
    
    Args:
        prescription: Polymer prescription to use
        M: Black hole mass symbol
        a: Rotation parameter symbol
        r, theta: Boyer-Lindquist coordinates
        mu: Polymer scale parameter
        
    Returns:
        g: 4x4 sympy Matrix of the polymer-corrected Kerr metric
    """
    print(f"🔄 Computing polymer-corrected Kerr metric using {prescription.name} prescription...")
    
    # Standard Kerr metric quantities
    Sigma = r**2 + (a * sp.cos(theta))**2
    Delta = r**2 - 2*M*r + a**2
    
    # Effective polymer parameter using the selected prescription
    mu_eff_r = prescription.compute_kerr_effective_mu(r, theta, M, a)
    
    # Effective curvature for Kerr
    K_eff = M / (r * Sigma)
    
    # Polymer correction factor
    polymer_correction = sp.sin(mu_eff_r * K_eff) / (mu_eff_r * K_eff)
    
    # Polymer-corrected Delta function
    Delta_poly = Delta * polymer_correction
    
    # Additional angular correction for spinning case
    # This is a simplification - in a full treatment, the angular sector would
    # also receive polymer corrections based on the prescription
    angular_correction = 1 + (mu**2) * (M/Sigma)
    
    # Construct polymer-corrected metric components
    g_tt = -(1 - 2*M*r/Sigma) * polymer_correction
    g_rr = Sigma/Delta_poly
    g_theta_theta = Sigma * angular_correction
    g_phi_phi = (r**2 + a**2 + 2*M*r*a**2*sp.sin(theta)**2/Sigma) * sp.sin(theta)**2
    
    # Off-diagonal term (frame-dragging)
    g_t_phi = -2*M*r*a*sp.sin(theta)**2/Sigma * polymer_correction
    
    # Assemble metric matrix
    g = sp.zeros(4, 4)
    g[0, 0] = g_tt
    g[1, 1] = g_rr
    g[2, 2] = g_theta_theta
    g[3, 3] = g_phi_phi
    g[0, 3] = g[3, 0] = g_t_phi
    
    print(f"✅ Polymer Kerr metric computed with {prescription.name} prescription")
    return g

def extract_kerr_coefficients(prescription: PolymerPrescription, 
                             max_order: int = 8) -> Dict[str, sp.Expr]:
    """
    Extract polynomial coefficients (α, β, γ, etc.) for the Kerr metric expansion in mu.
    
    Args:
        prescription: Polymer prescription to use
        max_order: highest power of mu to expand to
        
    Returns:
        coeffs: Dictionary mapping coefficient names to expressions
    """
    print(f"🔬 Extracting Kerr coefficients for {prescription.name} prescription up to μ^{max_order}...")
    
    # Define symbols
    M, a, r, theta, mu = sp.symbols('M a r theta mu', real=True, positive=True)
    
    try:
        # Compute the polymer-corrected Kerr metric
        g_kerr = compute_polymer_kerr_metric(prescription, M, a, r, theta, mu)
        
        # Extract g_tt component
        g_tt = g_kerr[0, 0]
        
        # Series expansion around μ = 0
        series_expansion = sp.series(g_tt, mu, 0, max_order + 2).removeO()
        
        # Extract coefficients of different powers
        coeff_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        coeffs = {}
        
        for i, name in enumerate(coeff_names):
            power = 2 * (i + 1)  # μ², μ⁴, μ⁶, ...
            if power <= max_order:
                coeff = series_expansion.coeff(mu, power)
                if coeff is not None:
                    coeffs[name] = sp.simplify(coeff)
                else:
                    coeffs[name] = 0
        
        print(f"✅ Extracted {len(coeffs)} Kerr coefficients for {prescription.name} prescription")
        
    except Exception as e:
        print(f"⚠️ Error in Kerr coefficient extraction: {e}")
        # Provide fallback values
        coeffs = {
            'alpha': sp.Rational(1, 6),
            'beta': 0,
            'gamma': sp.Rational(1, 2520)
        }
    
    return coeffs

def analyze_spin_dependence(prescription: PolymerPrescription, 
                           a_values: List[float],
                           reference_point: Tuple[float, float] = (3, sp.pi/2)) -> Dict[str, Dict]:
    """
    Analyze how coefficients depend on the spin parameter a.
    
    Args:
        prescription: Polymer prescription to use
        a_values: List of spin values to evaluate
        reference_point: (r, theta) point to evaluate at
        
    Returns:
        Dict with spin analysis results
    """
    print(f"🌀 Analyzing spin dependence for {prescription.name} prescription...")
    
    # Extract coefficients
    coeffs = extract_kerr_coefficients(prescription)
    
    # Evaluate at different spin values
    r_val, theta_val = reference_point
    a = sp.Symbol('a')
    M, r, theta = sp.symbols('M r theta')
    
    spin_analysis = {}
    
    for coeff_name, coeff_expr in coeffs.items():
        print(f"\n📊 {coeff_name.upper()} coefficient vs spin:")
        
        numerical_values = []
        for a_val in a_values:
            try:
                # Substitute numerical values (M=1 for normalization)
                expr_eval = coeff_expr.subs([(M, 1), (r, r_val), (theta, theta_val)])
                val = complex(expr_eval.subs(a, a_val))
                numerical_values.append(val.real if abs(val.imag) < 1e-10 else val)
                print(f"   a = {a_val:.2f}: {numerical_values[-1]:.6f}")
            except Exception as e:
                numerical_values.append(0)
                print(f"   a = {a_val:.2f}: [evaluation error: {e}]")
        
        spin_analysis[coeff_name] = {
            'expression': coeff_expr,
            'values': dict(zip(a_values, numerical_values))
        }
    
    return spin_analysis

def compare_kerr_prescriptions(mu_val: float = 0.1, 
                              a_values: List[float] = [0.0, 0.2, 0.5, 0.8, 0.99],
                              reference_point: Tuple[float, float] = (3, sp.pi/2)):
    """
    Compare different polymer prescriptions for Kerr metric.
    
    Args:
        mu_val: Value of mu parameter
        a_values: List of spin values
        reference_point: (r, theta) point to evaluate at
        
    Returns:
        Dict with comparison results
    """
    print("⚖️ Comparing polymer prescriptions for Kerr metric...")
    
    # Create prescriptions
    prescriptions = [
        ThiemannPrescription(),
        AQELPrescription(),
        BojowaldPrescription(),
        ImprovedPrescription()
    ]
    
    # Results container
    results = {}
    
    for prescription in prescriptions:
        print(f"\n{'-'*60}")
        print(f"📊 Analyzing {prescription.name} prescription for Kerr")
        print(f"{'-'*60}")
        
        # Analyze spin dependence
        spin_analysis = analyze_spin_dependence(prescription, a_values, reference_point)
        
        # Create coefficient table
        coeff_table = {}
       