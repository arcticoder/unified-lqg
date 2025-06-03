#!/usr/bin/env python3
"""
Example: Comprehensive Prescription Comparison for LQG Coefficient Extraction

This script demonstrates how to:
1. Run coefficient extraction with multiple prescriptions
2. Compare results across prescriptions
3. Generate visualizations and CSV output
4. Analyze phenomenological implications
5. Perform consistency checks

Usage:
    python example_compare_prescriptions.py

Output:
    - prescription_coefficient_comparison.csv
    - prescription_coefficient_comparison.png
    - Console output with detailed analysis
"""

import sys
import os
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from enhanced_alpha_beta_gamma_extraction import main
from alternative_polymer_prescriptions import (
    ThiemannPrescription, AQELPrescription, BojowaldPrescription, 
    ImprovedPrescription, compare_prescriptions
)

def demonstrate_individual_prescriptions():
    """Demonstrate individual prescription analysis."""
    print("🔬 INDIVIDUAL PRESCRIPTION DEMONSTRATION")
    print("=" * 60)
    
    prescriptions = [
        ThiemannPrescription(),
        AQELPrescription(),
        BojowaldPrescription(),
        ImprovedPrescription()
    ]
    
    for prescription in prescriptions:
        print(f"\n📋 {prescription.name} Prescription:")
        print(f"   Description: {prescription.description}")
        
        # Example polymer factor calculation
        try:
            import sympy as sp
            r, M, mu = sp.symbols('r M mu', positive=True)
            K_classical = M / (r * (2*M - r))
            classical_geometry = {'f_classical': 1 - 2*M/r}
            
            polymer_factor = prescription.get_polymer_factor(K_classical, classical_geometry)
            print(f"   Polymer factor: sin(μ_eff * K) / μ_eff")
            print(f"   Effective μ: {prescription.compute_effective_mu(classical_geometry)}")
            
            # Expand to show leading terms
            expansion = sp.series(polymer_factor, mu, 0, n=5).removeO()
            print(f"   Series expansion (first few terms): {expansion}")
            
        except Exception as e:
            print(f"   ⚠️  Calculation failed: {e}")

def run_comprehensive_comparison():
    """Run the full LQG coefficient extraction with all prescriptions."""
    print("\n🚀 COMPREHENSIVE COMPARISON")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run main analysis
    print("Running enhanced alpha-beta-gamma extraction with all prescriptions...")
    results = main()
    
    execution_time = time.time() - start_time
    
    # Summary analysis
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 40)
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Prescriptions analyzed: {len(results)}")
    
    # Extract key coefficients for comparison
    alpha_comparison = {}
    gamma_comparison = {}
    
    for prescription, data in results.items():
        coeffs = data['coefficients']
        if 'alpha' in coeffs:
            alpha_comparison[prescription] = coeffs['alpha']
        if 'gamma' in coeffs:
            gamma_comparison[prescription] = coeffs['gamma']
    
    print(f"\n🔍 Alpha coefficient comparison:")
    for prescription, alpha in alpha_comparison.items():
        print(f"   {prescription:<12}: α = {alpha}")
    
    print(f"\n🔍 Gamma coefficient comparison:")
    for prescription, gamma in gamma_comparison.items():
        print(f"   {prescription:<12}: γ = {gamma}")
    
    # Calculate relative differences
    if 'standard' in alpha_comparison:
        ref_alpha = alpha_comparison['standard']
        print(f"\n📈 Relative differences (vs standard):")
        for prescription, alpha in alpha_comparison.items():
            if prescription != 'standard' and ref_alpha != 0:
                try:
                    rel_diff = (float(alpha) - float(ref_alpha)) / float(ref_alpha) * 100
                    print(f"   {prescription:<12}: {rel_diff:+.1f}%")
                except (ValueError, TypeError):
                    print(f"   {prescription:<12}: Cannot calculate (symbolic)")
    
    return results

def analyze_phenomenological_implications(results):
    """Analyze the phenomenological implications of different prescriptions."""
    print(f"\n🌟 PHENOMENOLOGICAL IMPLICATIONS")
    print("=" * 60)
    
    # Test parameters for analysis
    M_bh = 1.0  # Solar masses
    mu_values = [0.1, 0.01, 0.001]  # Range of quantum parameters
    
    print(f"Analysis for M = {M_bh} M☉:")
    
    for mu_val in mu_values:
        print(f"\n🔬 Quantum parameter μ = {mu_val}:")
        print("-" * 30)
        
        for prescription, data in results.items():
            coeffs = data['coefficients']
            
            if 'alpha' in coeffs:
                try:
                    alpha_val = float(coeffs['alpha'])
                    
                    # Horizon shift estimate
                    horizon_shift = alpha_val * mu_val**2 * M_bh
                    
                    # ISCO shift estimate (rough)
                    r_isco_classical = 6.0  # 6M for Schwarzschild
                    isco_shift = alpha_val * mu_val**2 * M_bh * (M_bh/r_isco_classical)**2
                    
                    print(f"   {prescription:<12}: δr_h ≈ {horizon_shift:+.4f}M, δr_ISCO ≈ {isco_shift:+.4f}M")
                    
                except (ValueError, TypeError):
                    print(f"   {prescription:<12}: [symbolic result - cannot evaluate numerically]")

def perform_consistency_checks(results):
    """Perform consistency checks on the prescription results."""
    print(f"\n✅ CONSISTENCY CHECKS")
    print("=" * 60)
    
    # Check 1: All prescriptions should give some quantum correction
    print(f"\n🔍 Quantum correction check:")
    for prescription, data in results.items():
        coeffs = data['coefficients']
        has_alpha = 'alpha' in coeffs and coeffs['alpha'] != 0
        status = "✅ HAS CORRECTION" if has_alpha else "❌ NO CORRECTION"
        print(f"   {prescription:<12}: {status}")
    
    # Check 2: Sign consistency
    print(f"\n🔍 Sign consistency check:")
    alpha_signs = {}
    for prescription, data in results.items():
        coeffs = data['coefficients']
        if 'alpha' in coeffs:
            try:
                alpha_val = float(coeffs['alpha'])
                alpha_signs[prescription] = "+" if alpha_val > 0 else "-"
            except (ValueError, TypeError):
                alpha_signs[prescription] = "?"
    
    for prescription, sign in alpha_signs.items():
        print(f"   {prescription:<12}: α {sign}")
    
    # Check 3: Magnitude ordering
    print(f"\n🔍 Convergence pattern check:")
    for prescription, data in results.items():
        coeffs = data['coefficients']
        
        try:
            alpha_mag = abs(float(coeffs.get('alpha', 0)))
            gamma_mag = abs(float(coeffs.get('gamma', 0)))
            
            if alpha_mag > 0 and gamma_mag > 0:
                ordering = "✅ |α| > |γ|" if alpha_mag > gamma_mag else "⚠️  |γ| ≥ |α|"
                print(f"   {prescription:<12}: {ordering} (α={alpha_mag:.2e}, γ={gamma_mag:.2e})")
            elif alpha_mag > 0:
                print(f"   {prescription:<12}: ✅ α non-zero (γ=0)")
            else:
                print(f"   {prescription:<12}: ❌ No significant coefficients")
                
        except (ValueError, TypeError):
            print(f"   {prescription:<12}: ⚠️  Cannot check (symbolic results)")

def generate_summary_report(results):
    """Generate a summary report of the prescription comparison."""
    print(f"\n📄 SUMMARY REPORT")
    print("=" * 60)
    
    # Count successful extractions
    successful_extractions = 0
    for prescription, data in results.items():
        if 'alpha' in data['coefficients'] and data['coefficients']['alpha'] != 0:
            successful_extractions += 1
    
    print(f"Prescriptions tested: {len(results)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Success rate: {successful_extractions/len(results)*100:.1f}%")
    
    # Recommend best prescription
    print(f"\n🏆 Recommendations:")
    if 'thiemann' in results:
        print("   - Thiemann prescription provides the standard reference")
    if 'aqel' in results:
        print("   - AQEL prescription may show different geometric dependence")
    if 'bojowald' in results:
        print("   - Bojowald prescription includes curvature-dependent effects")
    if 'improved' in results:
        print("   - Improved prescription includes higher-order corrections")
    
    print(f"\n📁 Output files generated:")
    print("   - prescription_coefficient_comparison.csv")
    print("   - prescription_coefficient_comparison.png")
    print("   - Console output with detailed analysis")

def main_example():
    """Main example execution function."""
    print("🚀 LQG PRESCRIPTION COMPARISON EXAMPLE")
    print("=" * 80)
    
    start_time = time.time()
    
    # Step 1: Demonstrate individual prescriptions
    demonstrate_individual_prescriptions()
    
    # Step 2: Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    # Step 3: Analyze phenomenological implications
    analyze_phenomenological_implications(results)
    
    # Step 4: Perform consistency checks
    perform_consistency_checks(results)
    
    # Step 5: Generate summary report
    generate_summary_report(results)
    
    # Final timing
    total_time = time.time() - start_time
    
    print(f"\n🎯 EXAMPLE COMPLETED!")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print("Results saved in results dictionary with keys:")
    for prescription in results.keys():
        print(f"  - {prescription}")
    
    print(f"\nTo reproduce these results:")
    print("  python example_compare_prescriptions.py")
    
    print(f"\nFor unit testing of prescriptions:")
    print("  python test_alternate_prescriptions.py")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive example
    try:
        results = main_example()
        print("\n✅ Example completed successfully!")
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        print("Make sure all required modules are available.")
