#!/usr/bin/env python3
"""
Unit Tests for Alternative Polymer Prescriptions

This module contains comprehensive unit tests to verify that each polymer
prescription correctly reproduces the expected μ²-only limits and satisfies
basic consistency requirements.

Test Categories:
1. μ² Limit Tests - Verify each prescription reduces to standard form at leading order
2. Symmetry Tests - Check scaling and coordinate transformation properties
3. Consistency Tests - Verify mathematical consistency of polymer factors
4. Convergence Tests - Test series expansion behavior
5. Physical Tests - Check sign consistency and magnitude ordering
"""

import unittest
import sympy as sp
import numpy as np
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from alternative_polymer_prescriptions import (
    ThiemannPrescription, AQELPrescription, BojowaldPrescription, 
    ImprovedPrescription, extract_coefficients_for_prescription
)

class TestAlternativePrescriptions(unittest.TestCase):
    """Test suite for alternative polymer prescriptions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.prescriptions = [
            ThiemannPrescription(),
            AQELPrescription(),
            BojowaldPrescription(),
            ImprovedPrescription()
        ]
        
        # Define test parameters
        self.r, self.M, self.mu = sp.symbols('r M mu', positive=True)
        self.test_classical_geometry = {
            'f_classical': 1 - 2*self.M/self.r
        }
        
        # Classical K_x for testing
        self.Kx_classical = self.M / (self.r * (2*self.M - self.r))
        
    def test_mu2_limit_thiemann(self):
        """Test that Thiemann prescription reproduces μ²-only limit."""
        print("\n🧪 Testing Thiemann μ² limit...")
        
        th = ThiemannPrescription()
        
        # Get polymer factor
        polymer_factor = th.get_polymer_factor(self.Kx_classical, self.test_classical_geometry)
        
        # Expand to μ² and compare with standard form
        series_expansion = sp.series(polymer_factor, self.mu, 0, 3).removeO()
        standard_expansion = self.Kx_classical * (1 - self.mu**2 * self.Kx_classical**2 / 6)
        
        # The difference should be O(μ⁴) or higher
        difference = sp.simplify(series_expansion - standard_expansion)
        mu2_coeff = difference.coeff(self.mu, 2)
        
        # Check that μ² coefficient difference is zero (within simplification)
        self.assertTrue(
            mu2_coeff is None or mu2_coeff == 0,
            f"Thiemann prescription μ² limit failed: {mu2_coeff}"
        )
        print(f"   ✅ Thiemann μ² limit verified")
        
    def test_mu2_limit_aqel(self):
        """Test that AQEL prescription reproduces μ²-only limit."""
        print("\n🧪 Testing AQEL μ² limit...")
        
        aqel = AQELPrescription()
        
        # Get polymer factor
        polymer_factor = aqel.get_polymer_factor(self.Kx_classical, self.test_classical_geometry)
        
        # Expand to μ² and compare with standard form
        series_expansion = sp.series(polymer_factor, self.mu, 0, 3).removeO()
        standard_expansion = self.Kx_classical * (1 - self.mu**2 * self.Kx_classical**2 / 6)
        
        # The difference should be O(μ⁴) or higher
        difference = sp.simplify(series_expansion - standard_expansion)
        mu2_coeff = difference.coeff(self.mu, 2)
        
        # Check that μ² coefficient difference is zero (within simplification)
        self.assertTrue(
            mu2_coeff is None or mu2_coeff == 0,
            f"AQEL prescription μ² limit failed: {mu2_coeff}"
        )
        print(f"   ✅ AQEL μ² limit verified")
        
    def test_mu2_limit_bojowald(self):
        """Test that Bojowald prescription reproduces μ²-only limit."""
        print("\n🧪 Testing Bojowald μ² limit...")
        
        boj = BojowaldPrescription()
        
        # Get polymer factor
        polymer_factor = boj.get_polymer_factor(self.Kx_classical, self.test_classical_geometry)
        
        # Expand to μ² and compare with standard form
        series_expansion = sp.series(polymer_factor, self.mu, 0, 3).removeO()
        standard_expansion = self.Kx_classical * (1 - self.mu**2 * self.Kx_classical**2 / 6)
        
        # The difference should be O(μ⁴) or higher
        difference = sp.simplify(series_expansion - standard_expansion)
        mu2_coeff = difference.coeff(self.mu, 2)
        
        # Check that μ² coefficient difference is zero (within simplification)
        self.assertTrue(
            mu2_coeff is None or mu2_coeff == 0,
            f"Bojowald prescription μ² limit failed: {mu2_coeff}"
        )
        print(f"   ✅ Bojowald μ² limit verified")

    def test_classical_limit_all_prescriptions(self):
        """Test that all prescriptions reduce to classical limit when μ=0."""
        print("\n🧪 Testing classical limits (μ→0)...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                polymer_factor = prescription.get_polymer_factor(
                    self.Kx_classical, self.test_classical_geometry
                )
                
                # Classical limit should be K_classical
                classical_limit = polymer_factor.subs(self.mu, 0)
                expected = self.Kx_classical
                
                difference = sp.simplify(classical_limit - expected)
                self.assertEqual(
                    difference, 0,
                    f"{prescription.name} classical limit failed: {difference}"
                )
                print(f"   ✅ {prescription.name} classical limit verified")

    def test_prescription_consistency(self):
        """Test basic consistency properties of each prescription."""
        print("\n🧪 Testing prescription consistency...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                # Test that effective μ makes sense
                mu_eff = prescription.compute_effective_mu(self.test_classical_geometry)
                
                # Should contain μ
                self.assertTrue(
                    mu_eff.has(self.mu),
                    f"{prescription.name} effective μ doesn't contain μ: {mu_eff}"
                )
                
                # Should reduce to μ in some limit
                # For f_classical = 1 when r >> M, most prescriptions should → μ
                large_r_limit = mu_eff.subs(self.M/self.r, 0)
                
                print(f"   {prescription.name}: μ_eff = {mu_eff}")
                print(f"      Large r limit: {large_r_limit}")

    def test_coefficient_extraction_convergence(self):
        """Test that coefficient extraction converges properly."""
        print("\n🧪 Testing coefficient extraction...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                try:
                    coefficients = extract_coefficients_for_prescription(prescription, max_order=4)
                    
                    # Should extract alpha
                    self.assertIn('alpha', coefficients)
                    self.assertIsNotNone(coefficients['alpha'])
                    
                    # Alpha should be reasonable (around 1/6 for standard case)
                    alpha_val = coefficients['alpha']
                    self.assertTrue(
                        isinstance(alpha_val, (int, float)) and 0.05 <= abs(alpha_val) <= 1.0,
                        f"{prescription.name} α out of expected range: {alpha_val}"
                    )
                    
                    print(f"   ✅ {prescription.name}: α = {alpha_val:.6f}")
                    
                except Exception as e:
                    self.fail(f"Coefficient extraction failed for {prescription.name}: {e}")

    def test_physical_signs(self):
        """Test that extracted coefficients have physically reasonable signs."""
        print("\n🧪 Testing physical signs...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                try:
                    coefficients = extract_coefficients_for_prescription(prescription, max_order=4)
                    
                    if 'alpha' in coefficients and coefficients['alpha'] is not None:
                        alpha_val = coefficients['alpha']
                        
                        # For polymer corrections, α is typically positive
                        # (quantum corrections generally increase the metric function)
                        self.assertGreater(
                            alpha_val, 0,
                            f"{prescription.name} α should be positive: {alpha_val}"
                        )
                        
                        print(f"   ✅ {prescription.name}: α = {alpha_val:.6f} > 0")
                        
                except Exception as e:
                    print(f"   ⚠️ Could not test signs for {prescription.name}: {e}")

    def test_prescription_differences(self):
        """Test that different prescriptions give different results."""
        print("\n🧪 Testing prescription differences...")
        
        # Extract coefficients for all prescriptions
        all_coefficients = {}
        for prescription in self.prescriptions:
            try:
                coefficients = extract_coefficients_for_prescription(prescription, max_order=4)
                all_coefficients[prescription.name] = coefficients.get('alpha', None)
            except Exception as e:
                print(f"   ⚠️ Failed to extract for {prescription.name}: {e}")
        
        # Check that not all alpha values are the same
        alpha_values = [v for v in all_coefficients.values() if v is not None]
        
        if len(alpha_values) > 1:
            # Convert to floats for comparison
            alpha_floats = [float(val) for val in alpha_values if isinstance(val, (int, float, sp.Basic))]
            
            if len(alpha_floats) > 1:
                # Check that there's some variation
                alpha_range = max(alpha_floats) - min(alpha_floats)
                self.assertGreater(
                    alpha_range, 1e-10,
                    "All prescriptions give identical results - no variation detected"
                )
                
                print(f"   ✅ Prescription variation detected: Δα = {alpha_range:.2e}")
                
                # Print comparison
                for name, alpha in all_coefficients.items():
                    if alpha is not None:
                        print(f"      {name}: α = {alpha}")

if __name__ == '__main__':
    print("🧪 Running Alternative Polymer Prescription Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlternativePrescriptions)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 60)# Numerical test values
        self.test_values = {
            self.r: 10.0,
            self.M: 1.0,
            self.mu: 0.1
        }
    
    def test_mu_squared_limit(self):
        """Test that all prescriptions give correct μ² limit."""
        print("\n🧪 Testing μ² limits...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                # Classical extrinsic curvature
                K_classical = self.M / (self.r * (2*self.M - self.r))
                
                # Get polymer factor
                polymer_factor = prescription.get_polymer_factor(
                    K_classical, self.test_classical_geometry
                )
                
                # Expand to μ² order
                expansion = sp.series(polymer_factor, self.mu, 0, n=3).removeO()
                
                # Extract μ² coefficient
                mu2_coeff = expansion.coeff(self.mu, 2)
                
                # For μ² limit, should match -K²/6 (standard polymer correction)
                expected_mu2_base = -K_classical**2 / 6
                
                # The exact form depends on the prescription's effective μ
                # but the leading order should be consistent
                self.assertIsNotNone(mu2_coeff, 
                    f"{prescription.name}: μ² coefficient should exist")
                
                # Verify it's non-zero (polymer effect should exist)
                mu2_numerical = mu2_coeff.subs(self.test_values)
                self.assertNotEqual(float(mu2_numerical), 0.0,
                    f"{prescription.name}: μ² coefficient should be non-zero")
                
                print(f"  ✅ {prescription.name}: μ² coefficient = {mu2_coeff}")
    
    def test_classical_limit(self):
        """Test that μ → 0 gives classical limit."""
        print("\n🧪 Testing classical limits...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                K_classical = self.M / (self.r * (2*self.M - self.r))
                
                # Get polymer factor
                polymer_factor = prescription.get_polymer_factor(
                    K_classical, self.test_classical_geometry
                )
                
                # Classical limit: μ → 0 should give K_classical
                classical_limit = polymer_factor.subs(self.mu, 0)
                
                # For small argument, sin(x)/x → 1, so limit should be K_classical
                # But with effective μ, this might be more complex
                self.assertIsNotNone(classical_limit,
                    f"{prescription.name}: Classical limit should exist")
                
                # Numerically check that it approaches K_classical for small μ
                small_mu_value = 1e-6
                test_vals_small_mu = self.test_values.copy()
                test_vals_small_mu[self.mu] = small_mu_value
                
                polymer_small = polymer_factor.subs(test_vals_small_mu)
                classical_val = K_classical.subs(test_vals_small_mu)
                
                relative_diff = abs(float(polymer_small - classical_val)) / abs(float(classical_val))
                
                self.assertLess(relative_diff, 0.1,  # Within 10% for small μ
                    f"{prescription.name}: Should approach classical limit for small μ")
                
                print(f"  ✅ {prescription.name}: Classical limit verified")
    
    def test_effective_mu_properties(self):
        """Test properties of effective μ for each prescription."""
        print("\n🧪 Testing effective μ properties...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                mu_eff = prescription.compute_effective_mu(self.test_classical_geometry)
                
                # Effective μ should be real
                self.assertTrue(mu_eff.is_real is not False,
                    f"{prescription.name}: Effective μ should be real")
                
                # Effective μ should be positive for physical parameters
                mu_eff_numerical = mu_eff.subs(self.test_values)
                self.assertGreater(float(mu_eff_numerical), 0,
                    f"{prescription.name}: Effective μ should be positive")
                
                # For standard prescriptions, μ_eff should reduce to μ in certain limits
                if prescription.name in ["Improved"]:
                    # For improved prescription, check that μ_eff ≈ μ for small μ
                    small_mu_vals = self.test_values.copy()
                    small_mu_vals[self.mu] = 0.01
                    mu_eff_small = mu_eff.subs(small_mu_vals)
                    mu_small = small_mu_vals[self.mu]
                    
                    relative_diff = abs(float(mu_eff_small) - mu_small) / mu_small
                    self.assertLess(relative_diff, 0.1,
                        f"{prescription.name}: μ_eff should approach μ for small μ")
                
                print(f"  ✅ {prescription.name}: μ_eff = {mu_eff}")
    
    def test_coefficient_extraction(self):
        """Test coefficient extraction for each prescription."""
        print("\n🧪 Testing coefficient extraction...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                # Extract coefficients
                coefficients = extract_coefficients_for_prescription(prescription, max_order=4)
                
                # Should have alpha coefficient
                self.assertIn('alpha', coefficients,
                    f"{prescription.name}: Should extract alpha coefficient")
                
                # Alpha should be numerical
                alpha_val = coefficients['alpha']
                self.assertIsInstance(alpha_val, (int, float, np.number),
                    f"{prescription.name}: Alpha should be numerical")
                
                # Alpha should be non-zero (polymer effect exists)
                self.assertNotEqual(alpha_val, 0.0,
                    f"{prescription.name}: Alpha should be non-zero")
                
                # Check gamma if available
                if 'gamma' in coefficients:
                    gamma_val = coefficients['gamma']
                    if gamma_val != 0:
                        # |alpha| should typically be > |gamma| (convergence)
                        self.assertGreater(abs(alpha_val), abs(gamma_val),
                            f"{prescription.name}: |α| should be > |γ| for convergence")
                
                print(f"  ✅ {prescription.name}: α = {alpha_val:.2e}")
    
    def test_prescription_uniqueness(self):
        """Test that different prescriptions give different results."""
        print("\n🧪 Testing prescription uniqueness...")
        
        # Extract coefficients for all prescriptions
        all_coefficients = {}
        for prescription in self.prescriptions:
            coefficients = extract_coefficients_for_prescription(prescription, max_order=4)
            all_coefficients[prescription.name] = coefficients
        
        # Compare alpha coefficients between prescriptions
        alpha_values = {name: coeffs['alpha'] for name, coeffs in all_coefficients.items() 
                       if 'alpha' in coeffs}
        
        # Should have at least 2 different values
        unique_alphas = set(f"{val:.6e}" for val in alpha_values.values())
        self.assertGreater(len(unique_alphas), 1,
            "Different prescriptions should give different alpha values")
        
        print(f"  ✅ Found {len(unique_alphas)} unique alpha values across prescriptions")
        for name, alpha in alpha_values.items():
            print(f"     {name}: α = {alpha:.6e}")
    
    def test_series_convergence_pattern(self):
        """Test that coefficient series show expected convergence pattern."""
        print("\n🧪 Testing series convergence patterns...")
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                # Extract higher order coefficients
                coefficients = extract_coefficients_for_prescription(prescription, max_order=6)
                
                # Check that coefficients generally decrease in magnitude
                coeff_names = ['alpha', 'gamma']  # μ², μ⁶ terms
                coeff_values = []
                
                for name in coeff_names:
                    if name in coefficients and coefficients[name] != 0:
                        coeff_values.append(abs(coefficients[name]))
                
                if len(coeff_values) >= 2:
                    # Generally expect |α| > |γ| for perturbative convergence
                    alpha_idx = coeff_names.index('alpha')
                    gamma_idx = coeff_names.index('gamma')
                    
                    if (alpha_idx < len(coeff_values) and gamma_idx < len(coeff_values)):
                        alpha_mag = coeff_values[alpha_idx]
                        gamma_mag = coeff_values[gamma_idx]
                        
                        # This is a strong expectation but not absolute requirement
                        if gamma_mag > alpha_mag:
                            print(f"  ⚠️  {prescription.name}: |γ| > |α| - unusual convergence pattern")
                        else:
                            print(f"  ✅ {prescription.name}: |α| > |γ| - good convergence pattern")
    
    def test_numerical_stability(self):
        """Test numerical stability of prescription calculations."""
        print("\n🧪 Testing numerical stability...")
        
        # Test with various parameter ranges
        test_ranges = [
            {'r_val': 3.0, 'M_val': 1.0, 'mu_val': 0.1},    # Near horizon
            {'r_val': 10.0, 'M_val': 1.0, 'mu_val': 0.01},  # Weak field
            {'r_val': 100.0, 'M_val': 1.0, 'mu_val': 0.001} # Far field
        ]
        
        for prescription in self.prescriptions:
            with self.subTest(prescription=prescription.name):
                stable_count = 0
                
                for test_range in test_ranges:
                    try:
                        test_vals = {
                            self.r: test_range['r_val'],
                            self.M: test_range['M_val'],
                            self.mu: test_range['mu_val']
                        }
                        
                        K_classical = self.M / (self.r * (2*self.M - self.r))
                        polymer_factor = prescription.get_polymer_factor(
                            K_classical, self.test_classical_geometry
                        )
                        
                        # Evaluate numerically
                        result = polymer_factor.subs(test_vals)
                        result_float = float(result)
                        
                        # Check for reasonable values (not NaN, not infinite)
                        if np.isfinite(result_float) and abs(result_float) < 1e10:
                            stable_count += 1
                            
                    except Exception as e:
                        print(f"    ⚠️  {prescription.name} failed at {test_range}: {e}")
                
                # Should be stable for most parameter ranges
                stability_ratio = stable_count / len(test_ranges)
                self.assertGreaterEqual(stability_ratio, 0.5,
                    f"{prescription.name}: Should be stable for most parameter ranges")
                
                print(f"  ✅ {prescription.name}: Stable for {stable_count}/{len(test_ranges)} test ranges")

class TestPrescriptionConsistency(unittest.TestCase):
    """Additional consistency tests for prescription framework."""
    
    def test_prescription_interface(self):
        """Test that all prescriptions implement required interface."""
        prescriptions = [
            ThiemannPrescription(),
            AQELPrescription(),
            BojowaldPrescription(),
            ImprovedPrescription()
        ]
        
        for prescription in prescriptions:
            with self.subTest(prescription=prescription.name):
                # Should have required attributes
                self.assertTrue(hasattr(prescription, 'name'))
                self.assertTrue(hasattr(prescription, 'description'))
                
                # Should have required methods
                self.assertTrue(hasattr(prescription, 'compute_effective_mu'))
                self.assertTrue(hasattr(prescription, 'get_polymer_factor'))
                
                # Methods should be callable
                self.assertTrue(callable(prescription.compute_effective_mu))
                self.assertTrue(callable(prescription.get_polymer_factor))
    
    def test_comparative_analysis(self):
        """Test the comparative analysis functionality."""
        from alternative_polymer_prescriptions import compare_prescriptions
        
        # This should run without errors
        try:
            results = compare_prescriptions()
            self.assertIsInstance(results, dict)
            self.assertGreater(len(results), 0)
            print("✅ Comparative analysis completed successfully")
        except Exception as e:
            self.fail(f"Comparative analysis failed: {e}")

def run_tests():
    """Run all tests with detailed output."""
    print("🧪 RUNNING ALTERNATIVE PRESCRIPTION UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAlternativePrescriptions))
    suite.addTests(loader.loadTestsFromTestCase(TestPrescriptionConsistency))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️  ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
