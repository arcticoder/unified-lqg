#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic Framework Test - Tests core imports and functionality without full execution
"""

import sys
import os
import locale

# Set up UTF-8 encoding for Windows
if sys.platform == 'win32':
    # Set console encoding to UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    try:
        # For Windows, ensure we can handle UTF-8
        locale.setlocale(locale.LC_ALL, '')
    except:
        pass

def test_imports():
    """Test importing all the main framework components"""
    print("Testing framework imports...")
    
    try:
        from alternative_polymer_prescriptions import PrescriptionComparisonFramework
        print("[OK] PrescriptionComparisonFramework imported successfully")
    except Exception as e:
        print(f"[ERROR] PrescriptionComparisonFramework import failed: {e}")
        return False
    
    try:
        from lqg_mu10_mu12_extension import Mu10Mu12ExtendedAnalyzer
        print("[OK] Mu10Mu12ExtendedAnalyzer imported successfully")
    except Exception as e:
        print(f"[ERROR] Mu10Mu12ExtendedAnalyzer import failed: {e}")
        return False
    
    try:
        from comprehensive_lqg_validation import ComprehensiveLQGValidator
        print("[OK] ComprehensiveLQGValidator imported successfully")
    except Exception as e:
        print(f"[ERROR] ComprehensiveLQGValidator import failed: {e}")
        return False
    
    try:
        from advanced_constraint_algebra import AdvancedConstraintAlgebraAnalyzer
        print("[OK] AdvancedConstraintAlgebraAnalyzer imported successfully")
    except Exception as e:
        print(f"[ERROR] AdvancedConstraintAlgebraAnalyzer import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality without full pipeline execution"""
    print("\nTesting basic functionality...")
    
    try:
        # Test PrescriptionComparisonFramework
        from alternative_polymer_prescriptions import PrescriptionComparisonFramework
        framework = PrescriptionComparisonFramework()
        
        # Test basic methods exist
        assert hasattr(framework, 'run_comparison')
        assert hasattr(framework, 'export_csv')
        assert hasattr(framework, 'analyze_prescription')
        print("[OK] PrescriptionComparisonFramework methods available")
        
        # Test simple analysis
        result = framework.analyze_prescription('thiemann')
        print(f"[OK] Sample analysis result: {result.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False

def test_simple_coefficient_extraction():
    """Test coefficient extraction without full table generation"""
    print("\nTesting coefficient extraction...")
    
    try:
        from alternative_polymer_prescriptions import ThiemannPrescription, extract_kerr_coefficients
        
        prescription = ThiemannPrescription()
        print(f"[OK] Created {prescription.name} prescription")
          # Test coefficient extraction with simple parameters
        coeffs = extract_kerr_coefficients(prescription=prescription, max_order=6)
        print(f"[OK] Extracted coefficients: {list(coeffs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Coefficient extraction test failed: {e}")
        return False

def main():
    """Run basic framework tests"""
    print("="*60)
    print("LQG FRAMEWORK BASIC TEST")
    print("="*60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 3: Simple coefficient extraction
    if test_simple_coefficient_extraction():
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("[OK] All basic tests passed! Framework core is functional.")
        return True
    else:
        print(f"[ERROR] {total_tests - tests_passed} tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
