#!/usr/bin/env python3
"""
Unit tests for advanced refinement modules.

These tests validate the functionality of the three advanced refinement modules:
1. Constraint Anomaly Scanner
2. Continuum Benchmarking Framework
3. Enhanced Spin-Foam Validator

Run with: python -m unittest tests/test_advanced_refinements.py
"""

import unittest
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'advanced_refinements'))

# Import modules to test
try:
    from advanced_refinements.constraint_anomaly_scanner import ConstraintAnomalyScanner
    from advanced_refinements.continuum_benchmarking import ContinuumBenchmarkingFramework
    from advanced_refinements.enhanced_spinfoam_validation import EnhancedSpinFoamValidator
    from advanced_refinements.demo_all_refinements import AdvancedRefinementDemo
except ImportError as e:
    print(f"Import error: {e}")


class TestAdvancedRefinements(unittest.TestCase):
    """Test suite for advanced LQG refinement modules."""
    
    def setUp(self):
        """Set up test environment."""
        # Use small test lattice size
        self.test_lattice_size = 3
        self.output_dir = Path("test_outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.lattice_file = "examples/example_reduced_variables.json"
    
    def test_constraint_anomaly_scanner_init(self):
        """Test that ConstraintAnomalyScanner initializes correctly."""
        try:
            scanner = ConstraintAnomalyScanner(n_sites=self.test_lattice_size, 
                                              output_dir=str(self.output_dir))
            self.assertEqual(scanner.n_sites, self.test_lattice_size)
            self.assertTrue(hasattr(scanner, 'output_dir'))
        except Exception as e:
            self.fail(f"ConstraintAnomalyScanner initialization failed: {e}")
      def test_continuum_benchmarking_init(self):
        """Test that ContinuumBenchmarkingFramework initializes correctly."""
        try:
            benchmarker = ContinuumBenchmarkingFramework(
                output_dir=str(self.output_dir)
            )
            self.assertTrue(hasattr(benchmarker, 'output_dir'))
            self.assertTrue(hasattr(benchmarker, 'lattice_results'))
        except Exception as e:
            self.fail(f"ContinuumBenchmarkingFramework initialization failed: {e}")
    
    def test_enhanced_spinfoam_validator_init(self):
        """Test that EnhancedSpinFoamValidator initializes correctly."""
        try:
            from advanced_refinements.enhanced_spinfoam_validation import EnhancedSpinFoamValidator
            validator = EnhancedSpinFoamValidator(
                output_dir=str(self.output_dir)
            )
            self.assertTrue(hasattr(validator, 'output_dir'))
        except Exception as e:
            self.fail(f"EnhancedSpinFoamValidator initialization failed: {e}")
    
    def test_advanced_refinement_demo(self):
        """Test that AdvancedRefinementDemo initializes correctly."""
        try:
            from advanced_refinements.demo_all_refinements import AdvancedRefinementDemo
            demo = AdvancedRefinementDemo(N_values=[self.test_lattice_size])
            self.assertEqual(demo.N_values, [self.test_lattice_size])
            self.assertTrue(hasattr(demo, 'results'))
        except Exception as e:
            self.fail(f"AdvancedRefinementDemo initialization failed: {e}")


if __name__ == "__main__":
    unittest.main()
