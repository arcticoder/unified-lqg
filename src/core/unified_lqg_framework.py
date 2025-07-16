#!/usr/bin/env python3
"""
Unified LQG Polymer Black Hole Framework
========================================

A comprehensive pipeline for Loop Quantum Gravity polymer black hole analysis,
integrating all modules from coefficient extraction to phenomenological predictions.

Author: LQG Research Group
Date: June 2025
Version: 2.0.0
"""

import json
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import traceback

# Core analysis modules
try:
    from alternative_polymer_prescriptions import (
        PrescriptionComparisonFramework,
        generate_comprehensive_kerr_coefficient_table,
        compute_enhanced_kerr_horizon_shifts,
        compare_kerr_prescriptions,
        save_comprehensive_csv_table,
        save_horizon_shift_csv_table
    )
    from enhanced_kerr_analysis import EnhancedKerrAnalysis
    from kerr_newman_generalization import KerrNewmanGeneralization
    from loop_quantized_matter_coupling_kerr import LoopQuantizedMatterCouplingKerr
    from numerical_relativity_interface_rotating import NumericalRelativityInterfaceRotating
    from observational_constraints import ObservationalConstraints
    from lqg_mu10_mu12_extension import Mu10Mu12ExtendedAnalyzer
    from advanced_constraint_algebra import AdvancedConstraintAlgebraAnalyzer as AdvancedConstraintAlgebra
    from loop_quantized_matter_coupling import LoopQuantizedMatterCoupling
    from numerical_relativity_interface import NumericalRelativityInterface
    from comprehensive_lqg_validation import ComprehensiveLQGValidator
    
    # Set import success flags
    IMPORTS_AVAILABLE = {
        'PrescriptionComparisonFramework': True,
        'EnhancedKerrAnalysis': True,
        'Mu10Mu12ExtendedAnalyzer': True,
        'AdvancedConstraintAlgebra': True,
        'ComprehensiveLQGValidator': True
    }
except ImportError as e:
    print(f"Warning: Some modules not available: {e}")
    print("Some functionality may be limited.")
    
    # Define fallback classes
    class MockAnalyzer:
        def __init__(self):
            pass
        def extract_coefficients_mu12(self):
            return {}
        def analyze_convergence(self, max_order):
            return {"convergent": False}
        def generate_predictions(self):
            return {}
        def export_csv(self, results, filename="mock.csv"):
            return True
        def validate_framework(self):
            return {"valid": False}
    
    # Set missing classes to mock versions
    PrescriptionComparisonFramework = MockAnalyzer
    EnhancedKerrAnalysis = MockAnalyzer
    Mu10Mu12ExtendedAnalyzer = MockAnalyzer
    AdvancedConstraintAlgebra = MockAnalyzer
    ComprehensiveLQGValidator = MockAnalyzer
    
    IMPORTS_AVAILABLE = {
        'PrescriptionComparisonFramework': False,
        'EnhancedKerrAnalysis': False,
        'Mu10Mu12ExtendedAnalyzer': False,
        'AdvancedConstraintAlgebra': False,
        'ComprehensiveLQGValidator': False
    }

@dataclass
class FrameworkResults:
    """Container for unified framework results."""
    
    def __init__(self):
        self.prescription_results = {}
        self.kerr_analysis_results = {}
        self.kerr_newman_results = {}
        self.matter_coupling_results = {}
        self.numerical_relativity_results = {}
        self.observational_constraints_results = {}
        self.mu12_extension_results = {}
        self.constraint_algebra_results = {}
        self.validation_results = {}
        self.phenomenology_predictions = {}
        self.error_log = []
        self.execution_time = 0.0
        self.modules_executed = []
        self.modules_failed = []

class UnifiedLQGFramework:
    """
    Main framework class for running the complete LQG analysis pipeline.
    """
    
    def __init__(self, config_file: str = "unified_lqg_config.json"):
        """Initialize the framework with configuration."""
        self.config_file = config_file
        self.config = self._load_config()
        self.results = FrameworkResults()
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            print(f"[OK] Loaded configuration from {self.config_file}")
            return config
        except FileNotFoundError:
            print(f"ERROR: Configuration file {self.config_file} not found!")
            print("Creating default configuration...")
            self._create_default_config()
            return self._load_config()
        except json.JSONDecodeError as e:
            print(f"ERROR: Error parsing configuration file: {e}")
            sys.exit(1)
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "modules": {
                "prescription_comparison": {"enabled": True},
                "mu12_extension": {"enabled": False},
                "constraint_algebra": {"enabled": False}
            },
            "physical_parameters": {
                "mass_range": {"min": 0.1, "max": 10.0},
                "mu_values": [0.01, 0.05, 0.1]
            },
            "output_options": {
                "save_results": True,
                "output_dir": "unified_results"
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"[OK] Created default configuration at {self.config_file}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # Create output directory if it doesn't exist
        output_dir = Path(self.config.get("output_options", {}).get("output_dir", "unified_results"))
        output_dir.mkdir(exist_ok=True)
          # Setup file and console logging with UTF-8 encoding
        log_file = output_dir / "framework_execution.log"
        
        # Remove existing handlers to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
    
    def run_prescription_comparison(self) -> bool:
        """Run prescription comparison analysis."""
        if not self.config["modules"]["prescription_comparison"]["enabled"]:
            return True
            
        try:
            self.logger.info("[INFO] Running prescription comparison analysis...")
            
            # Initialize the comparison framework
            framework = PrescriptionComparisonFramework()
            
            # Run all prescriptions
            prescriptions = self.config["modules"]["prescription_comparison"].get(
                "prescriptions", ["standard", "thiemann", "aqel", "bojowald", "improved"]
            )
            
            results = {}
            for prescription in prescriptions:
                self.logger.info(f"   Analyzing {prescription} prescription...")
                try:
                    result = framework.analyze_prescription(prescription)
                    results[prescription] = result
                    self.logger.info(f"   [OK] {prescription}: α={result.get('alpha', 'N/A')}")
                except Exception as e:
                    self.logger.error(f"   [ERROR] {prescription} failed: {e}")
                    results[prescription] = {"error": str(e)}
            
            # Generate comparison plots and CSV
            if self.config["modules"]["prescription_comparison"].get("generate_plots", True):
                framework.generate_comparison_plots(results)
                
            if self.config["modules"]["prescription_comparison"].get("output_csv", True):
                framework.export_csv(results)
            
            self.results.prescription_results = results
            self.results.modules_executed.append("prescription_comparison")
            self.logger.info("[OK] Prescription comparison completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Prescription comparison failed: {e}")
            self.results.modules_failed.append("prescription_comparison")
            self.results.error_log.append(f"prescription_comparison: {e}")
            return False
    
    def run_mu12_extension(self) -> bool:
        """Run mu10/mu12 higher-order analysis."""
        if not self.config["modules"]["mu12_extension"]["enabled"]:
            return True
            
        try:
            self.logger.info("[CHART] Running mu10/mu12 extension analysis...")
            
            analyzer = Mu10Mu12ExtendedAnalyzer()
            
            # Extract higher-order coefficients
            max_order = self.config["modules"]["mu12_extension"].get("max_order", 12)
            coefficients = analyzer.extract_coefficients_mu12()
            
            # Perform convergence analysis
            if self.config["modules"]["mu12_extension"].get("convergence_analysis", True):
                convergence = analyzer.analyze_convergence(max_order)
                coefficients["convergence"] = convergence
            
            # Generate Padé approximants
            if self.config["modules"]["mu12_extension"].get("use_pade_approximants", True):
                pade = analyzer.generate_pade_approximants()
                coefficients["pade"] = pade
            
            self.results.mu12_extension_results = coefficients
            self.results.modules_executed.append("mu12_extension")
            self.logger.info("[OK] mu10/mu12 extension completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] mu10/mu12 extension failed: {e}")
            self.results.modules_failed.append("mu12_extension")
            self.results.error_log.append(f"mu12_extension: {e}")
            return False
    
    def run_constraint_algebra(self) -> bool:
        """Run constraint algebra closure analysis."""
        if not self.config["modules"]["constraint_algebra"]["enabled"]:
            return True
            
        try:
            self.logger.info("[LINK] Running constraint algebra analysis...")
            
            algebra = AdvancedConstraintAlgebra()
            
            # Analyze anomalies
            tolerance = self.config["modules"]["constraint_algebra"].get("anomaly_tolerance", 1e-10)
            anomalies = algebra.analyze_constraint_closure(tolerance)
            
            # Test different lattice sizes
            lattice_sites = self.config["modules"]["constraint_algebra"].get("lattice_sites", [3, 5, 7])
            lattice_results = {}
            
            for n_sites in lattice_sites:
                self.logger.info(f"   Testing {n_sites} lattice sites...")
                result = algebra.test_lattice_size(n_sites)
                lattice_results[n_sites] = result
            
            # Optimize regularization schemes
            schemes = self.config["modules"]["constraint_algebra"].get(
                "regularization_schemes", ["epsilon_1", "epsilon_2"]
            )
            optimization = algebra.optimize_regularization(schemes)
            
            results = {
                "anomalies": anomalies,
                "lattice_analysis": lattice_results,
                "regularization_optimization": optimization
            }
            
            self.results.constraint_algebra_results = results
            self.results.modules_executed.append("constraint_algebra")
            self.logger.info("[OK] Constraint algebra analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Constraint algebra analysis failed: {e}")
            self.results.modules_failed.append("constraint_algebra")
            self.results.error_log.append(f"constraint_algebra: {e}")
            return False
    
    def run_validation_suite(self) -> bool:
        """Run comprehensive validation tests."""
        try:
            self.logger.info("[TEST] Running validation suite...")
            
            validator = ComprehensiveLQGValidator()
            
            # Run all validation tests
            validation_results = validator.run_comprehensive_validation()
            
            # Check against expected results
            expected = self.config.get("expected_results", {})
            validation_results["expectations_met"] = validator.check_expectations(
                validation_results, expected
            )
            
            self.results.validation_results = validation_results
            self.results.modules_executed.append("validation")
            
            # Log validation summary
            passed = validation_results.get("tests_passed", 0)
            total = validation_results.get("total_tests", 0)
            self.logger.info(f"[OK] Validation completed: {passed}/{total} tests passed")
            
            return passed == total
            
        except Exception as e:
            self.logger.error(f"[ERROR] Validation suite failed: {e}")
            self.results.modules_failed.append("validation")
            self.results.error_log.append(f"validation: {e}")
            return False
    
    def generate_phenomenology_predictions(self) -> bool:
        """Generate phenomenological predictions."""
        try:
            self.logger.info("[UNIVERSE] Generating phenomenological predictions...")
            
            # Extract key coefficients
            alpha = 1/6  # Default theoretical value
            if self.results.prescription_results:
                # Use most reliable prescription result
                for prescription, result in self.results.prescription_results.items():
                    if "alpha" in result and isinstance(result["alpha"], (int, float)):
                        alpha = result["alpha"]
                        break
            
            # Calculate phenomenological quantities
            mu_values = self.config["physical_parameters"]["mu_values"]
            mass_range = self.config["physical_parameters"]["mass_range"]
            
            predictions = {}
            
            for mu in mu_values:
                for M in [mass_range["min"], 1.0, mass_range["max"]]:
                    key = f"mu_{mu}_M_{M}"
                    
                    # Horizon shift
                    delta_rh = -mu**2 / (6 * M)
                    
                    # QNM frequency shift
                    delta_omega_ratio = mu**2 / (12 * M**2)
                    
                    # ISCO modifications (rough estimate)
                    delta_r_isco = alpha * mu**2 * M**2 / (6 * M)**4  # Simplified
                    
                    predictions[key] = {
                        "horizon_shift": delta_rh,
                        "qnm_frequency_shift_ratio": delta_omega_ratio,
                        "isco_shift": delta_r_isco,
                        "alpha_used": alpha,
                        "mu": mu,
                        "M": M
                    }
            
            self.results.phenomenology_predictions = predictions
            self.logger.info("[OK] Phenomenological predictions generated")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Phenomenology prediction failed: {e}")
            self.results.error_log.append(f"phenomenology: {e}")
            return False
    
    def save_results(self) -> bool:
        """Save all results to files."""
        if not self.config["output_options"]["save_results"]:
            return True
            
        try:
            output_dir = Path(self.config["output_options"]["output_dir"])
            output_dir.mkdir(exist_ok=True)
            
            # Save comprehensive results as JSON
            results_dict = {
                "framework_config": self.config["framework_config"],
                "execution_summary": {
                    "modules_executed": self.results.modules_executed,
                    "modules_failed": self.results.modules_failed,
                    "execution_time": self.results.execution_time,
                    "error_log": self.results.error_log
                },
                "prescription_results": self.results.prescription_results,
                "mu12_extension_results": self.results.mu12_extension_results,
                "constraint_algebra_results": self.results.constraint_algebra_results,
                "validation_results": self.results.validation_results,
                "phenomenology_predictions": self.results.phenomenology_predictions
            }
            
            # Save main results file
            results_file = output_dir / "unified_lqg_results.json"
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            self.logger.info(f"[OK] Results saved to {results_file}")
            
            # Generate summary report
            self._generate_summary_report(output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save results: {e}")
            return False
    
    def _generate_summary_report(self, output_dir: Path):
        """Generate a human-readable summary report."""
        report_file = output_dir / "EXECUTION_SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write("# Unified LQG Framework Execution Summary\n\n")
            f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Execution Time**: {self.results.execution_time:.2f} seconds\n\n")
            
            f.write("## Module Status\n\n")
            f.write("| Module | Status | Notes |\n")
            f.write("|--------|--------|-------|\n")
            
            all_modules = list(self.config["modules"].keys())
            for module in all_modules:
                if module in self.results.modules_executed:
                    status = "[OK] SUCCESS"
                elif module in self.results.modules_failed:
                    status = "[ERROR] FAILED"
                elif not self.config["modules"][module].get("enabled", False):
                    status = "⏸️ DISABLED"
                else:
                    status = "⏭️ SKIPPED"
                
                f.write(f"| {module} | {status} | |\n")
            
            f.write(f"\n## Results Summary\n\n")
            
            # Prescription comparison results
            if self.results.prescription_results:
                f.write("### Prescription Comparison\n\n")
                f.write("| Prescription | α | β | γ |\n")
                f.write("|--------------|---|---|---|\n")
                for prescription, result in self.results.prescription_results.items():
                    if "error" not in result:
                        alpha = result.get("alpha", "N/A")
                        beta = result.get("beta", "N/A")
                        gamma = result.get("gamma", "N/A")
                        f.write(f"| {prescription} | {alpha} | {beta} | {gamma} |\n")
                f.write("\n")
            
            # Error log
            if self.results.error_log:
                f.write("## Errors Encountered\n\n")
                for error in self.results.error_log:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review any failed modules and check error logs\n")
            f.write("2. Enable additional modules for complete analysis\n")
            f.write("3. Validate results against theoretical expectations\n")
            f.write("4. Generate phenomenological predictions\n")
        
        self.logger.info(f"[OK] Summary report saved to {report_file}")
    
    def run_kerr_analysis(self) -> bool:
        """Run comprehensive Kerr black hole analysis."""
        if not self.config["modules"]["kerr_analysis"]["enabled"]:
            return True
            
        try:
            self.logger.info("[ANALYSIS] Running Kerr black hole analysis...")
            
            # Get configuration parameters
            kerr_config = self.config["modules"]["kerr_analysis"]
            spin_values = kerr_config.get("spin_values", [0.0, 0.2, 0.5, 0.8, 0.99])
            mu_values = self.config["physical_parameters"]["mu_values"]
            
            results = {}
            
            # Step 1: Generate comprehensive coefficient table
            if kerr_config.get("comprehensive_table", True):
                self.logger.info("   Generating comprehensive Kerr coefficient table...")
                table_results = generate_comprehensive_kerr_coefficient_table(
                    spin_values=spin_values,
                    prescriptions=None,  # Use all prescriptions
                    output_format="both"
                )
                results["coefficient_table"] = table_results
                self.logger.info("   [OK] Coefficient table generated")
            
            # Step 2: Horizon shift analysis
            if kerr_config.get("horizon_shift_analysis", True):
                self.logger.info("   Computing enhanced Kerr horizon shifts...")
                horizon_shifts = compute_enhanced_kerr_horizon_shifts(
                    prescriptions=None,
                    spin_values=spin_values,
                    mu_values=mu_values,
                    M_val=1.0
                )
                results["horizon_shifts"] = horizon_shifts
                self.logger.info("   [OK] Horizon shifts computed")
            
            # Step 3: Prescription comparison for Kerr
            self.logger.info("   Running Kerr prescription comparison...")
            prescription_comparison = compare_kerr_prescriptions(
                mu_val=0.1,
                a_values=spin_values,
                reference_point=(3, 1.57)  # π/2
            )
            results["prescription_comparison"] = prescription_comparison
            
            # Step 4: Save CSV outputs
            if kerr_config.get("output_csv", True):
                output_dir = Path(self.config["output_options"]["output_dir"])
                output_dir.mkdir(exist_ok=True)
                
                # Main coefficient table
                if "coefficient_table" in results:
                    csv_file = output_dir / "kerr_coefficient_table.csv"
                    save_comprehensive_csv_table(results["coefficient_table"], str(csv_file))
                    self.logger.info(f"   [OK] Coefficient table saved to {csv_file}")
                
                # Horizon shifts table
                if "horizon_shifts" in results:
                    horizon_csv = output_dir / "kerr_horizon_shifts.csv"
                    save_horizon_shift_csv_table(results["horizon_shifts"], str(horizon_csv))
                    self.logger.info(f"   [OK] Horizon shifts saved to {horizon_csv}")
            
            # Step 5: Extract most stable prescription
            if "coefficient_table" in results:
                most_stable = results["coefficient_table"].get("most_stable_prescription", "Unknown")
                results["most_stable_prescription"] = most_stable
                self.logger.info(f"   Most stable prescription: {most_stable}")
            
            self.results.kerr_analysis_results = results
            self.results.modules_executed.append("kerr_analysis")
            self.logger.info("[OK] Kerr analysis completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Kerr analysis failed: {e}")
            self.results.modules_failed.append("kerr_analysis")
            self.results.error_log.append(f"kerr_analysis: {e}")
            return False
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete LQG analysis pipeline."""
        start_time = time.time()
        
        self.logger.info("[START] Starting Unified LQG Framework Pipeline")
        self.logger.info("=" * 60)
        
        # Run enabled modules in sequence
        success = True
        
        # Core analysis modules
        if not self.run_prescription_comparison():
            success = False
        
        # Kerr black hole analysis (new integration)
        if not self.run_kerr_analysis():
            success = False
        
        if not self.run_mu12_extension():
            success = False
        
        if not self.run_constraint_algebra():
            success = False
        
        # Generate predictions based on results
        if not self.generate_phenomenology_predictions():
            success = False
        
        # Run validation suite
        if not self.run_validation_suite():
            success = False
        
        # Save all results
        self.results.execution_time = time.time() - start_time
        if not self.save_results():
            success = False
        
        # Final summary
        self.logger.info("=" * 60)
        if success:
            self.logger.info("🎉 Pipeline completed successfully!")
        else:
            self.logger.warning("[WARNING] Pipeline completed with some failures")
        
        self.logger.info(f"[TIME] Total execution time: {self.results.execution_time:.2f} seconds")
        self.logger.info(f"[OK] Modules executed: {len(self.results.modules_executed)}")
        self.logger.info(f"[ERROR] Modules failed: {len(self.results.modules_failed)}")
        
        return success


def main():
    """Main entry point for the unified framework."""
    parser = argparse.ArgumentParser(
        description="Unified LQG Polymer Black Hole Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_lqg_framework.py                    # Run with default config
  python unified_lqg_framework.py --config my.json  # Run with custom config
  python unified_lqg_framework.py --validate-only   # Run validation tests only
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="unified_lqg_config.json",
        help="Configuration file path (default: unified_lqg_config.json)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation tests only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize framework
        framework = UnifiedLQGFramework(args.config)
        
        if args.validate_only:
            # Run validation only
            success = framework.run_validation_suite()
            framework.save_results()
        else:
            # Run complete pipeline
            success = framework.run_complete_pipeline()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
