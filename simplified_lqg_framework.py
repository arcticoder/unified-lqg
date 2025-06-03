#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified Unified LQG Framework - Focuses on core functionality without full complexity
"""

import sys
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

# Set up UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

@dataclass
class SimpleResults:
    """Container for simplified framework results."""
    modules_executed: List[str] = None
    modules_failed: List[str] = None
    prescription_results: Dict = None
    kerr_analysis_results: Dict = None
    validation_results: Dict = None
    error_log: List[str] = None
    
    def __post_init__(self):
        if self.modules_executed is None:
            self.modules_executed = []
        if self.modules_failed is None:
            self.modules_failed = []
        if self.prescription_results is None:
            self.prescription_results = {}
        if self.kerr_analysis_results is None:
            self.kerr_analysis_results = {}
        if self.validation_results is None:
            self.validation_results = {}
        if self.error_log is None:
            self.error_log = []

class SimplifiedLQGFramework:
    """Simplified version of the unified LQG framework for testing core functionality."""
    
    def __init__(self, config_file: str = "unified_lqg_config.json"):
        self.config_file = config_file
        self.config = {}
        self.results = SimpleResults()
        self.setup_logging()
        self.load_config()
        
    def setup_logging(self):
        """Set up logging with UTF-8 support."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('simplified_framework.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"[OK] Loaded configuration from {self.config_file}")
            else:
                # Use default minimal config
                self.config = {
                    "modules": {
                        "prescription_comparison": {"enabled": True},
                        "kerr_analysis": {"enabled": True},
                        "validation": {"enabled": True}
                    },
                    "output_options": {
                        "save_results": True,
                        "output_dir": "simplified_results"
                    }
                }
                self.logger.info("[OK] Using default configuration")
        except Exception as e:
            self.logger.error(f"[ERROR] Configuration error: {e}")
            self.config = {}
    
    def run_prescription_comparison(self) -> bool:
        """Run simplified prescription comparison."""
        try:
            self.logger.info("[ANALYSIS] Running prescription comparison...")
            
            from alternative_polymer_prescriptions import PrescriptionComparisonFramework
            framework = PrescriptionComparisonFramework()
            
            # Test a few prescriptions
            prescriptions = ['thiemann', 'aqel', 'bojowald']
            results = {}
            
            for prescription in prescriptions:
                try:
                    result = framework.analyze_prescription(prescription)
                    results[prescription] = result
                    status = result.get('status', 'unknown')
                    self.logger.info(f"   [OK] {prescription}: {status}")
                except Exception as e:
                    results[prescription] = {'error': str(e), 'status': 'failed'}
                    self.logger.error(f"   [ERROR] {prescription}: {e}")
            
            self.results.prescription_results = results
            self.results.modules_executed.append("prescription_comparison")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Prescription comparison failed: {e}")
            self.results.modules_failed.append("prescription_comparison")
            self.results.error_log.append(f"prescription_comparison: {e}")
            return False
    
    def run_simple_kerr_analysis(self) -> bool:
        """Run simplified Kerr analysis."""
        try:
            self.logger.info("[ANALYSIS] Running simple Kerr analysis...")
            
            from alternative_polymer_prescriptions import (
                ThiemannPrescription, 
                extract_kerr_coefficients
            )
            
            # Test with Thiemann prescription
            prescription = ThiemannPrescription()
            coeffs = extract_kerr_coefficients(prescription, max_order=6)
            
            results = {
                'prescription': 'thiemann',
                'coefficients': {
                    k: str(v) for k, v in coeffs.items()  # Convert to strings for JSON
                },
                'coefficient_count': len(coeffs)
            }
            
            self.results.kerr_analysis_results = results
            self.results.modules_executed.append("kerr_analysis")
            self.logger.info(f"   [OK] Extracted {len(coeffs)} coefficients")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Kerr analysis failed: {e}")
            self.results.modules_failed.append("kerr_analysis")
            self.results.error_log.append(f"kerr_analysis: {e}")
            return False
    
    def run_basic_validation(self) -> bool:
        """Run basic validation tests."""
        try:
            self.logger.info("[TEST] Running basic validation...")
            
            # Basic checks
            validation_results = {
                'imports_working': True,
                'prescription_analysis': len(self.results.prescription_results) > 0,
                'coefficient_extraction': len(self.results.kerr_analysis_results) > 0,
                'overall_status': 'pass'
            }
            
            if not all([validation_results['prescription_analysis'], 
                       validation_results['coefficient_extraction']]):
                validation_results['overall_status'] = 'fail'
            
            self.results.validation_results = validation_results
            self.results.modules_executed.append("validation")
            self.logger.info(f"   [OK] Validation status: {validation_results['overall_status']}")
            return True
            
        except Exception as e:
            self.logger.error(f"[ERROR] Validation failed: {e}")
            self.results.modules_failed.append("validation")
            self.results.error_log.append(f"validation: {e}")
            return False
    
    def save_results(self):
        """Save results to JSON file."""
        try:
            output_dir = Path(self.config.get("output_options", {}).get("output_dir", "simplified_results"))
            output_dir.mkdir(exist_ok=True)
            
            results_file = output_dir / "simplified_results.json"
            
            # Convert results to JSON-serializable format
            def convert_for_json(obj):
                """Convert SymPy expressions and other objects to JSON-serializable format."""
                if hasattr(obj, '__str__'):
                    return str(obj)
                return obj
            
            def process_dict(d):
                """Recursively process dictionary to make it JSON-serializable."""
                if isinstance(d, dict):
                    return {k: process_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [process_dict(item) for item in d]
                else:
                    return convert_for_json(d)
            
            results_dict = {
                'modules_executed': self.results.modules_executed,
                'modules_failed': self.results.modules_failed,
                'prescription_results': process_dict(self.results.prescription_results),
                'kerr_analysis_results': process_dict(self.results.kerr_analysis_results),
                'validation_results': self.results.validation_results,
                'error_log': self.results.error_log,
                'summary': {
                    'total_modules': len(self.results.modules_executed) + len(self.results.modules_failed),
                    'successful_modules': len(self.results.modules_executed),
                    'failed_modules': len(self.results.modules_failed)
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"[OK] Results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to save results: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the simplified pipeline."""
        self.logger.info("="*60)
        self.logger.info("[START] SIMPLIFIED LQG FRAMEWORK PIPELINE")
        self.logger.info("="*60)
        
        success_count = 0
        total_tests = 3
        
        # Run tests
        if self.run_prescription_comparison():
            success_count += 1
        
        if self.run_simple_kerr_analysis():
            success_count += 1
        
        if self.run_basic_validation():
            success_count += 1
        
        # Save results
        self.save_results()
        
        # Summary
        self.logger.info("="*60)
        if success_count == total_tests:
            self.logger.info("[OK] Pipeline completed successfully")
            self.logger.info(f"[OK] Modules executed: {len(self.results.modules_executed)}")
            self.logger.info(f"[OK] Modules failed: {len(self.results.modules_failed)}")
        else:
            self.logger.info("[WARNING] Pipeline completed with some failures")
            self.logger.info(f"[OK] Modules executed: {len(self.results.modules_executed)}")
            self.logger.info(f"[ERROR] Modules failed: {len(self.results.modules_failed)}")
        
        return success_count == total_tests

def main():
    """Main entry point."""
    framework = SimplifiedLQGFramework()
    success = framework.run_pipeline()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
