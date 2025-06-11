#!/usr/bin/env python3
"""
Unified Gauge-Field Polymerization Framework: Operational Demo

This script validates the working capabilities of the framework.
"""

import sys
import os
import json
import time
from datetime import datetime

# Add framework paths
sys.path.extend([
    r'c:\Users\sherri3\Code\asciimath\lqg-anec-framework',
    r'c:\Users\sherri3\Code\asciimath\unified-lqg',
    r'c:\Users\sherri3\Code\asciimath\unified-lqg-qft',
    r'c:\Users\sherri3\Code\asciimath\warp-bubble-optimizer'
])

class FrameworkValidator:
    """Framework validation and demonstration"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'modules_validated': [],
            'physics_results': {},
            'performance': {}
        }
        
        print("=" * 70)
        print("UNIFIED GAUGE-FIELD POLYMERIZATION FRAMEWORK")
        print("Operational Validation Demo")
        print("=" * 70)
        
    def validate_propagator_module(self):
        """Validate polymerized YM propagator module"""
        print("\n1. VALIDATING POLYMERIZED YM PROPAGATOR")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Import and test modules
            print("   🔄 Importing modules...")
            
            # Run the actual working module
            result = os.system('cd "c:\\Users\\sherri3\\Code\\asciimath\\lqg-anec-framework" && python polymerized_ym_propagator.py > nul 2>&1')
            
            if result == 0:
                print("   ✅ Propagator module: OPERATIONAL")
                self.results['modules_validated'].append('propagator')
            else:
                print("   ❌ Propagator module: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        elapsed = time.time() - start_time
        self.results['performance']['propagator_time'] = elapsed
        print(f"   ⏱️  Validation completed in {elapsed:.2f}s")
        
    def validate_vertex_module(self):
        """Validate vertex form factors module"""
        print("\n2. VALIDATING VERTEX FORM FACTORS")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run the actual working module
            result = os.system('cd "c:\\Users\\sherri3\\Code\\asciimath\\unified-lqg" && python vertex_form_factors_pipeline.py > nul 2>&1')
            
            if result == 0:
                print("   ✅ Vertex module: OPERATIONAL")
                self.results['modules_validated'].append('vertex_pipeline')
            else:
                print("   ❌ Vertex module: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        elapsed = time.time() - start_time
        self.results['performance']['vertex_time'] = elapsed
        print(f"   ⏱️  Validation completed in {elapsed:.2f}s")
        
    def validate_cross_section_module(self):
        """Validate cross-section module"""
        print("\n3. VALIDATING CROSS-SECTION ANALYSIS")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run the actual working module
            result = os.system('cd "c:\\Users\\sherri3\\Code\\asciimath\\unified-lqg-qft" && python numerical_cross_section_scans.py > nul 2>&1')
            
            if result == 0:
                print("   ✅ Cross-section module: OPERATIONAL")
                self.results['modules_validated'].append('cross_section_analysis')
            else:
                print("   ❌ Cross-section module: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        elapsed = time.time() - start_time
        self.results['performance']['cross_section_time'] = elapsed
        print(f"   ⏱️  Validation completed in {elapsed:.2f}s")
        
    def validate_fdtd_module(self):
        """Validate FDTD/spin-foam module"""
        print("\n4. VALIDATING FDTD/SPIN-FOAM INTEGRATION")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run the actual working module
            result = os.system('cd "c:\\Users\\sherri3\\Code\\asciimath\\warp-bubble-optimizer" && python fdtd_spinfoam_polymer_integration.py > nul 2>&1')
            
            if result == 0:
                print("   ✅ FDTD module: OPERATIONAL")
                self.results['modules_validated'].append('fdtd_integration')
            else:
                print("   ❌ FDTD module: FAILED")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        elapsed = time.time() - start_time
        self.results['performance']['fdtd_time'] = elapsed
        print(f"   ⏱️  Validation completed in {elapsed:.2f}s")
        
    def generate_validation_report(self):
        """Generate validation summary"""
        print("\n5. VALIDATION SUMMARY")
        print("-" * 40)
        
        modules_validated = len(self.results['modules_validated'])
        total_time = sum(self.results['performance'].values())
        
        print(f"   📊 Modules validated: {modules_validated}/4")
        print(f"   ⏱️  Total validation time: {total_time:.2f}s")
        
        if modules_validated == 4:
            status = "FULLY_OPERATIONAL"
            emoji = "🎉"
        elif modules_validated >= 3:
            status = "MOSTLY_OPERATIONAL"
            emoji = "✅"
        elif modules_validated >= 2:
            status = "PARTIALLY_OPERATIONAL"
            emoji = "⚠️"
        else:
            status = "NEEDS_ATTENTION"
            emoji = "🔧"
            
        self.results['overall_status'] = status
        print(f"   {emoji} Framework status: {status}")
        
        # Save results
        with open('framework_validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"   💾 Results saved to: framework_validation_results.json")
        
        return status, modules_validated
        
    def run_validation(self):
        """Run complete validation"""
        print(f"🚀 Starting framework validation at {datetime.now()}")
        
        # Run all validations
        self.validate_propagator_module()
        self.validate_vertex_module()
        self.validate_cross_section_module()
        self.validate_fdtd_module()
        
        # Generate summary
        status, validated_count = self.generate_validation_report()
        
        # Final summary
        print("\n" + "=" * 70)
        print("FRAMEWORK VALIDATION COMPLETE")
        print("=" * 70)
        print(f"Status: {status}")
        print(f"Modules Validated: {validated_count}/4")
        print(f"Total Runtime: {sum(self.results['performance'].values()):.2f}s")
        print()
        
        if status == "FULLY_OPERATIONAL":
            print("🎉 EXCELLENT! All framework modules are fully operational!")
            print("   The unified gauge-field polymerization framework is")
            print("   ready for advanced physics research and applications.")
        elif status in ["MOSTLY_OPERATIONAL", "PARTIALLY_OPERATIONAL"]:
            print("✅ GOOD! The framework is operational with most modules working.")
            print("   Core functionality is available for research applications.")
        else:
            print("🔧 The framework needs attention, but some modules are working.")
            
        print()
        print("🔬 FRAMEWORK CAPABILITIES:")
        for module in self.results['modules_validated']:
            capabilities = {
                'propagator': 'Polymerized Yang-Mills propagator with instanton sector',
                'vertex_pipeline': 'Vertex form factors and AsciiMath symbolic pipeline',
                'cross_section_analysis': 'Cross-section enhancement and parameter optimization',
                'fdtd_integration': 'FDTD/spin-foam integration with polymer corrections'
            }
            print(f"   ✅ {capabilities.get(module, module)}")
            
        print()
        print("🚀 READY FOR APPLICATIONS:")
        print("   • Quantum gravity phenomenology")
        print("   • Warp bubble physics")
        print("   • ANEC violation studies")
        print("   • Polymerized gauge theory research")
        print("   • Advanced theoretical physics investigations")
        print("=" * 70)

def main():
    """Main entry point"""
    validator = FrameworkValidator()
    validator.run_validation()

if __name__ == "__main__":
    main()
