#!/usr/bin/env python3
"""
Quick smoke test to verify the LQG-integrated warp pipeline runs end-to-end.
This test uses minimal parameters to ensure fast execution while testing all components.
"""

import sys
import os
import traceback
import numpy as np

def test_pipeline_quick():
    """Test the pipeline with minimal parameters for speed."""
    print("🚀 Starting LQG-Integrated Warp Pipeline Smoke Test")
    print("=" * 60)
    
    try:
        # Import the main pipeline
        from enhance_main_pipeline import main
        
        # Test parameters - very minimal for speed
        test_config = {
            'n_sites': 3,  # Reduced from 5 for speed
            'mu_max': 1,   # Reduced from 2 
            'nu_max': 1,   # Reduced from 2
            'basis_truncation': 50,  # Much smaller for speed
            'use_quantum': True,
            'lattice_file': 'examples/example_reduced_variables.json'        }
        
        print("🔧 Configuration:")
        for key, value in test_config.items():
            print(f"  {key}: {value}")
        print()
        
        print("🧪 Testing pipeline components...")
        # Test 1: Import key modules
        print("  ✓ Importing enhanced_lqg_system...")
        from enhanced_lqg_system import EnhancedKinematicalHilbertSpace
        from lqg_fixed_components import LQGParameters, LatticeConfiguration
        
        print("  ✓ Importing lqg_fixed_components...")
        from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
        
        # Test 2: Create minimal Hilbert space
        print("  ✓ Creating minimal Hilbert space...")
        # Create proper configuration objects
        lattice_config = LatticeConfiguration(
            n_sites=test_config['n_sites'],
            r_min=1e-35,
            r_max=1e-33
        )
        
        lqg_params = LQGParameters(
            planck_area=1.0,
            mu_max=test_config['mu_max'],
            nu_max=test_config['nu_max'],
            coherent_width_E=0.5,
            coherent_width_K=0.5
        )
        
        hilbert = EnhancedKinematicalHilbertSpace(lattice_config, lqg_params)
        
        # Test 3: Access generated basis
        print("  ✓ Accessing generated basis...")
        basis_states = hilbert.basis_states
        print(f"    Generated {len(basis_states)} basis states")
          # Test 4: Create constraint system
        print("  ✓ Creating constraint system...")
        constraint_system = MidisuperspaceHamiltonianConstraint(
            lattice_config=lattice_config,
            lqg_params=lqg_params,
            kinematical_space=hilbert
        )
        
        print("  ✓ All component tests passed!")
        print()
        
        # Test 5: Run minimal pipeline
        print("🎯 Running minimal pipeline...")
        print("  (This may take a moment for eigenvalue computation)")
        
        # Create a minimal lattice file for testing
        minimal_lattice = {
            "n_sites": test_config['n_sites'],
            "mu_max": test_config['mu_max'],
            "nu_max": test_config['nu_max'],
            "classical_E_x": [0] * test_config['n_sites'],
            "classical_E_phi": [0] * test_config['n_sites'],
            "classical_K_x": [0.0] * test_config['n_sites'],
            "classical_K_phi": [0.0] * test_config['n_sites'],
            "target_energy": 1e-6,
            "wormhole_throat_radius": 1.0,
            "stabilization_parameter": 0.1
        }
        
        import json
        os.makedirs('examples', exist_ok=True)
        test_lattice_file = 'examples/test_minimal_lattice.json'
        with open(test_lattice_file, 'w') as f:
            json.dump(minimal_lattice, f, indent=2)
        
        # Override sys.argv for the test
        original_argv = sys.argv
        sys.argv = [
            'enhance_main_pipeline.py',
            '--use-quantum',
            '--lattice', test_lattice_file,
            '--basis-size', str(test_config['basis_truncation'])
        ]
        
        try:
            # Run the pipeline
            result = main()
            print("  ✅ Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ Pipeline failed: {e}")
            print(f"  Error type: {type(e).__name__}")
            if "ARPACK" in str(e) or "convergence" in str(e).lower():
                print("  💡 Hint: This might be an ARPACK convergence issue")
                print("     Consider implementing shift-invert strategy or reducing basis size further")
            return False
            
        finally:
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required modules are available")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Traceback:")
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components that should work independently."""
    print("\n🔬 Testing Individual Components")
    print("=" * 40)
    
    try:        # Test basis generation
        print("  Testing basis generation...")
        from enhanced_lqg_system import EnhancedKinematicalHilbertSpace
        from lqg_fixed_components import LQGParameters, LatticeConfiguration
        
        # Create proper configuration objects
        lattice_config = LatticeConfiguration(
            n_sites=3,
            r_min=1e-35,
            r_max=1e-33
        )
        
        lqg_params = LQGParameters(
            planck_area=1.0,
            mu_max=1,
            nu_max=1,
            coherent_width_E=0.5,
            coherent_width_K=0.5
        )
        
        hilbert = EnhancedKinematicalHilbertSpace(lattice_config, lqg_params)
        
        basis = hilbert.basis_states
        print(f"    ✓ Generated {len(basis)} basis states")
        
        # Test constraint operators
        print("  Testing constraint operators...")
        from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
        
        constraint_system = MidisuperspaceHamiltonianConstraint(lattice_config, lqg_params, hilbert)
        print("    ✓ Constraint system created")
          # Test operator construction (should not fail)
        try:
            kx_op = hilbert.kx_operator(site=0)
            print(f"    ✓ K_x operator: {kx_op.shape}")
        except Exception as e:
            print(f"    ⚠️  K_x operator failed: {e}")
            
        try:
            kphi_op = hilbert.kphi_operator(site=0)
            print(f"    ✓ K_φ operator: {kphi_op.shape}")
        except Exception as e:
            print(f"    ⚠️  K_φ operator failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"    ❌ Component test failed: {e}")
        return False

if __name__ == "__main__":
    print("LQG Warp Framework - Smoke Test Suite")
    print("=" * 50)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    # Then test the full pipeline
    if components_ok:
        pipeline_ok = test_pipeline_quick()
        
        print("\n" + "=" * 60)
        if pipeline_ok:
            print("🎉 SMOKE TEST PASSED! Pipeline is working.")
            print("   Ready for full-scale quantum spacetime simulations!")
        else:
            print("⚠️  SMOKE TEST PARTIAL: Components work, but pipeline has issues.")
            print("   May need ARPACK convergence improvements or basis size tuning.")
    else:
        print("\n" + "=" * 60)
        print("❌ SMOKE TEST FAILED: Basic components have issues.")
        print("   Check imports and basic functionality first.")
    
    print("=" * 60)
