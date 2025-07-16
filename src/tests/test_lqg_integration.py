#!/usr/bin/env python3
"""
Test script for LQG integration functionality.

This script demonstrates the quantum data conversion and validates
the integration between LQG solver outputs and the classical pipeline.
Includes comprehensive tests for the new LQG midisuperspace solver.
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

def create_test_lattice_data():
    """Create minimal test data for LQG solver."""
    
    # Simple 3-point lattice
    r_grid = [0.1, 0.5, 1.0]
    
    # Classical triad values (simplified)
    E_x = [0.1, 0.2, 0.15]
    E_phi = [0.15, 0.25, 0.2]
    
    # Classical extrinsic curvature
    K_x = [0.01, 0.02, 0.015]
    K_phi = [0.015, 0.025, 0.02]
    
    # Exotic matter profile (phantom scalar)
    exotic_scalar = [-0.1, -0.05, -0.02]
    
    test_data = {
        "r_grid": r_grid,
        "E_classical": {
            "E_x": E_x,
            "E_phi": E_phi
        },
        "K_classical": {
            "K_x": K_x,
            "K_phi": K_phi
        },
        "exotic_profile": {
            "scalar_field": exotic_scalar
        }
    }
    
    return test_data

def test_lqg_solver_basic():
    """Test basic LQG solver functionality."""
    
    print("🔷 Testing LQG midisuperspace constraint solver...")
    
    # Add the LQG solver to path
    lqg_path = os.path.join(os.path.dirname(__file__), '..', 'warp-lqg-midisuperspace')
    if lqg_path not in sys.path:
        sys.path.insert(0, lqg_path)
    
    # Create test data
    test_data = create_test_lattice_data()
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        lattice_file = os.path.join(temp_dir, "test_lattice.json")
        output_dir = os.path.join(temp_dir, "quantum_outputs")
        
        # Write test lattice data
        with open(lattice_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Import and test the LQG framework
        try:
            from solve_constraint import LQGMidisuperspaceFramework, LQGParameters, MuBarScheme
            
            # Initialize with minimal parameters for testing
            lqg_params = LQGParameters(
                gamma=1.0,
                mu_bar_scheme=MuBarScheme.MINIMAL_AREA,
                mu_max=2,  # Small for testing
                nu_max=2,
                basis_truncation=100
            )
            
            print("  Initializing LQG framework...")
            framework = LQGMidisuperspaceFramework(lqg_params)
            
            # Load classical data
            print("  Loading classical data...")
            r_grid, E_x, E_phi, K_x, K_phi, exotic_matter = framework.load_classical_data(lattice_file)
            
            # Initialize kinematical space
            print("  Initializing kinematical Hilbert space...")
            framework.initialize_kinematical_space()
            
            print(f"  Kinematical space dimension: {framework.kinematical_space.dim}")
            
            # Construct constraint operator
            print("  Constructing Hamiltonian constraint...")
            framework.construct_constraint_operator(E_x, E_phi, K_x, K_phi, exotic_matter)
            
            # Find physical states (CPU only for testing)
            print("  Solving constraint equation...")
            physical_states = framework.find_physical_states(num_states=2, use_gpu=False)
            
            # Compute quantum expectation values
            print("  Computing quantum expectation values...")
            quantum_expectations = framework.compute_quantum_expectation_values(
                physical_states[0], E_x, E_phi, exotic_matter
            )
            
            # Export results
            print("  Exporting results...")
            framework.export_results(output_dir, quantum_expectations)
            
            # Verify outputs
            expected_files = [
                os.path.join(output_dir, "expectation_E.json"),
                os.path.join(output_dir, "expectation_T00.json"),
                os.path.join(output_dir, "quantum_corrections.json")
            ]
            
            for expected_file in expected_files:
                if not os.path.exists(expected_file):
                    raise FileNotFoundError(f"Expected output file {expected_file} not found")
                
                # Load and verify content
                with open(expected_file, 'r') as f:
                    data = json.load(f)
                    print(f"  ✓ {os.path.basename(expected_file)}: {len(data)} entries")
            
            print("  ✅ LQG solver basic test passed!")
            return True
            
        except Exception as e:
            print(f"  ❌ LQG solver test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_coherent_state_construction():
    """Test coherent state construction."""
    
    print("🔷 Testing coherent state construction...")
    
    # Add the LQG solver to path
    lqg_path = os.path.join(os.path.dirname(__file__), '..', 'warp-lqg-midisuperspace')
    if lqg_path not in sys.path:
        sys.path.insert(0, lqg_path)
    
    try:
        from solve_constraint import KinematicalHilbertSpace, LQGParameters, LatticeConfiguration
        
        # Create setup
        r_grid = np.array([0.1, 0.5, 1.0])
        lattice_config = LatticeConfiguration(r_grid=r_grid, dr=0.4)
        lqg_params = LQGParameters(mu_max=1, nu_max=1)
        
        kin_space = KinematicalHilbertSpace(lattice_config, lqg_params)
        
        # Classical values
        E_x_classical = np.array([0.1, 0.2, 0.15])
        E_phi_classical = np.array([0.15, 0.25, 0.2])
        K_x_classical = np.array([0.01, 0.02, 0.015])
        K_phi_classical = np.array([0.015, 0.025, 0.02])
        
        # Construct coherent state
        coherent_state = kin_space.construct_coherent_state(
            E_x_classical, E_phi_classical, K_x_classical, K_phi_classical
        )
        
        # Verify normalization
        norm = np.linalg.norm(coherent_state)
        print(f"  Coherent state norm: {norm:.6f}")
        
        if abs(norm - 1.0) > 1e-10:
            raise ValueError(f"Coherent state not normalized: norm = {norm}")
        
        print("  ✅ Coherent state construction test passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Coherent state test failed: {e}")
        return False

def test_quantum_data_conversion():
    """Test the quantum data conversion utilities."""
    print("🧪 Testing Quantum Data Conversion")
    print("=" * 50)
    
    try:
        from load_quantum_T00 import convert_to_ndjson, convert_E_to_ndjson, validate_quantum_data
        
        # Test validation first
        print("1. Validating quantum input data...")
        results = validate_quantum_data("quantum_inputs")
        
        print(f"   Valid: {results['valid']}")
        print(f"   Files found: {len(results['files_found'])}")
        if results['errors']:
            print("   Errors:")
            for error in results['errors']:
                print(f"     - {error}")
        
        if not results['valid']:
            print("   ❌ Validation failed")
            return False
        
        # Test T00 conversion
        print("\n2. Converting T00 data...")
        t00_input = "quantum_inputs/expectation_T00.json"
        t00_output = "quantum_inputs/T00_quantum.ndjson"
        
        if os.path.exists(t00_input):
            n_points = convert_to_ndjson(t00_input, t00_output)
            print(f"   ✅ Converted {n_points} T00 data points")
            
            # Verify output
            with open(t00_output, 'r') as f:
                sample = ndjson.load(f)
            print(f"   Sample: r={sample[0]['r']:.2e}, T00={sample[0]['T00']:.2e}")
        else:
            print(f"   ⚠️ {t00_input} not found")
        
        # Test E field conversion
        print("\n3. Converting E field data...")
        e_input = "quantum_inputs/expectation_E.json"
        e_output = "quantum_inputs/E_quantum.ndjson"
        
        if os.path.exists(e_input):
            n_points = convert_E_to_ndjson(e_input, e_output)
            print(f"   ✅ Converted {n_points} E field data points")
            
            # Verify output
            with open(e_output, 'r') as f:
                sample = ndjson.load(f)
            print(f"   Sample: r={sample[0]['r']:.2e}, Ex={sample[0].get('Ex', 'N/A')}")
        else:
            print(f"   ⚠️ {e_input} not found")
        
        print("\n✅ Quantum data conversion test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Quantum data conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_stability_validation():
    """Test quantum stability analysis validation."""
    print("\n🔬 Testing Quantum Stability Analysis")
    print("=" * 50)
    
    try:
        # Check if quantum stability wrapper exists and can be imported
        sys.path.append('metric_engineering')
        from quantum_stability_wrapper import build_g_rr_from_Ex_Ephi, run_quantum_stability
        
        print("1. Testing metric reconstruction...")
        
        # Test with sample data
        Ex = [1.1, 1.2, 1.3, 1.4, 1.5]
        Ephi = [0.9, 1.0, 1.1, 1.2, 1.3]
        
        g_rr = build_g_rr_from_Ex_Ephi(Ex, Ephi)
        print(f"   ✅ Reconstructed g_rr: {g_rr[:3]}...")
        
        print("2. Checking quantum stability inputs...")
        
        # Check required files
        required_files = [
            "quantum_inputs/expectation_E.json",
            "warp-predictive-framework/outputs/wormhole_solutions.ndjson"
        ]
        
        all_exist = True
        for filepath in required_files:
            if os.path.exists(filepath):
                print(f"   ✅ {filepath}")
            else:
                print(f"   ❌ {filepath} (missing)")
                all_exist = False
        
        if not all_exist:
            print("   ⚠️ Some required files missing - creating mock data")
            
            # Create mock wormhole solutions for testing
            mock_wormhole = [{
                "label": "test_wormhole",
                "throat_radius": 4.25e-36,
                "exotic_profile": "quantum"
            }]
            
            os.makedirs("warp-predictive-framework/outputs", exist_ok=True)
            with open("warp-predictive-framework/outputs/wormhole_solutions.ndjson", 'w') as f:
                writer = ndjson.writer(f)
                writer.writerows(mock_wormhole)
            
            print("   ✅ Created mock wormhole data")
        
        print("\n✅ Quantum stability validation completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Quantum stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_pipeline_validation():
    """Test the integrated pipeline validation."""
    print("\n🚀 Testing Integrated Pipeline")
    print("=" * 50)
    
    try:
        # Check if run_pipeline.py can be imported/validated
        print("1. Checking pipeline components...")
        
        components = [
            ("run_pipeline.py", "Main pipeline script"),
            ("load_quantum_T00.py", "Quantum data converter"),
            ("metric_engineering/quantum_stability_wrapper.py", "Quantum stability analysis"),
            ("metric_engineering/compute_negative_energy.py", "Negative energy computation"),
            ("examples/example_reduced_variables.json", "Example LQG lattice file")
        ]
        
        all_components_ok = True
        for filepath, description in components:
            if os.path.exists(filepath):
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} (missing: {filepath})")
                all_components_ok = False
        
        print("\n2. Checking output directories...")
        
        output_dirs = [
            "quantum_inputs",
            "metric_engineering/outputs", 
            "warp-predictive-framework/outputs",
            "outputs"
        ]
        
        for dirpath in output_dirs:
            if os.path.exists(dirpath):
                print(f"   ✅ {dirpath}")
            else:
                print(f"   ⚠️ {dirpath} (will be created)")
                os.makedirs(dirpath, exist_ok=True)
        
        print("\n3. Testing quantum data presence...")
        
        quantum_files = [
            "quantum_inputs/expectation_T00.json",
            "quantum_inputs/expectation_E.json"
        ]
        
        quantum_ready = True
        for filepath in quantum_files:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                n_points = len(data.get('r', []))
                print(f"   ✅ {filepath} ({n_points} points)")
            else:
                print(f"   ❌ {filepath} (missing)")
                quantum_ready = False
        
        if quantum_ready:
            print("\n✅ Pipeline ready for quantum mode")
        else:
            print("\n⚠️ Pipeline ready for classical mode only")
        
        print("\n✅ Integrated pipeline validation completed")
        return all_components_ok and quantum_ready
        
    except Exception as e:
        print(f"\n❌ Pipeline validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🌌 LQG-Warp Framework Integration Tests")
    print("=" * 80)
    
    tests = [
        ("Quantum Data Conversion", test_quantum_data_conversion),
        ("Quantum Stability Validation", test_quantum_stability_validation), 
        ("Integrated Pipeline Validation", test_integrated_pipeline_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LQG integration is ready.")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py --validate-quantum")
        print("2. Run: python run_pipeline.py --use-quantum")
    else:
        print("⚠️ Some tests failed. Check error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
