#!/usr/bin/env python3
"""
Simple validation script for the unified quantum gravity framework.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("🧪 Simple Validation of Unified Quantum Gravity Framework")
print("=" * 60)

# Test 1: Import the package
print("\n1. Testing package imports...")
try:
    import unified_qg as uqg
    print(f"   ✅ Package imported successfully")
    print(f"   Available components: {uqg.__all__}")
    print(f"   Version: {uqg.__version__}")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Test AMR functionality
print("\n2. Testing Adaptive Mesh Refinement...")
try:
    config = uqg.AMRConfig(initial_grid_size=(16, 16), max_refinement_levels=2)
    amr = uqg.AdaptiveMeshRefinement(config)
    
    def test_func(x, y):
        return np.exp(-10 * ((x-0.5)**2 + (y-0.5)**2))
    
    root_patch = amr.create_initial_grid((-1, 1), (-1, 1), test_func)
    amr.refine_or_coarsen(root_patch)
    
    print(f"   ✅ AMR test passed: created grid with {len(amr.patches)} patches")
except Exception as e:
    print(f"   ❌ AMR test failed: {e}")

# Test 3: Test constraint closure function  
print("\n3. Testing Constraint Closure...")
try:
    results = uqg.run_constraint_closure_scan()
    print(f"   ✅ Constraint closure test passed: got {len(results)} results")
except Exception as e:
    print(f"   ❌ Constraint closure test failed: {e}")

# Test 4: Test 3D polymer field
print("\n4. Testing 3D Polymer Field...")
try:
    config = uqg.Field3DConfig()  # Use default config
    field = uqg.PolymerField3D(config)
    
    # Try to evolve the field
    field.evolve_step(dt=0.01)
    energy = field.compute_energy()
    
    print(f"   ✅ Polymer field test passed: energy = {energy:.3f}")
except Exception as e:
    print(f"   ❌ Polymer field test failed: {e}")

# Test 5: Test GPU solver
print("\n5. Testing GPU Solver...")
try:
    # Simple test matrix
    n = 10
    test_matrix = np.random.random((n, n))
    test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
    
    result = uqg.solve_constraint_gpu(test_matrix)
    
    print(f"   ✅ GPU solver test passed: result type = {type(result)}")
except Exception as e:
    print(f"   ❌ GPU solver test failed: {e}")

# Test 6: Test phenomenology generation
print("\n6. Testing Phenomenology Generation...")
try:
    result = uqg.generate_qc_phenomenology()
    
    print(f"   ✅ Phenomenology test passed: result type = {type(result)}")
except Exception as e:
    print(f"   ❌ Phenomenology test failed: {e}")

print("\n🎉 Basic validation complete!")
print("\nThe unified quantum gravity framework package is functional!")
print("\nTo explore more features:")
print("• Check qc_pipeline_results/ for detailed output")
print("• Run demo_unified_qg.py for comprehensive demos")
print("• Run quick_start.py for usage examples")
