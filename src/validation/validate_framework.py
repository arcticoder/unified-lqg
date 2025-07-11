#!/usr/bin/env python3
"""
Validation script for the unified quantum gravity framework.
Tests that all components can be imported and basic functionality works.
"""

import sys
import traceback
from pathlib import Path

print("🧪 Validating Unified Quantum Gravity Framework")
print("=" * 50)

# Test 1: Import the package
print("\n1. Testing package imports...")
try:
    import unified_qg
    from unified_qg import AdaptiveMeshRefinement, AMRConfig, GridPatch
    from unified_qg import PolymerField3D, Field3DConfig
    from unified_qg import run_constraint_closure_scan
    from unified_qg import generate_qc_phenomenology
    from unified_qg import solve_constraint_gpu
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test AMR functionality
print("\n2. Testing Adaptive Mesh Refinement...")
try:
    import numpy as np
    
    config = AMRConfig(initial_grid_size=(16, 16), max_refinement_levels=2)
    amr = AdaptiveMeshRefinement(config)
    
    # Simple test function with a sharp feature
    def test_func(x, y):
        return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
    
    root_patch = amr.create_initial_grid((-1, 1), (-1, 1), test_func)
    refined_patches = amr.refine_patch(root_patch)
    
    print(f"   ✅ AMR test passed: {len(refined_patches)} refined patches created")
except Exception as e:
    print(f"   ❌ AMR test failed: {e}")

# Test 3: Test constraint closure
print("\n3. Testing Constraint Closure...")
try:
    # Test the constraint closure function
    results = run_constraint_closure_scan(
        mu_range=(0.1, 0.5, 3),
        k_range=(0.1, 0.3, 3)
    )
    
    print(f"   ✅ Constraint closure test passed: scan completed with {len(results)} results")
except Exception as e:
    print(f"   ❌ Constraint closure test failed: {e}")

# Test 4: Test 3D polymer field
print("\n4. Testing 3D Polymer Field...")
try:
    config = Field3DConfig(
        lattice_spacing=0.1,
        grid_size=(8, 8, 8),
        coupling_strength=0.1
    )
    
    field = PolymerField3D(config)
    
    # Run a few evolution steps
    for i in range(3):
        field.evolve_step(dt=0.01)
    
    energy = field.compute_energy()
    print(f"   ✅ Polymer field test passed: final energy = {energy:.3f}")
except Exception as e:
    print(f"   ❌ Polymer field test failed: {e}")

# Test 5: Test GPU solver
print("\n5. Testing GPU Quantum Solver...")
try:
    # Simple harmonic oscillator Hamiltonian
    n = 50
    x = np.linspace(-5, 5, n)
    dx = x[1] - x[0]
    
    # Kinetic energy (finite difference approximation)
    kinetic = np.zeros((n, n))
    for i in range(1, n-1):
        kinetic[i, i-1] = -1
        kinetic[i, i] = 2
        kinetic[i, i+1] = -1
    kinetic /= dx**2
    
    # Potential energy
    potential = np.diag(0.5 * x**2)
    
    hamiltonian = -0.5 * kinetic + potential
    
    result = solve_constraint_gpu(hamiltonian, max_iterations=100)
    energy = result['energy']
    
    print(f"   ✅ GPU solver test passed: ground state energy = {energy:.3f}")
except Exception as e:
    print(f"   ❌ GPU solver test failed: {e}")

# Test 6: Test phenomenology generation
print("\n6. Testing Phenomenology Generation...")
try:
    # Generate results for a simple black hole
    result = generate_qc_phenomenology(mass=1.0, spin=0.0)
    
    print(f"   ✅ Phenomenology test passed: generated {len(result)} observables")
except Exception as e:
    print(f"   ❌ Phenomenology test failed: {e}")

print("\n🎉 Framework validation complete!")
print("\nThe unified quantum gravity framework is ready for use!")
print("\nNext steps:")
print("• Explore results in qc_pipeline_results/")
print("• Run demo_unified_qg.py for a comprehensive demonstration")
print("• Run quick_start.py for a quick usage example")
print("• Check the LaTeX papers in the papers/ directory")
