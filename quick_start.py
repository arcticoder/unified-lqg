#!/usr/bin/env python3
"""
Quick usage example for the unified_qg package.

This script demonstrates how to use the main components of the package.
"""

import unified_qg as uqg
import numpy as np


def main():
    print("🌟 UNIFIED QUANTUM GRAVITY PACKAGE - QUICK START")
    print("=" * 55)
    
    # Show package information
    print(f"Package version: {uqg.__version__}")
    print(f"Available components: {len(uqg.__all__)}")
    print(f"Components: {', '.join(uqg.__all__[:5])}...")
    
    # 1. AMR Example
    print("\n1. Adaptive Mesh Refinement Example:")
    print("-" * 40)
    
    amr_config = uqg.AMRConfig(initial_grid_size=(16, 16))
    amr = uqg.AdaptiveMeshRefinement(amr_config)
    print(f"   ✅ AMR system created with {amr_config.initial_grid_size} grid")
    
    # 2. 3D Polymer Field Example
    print("\n2. 3+1D Polymer Field Example:")
    print("-" * 40)
    
    field_config = uqg.Field3DConfig(grid_size=(8, 8, 8))
    polymer_field = uqg.PolymerField3D(field_config)
    print(f"   ✅ Polymer field created with {field_config.grid_size} grid")
    
    # 3. GPU Solver Status
    print("\n3. GPU Solver Status:")
    print("-" * 40)
    
    device_info = uqg.get_device_info()
    print(f"   PyTorch available: {device_info['torch_available']}")
    print(f"   CUDA available: {device_info['cuda_available']}")
    print(f"   GPU available: {uqg.is_gpu_available()}")
    
    # 4. Phenomenology Example
    print("\n4. Phenomenology Generation:")
    print("-" * 40)
    
    omega = uqg.compute_qnm_frequency(1.0, 0.5)
    isco = uqg.compute_isco_shift(1.0, 0.5)
    print(f"   QNM frequencies: {omega}")
    print(f"   ISCO radius: {isco:.3f}")
    
    # 5. Package utilities
    print("\n5. Package Utilities:")
    print("-" * 40)
    
    print("   ✅ Packaging utilities available")
    print("   ✅ Configuration creation available")
    
    print("\n🎉 All components are ready for use!")
    print("\nFor more examples, see demo_unified_qg.py")
    print("For documentation: help(unified_qg)")


if __name__ == "__main__":
    main()
