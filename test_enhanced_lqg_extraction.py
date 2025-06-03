#!/usr/bin/env python3
"""
Test script for enhanced LQG α, β, γ coefficient extraction
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from enhanced_alpha_beta_gamma_extraction import main
    print("Testing enhanced LQG coefficient extraction...")
    
    # Run the main extraction
    results = main()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*60)
    
    print("Results summary:")
    for key, value in results['coefficients'].items():
        print(f"  {key}: {value}")
    
    print(f"\nResummation successful: {results['resummation_success']}")
    print(f"Execution time: {results['execution_time']:.2f} seconds")

except Exception as e:
    print(f"Error during extraction: {e}")
    import traceback
    traceback.print_exc()
