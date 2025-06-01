#!/usr/bin/env python3
"""
Test script for the metric engineering control field design functionality.
"""

import sys
import os
import json

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metric_engineering.design_control_field import design_control_field, read_ndjson

def test_control_field_design():
    """Test the control field design functionality."""
    
    print("=== TESTING METRIC ENGINEERING CONTROL FIELD DESIGN ===\n")
    
    # Test with existing data
    wormhole_path = "../outputs/wormhole_solutions.ndjson"
    stability_path = "../outputs/stability_spectrum.ndjson"
    output_path = "outputs/test_control_fields.ndjson"
    
    # Run the control field design
    design_control_field(wormhole_path, stability_path, output_path)
    
    # Load and display results
    control_data = read_ndjson(output_path)
    
    print(f"\nGenerated {len(control_data)} control field proposals:")
    print("-" * 50)
    
    for i, field in enumerate(control_data, 1):
        print(f"Control Field {i}:")
        print(f"  Mode: {field['mode_label']}")
        print(f"  Parent: {field['parent_solution']}")
        print(f"  Eigenvalue: {field['eigenvalue']:.3f}")
        print(f"  Amplitude (ε): {field['control_field_amplitude']:.3e}")
        print(f"  Ansatz: {field['control_field_ansatz']}")
        print()
    
    print("✓ Control field design test completed successfully!")
    return True

if __name__ == "__main__":
    test_control_field_design()
