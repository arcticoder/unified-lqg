#!/usr/bin/env python3
"""
Demonstration of the metric_engineering pipeline workflow.

This script shows the typical sequence:
1. metric_refinement.py → refined_metrics.ndjson
2. compute_negative_energy.py → negative_energy_integrals.ndjson  
3. design_control_field.py → control_fields.ndjson

Run this to see the complete pipeline in action.
"""

import subprocess
import os
import sys

def run_pipeline_demo():
    """Run the complete metric engineering pipeline with test data."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    print("=== Metric Engineering Pipeline Demo ===\n")
    
    # Step 1: Generate some test input for metric refinement
    print("Step 1: Preparing test input for metric refinement...")
    test_input_path = "metric_engineering/outputs/test_negative_energy_input.ndjson"
    
    # Create minimal test input (simulating negative energy computation output)
    with open(test_input_path, 'w') as f:
        f.write('{"label": "wormhole_b0=1.60e-35_source=upstream_data", "negative_energy_integral": 1e-75}\n')
    
    print(f"Created test input: {test_input_path}")
    
    # Step 2: Run metric refinement (placeholder)
    print("\nStep 2: Running metric_refinement.py...")
    cmd = [
        sys.executable, "metric_engineering/metric_refinement.py",
        "--input", test_input_path,
        "--config", "metric_engineering/metric_config.am", 
        "--out", "metric_engineering/outputs/refined_metrics.ndjson"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ metric_refinement.py completed successfully")
        else:
            print(f"✗ metric_refinement.py failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Error running metric_refinement.py: {e}")
    
    # Step 3: Run negative energy computation
    print("\nStep 3: Running compute_negative_energy.py...")
    cmd = [
        sys.executable, "metric_engineering/compute_negative_energy.py",
        "--refined", "metric_engineering/outputs/refined_metrics.ndjson",
        "--out", "metric_engineering/outputs/negative_energy_integrals.ndjson"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ compute_negative_energy.py completed successfully")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"✗ compute_negative_energy.py failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Error running compute_negative_energy.py: {e}")
    
    # Step 4: Run control field design
    print("\nStep 4: Running design_control_field.py...")
    cmd = [
        sys.executable, "metric_engineering/design_control_field.py",
        "--wormhole", "outputs/wormhole_solutions.ndjson",
        "--stability", "outputs/stability_spectrum.ndjson", 
        "--out", "metric_engineering/outputs/control_fields.ndjson"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ design_control_field.py completed successfully")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"✗ design_control_field.py failed: {result.stderr}")
    except Exception as e:
        print(f"✗ Error running design_control_field.py: {e}")
    
    print("\n=== Pipeline Demo Complete ===")
    print("\nOutput files generated:")
    print("  - metric_engineering/outputs/refined_metrics.ndjson")
    print("  - metric_engineering/outputs/negative_energy_integrals.ndjson") 
    print("  - metric_engineering/outputs/control_fields.ndjson")
    
    print("\nNext steps for real implementation:")
    print("  1. Replace placeholder in metric_refinement.py with actual PDE solver")
    print("  2. Replace dummy integral in compute_negative_energy.py with numerical quadrature")
    print("  3. Replace heuristic in design_control_field.py with eigenfunction solver")

if __name__ == "__main__":
    run_pipeline_demo()
