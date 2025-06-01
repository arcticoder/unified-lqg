#!/usr/bin/env python3
"""
Example workflow demonstrating the metric engineering pipeline.

This script shows how to run the complete pipeline:
1. metric_refinement.py → refined_metrics.ndjson
2. compute_negative_energy.py → negative_energy_integrals.ndjson
3. design_control_field.py → control_fields.ndjson

Usage: python examples/example_workflow.py
"""

import subprocess
import os
import json

def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{description}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ Success: {result.stdout.strip()}")
    else:
        print(f"❌ Error: {result.stderr.strip()}")
    return result.returncode == 0

def main():
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)
    
    print("=== Metric Engineering Pipeline Demo ===")
    print(f"Working directory: {os.getcwd()}")
    
    # Step 1: Create sample refined metrics (normally from metric_refinement.py)
    print("\n1. Creating sample refined_metrics.ndjson...")
    sample_data = {
        "label": "wormhole_b0=5.0e-36_refined", 
        "parent_solution": "wormhole_b0=5.0e-36",
        "b0": 5.0e-36,
        "throat_radius": 5.0e-36,
        "refined_metric": {
            "g_tt": -1.005, 
            "g_rr": 1.01, 
            "g_thth": 2.5e-71, 
            "g_phph": 2.5e-71
        },
        "refinement_method": "perturbative_correction",
        "convergence_error": 5e-7
    }
    
    # Ensure directory exists
    os.makedirs("metric_engineering/outputs", exist_ok=True)
    
    with open("metric_engineering/outputs/refined_metrics.ndjson", "w") as f:
        f.write(json.dumps(sample_data) + "\n")
    print("✅ Created sample refined_metrics.ndjson")
    
    # Step 2: Compute negative energy integrals
    success = run_command(
        "python metric_engineering/compute_negative_energy.py "
        "--refined metric_engineering/outputs/refined_metrics.ndjson "
        "--out metric_engineering/outputs/negative_energy_integrals.ndjson",
        "2. Computing negative energy integrals..."
    )
    
    if not success:
        return
    
    # Step 3: Design control fields (requires stability data)
    print("\n3. Note: design_control_field.py requires stability_spectrum.ndjson")
    print("   Run: python metric_engineering/design_control_field.py \\")
    print("        --wormhole outputs/wormhole_solutions.ndjson \\")
    print("        --stability outputs/stability_spectrum.ndjson \\")
    print("        --out metric_engineering/outputs/control_fields.ndjson")
    
    print("\n=== Pipeline Demo Complete ===")
    print("Files created:")
    print("  - metric_engineering/outputs/refined_metrics.ndjson")
    print("  - metric_engineering/outputs/negative_energy_integrals.ndjson")

if __name__ == "__main__":
    main()
