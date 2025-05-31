#!/usr/bin/env python3
"""
Demo script showing the AsciiMath-based T^{00} integration workflow.

This demonstrates the complete pipeline from AsciiMath parsing to 
numerical integration with improved singularity handling.
"""

import os
import json

def create_demo_refined_metrics():
    """Create a sample refined_metrics.ndjson file for testing."""
    demo_metrics = [
        {
            "label": "wormhole_b0=1.0e-35_refined",
            "parent_solution": "wormhole_b0=1.0e-35", 
            "b0": 1.0e-35,
            "throat_radius": 1.0e-35,
            "ansatz_type": "morris_thorne_modified",
            "refinement_iterations": 5
        },
        {
            "label": "wormhole_b0=5.0e-36_refined",
            "parent_solution": "wormhole_b0=5.0e-36",
            "b0": 5.0e-36, 
            "throat_radius": 5.0e-36,
            "ansatz_type": "morris_thorne_modified",
            "refinement_iterations": 5
        }
    ]
    
    with open("demo_refined_metrics.ndjson", 'w') as f:
        for metric in demo_metrics:
            f.write(json.dumps(metric) + '\n')
    
    print("✓ Created demo_refined_metrics.ndjson")


def run_ascii_integration_demo():
    """Run the complete AsciiMath integration demo."""
    print("=" * 60)
    print("ASCIIMATH T^{00} INTEGRATION DEMO")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        "exotic_matter_density.am",
        "compute_negative_energy_am.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        print("Please create these files first.")
        return False
    
    # Create demo input if needed
    if not os.path.exists("demo_refined_metrics.ndjson"):
        create_demo_refined_metrics()
    
    # Run the AsciiMath version
    print("\n1. Running AsciiMath-based integration...")
    cmd_ascii = (
        "python compute_negative_energy_am.py "
        "--refined demo_refined_metrics.ndjson "
        "--am exotic_matter_density.am "
        "--out demo_results_ascii.ndjson "
        "--mode static "
        "--factor 10.0 "
        "--test"
    )
    
    print(f"Command: {cmd_ascii}")
    result = os.system(cmd_ascii)
    
    if result == 0:
        print("✓ AsciiMath integration completed successfully")
        
        # Show results
        if os.path.exists("demo_results_ascii.ndjson"):
            print("\n2. Results from AsciiMath integration:")
            with open("demo_results_ascii.ndjson", 'r') as f:
                for line in f:
                    result_data = json.loads(line)
                    label = result_data["label"]
                    integral = result_data["negative_energy_integral"]
                    b0 = result_data["b0"]
                    print(f"  {label}: ∫|T00|dV = {integral:.3e} (b0={b0:.1e})")
        
        # Compare with original LaTeX version if available
        print("\n3. Comparing with original LaTeX implementation...")
        cmd_latex = (
            "python compute_negative_energy.py "
            "--refined demo_refined_metrics.ndjson "
            "--tex exotic_matter_density.tex "
            "--out demo_results_latex.ndjson "
            "--factor 10.0"
        )
        
        latex_result = os.system(cmd_latex)
        if latex_result == 0 and os.path.exists("demo_results_latex.ndjson"):
            print("\n   Comparison:")
            print("   Method          | b0=1e-35      | b0=5e-36")
            print("   " + "-" * 45)
            
            # Load both result sets
            ascii_results = {}
            latex_results = {}
            
            with open("demo_results_ascii.ndjson", 'r') as f:
                for line in f:
                    data = json.loads(line)
                    ascii_results[data["b0"]] = data["negative_energy_integral"]
            
            with open("demo_results_latex.ndjson", 'r') as f:
                for line in f:
                    data = json.loads(line)
                    latex_results[data["b0"]] = data["negative_energy_integral"]
            
            for b0 in sorted(ascii_results.keys()):
                ascii_val = ascii_results[b0]
                latex_val = latex_results.get(b0, 0)
                rel_diff = abs(ascii_val - latex_val) / max(ascii_val, latex_val) if max(ascii_val, latex_val) > 0 else 0
                print(f"   AsciiMath       | {ascii_val:.3e}    | {ascii_val:.3e}")
                print(f"   LaTeX           | {latex_val:.3e}    | {latex_val:.3e}")
                print(f"   Rel. difference | {rel_diff:.1%}          | {rel_diff:.1%}")
        
        return True
    else:
        print("✗ AsciiMath integration failed")
        return False


def cleanup_demo_files():
    """Clean up temporary demo files."""
    demo_files = [
        "demo_refined_metrics.ndjson",
        "demo_results_ascii.ndjson", 
        "demo_results_latex.ndjson"
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up {file}")


if __name__ == "__main__":
    # Change to metric_engineering directory if needed
    if not os.path.basename(os.getcwd()).endswith("metric_engineering"):
        if os.path.exists("metric_engineering"):
            os.chdir("metric_engineering")
            print(f"Changed to: {os.getcwd()}")
    
    try:
        success = run_ascii_integration_demo()
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("\nNext steps:")
            print("1. Review the results in demo_results_ascii.ndjson")
            print("2. Run the test suite: python test_ascii_integration.py")
            print("3. Use the AsciiMath version in your production pipeline")
        else:
            print("⚠️  DEMO ENCOUNTERED ISSUES")
            print("\nTroubleshooting:")
            print("1. Check that all dependencies are installed")
            print("2. Verify the AsciiMath file format is correct")
            print("3. Run individual test components")
        
        # Ask about cleanup
        response = input("\nClean up demo files? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            cleanup_demo_files()
            
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
