#!/usr/bin/env python3
"""
demo_lqg_integration.py

Demonstration script showing how to run the new LQG midisuperspace 
constraint solver with proper holonomy corrections, inverse-triad 
regularization, and coherent state construction.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_lqg_demonstration():
    """Run a complete LQG demonstration using the example data."""
    
    print("🔷 LQG Midisuperspace Constraint Solver Demonstration")
    print("=" * 60)
    
    # Check if LQG example data exists
    example_file = "examples/lqg_example_reduced_variables.json"
    if not os.path.exists(example_file):
        print(f"❌ Example file {example_file} not found")
        return False
    
    print(f"📁 Using example data: {example_file}")
    
    # Create output directory
    output_dir = "outputs/lqg_demonstration"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the LQG solver with proper parameters
    print("\n🔬 Running LQG Constraint Solver...")
    
    lqg_solver_path = "../warp-lqg-midisuperspace/solve_constraint.py"
    
    cmd = [
        "python", lqg_solver_path,
        "--lattice", example_file,
        "--outdir", output_dir,
        "--mu-max", "3",
        "--nu-max", "3", 
        "--num-states", "5",
        "--mu-bar-scheme", "minimal_area",
        "--gamma", "1.0",
        "--refinement-study"
    ]
    
    # Add GPU flag if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.append("--use-gpu")
            print("  🚀 GPU acceleration enabled")
    except ImportError:
        print("  💻 Using CPU solver")
    
    try:
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("  ✅ LQG solver completed successfully!")
        print("  Output:")
        for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
            if line.strip():
                print(f"    {line}")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ LQG solver failed with return code {e.returncode}")
        print(f"  Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"  ❌ LQG solver script not found at {lqg_solver_path}")
        return False
    
    # Verify outputs
    print("\n📊 Verifying LQG Outputs...")
    
    expected_outputs = [
        "expectation_E.json",
        "expectation_T00.json", 
        "quantum_corrections.json"
    ]
    
    for output_file in expected_outputs:
        output_path = os.path.join(output_dir, output_file)
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
            print(f"  ✅ {output_file}: {len(data)} entries")
            
            # Show sample data
            if output_file == "quantum_corrections.json":
                if "quantum_corrections" in data:
                    qc = data["quantum_corrections"]
                    print(f"    Relative E_x correction: {qc.get('relative_correction_E_x', 'N/A')}")
                    print(f"    Relative E_φ correction: {qc.get('relative_correction_E_phi', 'N/A')}")
                    
                if "lqg_parameters" in data:
                    lqg_params = data["lqg_parameters"]
                    print(f"    Immirzi parameter γ: {lqg_params.get('gamma', 'N/A')}")
                    print(f"    Basis dimension: {lqg_params.get('basis_dimension', 'N/A')}")
        else:
            print(f"  ❌ {output_file}: Missing")
            return False
    
    print("\n🎯 Key LQG Features Demonstrated:")
    print("  ✅ Proper holonomy corrections with μ̄-scheme")
    print("  ✅ Thiemann's inverse-triad regularization")
    print("  ✅ Coherent states peaked on classical geometry")
    print("  ✅ Realistic exotic matter coupling")
    print("  ✅ Constraint algebra verification")
    print("  ✅ Quantum expectation value computation")
    
    return True

def compare_classical_vs_quantum():
    """Compare classical and quantum results."""
    
    print("\n🔬 Classical vs Quantum Comparison")
    print("=" * 40)
    
    classical_file = "examples/example_reduced_variables.json"
    quantum_outputs = "outputs/lqg_demonstration"
    
    if not os.path.exists(classical_file):
        print("❌ Classical data file not found")
        return
    
    if not os.path.exists(os.path.join(quantum_outputs, "expectation_E.json")):
        print("❌ Quantum outputs not found")
        return
    
    # Load classical data
    with open(classical_file, 'r') as f:
        classical_data = json.load(f)
    
    # Load quantum results
    with open(os.path.join(quantum_outputs, "expectation_E.json"), 'r') as f:
        quantum_E = json.load(f)
    
    with open(os.path.join(quantum_outputs, "expectation_T00.json"), 'r') as f:
        quantum_T00 = json.load(f)
    
    print("📊 Comparison Results:")
    
    # Compare E-field values
    if "E_classical" in classical_data and "E_x" in quantum_E:
        classical_E_x = classical_data["E_classical"]["E_x"]
        quantum_E_x = quantum_E["E_x"]
        
        if len(classical_E_x) == len(quantum_E_x):
            import numpy as np
            classical_array = np.array(classical_E_x)
            quantum_array = np.array(quantum_E_x)
            
            relative_diff = np.mean(np.abs(quantum_array - classical_array) / 
                                  (np.abs(classical_array) + 1e-12))
            
            print(f"  E_x relative difference: {relative_diff:.2%}")
        else:
            print("  E_x arrays have different lengths")
    
    # Show sample values
    print("\n📈 Sample Values:")
    if len(classical_data.get("r_grid", [])) > 0:
        i = 0  # First grid point
        r_val = classical_data["r_grid"][i]
        
        print(f"  At r = {r_val}:")
        
        if "E_classical" in classical_data:
            cl_Ex = classical_data["E_classical"]["E_x"][i]
            print(f"    Classical E_x: {cl_Ex:.6f}")
        
        if "E_x" in quantum_E and len(quantum_E["E_x"]) > i:
            qu_Ex = quantum_E["E_x"][i]
            print(f"    Quantum ⟨E_x⟩: {qu_Ex:.6f}")
        
        if "T00" in quantum_T00 and len(quantum_T00["T00"]) > i:
            qu_T00 = quantum_T00["T00"][i]
            print(f"    Quantum ⟨T^00⟩: {qu_T00:.6e}")

def show_refinement_results():
    """Show lattice refinement study results."""
    
    print("\n🔬 Lattice Refinement Study Results")
    print("=" * 40)
    
    corrections_file = "outputs/lqg_demonstration/quantum_corrections.json"
    
    if not os.path.exists(corrections_file):
        print("❌ Quantum corrections file not found")
        return
    
    with open(corrections_file, 'r') as f:
        data = json.load(f)
    
    if "refinement_study" in data:
        refinement = data["refinement_study"]
        print("📊 Refinement Study:")
        
        for level, results in refinement.items():
            print(f"  {level}: {results['n_sites']} sites, dr = {results['dr']:.4f}")
            print(f"    Convergence metric: {results['convergence_metric']:.2e}")
    else:
        print("⚠ Refinement study not performed")

def main():
    """Main demonstration function."""
    
    print("🌟 LQG Midisuperspace Warp Drive Framework")
    print("Demonstrating genuine Loop Quantum Gravity implementation")
    print("with proper holonomy corrections and coherent states")
    print()
    
    # Run the main demonstration
    success = run_lqg_demonstration()
    
    if success:
        # Additional analyses
        compare_classical_vs_quantum()
        show_refinement_results()
        
        print("\n🎉 LQG Demonstration Completed Successfully!")
        print("\nNext Steps:")
        print("  1. Run with different μ̄-schemes: improved_dynamics, adaptive")
        print("  2. Increase lattice resolution for continuum limit studies")
        print("  3. Integrate quantum T^00 into metric optimization pipeline")
        print("  4. Perform anomaly freedom verification studies")
        
        return 0
    else:
        print("\n❌ LQG Demonstration Failed")
        print("Check the error messages above for troubleshooting")
        return 1

if __name__ == "__main__":
    sys.exit(main())
