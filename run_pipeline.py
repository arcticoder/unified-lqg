#!/usr/bin/env python3
"""
             subprocess.run([
            "python",
            "../warp-lqg-midisuperspace/solve_constraint.py",
            "--lattice", lattice_file,
            "--out", "quantum_inputs",
            "--backend", "torch",
            "--gpu"
        ], check=True)process.run([
            "python3",
            "../warp-lqg-midisuperspace/solve_constraint.py",
            "--lattice", lattice_file,
            "--out", "quantum_inputs"
        ], check=True)te Predictive Framework Workflow with LQG Integration

This script demonstrates the full pipeline from upstream data import
through wormhole generation, stability analysis, lifetime computation,
and analogue-gravity mapping, with optional quantum corrections from
LQG midisuperspace solver.

INTEGRATION STEPS 5-7:
- Step 5: Extract quantum-corrected observables from LQG solver
- Step 6: Incorporate quantum-corrected stability analysis  
- Step 7: Build fully integrated simulation loop
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_lqg_if_requested(lattice_file):
    """Run LQG midisuperspace solver and prepare quantum inputs."""
    print("🔷 Running LQG midisuperspace solver …")
    
    # Ensure quantum_inputs directory exists
    os.makedirs("quantum_inputs", exist_ok=True)    # Run the LQG solver from the sibling repository
    try:
        # Use the working PyTorch GPU solver instead of the corrupted one
        subprocess.run([
            "python3",
            "../warp-lqg-midisuperspace/solve_constraint_pytorch.py",
            "--test",
            "--gpu"
        ], check=True)
        
        # Then run the GPU solver with proper options
        subprocess.run([
            "python3", 
            "../warp-lqg-midisuperspace/solve_constraint_gpu.py",
            "--lattice", lattice_file,
            "--out", "quantum_inputs",
            "--backend", "torch",
            "--n-states", "5"
        ], check=True)
        print("✓ LQG GPU solver completed successfully")
        
        # Verify outputs exist
        expected_files = [
            "quantum_inputs/expectation_T00.json",
            "quantum_inputs/expectation_E.json"
        ]
        
        for filepath in expected_files:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Expected LQG output {filepath} not found")
        
        print("✓ Quantum expectation value files verified")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ LQG solver failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ LQG solver error: {e}")
        return False

def convert_quantum_to_ndjson():
    """Convert quantum JSON outputs to NDJSON format for pipeline."""
    print("🔄 Converting quantum data to NDJSON format...")
    
    try:
        from load_quantum_T00 import convert_to_ndjson, convert_E_to_ndjson
        
        # Convert T00 data
        T00_input = "quantum_inputs/expectation_T00.json"
        T00_output = "quantum_inputs/T00_quantum.ndjson"
        if os.path.exists(T00_input):
            convert_to_ndjson(T00_input, T00_output)
            print(f"✓ Converted T00 data: {T00_output}")
        else:
            print(f"⚠ Warning: {T00_input} not found")
        
        # Convert E field data  
        E_input = "quantum_inputs/expectation_E.json"
        E_output = "quantum_inputs/E_quantum.ndjson"
        if os.path.exists(E_input):
            convert_E_to_ndjson(E_input, E_output)
            print(f"✓ Converted E field data: {E_output}")
        else:
            print(f"⚠ Warning: {E_input} not found")
            
        return True
        
    except Exception as e:
        print(f"✗ Quantum data conversion failed: {e}")
        return False

def run_classical_pipeline(use_quantum=False):
    """Run the classical framework pipeline with optional quantum corrections."""
    
    print("\n" + "=" * 60)
    print("RUNNING CLASSICAL FRAMEWORK PIPELINE")
    if use_quantum:
        print("WITH QUANTUM CORRECTIONS")
    print("=" * 60)
    
    try:
        # Stage 1: Metric refinement
        print("\n1️⃣ Metric Refinement")
        subprocess.run([
            "python3",
            "metric_engineering/metric_refinement.py",
            "--config", "metric_engineering/metric_config.am"
        ], check=True)
        print("✓ Metric refinement completed")

        # Stage 2: Wormhole generation
        print("\n2️⃣ Wormhole Generation")
        subprocess.run([
            "python3",
            "warp-predictive-framework/generate_wormhole.py",
            "--config", "warp-predictive-framework/predictive_config.am",
            "--out", "warp-predictive-framework/outputs/wormhole_solutions.ndjson"
        ], check=True)
        print("✓ Wormhole generation completed")

        # Stage 3: Stability analysis (quantum-corrected if requested)
        print("\n3️⃣ Stability Analysis")
        if use_quantum:
            print("   Using quantum-corrected stability analysis...")
            subprocess.run([
                "python3",
                "metric_engineering/quantum_stability_wrapper.py",
                "quantum_inputs/expectation_E.json",
                "warp-predictive-framework/outputs/wormhole_solutions.ndjson",
                "warp-predictive-framework/outputs/stability_spectrum.ndjson"
            ], check=True)
        else:
            subprocess.run([
                "python3",
                "warp-predictive-framework/analyze_stability.py",
                "--input", "warp-predictive-framework/outputs/wormhole_solutions.ndjson",
                "--config", "warp-predictive-framework/predictive_config.am",
                "--out", "warp-predictive-framework/outputs/stability_spectrum.ndjson"
            ], check=True)
        print("✓ Stability analysis completed")

        # Stage 4: Lifetime computation
        print("\n4️⃣ Lifetime Computation")
        subprocess.run([
            "python3",
            "warp-predictive-framework/compute_lifetime.py",
            "--input", "warp-predictive-framework/outputs/stability_spectrum.ndjson",
            "--config", "warp-predictive-framework/predictive_config.am",
            "--out", "warp-predictive-framework/outputs/lifetime_estimates.ndjson"
        ], check=True)
        print("✓ Lifetime computation completed")

        # Stage 5: Negative-energy integration (quantum-corrected if requested)
        print("\n5️⃣ Negative-Energy Integration")
        cmd = [
            "python3",
            "metric_engineering/compute_negative_energy.py",
            "--refined", "metric_engineering/outputs/refined_metrics.ndjson",
            "--out", "metric_engineering/outputs/negative_energy_integrals.ndjson",
            "--factor", "10.0"
        ]
        
        if use_quantum:
            print("   Using quantum-corrected T^00 data...")
            cmd += ["--quantum-ndjson", "quantum_inputs/T00_quantum.ndjson"]
        else:
            cmd += ["--am", "metric_engineering/exotic_matter_density.am"]
            
        subprocess.run(cmd, check=True)
        print("✓ Negative-energy integration completed")

        # Stage 6: Control field design
        print("\n6️⃣ Control Field Design")
        subprocess.run([
            "python3",
            "metric_engineering/design_control_field.py",
            "--wormhole", "warp-predictive-framework/outputs/wormhole_solutions.ndjson",
            "--stability", "warp-predictive-framework/outputs/stability_spectrum.ndjson",
            "--out", "metric_engineering/outputs/control_fields.ndjson"
        ], check=True)
        print("✓ Control field design completed")

        # Stage 7: Field-mode spectrum
        print("\n7️⃣ Field-Mode Spectrum")
        subprocess.run([
            "python3",
            "metric_engineering/compute_mode_spectrum.py",
            "--geometry", "metric_engineering/outputs/optimized_geometry.json",
            "--out", "metric_engineering/outputs/mode_spectrum.ndjson"
        ], check=True)
        print("✓ Field-mode spectrum completed")

        # Stage 8: Metamaterial blueprint
        print("\n8️⃣ Metamaterial Blueprint")
        subprocess.run([
            "python3",
            "metric_engineering/design_metamaterial_blueprint.py",
            "--modes", "metric_engineering/outputs/mode_spectrum.ndjson",
            "--throat_radius", "4.25e-36",
            "--outer_factor", "2.0",
            "--num_shells", "20",
            "--out", "metric_engineering/outputs/metamaterial_blueprint.json"
        ], check=True)
        print("✓ Metamaterial blueprint completed")
        
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Pipeline stage failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        return False

def run_full_pipeline(use_quantum=False, lattice_file="examples/example_reduced_variables.json", 
                     max_iterations=1, convergence_tolerance=1e-3):
    """
    Run the complete integrated pipeline with optional quantum corrections.
    
    Args:
        use_quantum: Whether to use LQG quantum corrections
        lattice_file: Lattice file for LQG midisuperspace
        max_iterations: Maximum iterations for convergence
        convergence_tolerance: Convergence tolerance for iterative mode
    """
    
    print("=" * 80)
    print("🌌 WARP DRIVE FRAMEWORK: COMPLETE PIPELINE")
    if use_quantum:
        print("🔬 WITH LQG QUANTUM CORRECTIONS")
    print("=" * 80)
    
    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("metric_engineering/outputs", exist_ok=True)
    os.makedirs("warp-predictive-framework/outputs", exist_ok=True)
    
    success = True
    
    for iteration in range(max_iterations):
        print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
        
        # Step 5: Extract quantum-corrected observables (if requested)
        if use_quantum:
            print("\n🔬 STEP 5: Extract Quantum-Corrected Observables")
            if not run_lqg_if_requested(lattice_file):
                success = False
                break
                
            if not convert_quantum_to_ndjson():
                success = False
                break
        
        # Step 6-7: Run classical pipeline with quantum corrections
        print(f"\n🚀 STEPS 6-7: Integrated Classical+Quantum Pipeline")
        if not run_classical_pipeline(use_quantum):
            success = False
            break
        
        # Check convergence for iterative mode
        if max_iterations > 1:
            print(f"\n🔍 Checking convergence (iteration {iteration + 1})...")
            
            # Load current negative energy results
            try:
                import ndjson
                with open("metric_engineering/outputs/negative_energy_integrals.ndjson") as f:
                    current_results = ndjson.load(f)
                
                if iteration > 0:
                    # Compare with previous iteration
                    prev_file = f"metric_engineering/outputs/negative_energy_integrals_iter{iteration-1}.ndjson"
                    if os.path.exists(prev_file):
                        with open(prev_file) as f:
                            prev_results = ndjson.load(f)
                        
                        # Check relative change in negative energy
                        current_energy = sum(r.get('negative_energy_integral', 0) for r in current_results)
                        prev_energy = sum(r.get('negative_energy_integral', 0) for r in prev_results)
                        
                        if prev_energy != 0:
                            relative_change = abs(current_energy - prev_energy) / abs(prev_energy)
                            print(f"   Relative energy change: {relative_change:.2e}")
                            
                            if relative_change < convergence_tolerance:
                                print(f"✓ Converged at iteration {iteration + 1}")
                                break
                
                # Save current iteration results
                import shutil
                shutil.copy(
                    "metric_engineering/outputs/negative_energy_integrals.ndjson",
                    f"metric_engineering/outputs/negative_energy_integrals_iter{iteration}.ndjson"
                )
                
            except Exception as e:
                print(f"⚠ Warning: Convergence check failed: {e}")
    
    # Final summary
    print("\n" + "=" * 80)
    if success:
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        print("\n📁 Generated Output Files:")
        output_files = [
            "warp-predictive-framework/outputs/wormhole_solutions.ndjson",
            "warp-predictive-framework/outputs/stability_spectrum.ndjson", 
            "warp-predictive-framework/outputs/lifetime_estimates.ndjson",
            "metric_engineering/outputs/negative_energy_integrals.ndjson",
            "metric_engineering/outputs/control_fields.ndjson",
            "metric_engineering/outputs/mode_spectrum.ndjson",
            "metric_engineering/outputs/metamaterial_blueprint.json"
        ]
        
        for filepath in output_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"  ✓ {filepath} ({size} bytes)")
            else:
                print(f"  ✗ {filepath} (missing)")
        
        if use_quantum:
            print("\n🔬 Quantum Integration Files:")
            quantum_files = [
                "quantum_inputs/expectation_T00.json",
                "quantum_inputs/expectation_E.json", 
                "quantum_inputs/T00_quantum.ndjson",
                "quantum_inputs/E_quantum.ndjson"
            ]
            
            for filepath in quantum_files:
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"  ✓ {filepath} ({size} bytes)")
                else:
                    print(f"  ✗ {filepath} (missing)")
        
    else:
        print("❌ PIPELINE FAILED")
        print("=" * 80)
        print("Check error messages above for details.")
    
    return success

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete warp framework pipeline with optional LQG quantum corrections"
    )
    parser.add_argument(
        "--use-quantum", action="store_true",
        help="Run LQG solver first and use quantum-corrected T00 and stability analysis"
    )
    parser.add_argument(
        "--lattice", default="examples/example_reduced_variables.json",
        help="Lattice file for LQG midisuperspace solver"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1,
        help="Maximum iterations for convergence (default: 1)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-3,
        help="Convergence tolerance for iterative mode (default: 1e-3)"
    )
    parser.add_argument(
        "--validate-quantum", action="store_true",
        help="Only validate quantum data conversion, don't run full pipeline"
    )
    
    args = parser.parse_args()
    
    if args.validate_quantum:
        print("🔍 Validating quantum data conversion...")
        try:
            from load_quantum_T00 import validate_quantum_data
            results = validate_quantum_data("quantum_inputs")
            
            print("Validation Results:")
            print(f"  Valid: {results['valid']}")
            print(f"  Files found: {len(results['files_found'])}")
            print(f"  Files missing: {len(results['files_missing'])}")
            
            if results["errors"]:
                print("  Errors:")
                for error in results["errors"]:
                    print(f"    - {error}")
            
            sys.exit(0 if results['valid'] else 1)
            
        except Exception as e:
            print(f"✗ Validation failed: {e}")
            sys.exit(1)
    
    success = run_full_pipeline(
        use_quantum=args.use_quantum,
        lattice_file=args.lattice,
        max_iterations=args.max_iterations,
        convergence_tolerance=args.tolerance
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
