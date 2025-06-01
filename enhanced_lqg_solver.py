#!/usr/bin/env python3
"""
Enhanced LQG Solver with Coherent State Fixes

This module provides an enhanced interface to the LQG midisuperspace solver
with fixes for coherent state construction:
1. Supports integer-valued fluxes matching quantum eigenvalues
2. Supports wider Gaussian widths for non-integer classical values
3. Automatically detects and applies appropriate fixes
"""

import os
import json
import numpy as np
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any

from lqg_fixed_components import LQGParameters, FluxBasisState, KinematicalHilbertSpace
from kinematical_hilbert import run_quantum_solver

def is_close_to_integer(values: List[float], tolerance: float = 0.05) -> bool:
    """
    Check if values are close to integer values within tolerance.
    
    Args:
        values: List of values to check
        tolerance: Maximum allowed distance from integer value
        
    Returns:
        True if all values are close to integers, False otherwise
    """
    return all(abs(val - round(val)) < tolerance for val in values)

def detect_flux_k_scaling(flux_values: List[float], k_values: List[float]) -> Optional[float]:
    """
    Detect the scaling factor between flux (E) values and K values.
    
    Args:
        flux_values: List of flux values (E_x or E_phi)
        k_values: List of K values (K_x or K_phi)
        
    Returns:
        Estimated scaling factor or None if no consistent scaling detected
    """
    # Filter out zero values to avoid division by zero
    non_zero_pairs = [(f, k) for f, k in zip(flux_values, k_values) if abs(f) > 1e-6]
    
    if not non_zero_pairs:
        return None
    
    # Calculate scaling factors
    scaling_factors = [k/f for f, k in non_zero_pairs]
    
    # Check if they're consistent
    avg_scaling = sum(scaling_factors) / len(scaling_factors)
    if all(abs(s - avg_scaling) < 0.02 for s in scaling_factors):
        return avg_scaling
    
    return None

def get_optimal_coherent_params(lattice_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze lattice data and determine optimal coherent state parameters.
    
    Args:
        lattice_data: Dictionary with E_x, E_phi, K_x, K_phi arrays
        
    Returns:
        (params_dict, diagnostics): LQG parameters dict and diagnostics info
    """
    E_x = lattice_data.get('E_x', [])
    E_phi = lattice_data.get('E_phi', [])
    K_x = lattice_data.get('K_x', [])
    K_phi = lattice_data.get('K_phi', [])
    
    # Diagnostics to return
    diagnostics = {
        "E_x_close_to_integer": False,
        "E_phi_close_to_integer": False,
        "detected_K_x_scaling": None,
        "detected_K_phi_scaling": None,
        "recommended_fix": None
    }
    
    # Check if values are close to integers
    diagnostics["E_x_close_to_integer"] = is_close_to_integer(E_x)
    diagnostics["E_phi_close_to_integer"] = is_close_to_integer(E_phi)
    
    # Detect K scaling factors
    diagnostics["detected_K_x_scaling"] = detect_flux_k_scaling(E_x, K_x)
    diagnostics["detected_K_phi_scaling"] = detect_flux_k_scaling(E_phi, K_phi)
    
    # Default parameters (baseline)
    params = {
        "mu_max": 2,
        "nu_max": 2,
        "basis_truncation": 100,
        "coherent_width_E": 0.5,
        "coherent_width_K": 0.5
    }
    
    # Determine which fix to apply
    if diagnostics["E_x_close_to_integer"] and diagnostics["E_phi_close_to_integer"]:
        # Data is already integer-friendly, use default parameters
        diagnostics["recommended_fix"] = "none_needed"
    else:
        # Non-integer data, use wider Gaussians
        params["coherent_width_E"] = 1.5
        params["coherent_width_K"] = 1.5
        diagnostics["recommended_fix"] = "wider_gaussian"
        
        # If scaling factors are detected, note them
        if diagnostics["detected_K_x_scaling"] is not None:
            diagnostics["recommended_fix"] += "_with_k_scaling"
    
    return params, diagnostics

def run_enhanced_lqg_solver(lattice_file: str, outdir: str = "outputs", 
                           verbose: bool = True) -> bool:
    """
    Run LQG solver with enhanced coherent state construction.
    
    Args:
        lattice_file: Path to lattice JSON file
        outdir: Output directory for LQG results
        verbose: Whether to print detailed output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directories exist
        os.makedirs(outdir, exist_ok=True)
        os.makedirs("quantum_inputs", exist_ok=True)
        
        # Load lattice data
        with open(lattice_file, 'r') as f:
            lattice_data = json.load(f)
        
        if verbose:
            print(f"Loaded lattice data from {lattice_file}")
            
        # Analyze lattice data and get optimal parameters
        lqg_params, diagnostics = get_optimal_coherent_params(lattice_data)
        
        if verbose:
            print("\nCoherent State Analysis:")
            print(f"- E_x close to integer: {diagnostics['E_x_close_to_integer']}")
            print(f"- E_phi close to integer: {diagnostics['E_phi_close_to_integer']}")
            if diagnostics['detected_K_x_scaling']:
                print(f"- Detected K_x scaling: {diagnostics['detected_K_x_scaling']:.4f}")
            if diagnostics['detected_K_phi_scaling']:
                print(f"- Detected K_phi scaling: {diagnostics['detected_K_phi_scaling']:.4f}")
            print(f"- Recommended fix: {diagnostics['recommended_fix']}")
            print(f"- Using coherent_width_E: {lqg_params['coherent_width_E']}")
            print(f"- Using coherent_width_K: {lqg_params['coherent_width_K']}")
        
        # Run quantum solver with optimized parameters
        success = run_quantum_solver(
            lattice_file=lattice_file,
            outdir=outdir,
            mu_max=lqg_params["mu_max"],
            nu_max=lqg_params["nu_max"],
            basis_truncation=lqg_params["basis_truncation"],
            coherent_width_E=lqg_params["coherent_width_E"],
            coherent_width_K=lqg_params["coherent_width_K"]
        )
        
        if success:
            # Save diagnostics for reference
            with open(os.path.join(outdir, "coherent_state_diagnostics.json"), 'w') as f:
                json.dump(diagnostics, f, indent=2)
                
            if verbose:
                print("✓ Enhanced LQG solver completed successfully")
                print("✓ Coherent state diagnostics saved")
            return True
        else:
            if verbose:
                print("✗ Quantum solver failed to complete")
            return False
            
    except Exception as e:
        if verbose:
            print(f"✗ Enhanced LQG solver error: {e}")
        return False

def verify_quantum_outputs():
    """Verify that quantum outputs exist and are valid."""
    expected_files = [
        "quantum_inputs/expectation_T00.json",
        "quantum_inputs/expectation_E.json"
    ]
    
    missing = [f for f in expected_files if not os.path.exists(f)]
    
    if missing:
        print(f"✗ Missing quantum files: {', '.join(missing)}")
        return False
    
    try:
        # Very basic validation - check if files contain valid JSON
        for f in expected_files:
            if os.path.exists(f):
                with open(f, 'r') as file:
                    json.load(file)
        
        print("✓ Quantum expectation value files verified")
        return True
        
    except json.JSONDecodeError:
        print("✗ Invalid JSON in quantum files")
        return False
    except Exception as e:
        print(f"✗ Error validating quantum files: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced LQG solver with coherent state fixes")
    parser.add_argument("--lattice", default="examples/lqg_example_integer_values.json", 
                        help="Lattice file for LQG midisuperspace")
    parser.add_argument("--outdir", default="outputs", 
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    success = run_enhanced_lqg_solver(args.lattice, args.outdir)
    
    if success:
        print("\nConverting quantum data to pipeline format...")
        from load_quantum_T00 import convert_to_ndjson, convert_E_to_ndjson
        
        convert_to_ndjson("quantum_inputs/expectation_T00.json", "quantum_inputs/T00_quantum.ndjson")
        convert_E_to_ndjson("quantum_inputs/expectation_E.json", "quantum_inputs/E_quantum.ndjson")
        
        print("✓ Enhanced LQG pipeline complete")
    else:
        print("✗ Enhanced LQG pipeline failed")
