#!/usr/bin/env python3
"""
Quantum T00 data conversion utilities for LQG integration.

This module handles conversion between the LQG midisuperspace solver output
and the formats expected by the classical warp framework pipeline.
"""

import json
import ndjson
import numpy as np
from scipy.interpolate import interp1d
import os
from pathlib import Path

def convert_to_ndjson(input_json, output_ndjson):
    """
    Convert quantum expectation values from JSON to NDJSON format.
    
    Args:
        input_json: Path to expectation_T00.json from LQG solver
        output_ndjson: Output path for T00_quantum.ndjson
    """
    print(f"Converting quantum T00 data: {input_json} → {output_ndjson}")
    
    # Load quantum expectation data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Expected format: {"r": [r₁, r₂, …], "T00": [T00₁, T00₂, …]}
    if "r" not in data or "T00" not in data:
        raise ValueError(f"Quantum data must contain 'r' and 'T00' arrays, found: {list(data.keys())}")
    
    r_values = np.array(data["r"])
    t00_values = np.array(data["T00"])
    
    if len(r_values) != len(t00_values):
        raise ValueError(f"Mismatched array lengths: r={len(r_values)}, T00={len(t00_values)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_ndjson), exist_ok=True)
    
    # Convert to NDJSON format
    with open(output_ndjson, 'w') as f:
        writer = ndjson.writer(f)
        for ri, t00i in zip(r_values, t00_values):
            writer.writerow({
                "r": float(ri),
                "T00": float(t00i),
                "source": "lqg_quantum",
                "units": "planck"
            })
    
    print(f"✓ Converted {len(r_values)} quantum data points")
    return len(r_values)

def convert_E_to_ndjson(input_json, output_ndjson):
    """
    Convert quantum E-field expectation values from JSON to NDJSON format.
    
    Args:
        input_json: Path to expectation_E.json from LQG solver
        output_ndjson: Output path for E_quantum.ndjson
    """
    print(f"Converting quantum E-field data: {input_json} → {output_ndjson}")
    
    # Load quantum expectation data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Expected format: {"r": [r₁, r₂, …], "E_x": [E_x₁, E_x₂, …], "E_phi": [E_phi₁, E_phi₂, …]}
    if not all(key in data for key in ["r", "E_x", "E_phi"]):
        raise ValueError(f"Invalid quantum E-field data format in {input_json}")
    
    # Write NDJSON format: Each line is {"r": r_i, "E_x": E_x_i, "E_phi": E_phi_i}
    with open(output_ndjson, 'w') as f:
        writer = ndjson.writer(f)
        for i, (r_i, ex_i, ephi_i) in enumerate(zip(data["r"], data["E_x"], data["E_phi"])):
            writer.writerow({"r": r_i, "E_x": ex_i, "E_phi": ephi_i})
    
    print(f"✓ Converted {len(data['r'])} E-field data points to {output_ndjson}")
    return output_ndjson

def build_T00_interpolator(quantum_ndjson):
    """
    Build a numerical interpolator for T00(r) from quantum NDJSON data.
    
    Args:
        quantum_ndjson: Path to T00_quantum.ndjson
        
    Returns:
        Callable T00_num(r) function for integration
    """
    with open(quantum_ndjson, 'r') as f:
        quantum_data = ndjson.load(f)
    
    if not quantum_data:
        raise ValueError(f"No quantum data found in {quantum_ndjson}")
    
    # Extract r and T00 arrays
    rs = np.array([entry["r"] for entry in quantum_data])
    t00s = np.array([entry["T00"] for entry in quantum_data])
    
    # Sort by r to ensure monotonic interpolation
    sort_idx = np.argsort(rs)
    rs = rs[sort_idx]
    t00s = t00s[sort_idx]
    
    # Build interpolator with extrapolation
    T00_num = interp1d(rs, t00s, kind='cubic', fill_value="extrapolate", bounds_error=False)
    
    print(f"✓ Built T00 interpolator from {len(rs)} quantum points")
    print(f"  Range: r ∈ [{rs.min():.2e}, {rs.max():.2e}] (Planck units)")
    print(f"  T00 range: [{t00s.min():.2e}, {t00s.max():.2e}]")
    
    return T00_num

def build_T00_interpolator_negative_energy(quantum_ndjson_path):
    """
    Build interpolation function for T^00(r) from quantum data.
    
    Args:
        quantum_ndjson_path: Path to T00_quantum.ndjson file
        
    Returns:
        callable: f(r) that returns interpolated T^00 value at any r
    """
    print(f"Building T^00 interpolator from {quantum_ndjson_path}")
    
    # Read quantum T00 data points
    with open(quantum_ndjson_path, 'r') as f:
        data = ndjson.load(f)
    
    # Extract r and T00 arrays
    r_values = np.array([point["r"] for point in data])
    T00_values = np.array([point["T00"] for point in data])
    
    if len(r_values) < 4:
        raise ValueError(f"Not enough data points in {quantum_ndjson_path} for reliable interpolation")
    
    # Sort by r if not already sorted
    if not np.all(np.diff(r_values) >= 0):
        sort_indices = np.argsort(r_values)
        r_values = r_values[sort_indices]
        T00_values = T00_values[sort_indices]
    
    # Create cubic spline interpolator with extrapolation
    T00_interp = interp1d(r_values, T00_values, kind='cubic', 
                          bounds_error=False, fill_value="extrapolate")
    
    print(f"✓ T^00 interpolator created from {len(data)} data points")
    print(f"  r range: [{r_values.min():.2e}, {r_values.max():.2e}]")
    print(f"  T^00 range: [{T00_values.min():.2e}, {T00_values.max():.2e}]")
    
    return T00_interp

def validate_quantum_data(quantum_inputs_dir):
    """
    Validate quantum data files in the given directory.
    
    Args:
        quantum_inputs_dir: Directory containing quantum data files
        
    Returns:
        dict: Validation results including valid flag and any errors
    """
    print(f"Validating quantum data in {quantum_inputs_dir}...")
    
    required_files = [
        "expectation_T00.json",
        "expectation_E.json",
        "T00_quantum.ndjson",
        "E_quantum.ndjson"
    ]
    
    # Check which files exist
    files_found = []
    files_missing = []
    errors = []
    
    for filename in required_files:
        filepath = os.path.join(quantum_inputs_dir, filename)
        if os.path.exists(filepath):
            files_found.append(filename)
            try:
                # Validate content format
                if filename.endswith('.json'):
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    if filename == "expectation_T00.json":
                        if not all(key in data for key in ["r", "T00"]):
                            errors.append(f"{filename}: Missing 'r' or 'T00' key")
                    elif filename == "expectation_E.json":
                        if not all(key in data for key in ["r", "E_x", "E_phi"]):
                            errors.append(f"{filename}: Missing 'r', 'E_x', or 'E_phi' key")
                
                elif filename.endswith('.ndjson'):
                    with open(filepath, 'r') as f:
                        data = ndjson.load(f)
                    
                    if len(data) == 0:
                        errors.append(f"{filename}: Empty NDJSON file")
                        
                    if filename == "T00_quantum.ndjson":
                        if not all("r" in entry and "T00" in entry for entry in data[:3]):
                            errors.append(f"{filename}: Missing 'r' or 'T00' field")
                    elif filename == "E_quantum.ndjson":
                        if not all("r" in entry and "E_x" in entry and "E_phi" in entry for entry in data[:3]):
                            errors.append(f"{filename}: Missing field(s) in entries")
            
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")
        else:
            files_missing.append(filename)
    
    # Check if we have at least the minimum required files
    valid = len(files_found) >= 2 and "expectation_T00.json" in files_found
    
    validation_results = {
        "valid": valid and not errors,
        "files_found": files_found,
        "files_missing": files_missing,
        "errors": errors
    }
    
    return validation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert quantum LQG data to pipeline format")
    parser.add_argument("--quantum-dir", default="quantum_inputs",
                       help="Directory containing quantum solver outputs")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate quantum data, don't convert")
    
    args = parser.parse_args()
    
    if args.validate_only:
        results = validate_quantum_data(args.quantum_dir)
        print("Quantum data validation results:")
        print(f"Valid: {results['valid']}")
        print(f"Files found: {len(results['files_found'])}")
        print(f"Files missing: {len(results['files_missing'])}")
        if results["errors"]:
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
    else:
        # Convert both T00 and E data
        quantum_dir = Path(args.quantum_dir)
        
        # Convert T00 data
        t00_input = quantum_dir / "expectation_T00.json"
        t00_output = quantum_dir / "T00_quantum.ndjson"
        if t00_input.exists():
            convert_to_ndjson(str(t00_input), str(t00_output))
        else:
            print(f"Warning: {t00_input} not found")
        
        # Convert E data
        e_input = quantum_dir / "expectation_E.json"
        e_output = quantum_dir / "E_quantum.ndjson"
        if e_input.exists():
            convert_E_to_ndjson(str(e_input), str(e_output))
        else:
            print(f"Warning: {e_input} not found")
