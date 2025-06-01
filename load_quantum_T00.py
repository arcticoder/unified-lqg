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
    Convert quantum E field expectation values from JSON to NDJSON format.
    
    Args:
        input_json: Path to expectation_E.json from LQG solver
        output_ndjson: Output path for E_quantum.ndjson
    """
    print(f"Converting quantum E data: {input_json} → {output_ndjson}")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Expected format might vary, but typically:
    # {"r": [...], "Ex": [...], "Ephi": [...]} for spherical midisuperspace
    required_keys = ["r"]
    if not all(key in data for key in required_keys):
        raise ValueError(f"Quantum E data must contain {required_keys}, found: {list(data.keys())}")
    
    r_values = np.array(data["r"])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_ndjson), exist_ok=True)
    
    # Convert to NDJSON format
    with open(output_ndjson, 'w') as f:
        writer = ndjson.writer(f)
        for i, ri in enumerate(r_values):
            entry = {"r": float(ri), "source": "lqg_quantum", "units": "planck"}
            
            # Add all E-field components that are present
            for key in data:
                if key.startswith('E') and len(data[key]) == len(r_values):
                    entry[key] = float(data[key][i])
            
            writer.writerow(entry)
    
    print(f"✓ Converted {len(r_values)} quantum E field data points")
    return len(r_values)

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

def validate_quantum_data(quantum_dir="quantum_inputs"):
    """
    Validate quantum input data before processing.
    
    Args:
        quantum_dir: Directory containing quantum solver outputs
        
    Returns:
        dict: Validation results
    """
    results = {
        "valid": True,
        "files_found": [],
        "files_missing": [],
        "data_points": {},
        "errors": []
    }
    
    expected_files = [
        "expectation_T00.json",
        "expectation_E.json"
    ]
    
    for filename in expected_files:
        filepath = Path(quantum_dir) / filename
        if filepath.exists():
            results["files_found"].append(str(filepath))
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if "r" in data:
                        results["data_points"][filename] = len(data["r"])
                    else:
                        results["errors"].append(f"{filename}: missing 'r' array")
                        results["valid"] = False
            except Exception as e:
                results["errors"].append(f"{filename}: {str(e)}")
                results["valid"] = False
        else:
            results["files_missing"].append(str(filepath))
            results["valid"] = False
    
    return results

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
