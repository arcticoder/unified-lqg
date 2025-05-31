#!/usr/bin/env python3
"""
Simple data import test script for warp predictive framework.
This demonstrates loading the key files you need from warp-sensitivity-analysis.
"""

import json
from pathlib import Path

def load_upstream_data():
    """Load the minimal required data from warp-sensitivity-analysis."""
    
    # Path to upstream repository
    upstream_path = Path(__file__).parent.parent / "warp-sensitivity-analysis"
    
    if not upstream_path.exists():
        print(f"Error: Upstream repository not found at {upstream_path}")
        return None
    
    print(f"✓ Found upstream repository at: {upstream_path}")
    
    data = {}
    
    # Load blackhole curvature data (actual filename)
    blackhole_path = upstream_path / "strong_curvature" / "blackhole_data.ndjson"
    if blackhole_path.exists():
        try:
            with open(blackhole_path) as f:
                # Load NDJSON format (one JSON object per line)
                data['blackhole_curvature'] = [json.loads(line) for line in f if line.strip()]
            print(f"✓ Loaded blackhole data: {len(data['blackhole_curvature'])} entries")
        except Exception as e:
            print(f"Warning: Could not load blackhole data: {e}")
    
    # Load cosmological data
    cosmo_path = upstream_path / "strong_curvature" / "cosmo_data.ndjson"
    if cosmo_path.exists():
        try:
            with open(cosmo_path) as f:
                # Load NDJSON format
                data['cosmo_data'] = [json.loads(line) for line in f if line.strip()]
            print(f"✓ Loaded cosmo data: {len(data['cosmo_data'])} entries")
        except Exception as e:
            print(f"Warning: Could not load cosmo data: {e}")
    
    # Load unified strong models if available
    unified_path = upstream_path / "strong_curvature" / "unified_strong_models.ndjson"
    if unified_path.exists():
        try:
            with open(unified_path) as f:
                data['unified_strong_models'] = [json.loads(line) for line in f if line.strip()]
            print(f"✓ Loaded unified models: {len(data['unified_strong_models'])} entries")
        except Exception as e:
            print(f"Warning: Could not load unified models: {e}")
    
    return data

def find_throat_parameters(data):
    """Extract throat parameters from loaded data."""
    
    if not data:
        return {'throat_radius': 5e-35, 'source': 'default'}
    
    # Look for the highest curvature entry in blackhole data
    best_candidate = None
    max_curvature = 0
    
    if 'blackhole_curvature' in data:
        for entry in data['blackhole_curvature']:
            # Look for curvature indicators
            curvature = 0
            if 'max_kretschmann' in entry:
                curvature = entry['max_kretschmann']
            elif 'kretschmann_peak' in entry:
                curvature = entry['kretschmann_peak']
            elif 'curvature_scale' in entry:
                curvature = entry['curvature_scale']
            
            if curvature > max_curvature:
                max_curvature = curvature
                best_candidate = entry
    
    if best_candidate:
        # Extract throat radius from mass or other parameters
        if 'mass' in best_candidate:
            # Estimate Schwarzschild radius as throat radius
            mass = best_candidate['mass']
            G = 6.67e-11  # m^3 kg^-1 s^-2
            c = 3e8       # m/s
            throat_radius = 2 * G * mass / (c**2)
        elif 'r_min' in best_candidate:
            throat_radius = best_candidate['r_min']
        elif 'characteristic_scale' in best_candidate:
            throat_radius = best_candidate['characteristic_scale']
        else:
            # Default to Planck scale
            throat_radius = 1.6e-35  # Planck length
        
        return {
            'throat_radius': throat_radius,
            'source': best_candidate.get('label', 'upstream_data'),
            'max_curvature': max_curvature,
            'background_params': best_candidate
        }
    
    # Fallback to default
    return {'throat_radius': 5e-35, 'source': 'default'}

def example_wormhole_generation():
    """Example of how to use upstream data for wormhole generation."""
    
    print("=== Warp Predictive Framework - Data Import Example ===\n")
    
    # Step 1: Load upstream data
    data = load_upstream_data()
    if not data:
        return
    
    # Step 2: Extract throat parameters
    throat_params = find_throat_parameters(data)
    
    print(f"\n✓ Selected throat parameters:")
    print(f"  - Radius: {throat_params['throat_radius']:.2e} m")
    print(f"  - Source: {throat_params['source']}")
    if 'max_curvature' in throat_params:
        print(f"  - Max curvature: {throat_params['max_curvature']:.2e}")
    
    # Step 3: Create Morris-Thorne wormhole solution
    b0 = throat_params['throat_radius']
    
    wormhole_solution = {
        "metric": "Morris-Thorne",
        "throat_radius": b0,
        "shape_function": f"b(r) = {b0**2:.2e} / r",
        "redshift_function": "Φ(r) = 0",
        "data_source": throat_params['source'],
        "coordinate_system": "isotropic",
        "signature": "(-,+,+,+)"
    }
    
    print(f"\n✓ Generated wormhole solution:")
    for key, value in wormhole_solution.items():
        print(f"  - {key}: {value}")
    
    # Step 4: Save solution
    output_path = Path("outputs") / "example_wormhole.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(wormhole_solution, f, indent=2)
    
    print(f"\n✓ Solution saved to: {output_path}")
    
    # Step 5: Show how to access specific background data
    if 'blackhole_curvature' in data and data['blackhole_curvature']:
        print(f"\n✓ Example background entry keys: {list(data['blackhole_curvature'][0].keys())}")
    
    print("\n" + "="*60)
    print("This demonstrates the basic workflow from your request:")
    print("1. Load strong_curvature/blackhole_data.ndjson")
    print("2. Load strong_curvature/cosmo_data.ndjson") 
    print("3. Pick background with maximum curvature")
    print("4. Extract b0 (throat radius) parameter")
    print("5. Generate Morris-Thorne wormhole")
    print("\nYou can now import this pattern into generate_wormhole.py")

if __name__ == "__main__":
    example_wormhole_generation()
