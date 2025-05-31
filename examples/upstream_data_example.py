#!/usr/bin/env python3
"""
Example script showing how to use upstream data from warp-sensitivity-analysis
for wormhole generation in the predictive framework.

This demonstrates the key workflow described in your request:
1. Load strong curvature data (blackhole_curvature.ndjson, cosmo_data.ndjson)
2. Find Planck-scale candidates
3. Extract throat parameters for wormhole generation
4. Optionally load semi-classical or consistency check data
"""

import json
from pathlib import Path
from data_import import DataImporter, load_for_wormhole_generation, get_upstream_data

def main():
    print("=== Warp Predictive Framework - Upstream Data Import Example ===\n")
    
    # Method 1: Minimal dataset for standalone wormhole generation
    print("1. Loading minimal dataset for wormhole generation...")
    try:
        wormhole_data = load_for_wormhole_generation()
        
        # Extract key information
        throat_params = wormhole_data['throat_parameters']
        candidates = wormhole_data['candidates']
        data = wormhole_data['data']
        
        print(f"✓ Throat radius: {throat_params['throat_radius']:.2e} m")
        print(f"✓ Source: {throat_params['source']}")
        print(f"✓ Background type: {throat_params['background_type']}")
        print(f"✓ Found {len(candidates)} Planck-scale candidates")
        print(f"✓ Available data keys: {list(data.keys())}")
        
        # Example: Use the first black hole curvature entry as background
        if 'blackhole_curvature' in data and data['blackhole_curvature']:
            background = data['blackhole_curvature'][0]
            print(f"✓ First background entry label: {background.get('label', 'unlabeled')}")
            
            # This is how you'd extract the throat radius (b0) parameter:
            b0 = throat_params['throat_radius']
            print(f"✓ Using b0 = {b0:.2e} m for Morris-Thorne wormhole")
            
    except FileNotFoundError:
        print("✗ Error: Upstream repository not found")
        print("Make sure warp-sensitivity-analysis is in the parent directory")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    print("\n" + "="*60 + "\n")
    
    # Method 2: Load specific datasets
    print("2. Loading specific datasets...")
    try:
        # Load only strong curvature data
        strong_data = get_upstream_data('strong')
        print(f"✓ Strong curvature data keys: {list(strong_data.keys())}")
        
        # Load semi-classical data (if you need PN corrections)
        semi_data = get_upstream_data('semi_classical')
        print(f"✓ Semi-classical data keys: {list(semi_data.keys())}")
        
        # Load consistency check data (if you need validation)
        consistency_data = get_upstream_data('consistency')
        print(f"✓ Consistency check data keys: {list(consistency_data.keys())}")
        
    except Exception as e:
        print(f"⚠ Warning: Some datasets not available: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Method 3: Direct DataImporter usage for advanced control
    print("3. Using DataImporter directly...")
    try:
        importer = DataImporter()
        print(f"✓ Upstream path: {importer.upstream_path}")
        
        # Find all Planck-scale candidates
        candidates = importer.find_planck_scale_candidates()
        print(f"✓ Planck-scale candidates: {len(candidates)}")
        
        for i, candidate in enumerate(candidates[:3]):  # Show first 3
            print(f"  {i+1}. {candidate['type']} - {candidate['label']}")
            print(f"     Max curvature: {candidate['max_curvature']:.2e} m^-2")
        
        # Get optimal parameters for different background types
        auto_params = importer.get_optimal_throat_parameters('auto')
        print(f"✓ Auto-selected background: {auto_params['source']}")
        
        # Try to get blackhole-specific background
        try:
            bh_params = importer.get_optimal_throat_parameters('blackhole')
            print(f"✓ Blackhole background: {bh_params['source']}")
        except:
            print("⚠ No blackhole backgrounds available")
        
        # Try to get cosmological background
        try:
            cosmo_params = importer.get_optimal_throat_parameters('cosmological')
            print(f"✓ Cosmological background: {cosmo_params['source']}")
        except:
            print("⚠ No cosmological backgrounds available")
            
    except Exception as e:
        print(f"✗ Error with DataImporter: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Method 4: Example wormhole generation workflow
    print("4. Example wormhole generation workflow...")
    try:
        # Step 1: Load data and get optimal parameters
        wormhole_data = load_for_wormhole_generation('auto')
        throat_params = wormhole_data['throat_parameters']
        
        # Step 2: Extract Morris-Thorne parameters
        b0 = throat_params['throat_radius']  # Throat radius
        shape_func = throat_params['shape_function']  # b(r) = b0^2/r
        redshift_func = throat_params['redshift_function']  # Φ(r) = 0
        
        print(f"✓ Morris-Thorne wormhole parameters:")
        print(f"  - Throat radius (b0): {b0:.2e} m")
        print(f"  - Shape function: {shape_func}")
        print(f"  - Redshift function: {redshift_func}")
        print(f"  - Data source: {throat_params['source']}")
        
        # Step 3: Create a simple wormhole solution dictionary
        wormhole_solution = {
            "metric_type": "Morris-Thorne",
            "throat_radius": b0,
            "shape_function": shape_func,
            "redshift_function": redshift_func,
            "data_source": throat_params['source'],
            "background_type": throat_params['background_type'],
            "max_curvature": throat_params.get('max_curvature', 0),
            "coordinates": "isotropic",
            "signature": "(-,+,+,+)"
        }
        
        print(f"✓ Wormhole solution generated")
        
        # Step 4: Save to output (if needed)
        output_path = Path("outputs") / "example_wormhole_solution.json"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(wormhole_solution, f, indent=2)
        
        print(f"✓ Solution saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Error in wormhole generation: {e}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("\nTo use this in your own code:")
    print("1. Import: from data_import import load_for_wormhole_generation")
    print("2. Load: wormhole_data = load_for_wormhole_generation()")
    print("3. Extract: b0 = wormhole_data['throat_parameters']['throat_radius']")
    print("4. Generate your Morris-Thorne wormhole with b0 as the throat radius")

if __name__ == "__main__":
    main()
