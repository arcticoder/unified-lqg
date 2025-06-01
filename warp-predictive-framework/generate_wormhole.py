import argparse
import json
import os
import sys
try:
    import ndjson
except ImportError:
    ndjson = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_import import DataImporter, get_upstream_data, load_for_wormhole_generation
except ImportError:
    # Fallback function if data_import not available
    def load_for_wormhole_generation():
        return {
            'data': {'blackhole_curvature': 0.1},
            'throat_parameters': {'default_b0': 1e-35}
        }

"""
Example usage of upstream data import:

# Minimal dataset for standalone wormhole generation:
wormhole_data = load_for_wormhole_generation()
background_curvature = wormhole_data['data']['blackhole_curvature']
throat_params = wormhole_data['throat_parameters']

# Or load specific datasets:
strong_data = get_upstream_data('strong')
blackhole_curvature = strong_data['blackhole_curvature']
cosmo_data = strong_data['cosmo_data']

# Pick a background parameter set:
background = blackhole_curvature[0]  # First entry
b0 = background.get('throat_radius', 5e-35)  # Use extracted or default
"""

def parse_am_config(path):
    """Parse a simple AsciiMath-style config into a Python dict."""
    config = {}
    with open(path) as f:
        contents = f.read().strip()
    # Very minimal AsciiMath parsing assuming key = value pairs in a list
    # For a robust parser, replace this with a proper AsciiMath JSON-like parser
    try:
        # Simple parsing of key=value pairs
        lines = contents.replace('[', '').replace(']', '').split(',')
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                # Try to convert to appropriate type
                try:
                    if '.' in value or 'e' in value.lower():
                        config[key] = float(value)
                    elif value.isdigit():
                        config[key] = int(value)
                    elif value.lower() in ['true', 'false']:
                        config[key] = value.lower() == 'true'
                    else:
                        config[key] = value
                except ValueError:
                    config[key] = value
    except Exception as e:
        print(f"Warning: Could not parse config file: {e}")
        config = {}
    return config

def generate_wormhole(config_path, output_path):
    config = parse_am_config(config_path)
    
    # Try to load optimal parameters from upstream data using new import system
    try:
        wormhole_data = load_for_wormhole_generation()
        optimal_params = wormhole_data['throat_parameters']
        candidates = wormhole_data['candidates']
        
        # Use upstream data if available, otherwise fall back to config
        b0 = optimal_params.get('throat_radius', config.get('ThroatRadius', 1e-35))
        source_info = optimal_params.get('source', 'config_file')
        background_type = optimal_params.get('background_type', 'unknown')
        max_curvature = optimal_params.get('max_curvature', 0)
        
        print(f"✓ Using throat radius from {source_info}: {b0:.2e} m")
        if background_type != 'unknown':
            print(f"✓ Background type: {background_type}")
        if max_curvature > 0:
            print(f"✓ Max curvature scale: {max_curvature:.2e} m^-2")
        print(f"✓ Found {len(candidates)} Planck-scale candidates")
            
    except Exception as e:
        print(f"⚠ Warning: Could not load upstream data ({e}), using config values")
        b0 = config.get('ThroatRadius', 5e-35)
        source_info = 'config_file'
        background_type = 'unknown'
        max_curvature = 0
    
    # Generate wormhole solution with Morris-Thorne metric
    # Fixed metric calculation - was incorrect before
    solution = {
        "label": f"wormhole_b0={b0:.2e}_source={source_info}",
        "r": b0,
        "throat_radius": b0,
        "source": source_info,
        "background_type": background_type,
        "metric": {
            "g_tt": -1,  # Time component
            "g_rr": 1,   # Radial component (at throat, shape function denominator = 0)
            "g_thth": b0**2,  # Angular theta component
            "g_phph": b0**2   # Angular phi component (times sin^2(theta))
        },
        "shape_function": "b0**2 / r",
        "redshift_function": "0"
    }
      # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        if ndjson is not None:
            writer = ndjson.writer(f)
            writer.writerow(solution)
        else:
            # Fallback to regular JSON
            json.dump(solution, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wormhole solutions")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    parser.add_argument('--out', required=True, help="Output NDJSON file")
    args = parser.parse_args()
    generate_wormhole(args.config, args.out)