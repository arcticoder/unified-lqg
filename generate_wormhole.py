import argparse
import ndjson
import json

def parse_am_config(path):
    """Parse a simple AsciiMath-style config into a Python dict."""
    config = {}
    with open(path) as f:
        contents = f.read().strip()
    # Very minimal AsciiMath parsing assuming key = value pairs in a list
    # For a robust parser, replace this with a proper AsciiMath JSON-like parser
    try:
        ast = eval(contents.replace('[', '').replace(']', ''))
        # If eval produces a tuple/list, ignore
    except Exception:
        ast = {}
    return ast

def generate_wormhole(config_path, output_path):
    config = parse_am_config(config_path)
    b0 = config.get('ThroatRadius', 1e-35)
    # For demonstration, generate a single radial point
    solution = {
        "label": f"wormhole_b0={b0}",
        "r": b0,
        "metric": {
            "g_tt": -1,
            "g_rr": 1 / (1 - (b0**2 / b0) / b0),
            "g_thth": b0**2,
            "g_phph": (b0**2) * (1)
        }
    }
    with open(output_path, 'w') as f:
        writer = ndjson.writer(f)
        writer.writerow(solution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate wormhole solutions")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    parser.add_argument('--out', required=True, help="Output NDJSON file")
    args = parser.parse_args()
    generate_wormhole(args.config, args.out)