import argparse
import ndjson

def analyze_stability(input_path, config_path, output_path):
    # Placeholder: read wormhole solutions and write a dummy stability spectrum
    with open(input_path) as f:
        data = ndjson.load(f)
    spectrum = []
    for entry in data:
        spectrum.append({
            "label": entry["label"] + "_mode0",
            "eigenvalue": -0.1,
            "eigenfunction_norm": 1.0,
            "growth_rate": 1e+42,
            "stable": False
        })
    with open(output_path, 'w') as f:
        writer = ndjson.writer(f)
        writer.writerows(spectrum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze wormhole stability")
    parser.add_argument('--input', required=True, help="Input wormhole solutions NDJSON")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    parser.add_argument('--out', required=True, help="Output stability spectrum NDJSON")
    args = parser.parse_args()
    analyze_stability(args.input, args.config, args.out)