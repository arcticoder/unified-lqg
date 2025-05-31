import argparse
import ndjson

def compute_lifetime(input_path, config_path, output_path):
    # Placeholder: read stability spectrum and write dummy lifetime estimates
    with open(input_path) as f:
        data = ndjson.load(f)
    estimates = []
    for entry in data:
        estimates.append({
            "label": entry["label"].replace("_mode0", ""),
            "initial_radius": float(entry["label"].split('=')[1].split('_')[0]),
            "dominant_growth_rate": entry["growth_rate"],
            "estimated_instability_time": 1e-42,
            "semi_classical_evap_time": 1e-36
        })
    with open(output_path, 'w') as f:
        writer = ndjson.writer(f)
        writer.writerows(estimates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wormhole lifetime estimates")
    parser.add_argument('--input', required=True, help="Input stability spectrum NDJSON")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    parser.add_argument('--out', required=True, help="Output lifetime estimates NDJSON")
    args = parser.parse_args()
    compute_lifetime(args.input, args.config, args.out)