import argparse
import ndjson
import math
import os

def read_ndjson(path):
    with open(path) as f:
        return ndjson.load(f)

def write_ndjson(obj_list, path):
    with open(path, 'w') as f:
        writer = ndjson.writer(f)
        writer.writerows(obj_list)

def compute_negative_energy(refined_metrics_path, output_path):
    """
    Placeholder: For each refined metric ansatz, compute a dummy negative-energy
    integral ∫ |T^{00}| dV over the throat region. Here we just assign
    NegativeEnergyIntegral = epsilon * b0^2 as a stand-in.
    """
    metrics = read_ndjson(refined_metrics_path)
    results = []

    for entry in metrics:
        label = entry.get("label", "unknown_refined")
        # Extract b0 from parent label (assuming "wormhole_b0=<value>_refined")
        parts = label.split("_refined")[0].split("=")
        try:
            b0 = float(parts[1])
        except (IndexError, ValueError):
            b0 = 1e-35
        # Dummy integral: epsilon * b0^2
        epsilon = 1e-5  # placeholder—replace with actual shape-function parameter
        negative_integral = epsilon * (b0 ** 2)

        results.append({
            "label": label,
            "negative_energy_integral": negative_integral
        })

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    write_ndjson(results, output_path)
    print(f"Wrote {len(results)} negative-energy entries to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute dummy negative-energy integrals for refined metrics."
    )
    parser.add_argument(
        '--refined', required=True,
        help="Path to refined_metrics.ndjson (from metric_refinement.py)."
    )
    parser.add_argument(
        '--out', required=True,
        help="Output NDJSON file for negative-energy integrals."
    )
    args = parser.parse_args()
    compute_negative_energy(args.refined, args.out)