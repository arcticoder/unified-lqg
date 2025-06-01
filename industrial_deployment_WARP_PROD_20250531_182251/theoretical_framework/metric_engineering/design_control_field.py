#!/usr/bin/env python3

import argparse
import ndjson
import json
import os

def read_ndjson(path):
    """Load a list of dicts from an NDJSON file."""
    with open(path, 'r') as f:
        return ndjson.load(f)

def write_ndjson(obj_list, path):
    """Write a list of dicts to an NDJSON file."""
    with open(path, 'w') as f:
        writer = ndjson.writer(f)
        for obj in obj_list:
            writer.writerow(obj)

def design_control_field(
    wormhole_solutions_path,
    stability_spectrum_path,
    output_path
):
    """
    For each unstable mode (eigenvalue < 0) in the stability spectrum, propose a 
    minimal control field δΦ(r) ∝ ψ_mode(r) that would shift the mode frequency 
    ω^2 ≥ 0. This is a placeholder routine: it does not solve PDEs, but records 
    an AsciiMath-style ansatz for δΦ(r).

    Inputs:
      - wormhole_solutions_path: NDJSON file with wormhole solutions
      - stability_spectrum_path: NDJSON file with {label, eigenvalue, eigenfunction_norm, growth_rate, ...}
    
    Output:
      - NDJSON file with one record per unstable mode:
        {
          "mode_label": "..._mode0",
          "eigenvalue": <negative number>,
          "control_field_amplitude": <epsilon>,
          "control_field_ansatz": "δΦ(r) = ε * ψ_mode(r)"
        }
      - ε (epsilon) is chosen heuristically as |eigenvalue| * 1e-2 (placeholder).
    """
    # Load inputs
    stability_data = read_ndjson(stability_spectrum_path)
    wormhole_data = read_ndjson(wormhole_solutions_path)

    control_records = []

    for mode in stability_data:
        ev = mode.get("eigenvalue", 0.0)
        if ev < 0:
            # Identify the corresponding wormhole label
            parent_label = mode["label"].rsplit("_", 1)[0]
            # Heuristic: set epsilon = |eigenvalue| * tolerance_factor
            epsilon = abs(ev) * 1e-2
            # AsciiMath ansatz (placeholder)
            ansatz = f"δΦ(r) = {epsilon:.3e} * ψ_{mode['label']}(r)"
            control_records.append({
                "mode_label": mode["label"],
                "parent_solution": parent_label,
                "eigenvalue": ev,
                "control_field_amplitude": epsilon,
                "control_field_ansatz": ansatz
            })

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    write_ndjson(control_records, output_path)
    print(f"Wrote {len(control_records)} control-field proposals to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design active control fields δΦ(r) to stabilize unstable wormhole modes."
    )
    parser.add_argument(
        '--wormhole', required=True,
        help="Path to wormhole_solutions.ndjson (static solutions)."
    )
    parser.add_argument(
        '--stability', required=True,
        help="Path to stability_spectrum.ndjson (eigenvalues & eigenfunctions)."
    )
    parser.add_argument(
        '--out', required=True,
        help="Output NDJSON file for control-field proposals."
    )

    args = parser.parse_args()
    design_control_field(
        wormhole_solutions_path=args.wormhole,
        stability_spectrum_path=args.stability,
        output_path=args.out
    )
