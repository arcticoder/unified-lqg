import argparse
import ndjson
import os
import numpy as np
from data_import import DataImporter, get_upstream_data

def compute_lifetime(input_path, config_path, output_path):
    # Read stability spectrum
    with open(input_path) as f:
        data = ndjson.load(f)
    
    # Try to load upstream semi-classical data for PN corrections
    try:
        importer = DataImporter()
        semi_classical_data = importer.load_semi_classical_data()
        strong_data = importer.load_strong_curvature_data()
        print("Loaded upstream data for lifetime computation")
        
        # Extract PN correction factors if available
        pn_factor = 1.0
        if 'pn_predictions' in semi_classical_data and semi_classical_data['pn_predictions']:
            # Use PN data to modify lifetime estimates
            pn_entry = semi_classical_data['pn_predictions'][0]
            if 'correction_factor' in pn_entry:
                pn_factor = pn_entry['correction_factor']
        
    except Exception as e:
        print(f"Warning: Could not load upstream data ({e}), using simplified estimates")
        semi_classical_data = {}
        strong_data = {}
        pn_factor = 1.0
    
    estimates = []
    for entry in data:
        # Extract basic parameters
        throat_radius = entry.get('throat_radius', 1e-35)
        growth_rate = entry.get('growth_rate', 1e42)
        eigenvalue = entry.get('eigenvalue', -0.1)
        background_type = entry.get('background_type', 'unknown')
        
        # Compute instability timescale
        if growth_rate > 0:
            instability_time = 1.0 / growth_rate
        else:
            instability_time = float('inf')  # Stable mode
        
        # Compute semi-classical evaporation time (Hawking-like)
        # T_evap ~ (M/M_planck)^3 * t_planck, but for wormhole throats
        # Use throat radius as characteristic scale
        planck_time = 5.39e-44  # seconds
        planck_length = 1.61e-35  # meters
        
        # Scale based on throat size and PN corrections
        evap_time_raw = planck_time * (throat_radius / planck_length)**3
        evap_time = evap_time_raw * pn_factor
        
        # Compare instability vs evaporation timescales
        dominant_process = "instability" if instability_time < evap_time else "evaporation"
        total_lifetime = min(instability_time, evap_time)
        
        # Create lifetime estimate
        estimate = {
            "label": entry["label"].replace("_mode0", ""),
            "parent_mode": entry["label"],
            "initial_radius": throat_radius,
            "background_type": background_type,
            "dominant_growth_rate": growth_rate,
            "dominant_eigenvalue": eigenvalue,
            "estimated_instability_time": instability_time,
            "semi_classical_evap_time": evap_time,
            "pn_correction_factor": pn_factor,
            "dominant_process": dominant_process,
            "total_lifetime": total_lifetime,
            "analysis_method": "upstream_informed" if semi_classical_data else "simplified"
        }
        
        estimates.append(estimate)    
    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        writer = ndjson.writer(f)
        for estimate in estimates:
            writer.writerow(estimate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute wormhole lifetime estimates")
    parser.add_argument('--input', required=True, help="Input stability spectrum NDJSON")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    parser.add_argument('--out', required=True, help="Output lifetime estimates NDJSON")
    args = parser.parse_args()
    compute_lifetime(args.input, args.config, args.out)