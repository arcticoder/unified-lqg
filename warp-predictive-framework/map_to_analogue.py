import argparse
import ndjson
import os
import numpy as np
from data_import import DataImporter, get_upstream_data

def map_to_analogue(input_path, config_path):
    # Read lifetime estimates
    with open(input_path) as f:
        lifetime_data = ndjson.load(f)
    
    # Try to load all upstream data for comprehensive analogue mapping
    try:
        importer = DataImporter()
        all_data = get_upstream_data('all')
        strong_data = all_data['strong_curvature']
        semi_classical_data = all_data['semi_classical']
        consistency_data = all_data['consistency_checks']
        
        print("Loaded comprehensive upstream data for analogue mapping")
        
        # Check data quality from consistency checks
        data_reliable = True
        if 'consistency_report' in consistency_data:
            # Use consistency check results to validate analogue mapping
            for check in consistency_data['consistency_report']:
                if not check.get('passed', True):
                    print(f"Warning: Consistency check failed: {check.get('test_name', 'unknown')}")
                    data_reliable = False
        
    except Exception as e:
        print(f"Warning: Could not load upstream data ({e}), using simplified mapping")
        all_data = {}
        data_reliable = False
    
    # Process each lifetime estimate
    analogue_predictions = []
    
    for entry in lifetime_data:
        throat_radius = entry.get('initial_radius', 1e-35)
        total_lifetime = entry.get('total_lifetime', 1e-42)
        background_type = entry.get('background_type', 'unknown')
        pn_factor = entry.get('pn_correction_factor', 1.0)
        
        # Map to BEC phonon analogue system
        # Scale throat radius to BEC healing length scale
        healing_length = 1e-6  # typical BEC healing length in meters
        throat_analog = throat_radius * (healing_length / 1e-35)  # Scale up
        
        # Map lifetime to phonon instability frequency
        if total_lifetime > 0:
            instability_freq = 1.0 / total_lifetime  # Hz
            # Scale to experimental frequencies (kHz range)
            lab_freq = instability_freq * 1e-39  # Bring to observable range
        else:
            lab_freq = 0.0
        
        # Map evaporation time to lab timescale
        evap_time_lab = entry.get('semi_classical_evap_time', 1e-36) * 1e36  # Scale to seconds
        
        # Include quality assessment
        quality = "high" if data_reliable and entry.get('analysis_method') == 'upstream_informed' else "low"
        
        prediction = {
            "wormhole_source": entry["label"],
            "background_type": background_type,
            "analogue_type": "BEC_phonon",
            "throat_radius_analog": f"{throat_analog:.2e} m",
            "instability_freq": f"2π × {lab_freq:.2e} Hz",
            "lab_evap_time": f"{evap_time_lab:.3f} s",
            "pn_correction_applied": pn_factor != 1.0,
            "prediction_quality": quality,
            "consistency_validated": data_reliable
        }
        
        analogue_predictions.append(prediction)
    
    # Generate AsciiMath output format
    output_lines = ["["]
    
    for i, pred in enumerate(analogue_predictions):
        if i > 0:
            output_lines.append(",")
        output_lines.extend([
            f"  wormhole_{i+1} = [",
            f"    source              = \"{pred['wormhole_source']}\",",
            f"    background_type     = \"{pred['background_type']}\",",
            f"    analogue_type       = \"{pred['analogue_type']}\",",
            f"    throat_radius_analog = \"{pred['throat_radius_analog']}\",",
            f"    instability_freq     = \"{pred['instability_freq']}\",",
            f"    lab_evap_time        = \"{pred['lab_evap_time']}\",",
            f"    pn_correction_applied = {str(pred['pn_correction_applied']).lower()},",
            f"    prediction_quality   = \"{pred['prediction_quality']}\",",
            f"    consistency_validated = {str(pred['consistency_validated']).lower()}",
            "  ]"
        ])
    
    output_lines.append("]")
    
    # Write to outputs
    output_path = "outputs/analogue_predictions.am"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines))
    
    # Also save as NDJSON for easier processing    ndjson_path = "outputs/analogue_predictions.ndjson"
    with open(ndjson_path, 'w') as f:
        writer = ndjson.writer(f)
        for prediction in analogue_predictions:
            writer.writerow(prediction)
    
    print(f"Analogue predictions written to {output_path} and {ndjson_path}")
    print(f"Generated {len(analogue_predictions)} analogue predictions")
    if data_reliable:
        print("Predictions are based on validated upstream data")
    else:
        print("Warning: Predictions are based on simplified models - upstream data validation failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map wormhole lifetimes to analogue outputs")
    parser.add_argument('--input', required=True, help="Input lifetime estimates NDJSON")
    parser.add_argument('--config', required=True, help="Path to predictive_config.am")
    args = parser.parse_args()
    map_to_analogue(args.input, args.config)