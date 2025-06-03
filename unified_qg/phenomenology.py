"""
Phenomenology generation module for quantum-corrected observables.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any


def compute_qnm_frequency(M: float, a: float) -> np.ndarray:
    """Placeholder: return quantum-corrected quasi-normal mode frequencies."""
    return np.array([0.5 / M, 1.2 / M])  # Dummy values


def compute_isco_shift(M: float, a: float) -> float:
    """Placeholder: compute shift in ISCO radius due to quantum corrections."""
    return 6.0 * M * (1 - 0.1 * a)  # Dummy formula


def compute_horizon_area_spectrum(M: float, a: float) -> np.ndarray:
    """Placeholder: return an array of possible horizon areas."""
    radii = np.linspace(2 * M, 4 * M, 10)
    areas = 4 * np.pi * radii**2 * (1 + 0.05 * a)  # Dummy spectrum
    return areas


def generate_qc_phenomenology(
    data_config: Dict[str, Any],
    output_dir: str = "qc_results"
) -> List[Dict[str, Any]]:
    """
    Use the unified pipeline to produce quantum-corrected observables:
    - Quasi-normal mode frequencies
    - ISCO shifts
    - Horizon-area spectra
    """
    os.makedirs(output_dir, exist_ok=True)

    phenomenology_results = []
    masses = data_config.get("masses", [1.0])
    spins = data_config.get("spins", [0.0, 0.5, 0.9])
    for M in masses:
        for a in spins:
            omega_qnm = compute_qnm_frequency(M, a)
            isco_radius = compute_isco_shift(M, a)
            horizon_spectrum = compute_horizon_area_spectrum(M, a)

            result = {
                "mass": M,
                "spin": a,
                "omega_qnm": omega_qnm.tolist(),
                "isco_radius": isco_radius,
                "horizon_spectrum": horizon_spectrum.tolist(),
            }
            filename = f"{output_dir}/qc_M{M:.2f}_a{a:.2f}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            phenomenology_results.append(result)
            print(f"Saved QC results for M={M}, a={a} to {filename}")

    return phenomenology_results
