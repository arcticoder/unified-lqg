#!/usr/bin/env python3
"""
Example: Compare Prescriptions
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from enhanced_alpha_beta_gamma_extraction import main
from alternative_polymer_prescriptions import (
    ThiemannPrescription, AQELPrescription, BojowaldPrescription
)

if __name__ == "__main__":
    print("Running LQG coefficient extraction with prescription comparison...")
    results = main()
    
    print("\nExample usage completed!")
    print("Results saved in results dictionary with keys:")
    for prescription in results.keys():
        print(f"  - {prescription}")
