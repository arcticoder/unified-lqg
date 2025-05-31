#!/usr/bin/env python3
"""
Complete Predictive Framework Workflow

This script demonstrates the full pipeline from upstream data import
through wormhole generation, stability analysis, lifetime computation,
and analogue-gravity mapping.
"""

import os
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_import import DataImporter, get_upstream_data
from generate_wormhole import generate_wormhole
from analyze_stability import analyze_stability
from compute_lifetime import compute_lifetime
from map_to_analogue import map_to_analogue
from design_control_field import design_control_field

def run_full_pipeline(config_path="predictive_config.am", verbose=True):
    """Run the complete predictive framework pipeline."""
    
    if verbose:
        print("=" * 60)
        print("WARP PREDICTIVE FRAMEWORK - FULL PIPELINE")
        print("=" * 60)
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
      # Define output file paths
    wormhole_output = "outputs/wormhole_solutions.ndjson"
    stability_output = "outputs/stability_spectrum.ndjson"
    lifetime_output = "outputs/lifetime_estimates.ndjson"
    control_output = "outputs/control_fields/control_fields.ndjson"
    
    try:
        # Step 1: Test upstream data connection
        if verbose:
            print("\n1. Testing upstream data connection...")
        
        importer = DataImporter()
        candidates = importer.find_planck_scale_candidates()
        
        if verbose:
            print(f"   Found {len(candidates)} Planck-scale candidates")
            for i, candidate in enumerate(candidates[:3]):  # Show top 3
                print(f"   {i+1}. {candidate['label']} ({candidate['type']}) - "
                      f"Max curvature: {candidate['max_curvature']:.2e}")
        
        # Step 2: Generate wormhole solutions
        if verbose:
            print("\n2. Generating wormhole solutions...")
        
        generate_wormhole(config_path, wormhole_output)
        
        if verbose:
            print(f"   Wormhole solutions written to {wormhole_output}")
        
        # Step 3: Analyze stability
        if verbose:
            print("\n3. Analyzing stability spectrum...")
        
        analyze_stability(wormhole_output, config_path, stability_output)
        
        if verbose:
            print(f"   Stability spectrum written to {stability_output}")
        
        # Step 4: Compute lifetimes
        if verbose:
            print("\n4. Computing lifetime estimates...")
          
        compute_lifetime(stability_output, config_path, lifetime_output)
        
        if verbose:
            print(f"   Lifetime estimates written to {lifetime_output}")
        
        # Step 5: Design control fields for unstable modes
        if verbose:
            print("\n5. Designing active control fields...")
        
        design_control_field(wormhole_output, stability_output, control_output)
        
        if verbose:
            print(f"   Control field proposals written to {control_output}")
          # Step 6: Map to analogue predictions
        if verbose:
            print("\n6. Mapping to analogue-gravity predictions...")
        
        map_to_analogue(lifetime_output, config_path)
        
        if verbose:
            print("   Analogue predictions written to outputs/")
        
        # Step 7: Summary        if verbose:
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print("\nOutput files generated:")
            print(f"  - {wormhole_output}")
            print(f"  - {stability_output}")
            print(f"  - {lifetime_output}")
            print(f"  - {control_output}")
            print("  - outputs/analogue_predictions.am")
            print("  - outputs/analogue_predictions.ndjson")
            
            # Show data quality summary
            try:
                all_data = get_upstream_data('all')
                strong_count = len(all_data['strong_curvature'].get('blackhole_curvature', []))
                semi_count = len(all_data['semi_classical'].get('pn_predictions', []))
                consistency_count = len(all_data['consistency_checks'].get('consistency_report', []))
                
                print(f"\nUpstream data summary:")
                print(f"  - Strong curvature entries: {strong_count}")
                print(f"  - Semi-classical entries: {semi_count}")
                print(f"  - Consistency checks: {consistency_count}")
                
            except Exception as e:
                print(f"\nWarning: Could not summarize upstream data: {e}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed at some step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_import():
    """Test the data import functionality."""
    print("Testing data import functionality...")
    
    try:
        importer = DataImporter()
        print(f"✓ Upstream path exists: {importer.upstream_path}")
        
        # Test strong curvature data
        strong_data = importer.load_strong_curvature_data()
        print(f"✓ Strong curvature data loaded: {list(strong_data.keys())}")
        
        # Test semi-classical data
        semi_data = importer.load_semi_classical_data()
        print(f"✓ Semi-classical data loaded: {list(semi_data.keys())}")
        
        # Test consistency data
        consistency_data = importer.load_consistency_check_data()
        print(f"✓ Consistency data loaded: {list(consistency_data.keys())}")
        
        # Test candidate finding
        candidates = importer.find_planck_scale_candidates()
        print(f"✓ Found {len(candidates)} Planck-scale candidates")
        
        # Test optimal parameters
        params = importer.get_optimal_throat_parameters()
        print(f"✓ Optimal throat parameters: {params}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data import test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the complete predictive framework pipeline")
    parser.add_argument('--config', default='predictive_config.am', 
                       help='Path to configuration file')
    parser.add_argument('--test-import', action='store_true',
                       help='Test data import functionality only')
    parser.add_argument('--quiet', action='store_true',
                       help='Run in quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    if args.test_import:
        success = test_data_import()
        sys.exit(0 if success else 1)
    
    success = run_full_pipeline(args.config, verbose=not args.quiet)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
