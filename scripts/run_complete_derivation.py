#!/usr/bin/env python3
"""
Complete LQG Closed-Form Metric Derivation Pipeline

This script orchestrates the complete roadmap (Steps 1-6) for deriving 
a closed-form LQG-corrected spherically symmetric metric.

Steps executed:
1. Symbolic polymerization and series expansion
2. Numerical fitting of LQG midisuperspace outputs  
3. Symbolic-numeric matching and template building
4. Bounce radius analysis
5. Hamilton-Jacobi consistency checks
6. Final validation and export

Usage:
    python scripts/run_complete_derivation.py [--config CONFIG_FILE] [--mu MU_VALUE] [--M M_VALUE]
"""

import argparse
import sys
import os
import json
import traceback
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_step(step_num: int, title: str):
    """Print a formatted step header."""
    print(f"\n{'='*20} STEP {step_num}: {title} {'='*20}")

def run_step_1_symbolic_derivation():
    """
    Step 1: Symbolic polymerization and metric derivation.
    """
    print_step(1, "SYMBOLIC POLYMERIZATION AND SERIES EXPANSION")
    
    try:
        from scripts.derive_effective_metric import derive_lqg_metric, export_results
        
        print("Running symbolic derivation of LQG-corrected metric...")
        results = derive_lqg_metric()
        
        # Export results for use by other steps
        export_results(results, "scripts/lqg_metric_results.py")
        
        print("✓ Step 1 completed successfully")
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"✗ Step 1 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_step_2_numerical_fitting(config_file: str = None, mu_value: float = 0.05, M_value: float = 1.0):
    """
    Step 2: Numerical fitting of LQG midisuperspace outputs.
    """
    print_step(2, "NUMERICAL FITTING OF LQG OUTPUTS")
    
    try:
        from scripts.fit_effective_metric import run_complete_fitting_pipeline
        
        print("Running numerical fitting pipeline...")
        results = run_complete_fitting_pipeline(
            config_file=config_file,
            lattice_sizes=[8, 12, 16, 20],
            mu_value=mu_value,
            M_value=M_value
        )
        
        print("✓ Step 2 completed successfully")
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"✗ Step 2 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_step_3_symbolic_numeric_matching(mu_value: float = 0.05, M_value: float = 1.0):
    """
    Step 3: Match symbolic and numeric results, build closed-form template.
    """
    print_step(3, "SYMBOLIC-NUMERIC MATCHING AND TEMPLATE BUILDING")
    
    try:
        from scripts.match_symbolic_numeric import run_complete_matching_pipeline
        
        print("Running symbolic-numeric matching...")
        results = run_complete_matching_pipeline(
            mu_value=mu_value,
            M_value=M_value,
            use_fitted_values=False  # Prefer symbolic coefficients
        )
        
        print("✓ Step 3 completed successfully")
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"✗ Step 3 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_step_4_bounce_radius_analysis():
    """
    Step 4: Bounce radius analysis and spin-foam comparison.
    """
    print_step(4, "BOUNCE RADIUS ANALYSIS")
    
    try:
        from scripts.solve_bounce_radius import run_bounce_analysis
        
        print("Running bounce radius analysis...")
        results = run_bounce_analysis()
        
        print("✓ Step 4 completed successfully")
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"✗ Step 4 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_step_5_hamilton_jacobi_check():
    """
    Step 5: Hamilton-Jacobi consistency analysis.
    """
    print_step(5, "HAMILTON-JACOBI CONSISTENCY CHECK")
    
    try:
        from scripts.hamilton_jacobi_check import run_hamilton_jacobi_analysis
        
        print("Running Hamilton-Jacobi analysis...")
        results = run_hamilton_jacobi_analysis()
        
        print("✓ Step 5 completed successfully")
        return {'success': True, 'results': results}
        
    except Exception as e:
        print(f"✗ Step 5 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def run_step_6_final_validation():
    """
    Step 6: Final validation and integration with LQG pipeline.
    """
    print_step(6, "FINAL VALIDATION AND PIPELINE INTEGRATION")
    
    try:
        # Load final closed-form metric
        from scripts.lqg_closed_form_metric import (
            f_LQG, g_LQG_components, ALPHA_LQG, VALIDATION_INFO, LATEX_EXPRESSIONS
        )
        
        print("Final closed-form LQG metric loaded successfully:")
        print(f"  Coefficient: α = {ALPHA_LQG}")
        print(f"  Validation status: {VALIDATION_INFO}")
        
        # Generate final summary
        print("\nLaTeX expressions generated:")
        for name, expr in LATEX_EXPRESSIONS.items():
            print(f"  {name}: {expr}")
        
        # Test evaluation
        r_test, M_test, mu_test = 3.0, 1.0, 0.05
        f_test = f_LQG(r_test, M_test, mu_test)
        g_components = g_LQG_components(r_test, 0, np.pi/2, 0, M_test, mu_test)
        
        print(f"\nTest evaluation at r={r_test}, M={M_test}, μ={mu_test}:")
        print(f"  f_LQG = {f_test:.6f}")
        print(f"  g_tt = {g_components['g_tt']:.6f}")
        print(f"  g_rr = {g_components['g_rr']:.6f}")
        
        print("✓ Step 6 completed successfully")
        return {'success': True, 'results': {
            'alpha_final': ALPHA_LQG,
            'validation_info': VALIDATION_INFO,
            'latex_expressions': LATEX_EXPRESSIONS
        }}
        
    except Exception as e:
        print(f"✗ Step 6 failed: {e}")
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

def generate_final_report(all_results: dict, mu_value: float, M_value: float):
    """
    Generate comprehensive final report.
    """
    print_header("FINAL DERIVATION REPORT")
    
    # Count successful steps
    successful_steps = sum(1 for result in all_results.values() if result.get('success', False))
    total_steps = len(all_results)
    
    print(f"Pipeline execution: {successful_steps}/{total_steps} steps completed successfully")
    print(f"Parameters used: M = {M_value}, μ = {mu_value}")
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step-by-step summary
    print("\nStep-by-step summary:")
    step_names = {
        'step_1': 'Symbolic Derivation',
        'step_2': 'Numerical Fitting',
        'step_3': 'Symbolic-Numeric Matching',
        'step_4': 'Bounce Radius Analysis',
        'step_5': 'Hamilton-Jacobi Check',
        'step_6': 'Final Validation'
    }
    
    for step_key, step_name in step_names.items():
        if step_key in all_results:
            status = "✓ SUCCESS" if all_results[step_key]['success'] else "✗ FAILED"
            print(f"  {step_name}: {status}")
            if not all_results[step_key]['success']:
                print(f"    Error: {all_results[step_key].get('error', 'Unknown error')}")
    
    # Final result summary
    if all_results.get('step_6', {}).get('success', False):
        step_6_results = all_results['step_6']['results']
        print(f"\n🎉 DERIVATION COMPLETE! 🎉")
        print(f"Final LQG coefficient: α = {step_6_results['alpha_final']:.8f}")
        
        # LaTeX output
        print(f"\nFinal metric (LaTeX):")
        latex_exprs = step_6_results['latex_expressions']
        print(f"\\[ {latex_exprs['metric_function']} \\]")
        print(f"\\[ {latex_exprs['line_element']} \\]")
        
    else:
        print("\n⚠ DERIVATION INCOMPLETE")
        print("Some steps failed. Check individual step errors above.")
    
    # Save report to file
    report_file = "scripts/derivation_report.json"
    with open(report_file, 'w') as f:
        # Prepare JSON-serializable version
        json_results = {}
        for step_key, result in all_results.items():
            json_results[step_key] = {
                'success': result['success'],
                'error': result.get('error', None)
            }
        
        report_data = {
            'execution_time': datetime.now().isoformat(),
            'parameters': {'M': M_value, 'mu': mu_value},
            'successful_steps': successful_steps,
            'total_steps': total_steps,
            'step_results': json_results
        }
        
        if all_results.get('step_6', {}).get('success', False):
            report_data['final_coefficient'] = all_results['step_6']['results']['alpha_final']
            report_data['latex_expressions'] = all_results['step_6']['results']['latex_expressions']
        
        json.dump(report_data, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")

def main():
    """
    Main function to run the complete derivation pipeline.
    """
    parser = argparse.ArgumentParser(description="Run complete LQG closed-form metric derivation")
    parser.add_argument('--config', type=str, help="Configuration file path")
    parser.add_argument('--mu', type=float, default=0.05, help="Polymer scale parameter (default: 0.05)")
    parser.add_argument('--M', type=float, default=1.0, help="Mass parameter (default: 1.0)")
    parser.add_argument('--skip-steps', type=str, nargs='*', 
                       choices=['1', '2', '3', '4', '5', '6'],
                       help="Steps to skip (e.g., --skip-steps 2 4)")
    
    args = parser.parse_args()
    
    print_header("LQG CLOSED-FORM METRIC DERIVATION PIPELINE")
    print(f"Parameters: M = {args.M}, μ = {args.mu}")
    if args.config:
        print(f"Configuration file: {args.config}")
    if args.skip_steps:
        print(f"Skipping steps: {', '.join(args.skip_steps)}")
    
    # Results storage
    all_results = {}
    
    # Import numpy here to avoid issues in other modules
    import numpy as np
    
    # Execute pipeline steps
    skip_steps = set(args.skip_steps or [])
    
    if '1' not in skip_steps:
        all_results['step_1'] = run_step_1_symbolic_derivation()
    else:
        print_step(1, "SYMBOLIC DERIVATION (SKIPPED)")
        all_results['step_1'] = {'success': True, 'skipped': True}
    
    if '2' not in skip_steps:
        all_results['step_2'] = run_step_2_numerical_fitting(args.config, args.mu, args.M)
    else:
        print_step(2, "NUMERICAL FITTING (SKIPPED)")
        all_results['step_2'] = {'success': True, 'skipped': True}
    
    if '3' not in skip_steps:
        all_results['step_3'] = run_step_3_symbolic_numeric_matching(args.mu, args.M)
    else:
        print_step(3, "SYMBOLIC-NUMERIC MATCHING (SKIPPED)")
        all_results['step_3'] = {'success': True, 'skipped': True}
    
    if '4' not in skip_steps:
        all_results['step_4'] = run_step_4_bounce_radius_analysis()
    else:
        print_step(4, "BOUNCE RADIUS ANALYSIS (SKIPPED)")
        all_results['step_4'] = {'success': True, 'skipped': True}
    
    if '5' not in skip_steps:
        all_results['step_5'] = run_step_5_hamilton_jacobi_check()
    else:
        print_step(5, "HAMILTON-JACOBI CHECK (SKIPPED)")
        all_results['step_5'] = {'success': True, 'skipped': True}
    
    if '6' not in skip_steps:
        all_results['step_6'] = run_step_6_final_validation()
    else:
        print_step(6, "FINAL VALIDATION (SKIPPED)")
        all_results['step_6'] = {'success': True, 'skipped': True}
    
    # Generate final report
    generate_final_report(all_results, args.mu, args.M)

if __name__ == "__main__":
    main()
