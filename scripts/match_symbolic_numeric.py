#!/usr/bin/env python3
"""
Match symbolic series to numeric fit and build closed-form LQG metric template.

This script implements Step 3 of the roadmap:
- Compare symbolic α coefficient with fitted γ parameter
- Build final closed-form template for LQG-corrected metric
- Generate LaTeX expressions and numerical evaluation functions
- Validate consistency between analytical and numerical approaches
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from typing import Dict, Any, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_symbolic_results() -> Dict[str, Any]:
    """
    Load symbolic results from derivation script.
    
    Returns:
        Dictionary with symbolic coefficients and expressions
    """
    try:
        from scripts.lqg_metric_results import alpha_star, f_LQG, evaluate_f_LQG
        print(f"Loaded symbolic coefficient: α = {alpha_star}")
        return {
            'alpha_star': alpha_star,
            'f_LQG': f_LQG,
            'evaluate_f_LQG': evaluate_f_LQG,
            'available': True
        }
    except ImportError:
        print("Warning: Symbolic results not available. Run derive_effective_metric.py first.")
        return {
            'alpha_star': 1/6,  # Placeholder
            'available': False
        }

def load_fitting_results() -> Dict[str, Any]:
    """
    Load numerical fitting results.
    
    Returns:
        Dictionary with fitting results
    """
    try:
        with open('scripts/fitting_results.json', 'r') as f:
            results = json.load(f)
        print("Loaded numerical fitting results")
        return results
    except FileNotFoundError:
        print("Warning: Fitting results not found. Run fit_effective_metric.py first.")
        return None

def compare_symbolic_vs_numeric(symbolic_results: Dict[str, Any],
                               fitting_results: Dict[str, Any],
                               mu_value: float,
                               M_value: float) -> Dict[str, Any]:
    """
    Compare symbolic coefficient with numerical fit.
    
    Args:
        symbolic_results: Results from symbolic derivation
        fitting_results: Results from numerical fitting
        mu_value: Polymer scale parameter
        M_value: Mass parameter
        
    Returns:
        Comparison results and validation status
    """
    print("Comparing symbolic vs numerical results...")
    
    if fitting_results is None or not fitting_results['fit_results']['success']:
        print("Cannot compare - fitting results unavailable or failed")
        return {'success': False}
    
    # Extract values
    alpha_symbolic = float(symbolic_results['alpha_star'])
    gamma_fitted = fitting_results['fit_results']['gamma']
    
    # Theoretical prediction: γ_theory = α * μ² * M²
    gamma_theory = alpha_symbolic * (mu_value**2) * (M_value**2)
    
    # Comparison metrics
    absolute_error = abs(gamma_fitted - gamma_theory)
    relative_error = absolute_error / abs(gamma_theory) if gamma_theory != 0 else float('inf')
    
    # Validation thresholds
    tolerance_strict = 0.05  # 5% for excellent agreement
    tolerance_acceptable = 0.15  # 15% for acceptable agreement
    
    validation_level = 'excellent' if relative_error < tolerance_strict else \
                      'acceptable' if relative_error < tolerance_acceptable else \
                      'poor'
    
    print(f"  Symbolic coefficient: α = {alpha_symbolic:.6f}")
    print(f"  Theoretical prediction: γ_theory = α*μ²*M² = {gamma_theory:.6f}")
    print(f"  Numerical fit: γ_fitted = {gamma_fitted:.6f}")
    print(f"  Absolute error: |Δγ| = {absolute_error:.6f}")
    print(f"  Relative error: {relative_error:.2%}")
    print(f"  Validation level: {validation_level.upper()}")
    
    return {
        'success': True,
        'alpha_symbolic': alpha_symbolic,
        'gamma_theory': gamma_theory,
        'gamma_fitted': gamma_fitted,
        'absolute_error': absolute_error,
        'relative_error': relative_error,
        'validation_level': validation_level,
        'parameters_used': {'mu': mu_value, 'M': M_value}
    }

def build_closed_form_template(comparison: Dict[str, Any],
                              use_fitted_values: bool = False) -> Dict[str, Any]:
    """
    Build the final closed-form LQG metric template.
    
    Args:
        comparison: Results from symbolic vs numeric comparison
        use_fitted_values: Whether to use fitted or theoretical coefficients
        
    Returns:
        Closed-form metric template and evaluation functions
    """
    print("Building closed-form LQG metric template...")
    
    if not comparison['success']:
        print("Cannot build template - comparison failed")
        return {'success': False}
    
    # Choose coefficient value
    if use_fitted_values:
        alpha_final = comparison['gamma_fitted'] / (comparison['parameters_used']['mu']**2 * 
                                                   comparison['parameters_used']['M']**2)
        coefficient_source = "fitted"
    else:
        alpha_final = comparison['alpha_symbolic']
        coefficient_source = "symbolic"
    
    print(f"Using {coefficient_source} coefficient: α = {alpha_final:.6f}")
    
    # Define closed-form metric function
    def f_LQG_template(r, M, mu):
        """
        LQG-corrected metric function to O(μ²).
        
        f_LQG(r) = 1 - 2M/r + α*μ²*M²/r⁴ + O(μ⁴)
        """
        return 1 - 2*M/r + alpha_final*(mu**2)*(M**2)/(r**4)
    
    def g_metric_template(r, t, theta, phi, M, mu):
        """
        Complete LQG-corrected metric tensor components.
        
        ds² = -f_LQG(r)dt² + dr²/f_LQG(r) + r²dΩ²
        """
        f_val = f_LQG_template(r, M, mu)
        return {
            'g_tt': -f_val,
            'g_rr': 1/f_val,
            'g_theta_theta': r**2,
            'g_phi_phi': r**2 * np.sin(theta)**2
        }
    
    # LaTeX template
    latex_template = {
        'metric_function': f"f_{{\\rm LQG}}(r) = 1 - \\frac{{2M}}{{r}} + {alpha_final:.6f}\\,\\frac{{\\mu^2 M^2}}{{r^4}} + \\mathcal{{O}}(\\mu^4)",
        'line_element': "ds^2 = -f_{\\rm LQG}(r)\\,dt^2 + \\frac{dr^2}{f_{\\rm LQG}(r)} + r^2\\,d\\Omega^2",
        'coefficient': f"\\alpha = {alpha_final:.6f}"
    }
    
    print("Closed-form template constructed:")
    print(f"  f_LQG(r) = 1 - 2M/r + ({alpha_final:.6f})*μ²M²/r⁴ + O(μ⁴)")
    print(f"  Source: {coefficient_source} derivation")
    
    return {
        'success': True,
        'alpha_final': alpha_final,
        'coefficient_source': coefficient_source,
        'f_LQG_template': f_LQG_template,
        'g_metric_template': g_metric_template,
        'latex_template': latex_template
    }

def validate_template_consistency(template: Dict[str, Any],
                                 test_parameters: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Validate the closed-form template for physical consistency.
    
    Args:
        template: Closed-form template results
        test_parameters: Test values for M, μ
        
    Returns:
        Validation results
    """
    print("Validating template consistency...")
    
    if not template['success']:
        print("Cannot validate - template construction failed")
        return {'success': False}
    
    if test_parameters is None:
        test_parameters = {'M': 1.0, 'mu': 0.05}
    
    M_test = test_parameters['M']
    mu_test = test_parameters['mu']
    
    # Test function
    f_template = template['f_LQG_template']
    
    # Test radial range
    r_test = np.linspace(2.1, 20.0, 100)
    f_values = f_template(r_test, M_test, mu_test)
    
    # Classical limit check (μ → 0)
    f_classical = 1 - 2*M_test/r_test
    f_mu_zero = f_template(r_test, M_test, 0.0)
    classical_error = np.max(np.abs(f_classical - f_mu_zero))
    
    # Positivity check (f > 0 for r > 2M)
    horizon_classical = 2*M_test
    r_exterior = r_test[r_test > horizon_classical + 0.1]  # Buffer beyond horizon
    f_exterior = f_template(r_exterior, M_test, mu_test)
    positivity_violations = np.sum(f_exterior <= 0)
    
    # Asymptotic flatness check (f → 1 as r → ∞)
    r_large = 100.0
    f_asymptotic = f_template(r_large, M_test, mu_test)
    asymptotic_error = abs(f_asymptotic - 1.0)
    
    # Quantum correction magnitude check
    r_quantum = 3*M_test  # Test at 3M
    f_quantum = f_template(r_quantum, M_test, mu_test)
    f_classical_at_3M = 1 - 2*M_test/r_quantum
    quantum_correction = abs(f_quantum - f_classical_at_3M)
    relative_correction = quantum_correction / abs(f_classical_at_3M)
    
    # Summary
    checks = {
        'classical_limit': classical_error < 1e-12,
        'positivity': positivity_violations == 0,
        'asymptotic_flatness': asymptotic_error < 1e-6,
        'reasonable_corrections': relative_correction < 0.1  # < 10% at 3M
    }
    
    all_passed = all(checks.values())
    
    print(f"  Classical limit (μ=0): {'✓' if checks['classical_limit'] else '✗'} (error: {classical_error:.2e})")
    print(f"  Positivity (r>2M): {'✓' if checks['positivity'] else '✗'} ({positivity_violations} violations)")
    print(f"  Asymptotic flatness: {'✓' if checks['asymptotic_flatness'] else '✗'} (error: {asymptotic_error:.2e})")
    print(f"  Reasonable corrections: {'✓' if checks['reasonable_corrections'] else '✗'} ({relative_correction:.1%} at r=3M)")
    print(f"  Overall validation: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    
    return {
        'success': True,
        'checks': checks,
        'all_passed': all_passed,
        'test_parameters': test_parameters,
        'detailed_results': {
            'classical_error': classical_error,
            'positivity_violations': positivity_violations,
            'asymptotic_error': asymptotic_error,
            'quantum_correction_magnitude': relative_correction
        }
    }

def generate_comparison_plots(template: Dict[str, Any],
                             comparison: Dict[str, Any],
                             save_path: str = None):
    """
    Generate plots comparing classical, symbolic, and fitted metrics.
    
    Args:
        template: Closed-form template
        comparison: Comparison results
        save_path: Path to save plots
    """
    if not template['success'] or not comparison['success']:
        print("Cannot generate plots - missing results")
        return
    
    # Parameters
    M_val = comparison['parameters_used']['M']
    mu_val = comparison['parameters_used']['mu']
    
    # Radial range
    r_plot = np.linspace(2.1, 10.0, 200)
    
    # Different metrics
    f_classical = 1 - 2*M_val/r_plot
    f_symbolic = template['f_LQG_template'](r_plot, M_val, mu_val)
    
    # Fitted version (if different)
    alpha_fitted = comparison['gamma_fitted'] / (mu_val**2 * M_val**2)
    f_fitted = 1 - 2*M_val/r_plot + alpha_fitted*(mu_val**2)*(M_val**2)/(r_plot**4)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Main comparison
    ax1.plot(r_plot, f_classical, ':', color='gray', linewidth=2, label='Classical Schwarzschild')
    ax1.plot(r_plot, f_symbolic, '-', color='blue', linewidth=2, label='Symbolic LQG')
    
    if abs(alpha_fitted - template['alpha_final']) > 1e-6:
        ax1.plot(r_plot, f_fitted, '--', color='red', linewidth=2, label='Fitted LQG')
    
    ax1.set_xlabel('r/M')
    ax1.set_ylabel('f(r)')
    ax1.set_title('LQG-Corrected vs Classical Metric')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([r_plot.min(), r_plot.max()])
    
    # Difference plot
    diff_symbolic = f_symbolic - f_classical
    ax2.plot(r_plot, diff_symbolic, '-', color='blue', linewidth=2, label='Symbolic - Classical')
    
    if abs(alpha_fitted - template['alpha_final']) > 1e-6:
        diff_fitted = f_fitted - f_classical
        ax2.plot(r_plot, diff_fitted, '--', color='red', linewidth=2, label='Fitted - Classical')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('r/M')
    ax2.set_ylabel('Δf(r)')
    ax2.set_title('Quantum Corrections')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add information box
    info_text = f"M = {M_val}, μ = {mu_val}\n"
    info_text += f"α_symbolic = {comparison['alpha_symbolic']:.6f}\n"
    if abs(alpha_fitted - template['alpha_final']) > 1e-6:
        info_text += f"α_fitted = {alpha_fitted:.6f}\n"
    info_text += f"Relative error: {comparison['relative_error']:.1%}\n"
    info_text += f"Validation: {comparison['validation_level']}"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def export_final_template(template: Dict[str, Any],
                         comparison: Dict[str, Any],
                         validation: Dict[str, Any],
                         filename: str = None):
    """
    Export the final closed-form template for use in the LQG pipeline.
    
    Args:
        template: Closed-form template
        comparison: Comparison results
        validation: Validation results
        filename: Output filename
    """
    if filename is None:
        filename = "scripts/lqg_closed_form_metric.py"
    
    with open(filename, 'w') as f:
        f.write('"""\n')
        f.write('Closed-form LQG-corrected spherically symmetric metric.\n')
        f.write('Auto-generated by match_symbolic_numeric.py\n')
        f.write('"""\n\n')
        f.write('import numpy as np\n\n')
        
        # Export coefficient
        f.write('# LQG correction coefficient (validated)\n')
        f.write(f'ALPHA_LQG = {template["alpha_final"]:.12f}  # {template["coefficient_source"]} derivation\n\n')
        
        # Export metric function
        f.write('def f_LQG(r, M, mu):\n')
        f.write('    """\n')
        f.write('    LQG-corrected metric function to O(μ²).\n')
        f.write('    \n')
        f.write('    f_LQG(r) = 1 - 2M/r + α*μ²*M²/r⁴ + O(μ⁴)\n')
        f.write('    \n')
        f.write('    Args:\n')
        f.write('        r: Radial coordinate\n')
        f.write('        M: Mass parameter\n')
        f.write('        mu: Polymer scale parameter\n')
        f.write('    \n')
        f.write('    Returns:\n')
        f.write('        LQG-corrected metric function values\n')
        f.write('    """\n')
        f.write(f'    return 1 - 2*M/r + ALPHA_LQG*(mu**2)*(M**2)/(r**4)\n\n')
        
        # Export metric tensor components
        f.write('def g_LQG_components(r, t, theta, phi, M, mu):\n')
        f.write('    """\n')
        f.write('    Complete LQG-corrected metric tensor components.\n')
        f.write('    \n')
        f.write('    ds² = -f_LQG(r)dt² + dr²/f_LQG(r) + r²dΩ²\n')
        f.write('    """\n')
        f.write('    f_val = f_LQG(r, M, mu)\n')
        f.write('    return {\n')
        f.write('        "g_tt": -f_val,\n')
        f.write('        "g_rr": 1/f_val,\n')
        f.write('        "g_theta_theta": r**2,\n')
        f.write('        "g_phi_phi": r**2 * np.sin(theta)**2\n')
        f.write('    }\n\n')
        
        # Export validation info
        f.write('# Validation information\n')
        f.write('VALIDATION_INFO = {\n')
        if validation['success']:
            for check_name, check_result in validation['checks'].items():
                f.write(f'    "{check_name}": {check_result},\n')
        f.write(f'    "coefficient_source": "{template["coefficient_source"]}",\n')
        if comparison['success']:
            f.write(f'    "relative_error": {comparison["relative_error"]:.6f},\n')
            f.write(f'    "validation_level": "{comparison["validation_level"]}"\n')
        f.write('}\n\n')
        
        # Export LaTeX
        f.write('# LaTeX expressions\n')
        f.write('LATEX_EXPRESSIONS = {\n')
        for key, expr in template['latex_template'].items():
            f.write(f'    "{key}": r"{expr}",\n')
        f.write('}\n')
    
    print(f"Final template exported to {filename}")

def run_complete_matching_pipeline(mu_value: float = 0.05,
                                  M_value: float = 1.0,
                                  use_fitted_values: bool = False) -> Dict[str, Any]:
    """
    Run the complete symbolic-numeric matching pipeline.
    
    Args:
        mu_value: Polymer scale parameter
        M_value: Mass parameter  
        use_fitted_values: Whether to use fitted or symbolic coefficients
        
    Returns:
        Complete matching results
    """
    print("="*60)
    print("SYMBOLIC-NUMERIC MATCHING PIPELINE")
    print("="*60)
    
    # Step 1: Load results
    symbolic_results = load_symbolic_results()
    fitting_results = load_fitting_results()
    
    # Step 2: Compare symbolic vs numeric
    comparison = compare_symbolic_vs_numeric(symbolic_results, fitting_results, mu_value, M_value)
    
    # Step 3: Build closed-form template
    template = build_closed_form_template(comparison, use_fitted_values)
    
    # Step 4: Validate template
    validation = validate_template_consistency(template, {'M': M_value, 'mu': mu_value})
    
    # Step 5: Generate comparison plots
    generate_comparison_plots(template, comparison, save_path='scripts/metric_comparison.png')
    
    # Step 6: Export final template
    export_final_template(template, comparison, validation)
    
    # Compile results
    complete_results = {
        'symbolic_results': symbolic_results,
        'fitting_results': fitting_results,
        'comparison': comparison,
        'template': template,
        'validation': validation
    }
    
    print("\n" + "="*60)
    print("MATCHING PIPELINE COMPLETE")
    print("="*60)
    
    if template['success'] and validation['success']:
        if validation['all_passed']:
            print("✓ SUCCESS: Closed-form template validated and exported!")
            print(f"  Final coefficient: α = {template['alpha_final']:.6f}")
            print(f"  Source: {template['coefficient_source']}")
            print(f"  Validation level: {comparison['validation_level'] if comparison['success'] else 'N/A'}")
        else:
            print("⚠ WARNING: Template constructed but validation failed")
    else:
        print("✗ ERROR: Template construction or validation failed")
    
    return complete_results

if __name__ == "__main__":
    # Run the complete matching pipeline
    results = run_complete_matching_pipeline()
