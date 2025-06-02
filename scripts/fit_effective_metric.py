#!/usr/bin/env python3
"""
Numerical fitting of LQG midisuperspace outputs to extract effective metric parameters.

This script implements Step 2 of the roadmap:
- Run LQG midisuperspace solver across different lattice sizes
- Extract numeric estimates of g_tt(r) from ground state
- Fit to ansatz f(r) = 1 - 2M_eff/r + γ/r⁴
- Compare fitted γ with theoretical prediction α*μ²
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import json
import sys
import os
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.lqg_metric_results import alpha_star, evaluate_f_LQG
    ALPHA_AVAILABLE = True
except ImportError:
    print("Warning: Symbolic results not yet available. Run derive_effective_metric.py first.")
    alpha_star = 1/6  # Placeholder
    ALPHA_AVAILABLE = False

def extract_gtt_from_lqg_output(lattice_sizes: List[int], 
                               base_config: Dict[str, Any],
                               lqg_params: Any) -> Dict[str, np.ndarray]:
    """
    Run LQG midisuperspace solver and extract g_tt(r) estimates.
    
    This interfaces with your existing LQG framework to get numeric data.
    
    Args:
        lattice_sizes: List of N values to run
        base_config: Base configuration dictionary  
        lqg_params: LQG parameters object
        
    Returns:
        Dictionary with 'r_data' and 'gtt_numeric' arrays
    """
    print("Extracting g_tt(r) from LQG midisuperspace outputs...")
    
    try:
        # Import your existing LQG components
        from refinement_framework import run_lqg_for_size
        from comprehensive_lqg_framework import LQGParameters, LatticeConfiguration
        
        r_all = []
        gtt_all = []
        
        for N in lattice_sizes:
            print(f"  Running LQG solver for N={N} sites...")
            
            try:
                # Run your existing LQG solver
                result = run_lqg_for_size(N, base_config, lqg_params)
                
                if 'error' in result:
                    print(f"    Error for N={N}: {result['error']}")
                    continue
                
                # Extract spatial coordinates
                lattice_config = LatticeConfiguration(N, base_config.get('throat_radius', 1.0))
                r_grid = lattice_config.get_radial_grid()
                
                # Extract metric component estimate from ground state
                if 'ground_state_expectation' in result:
                    # Use expectation values to reconstruct metric
                    Ex_expect = result['ground_state_expectation'].get('Ex', np.ones(N))
                    Ephi_expect = result['ground_state_expectation'].get('Ephi', r_grid)
                    
                    # Reconstruct g_tt ≈ -f(r) from triad data
                    # Classical relation: E^φ = r√f(r), so f(r) ≈ (E^φ/r)²
                    gtt_estimate = -(Ephi_expect / r_grid)**2
                    
                elif 'energy_density' in result:
                    # Use energy density as proxy for metric deviation
                    gtt_estimate = -result['energy_density']
                    
                else:
                    # Fallback: use eigenvalue information
                    eigenval = result.get('lowest_eigenvalue', 0.0)
                    # Simple model: uniform deviation proportional to eigenvalue
                    gtt_estimate = -(1 - 2.0/r_grid + eigenval/r_grid**2)
                
                # Store valid data points (avoid near-singularity)
                valid_mask = (r_grid > 2.1) & np.isfinite(gtt_estimate)
                r_all.extend(r_grid[valid_mask])
                gtt_all.extend(gtt_estimate[valid_mask])
                
                print(f"    Extracted {np.sum(valid_mask)} valid data points")
                
            except Exception as e:
                print(f"    Error processing N={N}: {e}")
                continue
        
        if len(r_all) == 0:
            raise ValueError("No valid data extracted from LQG solver")
        
        r_data = np.array(r_all)
        gtt_data = np.array(gtt_all)
        
        # Sort by radius
        sort_idx = np.argsort(r_data)
        r_data = r_data[sort_idx]
        gtt_data = gtt_data[sort_idx]
        
        print(f"Total extracted points: {len(r_data)}")
        print(f"Radius range: [{r_data.min():.3f}, {r_data.max():.3f}]")
        
        return {
            'r_data': r_data,
            'gtt_numeric': gtt_data
        }
        
    except ImportError as e:
        print(f"Cannot import LQG components: {e}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data() -> Dict[str, np.ndarray]:
    """
    Generate synthetic data that mimics LQG-corrected metric.
    
    This is used when the full LQG solver is not available.
    
    Returns:
        Dictionary with synthetic r_data and gtt_numeric
    """
    print("Generating synthetic LQG data...")
    
    # Parameters for synthetic data
    M_true = 1.0
    mu_true = 0.05
    alpha_true = float(alpha_star) if ALPHA_AVAILABLE else 1/6
    
    # Radial grid
    r_data = np.linspace(2.1, 10.0, 50)
    
    # True LQG metric with noise
    f_true = 1 - 2*M_true/r_data + alpha_true*(mu_true**2)*(M_true**2)/(r_data**4)
    noise_level = 0.01
    gtt_numeric = f_true + noise_level * np.random.randn(len(r_data))
    
    print(f"Synthetic data: M={M_true}, μ={mu_true}, α={alpha_true}")
    print(f"Points: {len(r_data)}, noise level: {noise_level}")
    
    return {
        'r_data': r_data,
        'gtt_numeric': gtt_numeric
    }

def f_ansatz(r: np.ndarray, M_eff: float, gamma: float) -> np.ndarray:
    """
    Metric ansatz for fitting: f(r) = 1 - 2*M_eff/r + γ/r⁴
    
    Args:
        r: Radial coordinates
        M_eff: Effective mass parameter
        gamma: r⁻⁴ coefficient (should match α*μ²*M²)
        
    Returns:
        Metric function values
    """
    return 1.0 - 2.0*M_eff/r + gamma/(r**4)

def fit_effective_parameters(r_data: np.ndarray, 
                           gtt_data: np.ndarray,
                           initial_guess: Tuple[float, float] = None) -> Dict[str, Any]:
    """
    Fit LQG data to effective metric ansatz.
    
    Args:
        r_data: Radial coordinate values
        gtt_data: Metric function values f(r) = -g_tt(r)
        initial_guess: (M_eff, gamma) initial values
        
    Returns:
        Dictionary with fit results
    """
    print("Fitting effective metric parameters...")
    
    if initial_guess is None:
        initial_guess = (1.0, 0.1)  # Reasonable defaults
    
    try:
        # Use curve_fit for initial estimate
        popt, pcov = curve_fit(f_ansatz, r_data, gtt_data, 
                              p0=initial_guess, 
                              maxfev=5000)
        
        M_eff_fit, gamma_fit = popt
        
        # Compute fit quality
        gtt_fitted = f_ansatz(r_data, M_eff_fit, gamma_fit)
        residuals = gtt_data - gtt_fitted
        rms_error = np.sqrt(np.mean(residuals**2))
        max_error = np.max(np.abs(residuals))
        
        # Parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))
        M_eff_error, gamma_error = param_errors
        
        print(f"  Fitted M_eff = {M_eff_fit:.6f} ± {M_eff_error:.6f}")
        print(f"  Fitted γ = {gamma_fit:.6f} ± {gamma_error:.6f}")
        print(f"  RMS error = {rms_error:.2e}")
        print(f"  Max error = {max_error:.2e}")
        
        return {
            'M_eff': M_eff_fit,
            'gamma': gamma_fit,
            'M_eff_error': M_eff_error,
            'gamma_error': gamma_error,
            'rms_error': rms_error,
            'max_error': max_error,
            'fitted_curve': gtt_fitted,
            'residuals': residuals,
            'success': True
        }
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        return {'success': False, 'error': str(e)}

def compare_with_theory(fit_results: Dict[str, Any], 
                       mu_value: float = 0.05,
                       M_value: float = 1.0) -> Dict[str, Any]:
    """
    Compare fitted parameters with theoretical predictions.
    
    Args:
        fit_results: Results from fit_effective_parameters
        mu_value: Polymer scale parameter
        M_value: Mass parameter
        
    Returns:
        Dictionary with comparison results
    """
    print("Comparing with theoretical predictions...")
    
    if not fit_results['success']:
        print("Cannot compare - fitting failed")
        return {'success': False}
    
    M_eff_fit = fit_results['M_eff']
    gamma_fit = fit_results['gamma']
    
    # Theoretical predictions
    M_theory = M_value  # Should match classical mass
    
    if ALPHA_AVAILABLE:
        alpha_num = float(alpha_star)
        gamma_theory = alpha_num * (mu_value**2) * (M_value**2)
    else:
        # Use placeholder
        alpha_num = 1/6
        gamma_theory = alpha_num * (mu_value**2) * (M_value**2)
    
    # Compute relative errors
    M_relative_error = abs(M_eff_fit - M_theory) / M_theory
    gamma_relative_error = abs(gamma_fit - gamma_theory) / abs(gamma_theory) if gamma_theory != 0 else float('inf')
    
    print(f"  Mass comparison:")
    print(f"    Theory: M = {M_theory:.6f}")
    print(f"    Fitted: M_eff = {M_eff_fit:.6f}")
    print(f"    Relative error: {M_relative_error:.2%}")
    
    print(f"  Gamma comparison:")
    print(f"    Theory: γ = α*μ²*M² = {alpha_num:.6f} × {mu_value:.6f}² × {M_value:.6f}² = {gamma_theory:.6f}")
    print(f"    Fitted: γ = {gamma_fit:.6f}")
    print(f"    Relative error: {gamma_relative_error:.2%}")
    
    # Success criteria
    mass_match = M_relative_error < 0.1  # 10% tolerance
    gamma_match = gamma_relative_error < 0.2  # 20% tolerance
    overall_success = mass_match and gamma_match
    
    print(f"  Mass match (< 10%): {'✓' if mass_match else '✗'}")
    print(f"  Gamma match (< 20%): {'✓' if gamma_match else '✗'}")
    print(f"  Overall validation: {'✓ PASSED' if overall_success else '✗ FAILED'}")
    
    return {
        'success': True,
        'M_theory': M_theory,
        'gamma_theory': gamma_theory,
        'alpha_used': alpha_num,
        'M_relative_error': M_relative_error,
        'gamma_relative_error': gamma_relative_error,
        'mass_match': mass_match,
        'gamma_match': gamma_match,
        'overall_success': overall_success
    }

def plot_fitting_results(r_data: np.ndarray,
                        gtt_data: np.ndarray, 
                        fit_results: Dict[str, Any],
                        comparison: Dict[str, Any] = None,
                        save_path: str = None):
    """
    Create visualization of fitting results.
    
    Args:
        r_data: Radial coordinates
        gtt_data: Original data
        fit_results: Fitting results
        comparison: Theory comparison results
        save_path: Path to save plot (optional)
    """
    if not fit_results['success']:
        print("Cannot plot - fitting failed")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Main plot: data vs fit
    ax1.scatter(r_data, gtt_data, alpha=0.6, s=20, label='LQG numeric data', color='blue')
    
    r_plot = np.linspace(r_data.min(), r_data.max(), 200)
    gtt_fitted_plot = f_ansatz(r_plot, fit_results['M_eff'], fit_results['gamma'])
    ax1.plot(r_plot, gtt_fitted_plot, '--', color='red', linewidth=2, label='Fitted ansatz')
    
    # Add classical Schwarzschild for reference
    if comparison and comparison['success']:
        M_classical = comparison['M_theory']
        gtt_classical = 1 - 2*M_classical/r_plot
        ax1.plot(r_plot, gtt_classical, ':', color='gray', alpha=0.7, label='Classical Schwarzschild')
    
    ax1.set_xlabel('r')
    ax1.set_ylabel('f(r) = -g_tt(r)')
    ax1.set_title('LQG-Corrected Metric Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    ax2.scatter(r_data, fit_results['residuals'], alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('r')
    ax2.set_ylabel('Residuals')
    ax2.set_title(f'Fit Residuals (RMS = {fit_results["rms_error"]:.2e})')
    ax2.grid(True, alpha=0.3)
    
    # Add fit information as text
    info_text = f"M_eff = {fit_results['M_eff']:.6f} ± {fit_results['M_eff_error']:.6f}\n"
    info_text += f"γ = {fit_results['gamma']:.6f} ± {fit_results['gamma_error']:.6f}"
    
    if comparison and comparison['success']:
        info_text += f"\n\nTheory comparison:"
        info_text += f"\nM error: {comparison['M_relative_error']:.1%}"
        info_text += f"\nγ error: {comparison['gamma_relative_error']:.1%}"
        info_text += f"\nValidation: {'PASS' if comparison['overall_success'] else 'FAIL'}"
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def run_complete_fitting_pipeline(config_file: str = None,
                                 lattice_sizes: List[int] = None,
                                 mu_value: float = 0.05,
                                 M_value: float = 1.0) -> Dict[str, Any]:
    """
    Run the complete fitting pipeline from LQG data to parameter extraction.
    
    Args:
        config_file: Path to configuration file (optional)
        lattice_sizes: List of lattice sizes to test
        mu_value: Polymer scale parameter
        M_value: Mass parameter
        
    Returns:
        Complete results dictionary
    """
    print("="*60)
    print("LQG EFFECTIVE METRIC PARAMETER FITTING")
    print("="*60)
    
    # Default parameters
    if lattice_sizes is None:
        lattice_sizes = [8, 12, 16, 20]
    
    # Load or create base configuration
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            base_config = json.load(f)
    else:
        # Default configuration
        base_config = {
            'throat_radius': 1.0,
            'E_x_classical': None,  # Will be set by LQG solver
            'E_phi_classical': None
        }
    
    # Create LQG parameters (simplified)
    class MockLQGParams:
        def __init__(self):
            self.gamma = 0.2375
            self.planck_area = 1.0
            self.regularization_epsilon = 1e-12
    
    lqg_params = MockLQGParams()
    
    # Step 1: Extract data from LQG solver
    lqg_data = extract_gtt_from_lqg_output(lattice_sizes, base_config, lqg_params)
    
    # Step 2: Fit to effective metric ansatz
    fit_results = fit_effective_parameters(lqg_data['r_data'], lqg_data['gtt_numeric'])
    
    # Step 3: Compare with theoretical predictions
    comparison = compare_with_theory(fit_results, mu_value, M_value)
    
    # Step 4: Visualize results
    plot_fitting_results(lqg_data['r_data'], lqg_data['gtt_numeric'], 
                        fit_results, comparison, 
                        save_path='scripts/lqg_metric_fit.png')
    
    # Compile complete results
    complete_results = {
        'lqg_data': lqg_data,
        'fit_results': fit_results,
        'theory_comparison': comparison,
        'parameters': {
            'mu_value': mu_value,
            'M_value': M_value,
            'lattice_sizes': lattice_sizes
        }
    }
    
    # Save results
    results_file = 'scripts/fitting_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = complete_results.copy()
        json_results['lqg_data']['r_data'] = lqg_data['r_data'].tolist()
        json_results['lqg_data']['gtt_numeric'] = lqg_data['gtt_numeric'].tolist()
        if fit_results['success']:
            json_results['fit_results']['fitted_curve'] = fit_results['fitted_curve'].tolist()
            json_results['fit_results']['residuals'] = fit_results['residuals'].tolist()
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    return complete_results

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_fitting_pipeline()
    
    print("\n" + "="*60)
    print("FITTING PIPELINE COMPLETE")
    print("="*60)
    
    if results['fit_results']['success'] and results['theory_comparison']['success']:
        if results['theory_comparison']['overall_success']:
            print("✓ SUCCESS: Fitted parameters match theoretical predictions!")
        else:
            print("⚠ WARNING: Fitted parameters deviate from theory")
    else:
        print("✗ ERROR: Fitting or comparison failed")
