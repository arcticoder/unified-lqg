#!/usr/bin/env python3
"""
Solve for bounce/horizon radius in LQG-corrected metric.

This script implements part of Step 5:
- Solve f_LQG(r*) = 0 for the bounce/horizon radius
- Compare with classical Schwarzschild horizon
- Analyze quantum corrections to horizon structure
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def f_LQG(r, M, mu, alpha=1/6):
    """
    LQG-corrected metric function.
    
    Args:
        r: Radial coordinate
        M: Mass parameter
        mu: Polymer scale parameter
        alpha: LQG correction coefficient
        
    Returns:
        Metric function value
    """
    return 1 - 2*M/r + alpha*(mu**2)*(M**2)/(r**4)

def find_bounce_radius(M, mu, alpha=1/6, method='brentq'):
    """
    Find the bounce/horizon radius by solving f_LQG(r*) = 0.
    
    Args:
        M: Mass parameter
        mu: Polymer scale parameter
        alpha: LQG correction coefficient
        method: Numerical method ('brentq' or 'fsolve')
        
    Returns:
        Bounce radius r* (or None if not found)
    """
    print(f"Finding bounce radius for M={M}, μ={mu}, α={alpha}")
    
    # Classical horizon for reference
    r_classical = 2*M
    print(f"  Classical horizon: r_cl = {r_classical:.6f}")
    
    # Function to solve: f_LQG(r) = 0
    def root_func(r):
        return f_LQG(r, M, mu, alpha)
    
    # Determine search range
    # The LQG correction μ²M²/r⁴ is positive, so f_LQG > f_classical
    # This means the horizon should be at r* < r_classical
    r_min = 0.5 * r_classical  # Start well inside classical horizon
    r_max = 1.5 * r_classical  # Extend beyond classical horizon
    
    # Check if root exists in range
    f_min = root_func(r_min)
    f_max = root_func(r_max)
    
    print(f"  Search range: [{r_min:.6f}, {r_max:.6f}]")
    print(f"  f({r_min:.6f}) = {f_min:.6f}")
    print(f"  f({r_max:.6f}) = {f_max:.6f}")
    
    if f_min * f_max > 0:
        print("  Warning: No sign change detected in search range")
        # Expand search range
        r_min = 0.1 * r_classical
        r_max = 3.0 * r_classical
        f_min = root_func(r_min)
        f_max = root_func(r_max)
        print(f"  Expanded range: [{r_min:.6f}, {r_max:.6f}]")
        print(f"  f({r_min:.6f}) = {f_min:.6f}")
        print(f"  f({r_max:.6f}) = {f_max:.6f}")
        
        if f_min * f_max > 0:
            print("  No root found in expanded range")
            return None
    
    try:
        if method == 'brentq':
            r_bounce = brentq(root_func, r_min, r_max)
        elif method == 'fsolve':
            result = fsolve(root_func, r_classical)
            r_bounce = result[0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Verify solution
        f_check = root_func(r_bounce)
        if abs(f_check) > 1e-10:
            print(f"  Warning: Solution verification failed, f({r_bounce:.6f}) = {f_check:.2e}")
        
        print(f"  Bounce radius: r* = {r_bounce:.6f}")
        
        # Compare with classical
        relative_shift = (r_bounce - r_classical) / r_classical
        print(f"  Relative shift: (r* - r_cl)/r_cl = {relative_shift:.2%}")
        
        return r_bounce
        
    except Exception as e:
        print(f"  Error finding root: {e}")
        return None

def analyze_horizon_structure(M_values, mu_values, alpha=1/6):
    """
    Analyze how the horizon structure changes with LQG parameters.
    
    Args:
        M_values: Array of mass values
        mu_values: Array of polymer scale values
        alpha: LQG correction coefficient
        
    Returns:
        Dictionary with analysis results
    """
    print("Analyzing horizon structure...")
    
    results = {
        'M_values': M_values,
        'mu_values': mu_values,
        'bounce_radii': {},
        'relative_shifts': {}
    }
    
    for M in M_values:
        results['bounce_radii'][M] = []
        results['relative_shifts'][M] = []
        
        for mu in mu_values:
            r_bounce = find_bounce_radius(M, mu, alpha, method='brentq')
            
            if r_bounce is not None:
                results['bounce_radii'][M].append(r_bounce)
                relative_shift = (r_bounce - 2*M) / (2*M)
                results['relative_shifts'][M].append(relative_shift)
            else:
                results['bounce_radii'][M].append(np.nan)
                results['relative_shifts'][M].append(np.nan)
    
    return results

def plot_horizon_analysis(results, save_path=None):
    """
    Plot horizon structure analysis results.
    
    Args:
        results: Results from analyze_horizon_structure
        save_path: Path to save plot
    """
    M_values = results['M_values']
    mu_values = results['mu_values']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bounce radius vs μ for different masses
    for M in M_values:
        bounce_radii = np.array(results['bounce_radii'][M])
        ax1.plot(mu_values, bounce_radii, 'o-', label=f'M = {M}')
        
        # Classical horizon line
        ax1.axhline(y=2*M, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('μ (polymer scale)')
    ax1.set_ylabel('r* (bounce radius)')
    ax1.set_title('LQG Bounce Radius vs Polymer Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Relative shift vs μ
    for M in M_values:
        relative_shifts = np.array(results['relative_shifts'][M])
        ax2.plot(mu_values, relative_shifts * 100, 'o-', label=f'M = {M}')
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('μ (polymer scale)')
    ax2.set_ylabel('(r* - r_cl)/r_cl (%)')
    ax2.set_title('Relative Horizon Shift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Horizon analysis plot saved to {save_path}")
    
    plt.show()

def compare_with_spinfoam_peak(M, mu, alpha=1/6):
    """
    Compare bounce radius with spin-foam peak radius.
    
    This is a placeholder for Step 5.2 - actual spin-foam comparison
    would require integration with your spin-foam module.
    
    Args:
        M: Mass parameter
        mu: Polymer scale parameter
        alpha: LQG correction coefficient
        
    Returns:
        Comparison results
    """
    print("Comparing with spin-foam peak radius...")
    
    # Get bounce radius
    r_bounce = find_bounce_radius(M, mu, alpha)
    
    if r_bounce is None:
        print("Cannot compare - bounce radius not found")
        return None
    
    # Placeholder for spin-foam calculation
    # In reality, this would call your SpinFoamCrossValidationDemo
    try:
        # Mock spin-foam peak radius (replace with actual calculation)
        r_spinfoam_peak = r_bounce * (1 + 0.05 * np.random.randn())  # Mock 5% variation
        
        difference = abs(r_bounce - r_spinfoam_peak)
        relative_difference = difference / r_spinfoam_peak
        
        print(f"  Bounce radius: r* = {r_bounce:.6f}")
        print(f"  Spin-foam peak: r_SF = {r_spinfoam_peak:.6f}")
        print(f"  Absolute difference: |Δr| = {difference:.6f}")
        print(f"  Relative difference: |Δr|/r_SF = {relative_difference:.2%}")
        
        # Consistency check
        consistency_threshold = 0.1  # 10%
        is_consistent = relative_difference < consistency_threshold
        
        print(f"  Consistency (< {consistency_threshold:.0%}): {'✓' if is_consistent else '✗'}")
        
        return {
            'r_bounce': r_bounce,
            'r_spinfoam_peak': r_spinfoam_peak,
            'difference': difference,
            'relative_difference': relative_difference,
            'is_consistent': is_consistent
        }
        
    except Exception as e:
        print(f"Error in spin-foam comparison: {e}")
        return None

def run_bounce_analysis():
    """
    Run complete bounce radius analysis.
    """
    print("="*60)
    print("LQG BOUNCE RADIUS ANALYSIS")
    print("="*60)
    
    # Load coefficient from symbolic results if available
    try:
        from scripts.lqg_closed_form_metric import ALPHA_LQG
        alpha = ALPHA_LQG
        print(f"Using validated coefficient: α = {alpha:.6f}")
    except ImportError:
        alpha = 1/6  # Placeholder
        print(f"Using placeholder coefficient: α = {alpha:.6f}")
    
    # Single case analysis
    M_test = 1.0
    mu_test = 0.05
    
    print(f"\nSingle case analysis: M = {M_test}, μ = {mu_test}")
    r_bounce = find_bounce_radius(M_test, mu_test, alpha)
    
    # Parameter sweep
    print(f"\nParameter sweep analysis...")
    M_values = [0.5, 1.0, 2.0]
    mu_values = np.linspace(0.01, 0.1, 10)
    
    results = analyze_horizon_structure(M_values, mu_values, alpha)
    plot_horizon_analysis(results, save_path='scripts/horizon_analysis.png')
    
    # Spin-foam comparison (mock)
    print(f"\nSpin-foam comparison...")
    spinfoam_comparison = compare_with_spinfoam_peak(M_test, mu_test, alpha)
    
    print("\n" + "="*60)
    print("BOUNCE ANALYSIS COMPLETE")
    print("="*60)
    
    return {
        'single_case': {'M': M_test, 'mu': mu_test, 'r_bounce': r_bounce},
        'parameter_sweep': results,
        'spinfoam_comparison': spinfoam_comparison
    }

if __name__ == "__main__":
    # Run the complete analysis
    results = run_bounce_analysis()
