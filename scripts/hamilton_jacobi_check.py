#!/usr/bin/env python3
"""
Hamilton-Jacobi consistency check for LQG-corrected metric.

This script implements Step 6 of the roadmap:
- Insert f_LQG(r) into Hamilton-Jacobi equation for test particle
- Check if geodesics can be integrated analytically
- Analyze closed-form solutions and special functions
"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def hamilton_jacobi_analysis():
    """
    Perform Hamilton-Jacobi analysis for LQG-corrected metric.
    
    For a test particle in the metric ds² = -f(r)dt² + dr²/f(r) + r²dΩ²,
    the Hamilton-Jacobi equation gives:
    
    -E²/f(r) + f(r)(∂S/∂r)² + m² = 0
    
    This gives: ∂S/∂r = ±√[E²/f² - m²/f]
    
    Returns:
        Dictionary with symbolic and numerical results
    """
    print("Performing Hamilton-Jacobi analysis for LQG metric...")
    
    # Define symbolic variables
    r, E, m, M, mu, alpha = sp.symbols('r E m M mu alpha', positive=True, real=True)
    
    # LQG-corrected metric function
    f_LQG = 1 - 2*M/r + alpha*mu**2*M**2/r**4
    
    print("LQG metric function:")
    sp.pprint(f_LQG)
    
    # Hamilton-Jacobi radial equation: ∂S/∂r = ±√[E²/f² - m²/f]
    print("\nHamilton-Jacobi radial momentum:")
    
    # Expression under square root
    under_sqrt = E**2/f_LQG**2 - m**2/f_LQG
    
    # Simplify by factoring out 1/f_LQG
    under_sqrt_factored = (E**2/f_LQG - m**2) / f_LQG
    
    print("p_r = ∂S/∂r = ±√[(E²/f - m²)/f]")
    sp.pprint(under_sqrt_factored)
    
    # Radial momentum
    p_r = sp.sqrt(under_sqrt_factored)
    
    print("\nRadial momentum expression:")
    sp.pprint(p_r)
    
    return {
        'f_LQG': f_LQG,
        'under_sqrt': under_sqrt_factored,
        'p_r': p_r,
        'symbols': {'r': r, 'E': E, 'm': m, 'M': M, 'mu': mu, 'alpha': alpha}
    }

def attempt_analytical_integration(hj_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to integrate the Hamilton-Jacobi equation analytically.
    
    Args:
        hj_results: Results from hamilton_jacobi_analysis
        
    Returns:
        Integration results and analysis
    """
    print("Attempting analytical integration...")
    
    # Extract expressions
    p_r = hj_results['p_r']
    f_LQG = hj_results['f_LQG']
    symbols = hj_results['symbols']
    r, E, m, M, mu, alpha = symbols['r'], symbols['E'], symbols['m'], symbols['M'], symbols['mu'], symbols['alpha']
    
    # The action integral: S = ∫ p_r dr
    print("Attempting to integrate: S = ∫ p_r dr")
    
    try:
        # Try direct integration
        print("Trying direct symbolic integration...")
        S_integral = sp.integrate(p_r, r)
        
        if S_integral.has(sp.Integral):
            print("Direct integration returned unevaluated integral")
            integration_success = False
        else:
            print("Direct integration successful!")
            sp.pprint(S_integral)
            integration_success = True
            
    except Exception as e:
        print(f"Direct integration failed: {e}")
        S_integral = None
        integration_success = False
    
    # Try series expansion approach
    print("\nTrying series expansion approach...")
    
    try:
        # Expand p_r in small μ
        print("Expanding p_r in small μ...")
        p_r_series = sp.series(p_r, mu, 0, 3).removeO()
        
        print("Series expansion of p_r:")
        sp.pprint(p_r_series)
        
        # Try to integrate term by term
        print("Integrating series term by term...")
        S_series = sp.integrate(p_r_series, r)
        
        print("Integrated series:")
        sp.pprint(S_series)
        
        series_success = True
        
    except Exception as e:
        print(f"Series expansion integration failed: {e}")
        S_series = None
        series_success = False
    
    # Analyze integrability
    print("\nAnalyzing integrability...")
    
    # Check for known special functions
    special_functions_found = []
    
    if S_integral and not S_integral.has(sp.Integral):
        # Look for special functions in the result
        if S_integral.has(sp.log):
            special_functions_found.append('logarithmic')
        if S_integral.has(sp.atan) or S_integral.has(sp.atanh):
            special_functions_found.append('inverse trigonometric')
        if S_integral.has(sp.elliptic_f) or S_integral.has(sp.elliptic_e):
            special_functions_found.append('elliptic')
    
    return {
        'integration_success': integration_success,
        'S_integral': S_integral,
        'series_success': series_success,
        'S_series': S_series,
        'p_r_series': p_r_series if series_success else None,
        'special_functions': special_functions_found
    }

def effective_potential_analysis(hj_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze the effective potential for radial motion.
    
    The effective potential is: V_eff(r) = f(r)[m² + L²/r²]
    where L is the angular momentum.
    
    Args:
        hj_results: Results from Hamilton-Jacobi analysis
        
    Returns:
        Effective potential analysis
    """
    print("Analyzing effective potential...")
    
    # Extract symbols and expressions
    f_LQG = hj_results['f_LQG']
    symbols = hj_results['symbols']
    r, M, mu, alpha = symbols['r'], symbols['M'], symbols['mu'], symbols['alpha']
    
    # Add angular momentum
    m, L = sp.symbols('m L', positive=True, real=True)
    
    # Effective potential
    V_eff = f_LQG * (m**2 + L**2/r**2)
    
    print("Effective potential:")
    sp.pprint(V_eff)
    
    # Find critical points (extrema)
    print("\nFinding critical points...")
    dV_dr = sp.diff(V_eff, r)
    
    print("dV_eff/dr:")
    sp.pprint(dV_dr)
    
    # Solve for critical points
    try:
        critical_points = sp.solve(dV_dr, r)
        print(f"\nCritical points: {critical_points}")
        
        # Filter for positive real solutions
        physical_critical_points = []
        for point in critical_points:
            if point.is_positive or (point.is_real and point > 0):
                physical_critical_points.append(point)
        
        print(f"Physical critical points: {physical_critical_points}")
        
    except Exception as e:
        print(f"Could not solve for critical points: {e}")
        critical_points = []
        physical_critical_points = []
    
    # Analyze stability of circular orbits
    print("\nAnalyzing orbital stability...")
    d2V_dr2 = sp.diff(V_eff, r, 2)
    
    stability_analysis = {
        'V_eff': V_eff,
        'dV_dr': dV_dr,
        'd2V_dr2': d2V_dr2,
        'critical_points': critical_points,
        'physical_critical_points': physical_critical_points
    }
    
    return stability_analysis

def numerical_geodesic_comparison(alpha_value: float = 1/6,
                                 M_value: float = 1.0,
                                 mu_value: float = 0.05) -> Dict[str, Any]:
    """
    Compare geodesics in classical vs LQG metrics numerically.
    
    Args:
        alpha_value: LQG coefficient value
        M_value: Mass parameter
        mu_value: Polymer scale parameter
        
    Returns:
        Numerical comparison results
    """
    print("Performing numerical geodesic comparison...")
    
    # Radial range for analysis
    r_vals = np.linspace(2.1, 10.0, 100)
    
    # Classical and LQG metric functions
    f_classical = 1 - 2*M_value/r_vals
    f_LQG_num = 1 - 2*M_value/r_vals + alpha_value*(mu_value**2)*(M_value**2)/(r_vals**4)
    
    # Test particle parameters
    E_test = 0.95  # Slightly bound orbit
    m_test = 1.0   # Rest mass
    L_test = 3.0   # Angular momentum
    
    # Effective potentials
    V_eff_classical = f_classical * (m_test**2 + L_test**2/r_vals**2)
    V_eff_LQG = f_LQG_num * (m_test**2 + L_test**2/r_vals**2)
    
    # Radial momentum (for E > V_eff)
    p_r_classical = np.sqrt(np.maximum(0, E_test**2/f_classical - V_eff_classical))
    p_r_LQG = np.sqrt(np.maximum(0, E_test**2/f_LQG_num - V_eff_LQG))
    
    # Find turning points (where p_r = 0)
    def find_turning_points(r_array, p_r_array):
        turning_points = []
        for i in range(1, len(p_r_array)):
            if p_r_array[i-1] > 0 and p_r_array[i] == 0:
                turning_points.append(r_array[i])
        return turning_points
    
    turning_classical = find_turning_points(r_vals, p_r_classical)
    turning_LQG = find_turning_points(r_vals, p_r_LQG)
    
    print(f"  Classical turning points: {turning_classical}")
    print(f"  LQG turning points: {turning_LQG}")
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Effective potential comparison
    ax1.plot(r_vals, V_eff_classical, '-', label='Classical', color='blue')
    ax1.plot(r_vals, V_eff_LQG, '--', label='LQG', color='red')
    ax1.axhline(y=E_test**2, color='green', linestyle=':', label=f'E² = {E_test**2}')
    
    ax1.set_xlabel('r/M')
    ax1.set_ylabel('V_eff')
    ax1.set_title('Effective Potential Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Radial momentum comparison
    ax2.plot(r_vals, p_r_classical, '-', label='Classical', color='blue')
    ax2.plot(r_vals, p_r_LQG, '--', label='LQG', color='red')
    
    ax2.set_xlabel('r/M')
    ax2.set_ylabel('p_r')
    ax2.set_title('Radial Momentum Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scripts/geodesic_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'r_vals': r_vals,
        'V_eff_classical': V_eff_classical,
        'V_eff_LQG': V_eff_LQG,
        'p_r_classical': p_r_classical,
        'p_r_LQG': p_r_LQG,
        'turning_classical': turning_classical,
        'turning_LQG': turning_LQG,
        'test_parameters': {'E': E_test, 'm': m_test, 'L': L_test}
    }

def run_hamilton_jacobi_analysis():
    """
    Run complete Hamilton-Jacobi consistency analysis.
    """
    print("="*60)
    print("HAMILTON-JACOBI CONSISTENCY ANALYSIS")
    print("="*60)
    
    # Step 1: Basic Hamilton-Jacobi analysis
    hj_results = hamilton_jacobi_analysis()
    
    # Step 2: Attempt analytical integration
    integration_results = attempt_analytical_integration(hj_results)
    
    # Step 3: Effective potential analysis
    potential_results = effective_potential_analysis(hj_results)
    
    # Step 4: Numerical comparison
    numerical_results = numerical_geodesic_comparison()
    
    # Summary
    print("\n" + "="*60)
    print("HAMILTON-JACOBI ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Analytical integration: {'✓' if integration_results['integration_success'] else '✗'}")
    print(f"Series expansion: {'✓' if integration_results['series_success'] else '✗'}")
    
    if integration_results['special_functions']:
        print(f"Special functions found: {', '.join(integration_results['special_functions'])}")
    else:
        print("No standard special functions identified")
    
    if potential_results['physical_critical_points']:
        print(f"Stable circular orbits: {len(potential_results['physical_critical_points'])} found")
    else:
        print("No stable circular orbits found in symbolic analysis")
    
    # Compile complete results
    complete_results = {
        'hamilton_jacobi': hj_results,
        'integration': integration_results,
        'effective_potential': potential_results,
        'numerical_comparison': numerical_results
    }
    
    return complete_results

if __name__ == "__main__":
    # Run the complete analysis
    results = run_hamilton_jacobi_analysis()
