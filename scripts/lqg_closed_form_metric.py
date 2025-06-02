#!/usr/bin/env python3
"""
LQG Closed-Form Metric Template

This module provides the final closed-form LQG-corrected metric for spherically 
symmetric spacetimes, derived from symbolic polymer quantization and validated 
against numerical LQG midisuperspace results.

Metric function: f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + O(μ⁴)
where α is the dimensionless LQG coefficient from polymer quantization.

Usage:
    from scripts.lqg_closed_form_metric import f_LQG, g_LQG_components, ALPHA_LQG
    
    M, mu = 1.0, 0.05
    r = 3.0
    f_value = f_LQG(r, M, mu)
    metric_components = g_LQG_components(r, 0, np.pi/2, 0, M, mu)
"""

import numpy as np
from typing import Dict, Any, Union, Optional

# LQG coefficient from symbolic derivation
# This is typically α ≈ 1/6 from sin(μK)/μ expansion
ALPHA_LQG = 1/6

def f_LQG(r: Union[float, np.ndarray], 
          M: float, 
          mu: float, 
          alpha: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    LQG-corrected metric function to O(μ²).
    
    f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + O(μ⁴)
    
    Args:
        r: Radial coordinate(s)
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Metric function value(s)
    """
    if alpha is None:
        alpha = ALPHA_LQG
    
    r = np.asarray(r)
    return 1 - 2*M/r + alpha*(mu**2)*(M**2)/(r**4)


def g_LQG_components(r: Union[float, np.ndarray], 
                     t: Union[float, np.ndarray], 
                     theta: Union[float, np.ndarray], 
                     phi: Union[float, np.ndarray],
                     M: float, 
                     mu: float,
                     alpha: Optional[float] = None) -> Dict[str, Union[float, np.ndarray]]:
    """
    Complete LQG-corrected metric tensor components.
    
    ds² = -f_LQG(r)dt² + dr²/f_LQG(r) + r²dΩ²
    
    Args:
        r, t, theta, phi: Coordinates
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Dictionary with metric components
    """
    f_val = f_LQG(r, M, mu, alpha)
    
    return {
        'g_tt': -f_val,
        'g_rr': 1/f_val,
        'g_theta_theta': r**2,
        'g_phi_phi': r**2 * np.sin(theta)**2,
        'g_tr': 0,
        'g_t_theta': 0,
        'g_t_phi': 0,
        'g_r_theta': 0,
        'g_r_phi': 0,
        'g_theta_phi': 0
    }


def effective_potential_LQG(r: Union[float, np.ndarray], 
                           m: float, 
                           L: float, 
                           M: float, 
                           mu: float,
                           alpha: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Effective potential for radial motion in LQG-corrected spacetime.
    
    V_eff(r) = f_LQG(r) * [m² + L²/r²]
    
    Args:
        r: Radial coordinate(s)
        m: Test particle rest mass
        L: Angular momentum parameter
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Effective potential value(s)
    """
    f_val = f_LQG(r, M, mu, alpha)
    r = np.asarray(r)
    return f_val * (m**2 + L**2/r**2)


def radial_momentum_LQG(r: Union[float, np.ndarray], 
                       E: float, 
                       m: float, 
                       M: float, 
                       mu: float,
                       alpha: Optional[float] = None) -> Union[float, np.ndarray]:
    """
    Radial momentum for geodesics in LQG-corrected spacetime.
    
    p_r = √[(E²/f_LQG - m²)/f_LQG]
    
    Args:
        r: Radial coordinate(s)
        E: Energy parameter
        m: Test particle rest mass
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Radial momentum value(s)
    """
    f_val = f_LQG(r, M, mu, alpha)
    discriminant = (E**2/f_val - m**2) / f_val
    
    # Handle negative discriminant (classically forbidden region)
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=complex)
    valid_mask = discriminant >= 0
    
    if np.any(valid_mask):
        result[valid_mask] = np.sqrt(discriminant[valid_mask])
    
    return np.real(result) if np.all(np.isreal(result)) else result


def solve_bounce_radius(M: float, 
                       mu: float, 
                       alpha: Optional[float] = None,
                       method: str = 'brentq') -> Optional[float]:
    """
    Solve f_LQG(r_*) = 0 for the quantum-corrected horizon/bounce radius.
    
    Args:
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        method: Root-finding method ('brentq' or 'newton')
        
    Returns:
        Bounce radius r_* or None if not found
    """
    from scipy.optimize import brentq, newton
    
    if alpha is None:
        alpha = ALPHA_LQG
    
    def f_root(r):
        return f_LQG(r, M, mu, alpha)
    
    try:
        r_classical = 2.0 * M
        
        if method == 'brentq':
            # Try to bracket the root
            r_low = 0.8 * r_classical
            r_high = 1.2 * r_classical
            
            if f_root(r_low) * f_root(r_high) < 0:
                return brentq(f_root, r_low, r_high)
            else:
                # Expand search range
                r_low = 0.5 * r_classical
                r_high = 1.5 * r_classical
                if f_root(r_low) * f_root(r_high) < 0:
                    return brentq(f_root, r_low, r_high)
                
        elif method == 'newton':
            # Use Newton's method starting near classical horizon
            def f_prime(r):
                return 2*M/(r**2) - 4*alpha*(mu**2)*(M**2)/(r**5)
            
            return newton(f_root, r_classical * 0.95, fprime=f_prime)
            
    except Exception as e:
        print(f"Error solving for bounce radius: {e}")
        return None
    
    return None


def horizon_shift_analysis(M: float, 
                          mu: float, 
                          alpha: Optional[float] = None) -> Dict[str, float]:
    """
    Analyze the quantum correction to the horizon structure.
    
    Args:
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Dictionary with horizon analysis results
    """
    if alpha is None:
        alpha = ALPHA_LQG
    
    r_classical = 2.0 * M
    r_bounce = solve_bounce_radius(M, mu, alpha)
    
    if r_bounce is not None:
        delta_r = r_bounce - r_classical
        relative_shift = delta_r / r_classical
        
        return {
            'r_classical': r_classical,
            'r_bounce': r_bounce,
            'delta_r': delta_r,
            'relative_shift': relative_shift,
            'mu_value': mu,
            'alpha_used': alpha,
            'success': True
        }
    else:
        return {
            'r_classical': r_classical,
            'r_bounce': None,
            'error': 'Could not find bounce radius',
            'success': False
        }


def validate_classical_limit(M: float, 
                           mu_values: np.ndarray,
                           r_test: float = 3.0,
                           tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate that f_LQG → 1 - 2M/r as μ → 0.
    
    Args:
        M: Black hole mass parameter
        mu_values: Array of polymer scale values to test
        r_test: Test radius for evaluation
        tolerance: Tolerance for classical limit validation
        
    Returns:
        Validation results
    """
    f_classical = 1 - 2*M/r_test
    f_lqg_values = []
    errors = []
    
    for mu in mu_values:
        f_lqg = f_LQG(r_test, M, mu)
        f_lqg_values.append(f_lqg)
        error = abs(f_lqg - f_classical)
        errors.append(error)
    
    max_error = max(errors)
    validation_passed = max_error < tolerance
    
    return {
        'mu_values': mu_values,
        'f_classical': f_classical,
        'f_lqg_values': np.array(f_lqg_values),
        'errors': np.array(errors),
        'max_error': max_error,
        'tolerance': tolerance,
        'validation_passed': validation_passed,
        'test_radius': r_test
    }


def export_latex_expressions(alpha: Optional[float] = None) -> Dict[str, str]:
    """
    Generate LaTeX expressions for the LQG-corrected metric.
    
    Args:
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Dictionary with LaTeX expressions
    """
    if alpha is None:
        alpha = ALPHA_LQG
    
    alpha_str = f"{alpha:.6f}" if alpha != 1/6 else r"\frac{1}{6}"
    
    return {
        'metric_function': f"f_{{\\rm LQG}}(r) = 1 - \\frac{{2M}}{{r}} + {alpha_str}\\,\\frac{{\\mu^2 M^2}}{{r^4}} + \\mathcal{{O}}(\\mu^4)",
        'line_element': "ds^2 = -f_{\\rm LQG}(r)\\,dt^2 + \\frac{dr^2}{f_{\\rm LQG}(r)} + r^2\\,d\\Omega^2",
        'coefficient': f"\\alpha = {alpha_str}",
        'effective_potential': "V_{\\rm eff}(r) = f_{\\rm LQG}(r)\\left[m^2 + \\frac{L^2}{r^2}\\right]",
        'radial_momentum': "p_r = \\sqrt{\\frac{E^2/f_{\\rm LQG}(r) - m^2}{f_{\\rm LQG}(r)}}"
    }


# Convenience functions for common use cases
def schwarzschild_comparison(r_values: np.ndarray, 
                           M: float, 
                           mu: float,
                           alpha: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    Compare LQG-corrected metric with classical Schwarzschild.
    
    Args:
        r_values: Array of radial coordinates
        M: Black hole mass parameter
        mu: Polymer scale parameter
        alpha: LQG coefficient (defaults to ALPHA_LQG = 1/6)
        
    Returns:
        Dictionary with comparison data
    """
    f_classical = 1 - 2*M/r_values
    f_lqg = f_LQG(r_values, M, mu, alpha)
    
    return {
        'r_values': r_values,
        'f_classical': f_classical,
        'f_lqg': f_lqg,
        'difference': f_lqg - f_classical,
        'relative_difference': (f_lqg - f_classical) / f_classical,
        'parameters': {'M': M, 'mu': mu, 'alpha': alpha if alpha is not None else ALPHA_LQG}
    }


if __name__ == "__main__":
    # Demo usage
    print("LQG Closed-Form Metric Demo")
    print("="*40)
    
    # Parameters
    M = 1.0
    mu = 0.05
    r_test = 3.0
    
    # Basic metric evaluation
    f_value = f_LQG(r_test, M, mu)
    print(f"f_LQG({r_test}) = {f_value:.6f}")
    
    # Classical comparison
    f_classical = 1 - 2*M/r_test
    correction = f_value - f_classical
    print(f"Classical f({r_test}) = {f_classical:.6f}")
    print(f"LQG correction = {correction:.6e}")
    
    # Horizon analysis
    horizon_analysis = horizon_shift_analysis(M, mu)
    if horizon_analysis['success']:
        print(f"Classical horizon: r = {horizon_analysis['r_classical']:.6f}")
        print(f"LQG bounce: r = {horizon_analysis['r_bounce']:.6f}")
        print(f"Relative shift: {horizon_analysis['relative_shift']:+.2%}")
    
    # LaTeX expressions
    latex_exprs = export_latex_expressions()
    print("\nLaTeX metric function:")
    print(latex_exprs['metric_function'])
