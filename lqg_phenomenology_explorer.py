#!/usr/bin/env python3
"""
LQG Phenomenology Explorer

This script explores the phenomenological consequences of LQG-corrected metrics,
including observational signatures and constraints on the polymer parameter μ.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

def evaluate_f_lqg(r_vals: np.ndarray, M: float, mu: float, 
                   alpha: float, beta: float = 0, gamma: float = 0) -> np.ndarray:
    """
    Evaluate the LQG-corrected metric function numerically.
    
    Args:
        r_vals: Radial coordinate values
        M: Mass parameter
        mu: Polymer parameter
        alpha: Leading LQG coefficient
        beta: Next-to-leading coefficient
        gamma: Next-to-next-to-leading coefficient
        
    Returns:
        f_LQG(r) values
    """
    f_classical = 1 - 2*M/r_vals
    
    # LQG corrections
    correction_alpha = alpha * (mu**2) * (M**2) / (r_vals**4)
    correction_beta = beta * (mu**4) * (M**3) / (r_vals**7) if beta != 0 else 0
    correction_gamma = gamma * (mu**6) * (M**4) / (r_vals**10) if gamma != 0 else 0
    
    f_lqg = f_classical + correction_alpha + correction_beta + correction_gamma
    
    return f_lqg

def evaluate_f_lqg_resummed(r_vals: np.ndarray, M: float, mu: float, 
                           alpha: float, beta: float) -> np.ndarray:
    """
    Evaluate the resummed LQG metric function.
    
    f_LQG(r) = 1 - 2M/r + [α·μ²M²/r⁴] / [1 - (β/α²)·μ²]
    """
    f_classical = 1 - 2*M/r_vals
    
    if alpha != 0:
        c = beta / (alpha**2)
        denominator = 1 - c * (mu**2)
        
        # Check for singularities
        if abs(denominator) < 1e-12:
            print(f"Warning: Resummation singularity at μ² = {1/c:.6f}")
            denominator = 1e-12
        
        correction_resummed = (alpha * (mu**2) * (M**2) / (r_vals**4)) / denominator
        f_lqg = f_classical + correction_resummed
    else:
        f_lqg = f_classical
    
    return f_lqg

def find_horizon_radius(M: float, mu: float, alpha: float, 
                       beta: float = 0, method: str = 'polynomial') -> float:
    """
    Find the horizon radius where f_LQG(r) = 0.
    
    Args:
        M: Mass parameter
        mu: Polymer parameter
        alpha: Leading LQG coefficient
        beta: Next-to-leading coefficient
        method: 'polynomial' or 'resummed'
        
    Returns:
        Horizon radius
    """
    # Initial guess near classical horizon
    r_guess = 2.0 * M
    
    def f_to_solve(r):
        if method == 'polynomial':
            return evaluate_f_lqg(np.array([r]), M, mu, alpha, beta)[0]
        else:
            return evaluate_f_lqg_resummed(np.array([r]), M, mu, alpha, beta)[0]
    
    # Simple Newton-Raphson iteration
    r = r_guess
    for _ in range(20):
        f_val = f_to_solve(r)
        if abs(f_val) < 1e-10:
            break
        
        # Numerical derivative
        dr = 1e-8
        f_prime = (f_to_solve(r + dr) - f_to_solve(r - dr)) / (2 * dr)
        
        if abs(f_prime) < 1e-15:
            break
            
        r_new = r - f_val / f_prime
        
        if abs(r_new - r) < 1e-12:
            break
            
        r = r_new
    
    return r

def compute_redshift_correction(r: float, M: float, mu: float, alpha: float) -> float:
    """
    Compute the gravitational redshift correction factor.
    
    z_LQG = z_Schwarzschild * sqrt(f_LQG / f_Schwarzschild)
    """
    f_classical = 1 - 2*M/r
    f_lqg = evaluate_f_lqg(np.array([r]), M, mu, alpha)[0]
    
    if f_classical > 0 and f_lqg > 0:
        correction_factor = np.sqrt(f_lqg / f_classical)
        return correction_factor
    else:
        return 1.0

def compute_photon_sphere_shift(M: float, mu: float, alpha: float) -> float:
    """
    Compute the shift in photon sphere radius.
    
    The photon sphere is where d/dr[r²/f(r)] = 0
    For small corrections: r_ph ≈ 3M * [1 - correction]
    """
    # Approximate shift for small μ
    # r_ph,LQG ≈ 3M * [1 - α*μ²M²/(27M⁴)] = 3M * [1 - α*μ²/27]
    
    shift_factor = alpha * (mu**2) / 27
    r_ph_lqg = 3 * M * (1 - shift_factor)
    
    return r_ph_lqg

def plot_metric_comparison(M: float = 1.0, mu_vals: List[float] = [0.01, 0.05, 0.1],
                          alpha: float = 1/6, beta: float = 0):
    """
    Plot comparison of classical and LQG-corrected metrics.
    """
    r_vals = np.linspace(2.01, 10, 1000) * M
    
    plt.figure(figsize=(12, 8))
    
    # Classical Schwarzschild
    f_classical = 1 - 2*M/r_vals
    plt.plot(r_vals/M, f_classical, 'k-', linewidth=2, label='Classical Schwarzschild')
    
    # LQG corrections for different μ values
    colors = ['red', 'blue', 'green', 'orange']
    for i, mu in enumerate(mu_vals):
        f_lqg = evaluate_f_lqg(r_vals, M, mu, alpha, beta)
        plt.plot(r_vals/M, f_lqg, '--', color=colors[i], linewidth=2, 
                label=f'LQG μ={mu} (α={alpha:.3f})')
    
    plt.xlabel('r/M')
    plt.ylabel('f(r)')
    plt.title('Metric Function: Classical vs LQG-Corrected')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(2, 10)
    plt.ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig('lqg_metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_horizon_shifts(M: float = 1.0, alpha: float = 1/6):
    """
    Analyze horizon radius shifts for different μ values.
    """
    mu_vals = np.logspace(-3, -1, 50)  # μ from 0.001 to 0.1
    
    horizon_shifts = []
    for mu in mu_vals:
        try:
            r_h_lqg = find_horizon_radius(M, mu, alpha)
            shift = r_h_lqg - 2*M
            horizon_shifts.append(shift)
        except:
            horizon_shifts.append(np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(mu_vals, np.array(horizon_shifts)/M, 'b-', linewidth=2)
    plt.xlabel('Polymer parameter μ')
    plt.ylabel('Horizon shift (r_h - 2M)/M')
    plt.title(f'LQG Horizon Radius Shift (α = {alpha:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lqg_horizon_shifts.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mu_vals, horizon_shifts

def generate_observational_constraints():
    """
    Generate observational constraints on the polymer parameter μ.
    """
    print("OBSERVATIONAL CONSTRAINTS ON μ")
    print("="*50)
    
    # Typical values
    M_solar = 1.989e30  # kg
    c = 299792458  # m/s
    G = 6.674e-11  # N⋅m²/kg²
    
    # Geometric units: M = GM/c² 
    M_geom = G * M_solar / c**2  # meters
    
    print(f"Solar mass in geometric units: M☉ = {M_geom:.3e} m")
    
    # For stellar-mass black holes (M ~ 10 M☉)
    M_stellar = 10 * M_geom
    
    print(f"\nStellar-mass BH (10 M☉): M = {M_stellar:.3e} m")
    print("Horizon radius: r_h = 2M = {:.3e} m".format(2 * M_stellar))
    
    # Observational precision requirements
    print("\nObservational precision constraints:")
    
    # 1. GPS satellite precision (~1 m altitude accuracy)
    gps_precision = 1.0  # meter precision
    r_gps = 6.371e6 + 2e4  # Earth radius + GPS altitude
    print(f"GPS precision at r = {r_gps:.1e} m: Δf/f < {gps_precision/r_gps:.1e}")
    
    # 2. Event Horizon Telescope precision (~10% for M87*)
    eht_precision = 0.1  # 10% precision
    print(f"EHT precision for supermassive BH: Δf/f < {eht_precision}")
    
    # 3. LIGO gravitational wave precision
    ligo_precision = 1e-21  # strain sensitivity
    print(f"LIGO strain precision: h < {ligo_precision}")
    
    # Constraints on μ
    alpha = 1/6  # Typical LQG value
    
    print(f"\nConstraints on μ (assuming α = {alpha:.3f}):")
    
    # For stellar-mass BH at horizon
    max_correction = 0.01  # 1% maximum allowed correction
    r_test = 3 * M_stellar  # Test at 3M
    
    # Δf/f = α*μ²*M²/r⁴ / (1 - 2M/r) < max_correction
    f_classical = 1 - 2*M_stellar/r_test
    mu_max = np.sqrt(max_correction * f_classical * (r_test**4) / (alpha * M_stellar**2))
    
    print(f"From 1% precision at r = 3M: μ < {mu_max:.3e}")
    
    # Convert to Planck units
    l_planck = 1.616e-35  # Planck length in meters
    mu_planck_units = mu_max / l_planck
    
    print(f"In Planck units: μ < {mu_planck_units:.3e} l_Pl")

def main():
    """
    Execute complete phenomenology analysis.
    """
    print("LQG PHENOMENOLOGY EXPLORER")
    print("="*40)
    
    # Standard parameters
    M = 1.0  # Geometric units
    alpha = 1/6  # From sin(μK)/μ expansion
    beta = 0  # Placeholder
    mu_vals = [0.01, 0.05, 0.1]
    
    print(f"Parameters: M = {M}, α = {alpha:.3f}")
    print(f"μ values: {mu_vals}")
    
    # 1. Plot metric comparisons
    print("\n1. Generating metric comparison plots...")
    try:
        plot_metric_comparison(M, mu_vals, alpha, beta)
        print("   ✓ Metric comparison plot saved")
    except Exception as e:
        print(f"   ✗ Plot generation failed: {e}")
    
    # 2. Analyze horizon shifts
    print("\n2. Analyzing horizon shifts...")
    try:
        mu_range, shifts = analyze_horizon_shifts(M, alpha)
        print("   ✓ Horizon shift analysis completed")
    except Exception as e:
        print(f"   ✗ Horizon analysis failed: {e}")
    
    # 3. Photon sphere analysis
    print("\n3. Photon sphere analysis:")
    for mu in mu_vals:
        r_ph = compute_photon_sphere_shift(M, mu, alpha)
        shift_percent = 100 * (r_ph - 3*M) / (3*M)
        print(f"   μ = {mu}: r_ph = {r_ph:.6f}M (shift: {shift_percent:+.3f}%)")
    
    # 4. Redshift corrections
    print("\n4. Redshift corrections at r = 3M:")
    for mu in mu_vals:
        correction = compute_redshift_correction(3*M, M, mu, alpha)
        correction_percent = 100 * (correction - 1)
        print(f"   μ = {mu}: z_LQG/z_GR = {correction:.6f} ({correction_percent:+.3f}%)")
    
    # 5. Observational constraints
    print("\n5. Generating observational constraints...")
    generate_observational_constraints()
    
    print(f"\n✓ Phenomenology analysis complete!")

if __name__ == "__main__":
    main()
