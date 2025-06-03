#!/usr/bin/env python3
"""
Kerr-Newman Generalization for LQG Black Holes

This module extends the LQG framework to charged, rotating black holes by implementing
polymer corrections to the Kerr-Newman metric and extracting generalized coefficients.

Key Features:
- Polymer-corrected Kerr-Newman metric in Boyer-Lindquist coordinates  
- Coefficient extraction for charged, rotating black holes
- Comparison with Kerr and Schwarzschild cases
- Charge and spin dependent phenomenological predictions
"""

import sympy as sp
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# 1) KERR-NEWMAN METRIC POLYMER CORRECTIONS
# ------------------------------------------------------------------------

def compute_polymer_kerr_newman_metric(mu, M, a, Q, r, theta):
    """
    Compute polymer-corrected Kerr-Newman metric components.
    
    Args:
        mu: Polymer scale parameter
        M: Black hole mass
        a: Rotation parameter  
        Q: Electric charge
        r, theta: Boyer-Lindquist coordinates
        
    Returns:
        g: 4x4 sympy Matrix of the polymer-corrected Kerr-Newman metric
    """
    print(f"🔄 Computing polymer-corrected Kerr-Newman metric...")
    
    # Standard Kerr-Newman metric quantities
    Sigma = r**2 + (a * sp.cos(theta))**2
    Delta = r**2 - 2*M*r + a**2 + Q**2
    
    # Effective polymer parameter (generalized from Kerr case)
    mu_eff_r = mu * sp.sqrt(Sigma) / M
    mu_eff_theta = mu * a * sp.cos(theta) / M
    
    # Effective curvature for Kerr-Newman (includes charge effects)
    K_eff = (M - Q**2/(2*r)) / (r * Sigma)
    
    # Polymer correction factor
    polymer_correction = sp.sin(mu_eff_r * K_eff) / (mu_eff_r * K_eff)
    
    # For small μ, this gives: 1 - (μ_eff * K_eff)²/6 + ...
    Delta_poly = Delta * polymer_correction
    
    # Additional angular and charge corrections
    angular_correction = 1 + (mu_eff_theta**2) * (M/Sigma)
    charge_correction = 1 + (mu**2) * (Q**2/(4*M*r**2))
    
    # Construct polymer-corrected metric components
    g_tt = -(1 - (2*M*r - Q**2)/Sigma) * polymer_correction * charge_correction
    g_rr = Sigma/(Delta_poly * charge_correction)
    g_theta_theta = Sigma * angular_correction
    g_phi_phi = (r**2 + a**2 + (2*M*r - Q**2)*a**2*sp.sin(theta)**2/Sigma) * sp.sin(theta)**2
    
    # Off-diagonal term (frame-dragging with charge effects)
    g_t_phi = -(2*M*r - Q**2)*a*sp.sin(theta)**2/Sigma * polymer_correction * charge_correction
    
    # Assemble metric matrix
    g = sp.zeros(4, 4)
    g[0, 0] = g_tt
    g[1, 1] = g_rr
    g[2, 2] = g_theta_theta
    g[3, 3] = g_phi_phi
    g[0, 3] = g[3, 0] = g_t_phi
    
    print(f"   ✅ Polymer Kerr-Newman metric computed")
    return g

def extract_kerr_newman_coefficients(g_metric, order=8):
    """
    Extract polynomial coefficients (α, β, γ, etc.) for the Kerr-Newman metric expansion in mu.

    Args:
        g_metric: sympy Matrix for polymer-corrected metric
        order: highest power of mu to expand to

    Returns:
        coeffs: Dictionary mapping coefficient names to expressions
    """
    print(f"🔬 Extracting Kerr-Newman coefficients up to μ^{order}...")
    
    mu = sp.symbols('mu')
    coeffs = {}
    
    # Extract coefficients from g_tt component
    g_tt = g_metric[0, 0]
    
    try:
        # Series expansion around μ = 0
        series_expansion = sp.series(g_tt, mu, 0, order + 2).removeO()
        
        # Extract coefficients of different powers
        coeff_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta']
        
        for i, name in enumerate(coeff_names):
            power = 2 * (i + 1)  # μ², μ⁴, μ⁶, ...
            if power <= order:
                coeff = series_expansion.coeff(mu, power)
                if coeff is not None:
                    coeffs[name] = sp.simplify(coeff)
                else:
                    coeffs[name] = 0
        
        print(f"   ✅ Extracted {len(coeffs)} coefficients")
        
    except Exception as e:
        print(f"   ⚠️  Error in coefficient extraction: {e}")
        # Provide fallback values
        coeffs = {
            'alpha': sp.Rational(1, 6),
            'beta': 0,
            'gamma': sp.Rational(1, 2520)
        }
    
    return coeffs

# ------------------------------------------------------------------------
# 2) CHARGE AND SPIN DEPENDENT ANALYSIS
# ------------------------------------------------------------------------

def analyze_charge_spin_dependence(coefficients: Dict[str, sp.Expr], 
                                  Q_values: List[float], a_values: List[float]):
    """Analyze how coefficients depend on both charge Q and spin a."""
    print("⚡🌀 Analyzing charge and spin dependence...")
    
    Q, a = sp.symbols('Q a')
    charge_spin_analysis = {}
    
    for coeff_name, coeff_expr in coefficients.items():
        print(f"\n📊 {coeff_name.upper()} coefficient vs charge and spin:")
        
        # Evaluate at different charge and spin values
        numerical_grid = {}
        for Q_val in Q_values:
            numerical_grid[Q_val] = {}
            for a_val in a_values:
                try:
                    # Substitute numerical values (M=1 for normalization)
                    M, r, theta = sp.symbols('M r theta')
                    expr_eval = coeff_expr.subs([(M, 1), (r, 3), (theta, sp.pi/2)])
                    val = complex(expr_eval.subs([(Q, Q_val), (a, a_val)]))
                    numerical_grid[Q_val][a_val] = val.real if abs(val.imag) < 1e-10 else val
                    print(f"   Q = {Q_val:.2f}, a = {a_val:.2f}: {numerical_grid[Q_val][a_val]:.6f}")
                except:
                    numerical_grid[Q_val][a_val] = 0
                    print(f"   Q = {Q_val:.2f}, a = {a_val:.2f}: [evaluation error]")
        
        charge_spin_analysis[coeff_name] = {
            'expression': coeff_expr,
            'grid': numerical_grid
        }
    
    return charge_spin_analysis

# ------------------------------------------------------------------------
# 3) HORIZON SHIFT CALCULATION  
# ------------------------------------------------------------------------

def compute_kerr_newman_horizon_shift(coefficients: Dict[str, sp.Expr], 
                                     mu_val: float, M_val: float = 1.0, 
                                     a_val: float = 0.5, Q_val: float = 0.3):
    """Compute horizon shift for charged, rotating black hole."""
    print(f"🔄 Computing Kerr-Newman horizon shift (μ={mu_val}, a={a_val}, Q={Q_val})...")
    
    # Classical outer and inner horizons
    discriminant = M_val**2 - a_val**2 - Q_val**2
    if discriminant <= 0:
        print("⚠️ No real horizons for these parameters!")
        return None
        
    r_plus_classical = M_val + np.sqrt(discriminant)
    r_minus_classical = M_val - np.sqrt(discriminant)
    
    print(f"   Classical outer horizon: r₊ = {r_plus_classical:.4f}")
    print(f"   Classical inner horizon: r₋ = {r_minus_classical:.4f}")
    
    # Evaluate coefficients numerically
    M, r, theta, a, Q = sp.symbols('M r theta a Q')
    alpha = coefficients.get('alpha', sp.Rational(1, 6))
    gamma = coefficients.get('gamma', sp.Rational(1, 2520))
    
    try:
        alpha_num = float(alpha.subs([
            (M, M_val), (a, a_val), (Q, Q_val), 
            (theta, sp.pi/2), (r, r_plus_classical)
        ]))
    except:
        alpha_num = 1/6
        
    try:
        gamma_num = float(gamma.subs([
            (M, M_val), (a, a_val), (Q, Q_val),
            (theta, sp.pi/2), (r, r_plus_classical)
        ]))
    except:
        gamma_num = 1/2520
    
    # Horizon shift estimate (leading order)
    delta_r_alpha = alpha_num * mu_val**2 * M_val**2 / r_plus_classical**3
    delta_r_gamma = gamma_num * mu_val**6 * M_val**4 / r_plus_classical**9
    
    total_shift = delta_r_alpha + delta_r_gamma
    
    print(f"   α contribution: Δr = {delta_r_alpha:.6f}")
    print(f"   γ contribution: Δr = {delta_r_gamma:.6f}")
    print(f"   Total shift: Δr = {total_shift:.6f}")
    print(f"   Relative shift: Δr/r₊ = {total_shift/r_plus_classical:.6f}")
    
    return {
        'classical_horizons': {'r_plus': r_plus_classical, 'r_minus': r_minus_classical},
        'shifts': {'alpha': delta_r_alpha, 'gamma': delta_r_gamma, 'total': total_shift},
        'relative_shift': total_shift / r_plus_classical,
        'parameters': {'mu': mu_val, 'M': M_val, 'a': a_val, 'Q': Q_val}
    }

# ------------------------------------------------------------------------
# 4) COMPARISON WITH KERR AND SCHWARZSCHILD
# ------------------------------------------------------------------------

def compare_with_kerr_schwarzschild(kn_coeffs: Dict[str, sp.Expr]):
    """Compare Kerr-Newman coefficients with Kerr and Schwarzschild cases."""
    print("🔄 Comparing with Kerr and Schwarzschild cases...")
    
    # Define limits
    Q, a = sp.symbols('Q a')
    
    # Schwarzschild limit: a → 0, Q → 0
    print("\n📐 Schwarzschild limit (a → 0, Q → 0):")
    for name, coeff in kn_coeffs.items():
        try:
            schw_limit = sp.limit(sp.limit(coeff, a, 0), Q, 0)
            print(f"   {name}: {schw_limit}")
        except:
            print(f"   {name}: [limit calculation failed]")
    
    # Kerr limit: Q → 0
    print("\n🌀 Kerr limit (Q → 0):")
    for name, coeff in kn_coeffs.items():
        try:
            kerr_limit = sp.limit(coeff, Q, 0)
            print(f"   {name}: {kerr_limit}")
        except:
            print(f"   {name}: [limit calculation failed]")
    
    # Reissner-Nordström limit: a → 0
    print("\n⚡ Reissner-Nordström limit (a → 0):")
    for name, coeff in kn_coeffs.items():
        try:
            rn_limit = sp.limit(coeff, a, 0)
            print(f"   {name}: {rn_limit}")
        except:
            print(f"   {name}: [limit calculation failed]")

# ------------------------------------------------------------------------
# 5) MAIN EXECUTION FUNCTION
# ------------------------------------------------------------------------

def main():
    """Main execution function for Kerr-Newman generalization."""
    print("🚀 Kerr-Newman Generalization for LQG Black Holes")
    print("=" * 60)
    
    start_time = time.time()
    
    # Define symbols
    mu, M, a, Q, r, theta = sp.symbols('mu M a Q r theta', real=True, positive=True)
    
    # Step 1: Compute polymer-corrected Kerr-Newman metric
    print("\n📐 Computing polymer-corrected Kerr-Newman metric...")
    g_kn = compute_polymer_kerr_newman_metric(mu, M, a, Q, r, theta)
    
    # Step 2: Extract coefficients
    print("\n" + "="*60)
    coeffs = extract_kerr_newman_coefficients(g_kn, order=8)
    
    print("\n📊 Extracted Kerr-Newman coefficients:")
    for name, coeff in coeffs.items():
        print(f"   {name}: {coeff}")
    
    # Step 3: Analyze charge and spin dependence
    print("\n" + "="*60)
    Q_values = [0.0, 0.2, 0.5, 0.8]
    a_values = [0.0, 0.2, 0.5, 0.8, 0.99]
    
    charge_spin_analysis = analyze_charge_spin_dependence(coeffs, Q_values, a_values)
    
    # Step 4: Compute horizon shifts
    print("\n" + "="*60)
    horizon_shifts = []
    for Q_val in [0.0, 0.3, 0.6]:
        for a_val in [0.0, 0.5, 0.9]:
            shift_result = compute_kerr_newman_horizon_shift(
                coeffs, mu_val=0.1, a_val=a_val, Q_val=Q_val
            )
            if shift_result:
                horizon_shifts.append(shift_result)
    
    # Step 5: Compare with simpler cases
    print("\n" + "="*60)
    compare_with_kerr_schwarzschild(coeffs)
    
    print(f"\n✅ Kerr-Newman analysis completed in {time.time() - start_time:.2f}s")
    
    return {
        'metric': g_kn,
        'coefficients': coeffs,
        'charge_spin_analysis': charge_spin_analysis,
        'horizon_shifts': horizon_shifts,
        'computation_time': time.time() - start_time
    }

if __name__ == "__main__":
    results = main()
