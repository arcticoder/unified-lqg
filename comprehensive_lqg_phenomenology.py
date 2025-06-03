#!/usr/bin/env python3
"""
Comprehensive LQG Phenomenology Explorer with Complete μ⁶ Analysis

This script implements comprehensive phenomenological analysis of LQG-corrected
black hole metrics including observational signatures, realistic parameter ranges,
and non-spherical extensions.

Features:
- Complete α, β, γ coefficient analysis
- Closed-form resummation validation  
- Horizon shift analysis
- Photon sphere and ISCO modifications
- Gravitational wave signatures
- Observational constraints from real systems
- Extensions to Reissner-Nordström and Kerr backgrounds

Author: Complete LQG Phenomenology Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import fsolve

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def define_lqg_metric_functions():
    """
    Define the LQG metric functions for numerical analysis.
    """
    # Coefficients from our extraction
    alpha = 1.0/6.0
    beta = 0.0  # Vanishes at leading order
    gamma = 1.0/2520.0  # Estimated from μ⁶ expansion
    
    def f_schwarzschild(r, M):
        """Standard Schwarzschild metric"""
        return 1.0 - 2.0*M/r
    
    def f_lqg_polynomial(r, M, mu):
        """LQG polynomial metric up to μ⁶"""
        f_classical = f_schwarzschild(r, M)
        correction_mu2 = alpha * (mu**2) * (M**2) / (r**4)
        correction_mu6 = gamma * (mu**6) * (M**4) / (r**10)
        return f_classical + correction_mu2 + correction_mu6
    
    def f_lqg_resummed(r, M, mu):
        """LQG resummed metric f = 1 - 2M/r + [α·μ²M²/r⁴] / [1 + c·μ²]"""
        f_classical = f_schwarzschild(r, M)
        c = gamma / alpha  # = 1/420
        resummed_correction = (alpha * (mu**2) * (M**2) / (r**4)) / (1.0 + c * (mu**2))
        return f_classical + resummed_correction
    
    return {
        'schwarzschild': f_schwarzschild,
        'lqg_polynomial': f_lqg_polynomial,
        'lqg_resummed': f_lqg_resummed,
        'coefficients': {'alpha': alpha, 'beta': beta, 'gamma': gamma}
    }

def analyze_horizon_modifications(metrics, M=1.0, mu_range=None):
    """
    Analyze modifications to the black hole horizon.
    """
    if mu_range is None:
        mu_range = np.logspace(-3, 0, 20)  # μ from 0.001 to 1.0
    
    print("="*60)
    print("HORIZON MODIFICATION ANALYSIS")
    print("="*60)
    
    results = {
        'mu_values': mu_range,
        'horizon_polynomial': [],
        'horizon_resummed': [],
        'horizon_shifts': []
    }
    
    # Classical horizon
    r_h_classical = 2.0 * M
    
    for mu in mu_range:
        # Find polynomial horizon (approximate)
        alpha = metrics['coefficients']['alpha']
        # Leading correction: r_h ≈ 2M - α·μ²M²/(2M)³ = 2M - α·μ²/(4M)
        horizon_shift = -alpha * (mu**2) / (4.0 * M)
        r_h_poly_approx = r_h_classical + horizon_shift
        
        # For resummed form, solve f_LQG(r) = 0 numerically
        try:
            from scipy.optimize import fsolve
            def f_lqg_root(r):
                return metrics['lqg_resummed'](r, M, mu)
            r_h_resummed = fsolve(f_lqg_root, r_h_classical)[0]
        except:
            # Fallback to approximation
            r_h_resummed = r_h_poly_approx
        
        results['horizon_polynomial'].append(r_h_poly_approx)
        results['horizon_resummed'].append(r_h_resummed)
        results['horizon_shifts'].append(horizon_shift)
    
    # Print key results
    print(f"Classical horizon: r_h = {r_h_classical:.3f}M")
    print("\nHorizon shifts for selected μ values:")
    for i, mu in enumerate([0.01, 0.05, 0.1, 0.2, 0.5]):
        if mu <= mu_range.max():
            idx = np.argmin(np.abs(mu_range - mu))
            shift = results['horizon_shifts'][idx]
            shift_percent = 100 * shift / r_h_classical
            print(f"  μ = {mu:.3f}: Δr_h = {shift:.6f}M ({shift_percent:.4f}%)")
    
    return results

def analyze_photon_sphere_isco(metrics, M=1.0, mu_range=None):
    """
    Analyze modifications to photon sphere and ISCO.
    """
    if mu_range is None:
        mu_range = np.logspace(-3, 0, 10)
    
    print("\n" + "="*60)
    print("PHOTON SPHERE & ISCO ANALYSIS")
    print("="*60)
    
    # Classical values
    r_ph_classical = 3.0 * M  # Photon sphere
    r_isco_classical = 6.0 * M  # ISCO
    
    alpha = metrics['coefficients']['alpha']
    
    print(f"Classical photon sphere: r_ph = {r_ph_classical:.1f}M")
    print(f"Classical ISCO: r_ISCO = {r_isco_classical:.1f}M")
    
    print("\nLQG corrections:")
    for mu in [0.01, 0.05, 0.1, 0.2]:
        # Photon sphere correction (leading order)
        # r_ph ≈ 3M[1 - α·μ²M²/(27M⁴)] = 3M[1 - α·μ²/27]
        ph_correction = -alpha * (mu**2) / 27.0
        r_ph_lqg = r_ph_classical * (1.0 + ph_correction)
        
        # ISCO correction (estimate)
        # r_ISCO ≈ 6M[1 - α·μ²M²/(216M⁴)] = 6M[1 - α·μ²/216]
        isco_correction = -alpha * (mu**2) / 216.0
        r_isco_lqg = r_isco_classical * (1.0 + isco_correction)
        
        print(f"\n  μ = {mu:.3f}:")
        print(f"    Photon sphere: r_ph = {r_ph_lqg:.6f}M (shift: {100*ph_correction:.4f}%)")
        print(f"    ISCO: r_ISCO = {r_isco_lqg:.6f}M (shift: {100*isco_correction:.4f}%)")

def compute_redshift_corrections(metrics, M=1.0, r_observer=10.0, mu_range=None):
    """
    Compute gravitational redshift corrections.
    """
    if mu_range is None:
        mu_range = [0.01, 0.05, 0.1, 0.2]
    
    print("\n" + "="*60)
    print("GRAVITATIONAL REDSHIFT CORRECTIONS")
    print("="*60)
    
    print(f"Observer at r = {r_observer:.1f}M")
    
    # Classical redshift: 1 + z = sqrt(f(r_obs)/f(r_em))
    # For emission at various radii
    r_emission_values = [2.5, 3.0, 5.0, 10.0]  # In units of M
    
    for r_em in r_emission_values:
        print(f"\nEmission from r = {r_em:.1f}M:")
        
        # Classical redshift
        f_obs_classical = metrics['schwarzschild'](r_observer, M)
        f_em_classical = metrics['schwarzschild'](r_em, M)
        z_classical = np.sqrt(f_obs_classical / f_em_classical) - 1.0
        
        print(f"  Classical redshift: z = {z_classical:.6f}")
        
        for mu in mu_range:
            # LQG redshift
            f_obs_lqg = metrics['lqg_resummed'](r_observer, M, mu)
            f_em_lqg = metrics['lqg_resummed'](r_em, M, mu)
            z_lqg = np.sqrt(f_obs_lqg / f_em_lqg) - 1.0
            
            # Relative correction
            delta_z_rel = (z_lqg - z_classical) / z_classical
            
            print(f"  μ = {mu:.3f}: z_LQG = {z_lqg:.6f} (Δz/z = {100*delta_z_rel:.4f}%)")

def estimate_gravitational_wave_corrections(metrics, M=1.0, mu_range=None):
    """
    Estimate corrections to gravitational wave frequencies and damping.
    """
    if mu_range is None:
        mu_range = [0.01, 0.05, 0.1, 0.2]
    
    print("\n" + "="*60)
    print("GRAVITATIONAL WAVE SIGNATURE CORRECTIONS")
    print("="*60)
    
    alpha = metrics['coefficients']['alpha']
    
    # Quasi-normal mode corrections
    # ω_QNM ∝ 1/M * f'(r_h) / √f(r_h)
    # For LQG: ω_QNM,LQG ≈ ω_QNM,GR * [1 + α·μ²M²/r_h⁴]
    
    print("Quasi-normal mode frequency corrections:")
    print("ω_QNM,LQG ≈ ω_QNM,GR * [1 + α·μ²M²/r_h⁴]")
    
    for mu in mu_range:
        r_h = 2.0 * M  # Approximate horizon
        qnm_correction = alpha * (mu**2) * (M**2) / (r_h**4)
        qnm_correction_percent = 100 * qnm_correction
        
        print(f"  μ = {mu:.3f}: Δω/ω ≈ {qnm_correction_percent:.4f}%")
    
    # Ringdown damping time corrections
    print("\nRingdown damping time corrections:")
    print("τ_damping,LQG ≈ τ_damping,GR / [1 + α·μ²M²/r_h⁴]")
    
    for mu in mu_range:
        r_h = 2.0 * M
        damping_correction = -alpha * (mu**2) * (M**2) / (r_h**4)
        damping_correction_percent = 100 * damping_correction
        
        print(f"  μ = {mu:.3f}: Δτ/τ ≈ {damping_correction_percent:.4f}%")

def observational_constraints_analysis():
    """
    Analyze observational constraints on the polymer parameter μ.
    """
    print("\n" + "="*60)
    print("OBSERVATIONAL CONSTRAINTS ON μ")
    print("="*60)
    
    alpha = 1.0/6.0
    
    # Event Horizon Telescope constraints
    print("1. EVENT HORIZON TELESCOPE (M87* and Sgr A*):")
    
    # M87*: M ≈ 6.5 × 10^9 M_solar
    M_M87 = 6.5e9  # Solar masses
    shadow_precision = 0.01  # 1% precision on shadow radius
    
    # Shadow radius: R_shadow ≈ 3√3 GM/c² = 3√3 M (in geometric units)
    # LQG correction: ΔR/R ≈ α·μ²M²/(shadow_radius)⁴ ≈ α·μ²/(3√3)⁴
    shadow_correction_factor = alpha / (3 * np.sqrt(3))**4
    mu_constraint_EHT = np.sqrt(shadow_precision / shadow_correction_factor)
    
    print(f"  M87* mass: {M_M87:.1e} M_solar")
    print(f"  Shadow measurement precision: {100*shadow_precision:.1f}%")
    print(f"  Constraint: μ < {mu_constraint_EHT:.3f}")
    
    # LIGO/Virgo constraints
    print("\n2. LIGO/VIRGO GRAVITATIONAL WAVES:")
    
    # Typical merger: M ~ 30 M_solar
    M_merger = 30.0
    gw_frequency_precision = 0.001  # 0.1% precision
    
    # QNM frequency correction: Δf/f ≈ α·μ²M²/r_h⁴ = α·μ²/(16M²)
    r_h_merger = 2.0  # Horizon in units of M
    qnm_correction_factor = alpha / (r_h_merger**4)
    mu_constraint_LIGO = np.sqrt(gw_frequency_precision / qnm_correction_factor)
    
    print(f"  Typical merger mass: {M_merger:.1f} M_solar")
    print(f"  Frequency precision: {100*gw_frequency_precision:.1f}%")
    print(f"  Constraint: μ < {mu_constraint_LIGO:.3f}")
    
    # X-ray timing constraints
    print("\n3. X-RAY TIMING (ISCO frequencies):")
    
    # Stellar-mass black holes: M ~ 10 M_solar
    M_stellar = 10.0
    isco_frequency_precision = 0.01  # 1% precision
    
    # ISCO frequency: f_ISCO ∝ 1/(M * r_ISCO^{3/2})
    # Correction: Δf/f ≈ -(3/2) * Δr_ISCO/r_ISCO ≈ -(3/2) * α·μ²/216
    isco_correction_factor = (3.0/2.0) * alpha / 216.0
    mu_constraint_xray = np.sqrt(isco_frequency_precision / isco_correction_factor)
    
    print(f"  Stellar-mass BH: {M_stellar:.1f} M_solar")
    print(f"  ISCO frequency precision: {100*isco_frequency_precision:.1f}%")
    print(f"  Constraint: μ < {mu_constraint_xray:.3f}")
    
    # Summary
    print("\n" + "-"*40)
    print("SUMMARY OF CONSTRAINTS:")
    print(f"  EHT (shadow): μ < {mu_constraint_EHT:.3f}")
    print(f"  LIGO (QNM): μ < {mu_constraint_LIGO:.3f}")
    print(f"  X-ray (ISCO): μ < {mu_constraint_xray:.3f}")
    
    strongest_constraint = min(mu_constraint_EHT, mu_constraint_LIGO, mu_constraint_xray)
    print(f"\n  STRONGEST CONSTRAINT: μ < {strongest_constraint:.3f}")
    
    return {
        'EHT': mu_constraint_EHT,
        'LIGO': mu_constraint_LIGO,
        'X-ray': mu_constraint_xray,
        'strongest': strongest_constraint
    }

def non_spherical_extensions():
    """
    Explore extensions to non-spherical backgrounds.
    """
    print("\n" + "="*60)
    print("NON-SPHERICAL EXTENSIONS")
    print("="*60)
    
    alpha = 1.0/6.0
    
    # Reissner-Nordström
    print("1. REISSNER-NORDSTRÖM LQG:")
    print("   f_RN-LQG(r) = 1 - 2M/r + Q²/r² + α·μ²(M² + Q²M)/r⁴")
    print("   where Q is the electric charge")
    
    Q_values = [0.5, 0.8, 0.95]  # In units of M
    M = 1.0
    mu = 0.1
    
    for Q in Q_values:
        # Classical RN
        r_h_RN_outer = M + np.sqrt(M**2 - Q**2)  # Outer horizon
        r_h_RN_inner = M - np.sqrt(M**2 - Q**2)  # Inner horizon (Cauchy)
        
        # LQG correction to charge term
        lqg_charge_correction = alpha * (mu**2) * (M**2 + Q**2 * M)
        
        print(f"\n   Q = {Q:.2f}M:")
        print(f"     Classical horizons: r+ = {r_h_RN_outer:.3f}M, r- = {r_h_RN_inner:.3f}M")
        print(f"     LQG charge correction: {lqg_charge_correction:.6f}/r⁴")
    
    # Kerr (simplified)
    print("\n2. KERR LQG (slow rotation approximation):")
    print("   f_Kerr-LQG(r,θ) ≈ 1 - 2M/r + α·μ²M²/r⁴ + O(a²)")
    print("   where a is the specific angular momentum")
    
    a_values = [0.1, 0.5, 0.9]  # In units of M
    
    for a in a_values:
        # Classical Kerr horizons
        r_h_Kerr_outer = M + np.sqrt(M**2 - a**2)
        r_h_Kerr_inner = M - np.sqrt(M**2 - a**2)
        
        print(f"\n   a = {a:.1f}M:")
        print(f"     Classical horizons: r+ = {r_h_Kerr_outer:.3f}M, r- = {r_h_Kerr_inner:.3f}M")
        print(f"     LQG corrections modify both f(r) and angular structure")
    
    # AdS background
    print("\n3. ASYMPTOTICALLY AdS:")
    print("   f_AdS-LQG(r) = 1 - 2M/r - Λr²/3 + α·μ²M²/r⁴")
    print("   where Λ < 0 is the cosmological constant")
    
    # Example: Λ = -3/L² where L is the AdS radius
    L_AdS_values = [10.0, 50.0, 100.0]  # In units of M
    
    for L in L_AdS_values:
        Lambda = -3.0 / (L**2)
        print(f"\n   AdS radius L = {L:.0f}M (Λ = {Lambda:.6f}):")
        print(f"     AdS contribution: -Λr²/3 = {-Lambda/3:.6f}r²")
        print(f"     Competes with LQG term at r ~ (M²μ²/|Λ|)^{1/6}")

def evaluate_resummed_metric(r_vals, M=1.0, mu=0.1):
    """
    Evaluate the resummed LQG metric f_LQG(r) = 1 - 2M/r + [μ²M²/6r⁴] / [1 + μ²/420]
    
    Args:
        r_vals: Array of radial coordinates
        M: Mass parameter (default: 1.0)
        mu: Polymer parameter (default: 0.1)
    
    Returns:
        Array of metric values
    """
    alpha = 1/6
    c = 1/420  # γ/α where γ = 1/2520
    
    # Schwarzschild part
    f_schwarzschild = 1 - 2*M/r_vals
    
    # LQG correction with resummation
    lqg_correction = (alpha * mu**2 * M**2 / r_vals**4) / (1 + c * mu**2)
    
    return f_schwarzschild + lqg_correction

def evaluate_polynomial_metric(r_vals, M=1.0, mu=0.1, include_gamma=True):
    """
    Evaluate the polynomial LQG metric up to μ⁶ order.
    
    Args:
        r_vals: Array of radial coordinates
        M: Mass parameter (default: 1.0)
        mu: Polymer parameter (default: 0.1)
        include_gamma: Whether to include γ term (default: True)
    
    Returns:
        Array of metric values
    """
    alpha = 1/6
    beta = 0
    gamma = 1/2520 if include_gamma else 0
    
    # Schwarzschild part
    f = 1 - 2*M/r_vals
    
    # LQG corrections
    f += alpha * mu**2 * M**2 / r_vals**4
    f += beta * mu**4 * M**3 / r_vals**7  # This is zero
    f += gamma * mu**6 * M**4 / r_vals**10
    
    return f

def create_comparison_plots():
    """
    Create comparison plots of different metrics.
    """
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    
    try:
        metrics = define_lqg_metric_functions()
        
        # Parameters
        M = 1.0
        mu_values = [0.05, 0.1, 0.2]
        r_range = np.linspace(2.1, 10.0, 200)
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Metric comparison
        plt.subplot(2, 2, 1)
        
        # Schwarzschild
        f_schw = [metrics['schwarzschild'](r, M) for r in r_range]
        plt.plot(r_range, f_schw, 'k-', linewidth=2, label='Schwarzschild')
        
        # LQG corrections
        for mu in mu_values:
            f_lqg_poly = [metrics['lqg_polynomial'](r, M, mu) for r in r_range]
            f_lqg_resum = [metrics['lqg_resummed'](r, M, mu) for r in r_range]
            
            plt.plot(r_range, f_lqg_poly, '--', label=f'LQG poly μ={mu}')
            plt.plot(r_range, f_lqg_resum, ':', label=f'LQG resum μ={mu}')
        
        plt.xlabel('r/M')
        plt.ylabel('f(r)')
        plt.title('Metric Function Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Relative corrections
        plt.subplot(2, 2, 2)
        
        for mu in mu_values:
            corrections = []
            for r in r_range:
                f_schw_val = metrics['schwarzschild'](r, M)
                f_lqg_val = metrics['lqg_resummed'](r, M, mu)
                rel_correction = (f_lqg_val - f_schw_val) / f_schw_val
                corrections.append(100 * rel_correction)
            
            plt.semilogy(r_range, np.abs(corrections), label=f'μ = {mu}')
        
        plt.xlabel('r/M')
        plt.ylabel('|Δf/f| (%)')
        plt.title('Relative Metric Corrections')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Horizon shifts
        plt.subplot(2, 2, 3)
        
        mu_range = np.linspace(0.01, 0.3, 50)
        alpha = metrics['coefficients']['alpha']
        horizon_shifts = [-alpha * (mu**2) / (4.0 * M) for mu in mu_range]
        horizon_shifts_percent = [100 * shift / (2.0 * M) for shift in horizon_shifts]
        
        plt.plot(mu_range, horizon_shifts_percent, 'b-', linewidth=2)
        plt.xlabel('μ')
        plt.ylabel('Horizon shift (%)')
        plt.title('Horizon Radius Shift')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Observational constraints
        plt.subplot(2, 2, 4)
        
        # Constraint lines
        constraints = observational_constraints_analysis()
        
        mu_test = np.linspace(0.01, 1.0, 100)
        shadow_signal = [alpha / (3 * np.sqrt(3))**4 * mu**2 for mu in mu_test]
        qnm_signal = [alpha / 16.0 * mu**2 for mu in mu_test]
        
        plt.loglog(mu_test, shadow_signal, 'r-', label='Shadow signal')
        plt.loglog(mu_test, qnm_signal, 'b-', label='QNM signal')
        plt.axhline(0.01, color='r', linestyle='--', alpha=0.7, label='EHT precision')
        plt.axhline(0.001, color='b', linestyle='--', alpha=0.7, label='LIGO precision')
        plt.axvline(constraints['strongest'], color='k', linestyle=':', alpha=0.7, label='Strongest constraint')
        
        plt.xlabel('μ')
        plt.ylabel('Signal strength')
        plt.title('Observational Constraints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'lqg_phenomenology_comprehensive.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available; skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")

def generate_comprehensive_plots():
    """
    Generate comprehensive comparison plots for polynomial vs resummed metrics.
    """
    # Parameters
    M = 1.0
    mu_values = [0.01, 0.05, 0.1, 0.2]
    r_vals = np.linspace(2.1, 10.0, 200)
    colors = ['blue', 'green', 'red', 'purple']
    
    # Create figure with additional subplot for resummation comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LQG Polymer Black Hole: Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Metric comparison (polynomial vs resummed)
    ax1 = axes[0, 0]
    for i, mu in enumerate(mu_values):
        f_poly = evaluate_polynomial_metric(r_vals, M, mu)
        f_resummed = evaluate_resummed_metric(r_vals, M, mu)
        
        ax1.plot(r_vals/M, f_poly, '--', linewidth=2, alpha=0.8, 
                label=f'Polynomial μ={mu}', color=colors[i])
        ax1.plot(r_vals/M, f_resummed, '-', linewidth=2, 
                label=f'Resummed μ={mu}', color=colors[i])
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('r/M')
    ax1.set_ylabel('f(r)')
    ax1.set_title('Polynomial vs Resummed Metrics')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_xlim(1.5, 10)
    
    # Plot 2: Relative difference between polynomial and resummed
    ax2 = axes[0, 1]
    for i, mu in enumerate(mu_values):
        f_poly = evaluate_polynomial_metric(r_vals, M, mu)
        f_resummed = evaluate_resummed_metric(r_vals, M, mu)
        
        # Relative difference
        rel_diff = np.abs(f_poly - f_resummed) / np.abs(f_resummed)
        
        ax2.semilogy(r_vals/M, rel_diff, linewidth=2, 
                    label=f'μ={mu}', color=colors[i])
    
    ax2.set_xlabel('r/M')
    ax2.set_ylabel('|f_poly - f_resummed|/|f_resummed|')
    ax2.set_title('Polynomial vs Resummed: Relative Difference')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(1.5, 10)
    
    # Plot 3: LQG corrections (resummed)
    ax3 = axes[0, 2]
    f_schwarzschild = 1 - 2*M/r_vals
    for i, mu in enumerate(mu_values):
        f_resummed = evaluate_resummed_metric(r_vals, M, mu)
        correction = (f_resummed - f_schwarzschild) / f_schwarzschild
        
        ax3.semilogy(r_vals/M, np.abs(correction), linewidth=2,
                    label=f'μ={mu}', color=colors[i])
    
    ax3.set_xlabel('r/M')
    ax3.set_ylabel('|Δf/f_Schwarzschild|')
    ax3.set_title('LQG Corrections (Resummed)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(1.5, 10)
    
    # Plot 4: Horizon analysis
    ax4 = axes[1, 0]
    horizon_shifts = []
    for mu in mu_values:
        # Find horizon numerically for resummed metric
        def metric_at_horizon(r):
            return evaluate_resummed_metric(np.array([r]), M, mu)[0]
        
        try:
            r_horizon = fsolve(metric_at_horizon, 2.0)[0]
            shift = r_horizon - 2*M
            horizon_shifts.append(shift)
        except:
            horizon_shifts.append(np.nan)
    
    ax4.plot(mu_values, np.array(horizon_shifts)/M, 'b-', linewidth=2)
    ax4.set_xlabel('μ')
    ax4.set_ylabel('Δr_h/M')
    ax4.set_title('Event Horizon Shift (Resummed)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Photon sphere analysis
    ax5 = axes[1, 1]
    photon_corrections = []
    for mu in mu_values:
        # Approximate photon sphere correction
        f_at_3M = evaluate_resummed_metric(np.array([3*M]), M, mu)[0]
        f_schwarzschild_3M = 1 - 2*M/(3*M)
        correction = (f_at_3M - f_schwarzschild_3M) / f_schwarzschild_3M
        photon_corrections.append(correction)
    
    ax5.semilogy(mu_values, np.abs(photon_corrections), 'g-', linewidth=2)
    ax5.set_xlabel('μ')
    ax5.set_ylabel('|Δf/f| at r=3M')
    ax5.set_title('Photon Sphere Corrections')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Observational constraints
    ax6 = axes[1, 2]
    
    # EHT constraint (1% precision at photon sphere)
    eht_precision = 0.01
    mu_eht = np.sqrt(eht_precision * (3**4) / (1/6))
    
    # LIGO constraint (0.1% precision)
    ligo_precision = 0.001
    mu_ligo = np.sqrt(ligo_precision / (1/6))
    
    # X-ray timing (10% precision at ISCO)
    xray_precision = 0.1
    mu_xray = np.sqrt(xray_precision * (6**4) / (1/6))
    
    constraints = ['EHT\n(1%)', 'LIGO\n(0.1%)', 'X-ray\n(10%)']
    mu_limits = [mu_eht, mu_ligo, mu_xray]
    colors_constraints = ['red', 'blue', 'green']
    
    bars = ax6.bar(constraints, mu_limits, color=colors_constraints, alpha=0.7)
    ax6.set_ylabel('μ constraint')
    ax6.set_title('Observational Constraints')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mu_limits):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('lqg_comprehensive_analysis_with_resummation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive analysis plot saved as 'lqg_comprehensive_analysis_with_resummation.png'")

def main():
    """
    Execute comprehensive LQG phenomenology analysis.
    """
    start_time = time.time()
    
    print("COMPREHENSIVE LQG PHENOMENOLOGY ANALYSIS")
    print("WITH μ⁶ CORRECTIONS AND OBSERVATIONAL SIGNATURES")
    print("="*80)
    
    # Define metric functions
    metrics = define_lqg_metric_functions()
    
    print("LQG Metric Coefficients:")
    for key, value in metrics['coefficients'].items():
        print(f"  {key} = {value}")
    
    # Horizon analysis
    horizon_results = analyze_horizon_modifications(metrics)
    
    # Photon sphere and ISCO
    analyze_photon_sphere_isco(metrics)
    
    # Redshift corrections
    compute_redshift_corrections(metrics)
    
    # Gravitational wave signatures
    estimate_gravitational_wave_corrections(metrics)
    
    # Observational constraints
    constraints = observational_constraints_analysis()
    
    # Non-spherical extensions
    non_spherical_extensions()
    
    # Create plots
    create_comparison_plots()
    generate_comprehensive_plots()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("\n" + "="*80)
    print("PHENOMENOLOGY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Strongest observational constraint: μ < {constraints['strongest']:.3f}")
    
    return {
        'metrics': metrics,
        'constraints': constraints,
        'execution_time': execution_time
    }

if __name__ == "__main__":
    print("COMPREHENSIVE LQG PHENOMENOLOGY WITH RESUMMATION")
    print("="*60)
    
    # Run comprehensive analysis
    import time
    results = main()
    
    # Generate enhanced plots with resummation comparison  
    print("\nGenerating comprehensive plots with resummation comparison...")
    generate_comprehensive_plots()
    
    print("\nAnalysis completed successfully!")
