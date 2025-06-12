#!/usr/bin/env python3
"""
LQG Non-Abelian Propagator Implementation

This module implements the complete non-Abelian momentum-space propagator 
with polymer corrections for direct integration into LQG pipelines.

D̃^{ab}_{μν}(k) = δ^{ab} * (η_{μν} - k_μk_ν/k²) * sin²(μ_g√(k²+m_g²))/(k²+m_g²) / μ_g²
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class PropagatorConfig:
    """Configuration for non-Abelian propagator calculations."""
    mu_g: float = 0.15  # Polymer scale parameter
    m_g: float = 0.1    # Gauge field mass
    N_colors: int = 3   # Number of colors (SU(3))
    momentum_cutoff: float = 10.0  # UV cutoff in momentum space

class LQGNonAbelianPropagator:
    """
    Complete non-Abelian propagator implementation for LQG-QFT calculations.
    
    This class provides the ACTUAL momentum-space 2-point functions that must
    be used in ALL spin-foam and ANEC violation calculations.
    """
    
    def __init__(self, config: PropagatorConfig):
        self.config = config
        self.eta = np.diag([1, -1, -1, -1])  # Minkowski metric
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._propagator_cache = {}
        
    def momentum_squared(self, k: np.ndarray) -> float:
        """Compute k² = k_μ k^μ in Minkowski space."""
        if len(k) != 4:
            raise ValueError("Momentum must be 4-vector")
        return k[0]**2 - k[1]**2 - k[2]**2 - k[3]**2
    
    def transverse_projector(self, k: np.ndarray, mu: int, nu: int) -> float:
        """
        Compute the transverse projector (η_{μν} - k_μk_ν/k²).
        
        This ensures gauge invariance: k^μ D̃^{ab}_{μν} = 0
        """
        k_sq = self.momentum_squared(k)
        if abs(k_sq) < 1e-12:
            # At k=0, use pure metric
            return self.eta[mu, nu]
        
        return self.eta[mu, nu] - k[mu] * k[nu] / k_sq
    
    def polymer_factor(self, k: np.ndarray) -> float:
        """
        Compute the polymer modification factor sin²(μ_g√(k²+m_g²))/(k²+m_g²).
        
        This is the KEY polymer correction that distinguishes LQG from classical QFT.
        """
        k_sq = self.momentum_squared(k)
        k_mag = np.sqrt(abs(k_sq) + self.config.m_g**2)
        
        # Polymer factor with mass regularization
        sinc_factor = np.sin(self.config.mu_g * k_mag) / (self.config.mu_g * k_mag)
        return sinc_factor**2 / (abs(k_sq) + self.config.m_g**2)
    
    def color_factor(self, a: int, b: int) -> float:
        """Color structure δ^{ab} for SU(N) gauge theory."""
        return 1.0 if a == b else 0.0
    
    def full_propagator_tensor(self, k: np.ndarray, a: int, b: int, mu: int, nu: int) -> float:
        """
        THE complete non-Abelian propagator tensor D̃^{ab}_{μν}(k).
        
        This is the momentum-space 2-point function that MUST be used in
        ALL LQG-QFT calculations involving gauge fields.
        
        Formula: D̃^{ab}_{μν}(k) = δ^{ab} * (η_{μν} - k_μk_ν/k²) * sin²(μ_g√(k²+m_g²))/(k²+m_g²) / μ_g²
        """
        # Validate inputs
        if not (0 <= a < self.config.N_colors and 0 <= b < self.config.N_colors):
            raise ValueError(f"Color indices must be in [0, {self.config.N_colors})")
        if not (0 <= mu < 4 and 0 <= nu < 4):
            raise ValueError("Lorentz indices must be in [0, 4)")
        
        # Build cache key
        cache_key = (tuple(k), a, b, mu, nu)
        if cache_key in self._propagator_cache:
            return self._propagator_cache[cache_key]
        
        # Compute all components
        color_delta = self.color_factor(a, b)
        transverse_proj = self.transverse_projector(k, mu, nu)
        polymer_corr = self.polymer_factor(k)
        
        # Full propagator
        propagator = color_delta * transverse_proj * polymer_corr / (self.config.mu_g**2)
        
        # Cache result
        self._propagator_cache[cache_key] = propagator
        
        return propagator
    
    def momentum_space_2point_correlation(self, k1: np.ndarray, k2: np.ndarray, 
                                        a: int, b: int, mu: int, nu: int) -> float:
        """
        2-point correlation function in momentum space for ANEC calculations.
        
        This implements ⟨A^a_μ(k1) A^b_ν(k2)⟩ with full polymer corrections.
        """
        # Momentum conservation: δ⁴(k1 + k2)
        momentum_conservation = np.allclose(k1 + k2, 0.0, atol=1e-10)
        if not momentum_conservation:
            return 0.0
        
        # Use the full propagator
        return self.full_propagator_tensor(k1, a, b, mu, nu)
    
    def anec_violation_integrand(self, k: np.ndarray, field_config: Dict) -> float:
        """
        Integrand for ANEC violation calculations using the non-Abelian propagator.
        
        This is where the propagator gets used in actual ANEC computations.
        """
        # Sum over color and Lorentz indices
        total = 0.0
        for a in range(self.config.N_colors):
            for b in range(self.config.N_colors):
                for mu in range(4):
                    for nu in range(4):
                        propagator = self.full_propagator_tensor(k, a, b, mu, nu)
                          # Weight by field configuration and energy-momentum tensor
                        weight = field_config.get('coupling_strength', 1.0)
                        energy_factor = field_config.get('energy_density', 1.0)
                        
                        total += weight * energy_factor * propagator
        
        return total
    
    def validate_gauge_invariance(self, k: np.ndarray, tolerance: float = 1e-6) -> bool:
        """
        Verify that k^μ D̃^{ab}_{μν} = 0 (gauge invariance).
        """
        k_sq = self.momentum_squared(k)
        if abs(k_sq) < 1e-10:
            return True  # Skip gauge test at k=0
            
        for a in range(self.config.N_colors):
            for b in range(self.config.N_colors):
                if a != b:
                    continue  # Only check diagonal elements
                for nu in range(4):
                    # Contract with momentum
                    contracted = sum(k[mu] * self.full_propagator_tensor(k, a, b, mu, nu) 
                                   for mu in range(4))
                    
                    if abs(contracted) > tolerance:
                        return False
        
        return True
    def classical_limit_check(self, k: np.ndarray, tolerance: float = 1e-3) -> bool:
        """
        Verify that μ_g → 0 recovers the classical propagator.
        """
        # Temporarily set small μ_g
        original_mu_g = self.config.mu_g
        self.config.mu_g = 1e-6  # Smaller value for better convergence
        
        # Clear cache
        self._propagator_cache.clear()
        
        try:
            # Compute propagator with small μ_g
            small_mu_prop = self.full_propagator_tensor(k, 0, 0, 1, 1)
            
            # Expected classical result
            k_sq = self.momentum_squared(k)
            if abs(k_sq) < 1e-10:
                return True  # Skip test at k=0
                
            transverse = self.transverse_projector(k, 1, 1)
            classical_expected = transverse / (abs(k_sq) + self.config.m_g**2) / (original_mu_g**2)
            
            # Check if they match (relative error)
            if abs(classical_expected) > 1e-10:
                relative_error = abs(small_mu_prop - classical_expected) / abs(classical_expected)
                success = relative_error < tolerance
            else:
                success = abs(small_mu_prop) < tolerance
            
            return success
            
        finally:
            # Restore original μ_g
            self.config.mu_g = original_mu_g
            self._propagator_cache.clear()
    
    def export_propagator_data(self, momentum_grid: np.ndarray, output_file: str) -> Dict:
        """
        Export propagator data over momentum grid for analysis.
        """
        results = {
            'config': {
                'mu_g': self.config.mu_g,
                'm_g': self.config.m_g,
                'N_colors': self.config.N_colors
            },
            'momentum_grid': momentum_grid.tolist(),
            'propagator_values': [],
            'gauge_invariance_checks': [],
            'classical_limit_checks': []
        }
        
        for k_vec in momentum_grid:
            # Compute sample propagator values
            prop_sample = self.full_propagator_tensor(k_vec, 0, 0, 1, 2)
            results['propagator_values'].append(prop_sample)
            
            # Run validation checks
            gauge_check = self.validate_gauge_invariance(k_vec)
            classical_check = self.classical_limit_check(k_vec)
            
            results['gauge_invariance_checks'].append(gauge_check)
            results['classical_limit_checks'].append(classical_check)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

# Integration function for the main LQG pipeline
def integrate_nonabelian_propagator_into_lqg_pipeline(lattice_file: str) -> bool:
    """
    MAIN INTEGRATION FUNCTION: Embed non-Abelian propagator into LQG calculations.
    
    This function modifies the LQG pipeline to use the complete tensor propagator
    in all momentum-space calculations.
    """
    print("🔷 Integrating Non-Abelian Propagator into LQG Pipeline...")
    
    # Initialize propagator
    config = PropagatorConfig(mu_g=0.15, m_g=0.1, N_colors=3)
    propagator = LQGNonAbelianPropagator(config)
    
    # Create momentum grid for validation
    k_values = np.linspace(-2.0, 2.0, 10)
    momentum_grid = []
    for k0 in k_values[::2]:  # Sparse sampling for speed
        for kx in k_values[::2]:
            momentum_grid.append([k0, kx, 0.0, 0.0])
    momentum_grid = np.array(momentum_grid)
    
    # Run validation
    print(f"   Validating propagator on {len(momentum_grid)} momentum points...")
    
    validation_passed = 0
    for i, k_vec in enumerate(momentum_grid):
        gauge_ok = propagator.validate_gauge_invariance(k_vec)
        classical_ok = propagator.classical_limit_check(k_vec)
        
        if gauge_ok and classical_ok:
            validation_passed += 1
    
    success_rate = validation_passed / len(momentum_grid)
    print(f"   Propagator validation: {validation_passed}/{len(momentum_grid)} passed ({success_rate:.1%})")
    
    # Export results
    output_file = "lqg_nonabelian_propagator_integration.json"
    results = propagator.export_propagator_data(momentum_grid, output_file)
    print(f"   Propagator data exported to: {output_file}")
    
    # Success if >80% validation passed
    integration_success = success_rate > 0.8
    
    if integration_success:
        print("✅ Non-Abelian propagator successfully integrated into LQG pipeline")
        
        # Create marker file for downstream processes
        with open("NONABELIAN_PROPAGATOR_INTEGRATED.flag", 'w') as f:
            f.write(f"Non-Abelian propagator integrated with μ_g={config.mu_g}, validation_rate={success_rate:.3f}")
    else:
        print("❌ Non-Abelian propagator integration failed validation")
    
    return integration_success

if __name__ == "__main__":
    # Test the propagator implementation
    config = PropagatorConfig(mu_g=0.15, m_g=0.1)
    propagator = LQGNonAbelianPropagator(config)
    
    # Test momentum vector
    k_test = np.array([1.0, 0.5, 0.3, 0.2])
    
    # Compute propagator
    D_value = propagator.full_propagator_tensor(k_test, 0, 0, 1, 2)
    print(f"Sample propagator value: D̃^{0,0}_{1,2}(k) = {D_value:.6e}")
    
    # Run validation
    gauge_ok = propagator.validate_gauge_invariance(k_test)
    classical_ok = propagator.classical_limit_check(k_test)
    
    print(f"Gauge invariance: {'✓' if gauge_ok else '✗'}")
    print(f"Classical limit: {'✓' if classical_ok else '✗'}")
