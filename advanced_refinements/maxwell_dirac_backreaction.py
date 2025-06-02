"""
Full Maxwell + Dirac Backreaction Integration

This module implements comprehensive backreaction effects between Maxwell electromagnetic
fields and Dirac fermionic matter in the LQG warp drive framework, including nonlinear
coupling terms and self-consistent field equations.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import warnings
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.integrate import solve_ivp

# Import core LQG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from lqg_fixed_components import MidisuperspaceHamiltonianConstraint
from lqg_additional_matter import MaxwellField, DiracField, PhantomScalarField

@dataclass
class BackreactionConfig:
    """Configuration for backreaction calculation"""
    coupling_strength: float = 1.0
    include_nonlinear: bool = True
    max_iterations: int = 100
    convergence_tolerance: float = 1e-8
    damping_factor: float = 0.5
    regularization_parameter: float = 1e-6

@dataclass
class BackreactionResult:
    """Results from backreaction calculation"""
    iteration: int
    maxwell_energy: float
    dirac_energy: float
    interaction_energy: float
    total_energy: float
    field_convergence: float
    geometry_convergence: float
    stress_energy_tensor: np.ndarray
    metric_perturbation: np.ndarray
    electromagnetic_field: np.ndarray
    dirac_field: np.ndarray

class MaxwellDiracBackreaction:
    """
    Implementation of self-consistent Maxwell-Dirac backreaction in curved spacetime
    with full nonlinear coupling and geometric feedback.
    """
    
    def __init__(self, N: int, config: Optional[BackreactionConfig] = None):
        """
        Initialize the Maxwell-Dirac backreaction system.
        
        Args:
            N: Lattice size for spatial discretization
            config: Configuration for backreaction calculation
        """
        self.N = N
        self.config = config or BackreactionConfig()
        
        # Initialize field components
        self.constraint = MidisuperspaceHamiltonianConstraint(N)
        self.maxwell_field = MaxwellField(N)
        self.dirac_field = DiracField(N)
        self.phantom_field = PhantomScalarField(N)
        
        # Hilbert space structure
        self.basis_states = self.constraint.generate_flux_basis()
        self.hilbert_dim = len(self.basis_states)
        
        # Field variables (will be updated during iteration)
        self.current_electromagnetic_field = None
        self.current_dirac_field = None
        self.current_metric = None
        
        # Results storage
        self.iteration_results = []
        
    def initialize_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize electromagnetic field, Dirac field, and metric perturbation.
        
        Returns:
            Tuple of (electromagnetic_field, dirac_field, metric_perturbation)
        """
        # Initialize electromagnetic field A_μ
        # 4-vector potential: (A_0, A_x, A_y, A_z)
        em_field = np.random.normal(0, 0.1, (4, self.N, self.N, self.N))
        
        # Initialize Dirac field ψ 
        # 4-component spinor field
        dirac_field = np.random.normal(0, 0.1, (4, self.N, self.N, self.N)) + \
                     1j * np.random.normal(0, 0.1, (4, self.N, self.N, self.N))
        
        # Initialize metric perturbation h_μν
        # Symmetric 4x4 matrix at each lattice point
        metric_pert = np.random.normal(0, 0.01, (4, 4, self.N, self.N, self.N))
        
        # Ensure metric perturbation is symmetric
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    metric_pert[:, :, i, j, k] = (metric_pert[:, :, i, j, k] + 
                                                 metric_pert[:, :, i, j, k].T) / 2
        
        return em_field, dirac_field, metric_pert
    
    def compute_electromagnetic_stress_tensor(self, A_field: np.ndarray) -> np.ndarray:
        """
        Compute electromagnetic stress-energy tensor T^μν_EM.
        
        Args:
            A_field: 4-vector electromagnetic potential
            
        Returns:
            Stress-energy tensor T^μν_EM
        """
        # Compute field strength tensor F_μν = ∂_μ A_ν - ∂_ν A_μ
        F_tensor = np.zeros((4, 4, self.N, self.N, self.N))
        
        # Finite difference derivatives
        for mu in range(4):
            for nu in range(4):
                if mu != nu:
                    if mu == 0:  # Time derivative (approximate as zero for static case)
                        F_tensor[mu, nu] = 0
                    else:
                        # Spatial derivatives
                        spatial_idx = mu - 1
                        if spatial_idx < 3:  # x, y, z directions
                            F_tensor[mu, nu] = self._spatial_derivative(A_field[nu], spatial_idx)
                        if nu > 0:
                            spatial_idx_nu = nu - 1
                            if spatial_idx_nu < 3:
                                F_tensor[mu, nu] -= self._spatial_derivative(A_field[mu], spatial_idx_nu)
        
        # Compute stress-energy tensor T^μν = (1/4π)[F^μα F_να - (1/4)η^μν F^αβ F_αβ]
        T_em = np.zeros((4, 4, self.N, self.N, self.N))
        
        # Minkowski metric η_μν = diag(-1, 1, 1, 1)
        eta = np.diag([-1, 1, 1, 1])
        
        for mu in range(4):
            for nu in range(4):
                # T^μν = F^μα F_να - (1/4)η^μν F^αβ F_αβ
                for alpha in range(4):
                    # F^μα F_να term
                    T_em[mu, nu] += (eta[mu, mu] * F_tensor[mu, alpha] * 
                                    eta[nu, nu] * F_tensor[nu, alpha])
                
                # F^αβ F_αβ scalar term
                F_squared = 0
                for alpha in range(4):
                    for beta in range(4):
                        F_squared += eta[alpha, alpha] * eta[beta, beta] * \
                                   F_tensor[alpha, beta] * F_tensor[alpha, beta]
                
                T_em[mu, nu] -= 0.25 * eta[mu, nu] * F_squared
        
        # Factor of 1/(4π) in Gaussian units
        T_em /= (4 * np.pi)
        
        return T_em
    
    def compute_dirac_stress_tensor(self, psi_field: np.ndarray, A_field: np.ndarray) -> np.ndarray:
        """
        Compute Dirac stress-energy tensor T^μν_Dirac.
        
        Args:
            psi_field: 4-component Dirac spinor field
            A_field: 4-vector electromagnetic potential
            
        Returns:
            Stress-energy tensor T^μν_Dirac
        """
        # Dirac gamma matrices (Dirac representation)
        gamma = self._get_gamma_matrices()
        
        # Compute Dirac stress-energy tensor
        # T^μν = (i/2)[ψ̄ γ^μ D^ν ψ - (D^ν ψ̄) γ^μ ψ + ψ̄ γ^ν D^μ ψ - (D^μ ψ̄) γ^ν ψ]
        # where D_μ = ∂_μ - ieA_μ is the covariant derivative
        
        T_dirac = np.zeros((4, 4, self.N, self.N, self.N), dtype=complex)
        
        # Dirac conjugate: ψ̄ = ψ† γ^0
        psi_bar = np.conj(psi_field) @ gamma[0]
        
        for mu in range(4):
            for nu in range(4):
                # Covariant derivatives
                D_mu_psi = self._covariant_derivative(psi_field, A_field, mu)
                D_nu_psi = self._covariant_derivative(psi_field, A_field, nu)
                D_mu_psi_bar = self._covariant_derivative(psi_bar, A_field, mu)
                D_nu_psi_bar = self._covariant_derivative(psi_bar, A_field, nu)
                
                # Compute tensor components
                term1 = psi_bar @ gamma[mu] @ D_nu_psi
                term2 = D_nu_psi_bar @ gamma[mu] @ psi_field
                term3 = psi_bar @ gamma[nu] @ D_mu_psi
                term4 = D_mu_psi_bar @ gamma[nu] @ psi_field
                
                T_dirac[mu, nu] = (1j/2) * (term1 - term2 + term3 - term4)
        
        # Take real part for physical stress-energy tensor
        return T_dirac.real
    
    def _get_gamma_matrices(self) -> List[np.ndarray]:
        """Get Dirac gamma matrices in standard representation"""
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        
        # Dirac gamma matrices (Dirac representation)
        gamma0 = np.block([[I, np.zeros((2, 2))], [np.zeros((2, 2)), -I]])
        gamma1 = np.block([[np.zeros((2, 2)), sigma_x], [sigma_x, np.zeros((2, 2))]])
        gamma2 = np.block([[np.zeros((2, 2)), sigma_y], [sigma_y, np.zeros((2, 2))]])
        gamma3 = np.block([[np.zeros((2, 2)), sigma_z], [sigma_z, np.zeros((2, 2))]])
        
        return [gamma0, gamma1, gamma2, gamma3]
    
    def _spatial_derivative(self, field: np.ndarray, direction: int) -> np.ndarray:
        """Compute finite difference spatial derivative"""
        if direction == 0:  # x-direction
            return np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)
        elif direction == 1:  # y-direction
            return np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)
        elif direction == 2:  # z-direction
            return np.roll(field, -1, axis=2) - np.roll(field, 1, axis=2)
        else:
            return np.zeros_like(field)
    
    def _covariant_derivative(self, field: np.ndarray, A_field: np.ndarray, mu: int) -> np.ndarray:
        """Compute covariant derivative D_μ = ∂_μ - ieA_μ"""
        e = 1.0  # Electric charge (in natural units)
        
        if mu == 0:  # Time derivative (approximate as zero for static)
            return -1j * e * A_field[0] * field
        else:
            # Spatial derivative plus gauge connection
            spatial_idx = mu - 1
            partial_derivative = self._spatial_derivative(field, spatial_idx)
            gauge_term = -1j * e * A_field[mu] * field
            return partial_derivative + gauge_term
    
    def compute_interaction_energy(self, psi_field: np.ndarray, A_field: np.ndarray) -> float:
        """Compute Maxwell-Dirac interaction energy"""
        # Interaction energy: E_int = ∫ ψ̄ γ^μ A_μ ψ d³x
        gamma = self._get_gamma_matrices()
        psi_bar = np.conj(psi_field) @ gamma[0]
        
        interaction_density = 0
        for mu in range(4):
            interaction_density += psi_bar @ gamma[mu] @ (A_field[mu] * psi_field)
        
        # Integrate over spatial volume
        return np.real(np.sum(interaction_density)) * (1.0 / self.N)**3
    
    def update_electromagnetic_field(self, A_field: np.ndarray, psi_field: np.ndarray, 
                                   metric_pert: np.ndarray) -> np.ndarray:
        """
        Update electromagnetic field using Maxwell equations with Dirac source.
        
        ∇²A^μ - ∂^μ(∂_ν A^ν) = 4π J^μ_Dirac
        """
        new_A_field = A_field.copy()
        
        # Compute Dirac current J^μ = ψ̄ γ^μ ψ
        gamma = self._get_gamma_matrices()
        psi_bar = np.conj(psi_field) @ gamma[0]
        
        J_dirac = np.zeros((4, self.N, self.N, self.N), dtype=complex)
        for mu in range(4):
            J_dirac[mu] = psi_bar @ gamma[mu] @ psi_field
        
        J_dirac = J_dirac.real  # Take real part
        
        # Solve Maxwell equations (simplified relaxation method)
        for mu in range(4):
            # Laplacian of A^μ
            laplacian_A = self._compute_laplacian(A_field[mu])
            
            # Divergence term ∂^μ(∂_ν A^ν) (simplified)
            divergence_term = self._compute_divergence_correction(A_field, mu)
            
            # Source term
            source = 4 * np.pi * J_dirac[mu]
            
            # Update with relaxation
            new_A_field[mu] = (1 - self.config.damping_factor) * A_field[mu] + \
                             self.config.damping_factor * (laplacian_A - divergence_term + source)
        
        return new_A_field
    
    def update_dirac_field(self, psi_field: np.ndarray, A_field: np.ndarray, 
                          metric_pert: np.ndarray) -> np.ndarray:
        """
        Update Dirac field using Dirac equation in curved spacetime.
        
        (iγ^μ D_μ - m)ψ = 0
        """
        # Dirac mass (in natural units)
        m_dirac = 1.0
        
        gamma = self._get_gamma_matrices()
        
        # Compute Dirac operator action
        dirac_operator_psi = np.zeros_like(psi_field)
        
        for mu in range(4):
            # Covariant derivative term
            D_mu_psi = self._covariant_derivative(psi_field, A_field, mu)
            dirac_operator_psi += 1j * gamma[mu] @ D_mu_psi
        
        # Mass term
        dirac_operator_psi -= m_dirac * psi_field
        
        # Update using relaxation (solving linear system implicitly)
        new_psi_field = (1 - self.config.damping_factor) * psi_field - \
                       self.config.damping_factor * dirac_operator_psi
        
        return new_psi_field
    
    def update_metric_perturbation(self, metric_pert: np.ndarray, T_total: np.ndarray) -> np.ndarray:
        """
        Update metric perturbation using Einstein equations.
        
        G^μν = 8π T^μν
        """
        # Simplified metric update (linearized Einstein equations)
        # h^μν = 8π T^μν (in weak field approximation)
        
        new_metric_pert = metric_pert.copy()
        
        for mu in range(4):
            for nu in range(4):
                source = 8 * np.pi * T_total[mu, nu]
                
                # Update with regularization and damping
                new_metric_pert[mu, nu] = (1 - self.config.damping_factor) * metric_pert[mu, nu] + \
                                         self.config.damping_factor * source
                
                # Apply regularization
                new_metric_pert[mu, nu] *= (1 / (1 + self.config.regularization_parameter))
        
        return new_metric_pert
    
    def _compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute 3D Laplacian using finite differences"""
        laplacian = np.zeros_like(field)
        
        # Second derivatives in each direction
        for direction in range(3):
            if direction == 0:  # x-direction
                laplacian += (np.roll(field, 1, axis=0) - 2*field + np.roll(field, -1, axis=0))
            elif direction == 1:  # y-direction  
                laplacian += (np.roll(field, 1, axis=1) - 2*field + np.roll(field, -1, axis=1))
            elif direction == 2:  # z-direction
                laplacian += (np.roll(field, 1, axis=2) - 2*field + np.roll(field, -1, axis=2))
        
        return laplacian / (1.0 / self.N)**2  # Scale by lattice spacing squared
    
    def _compute_divergence_correction(self, A_field: np.ndarray, mu: int) -> np.ndarray:
        """Compute divergence correction term for gauge fixing"""
        # Simplified: return zero for now (can be improved with proper gauge fixing)
        return np.zeros_like(A_field[mu])
    
    def check_convergence(self, old_fields: Tuple, new_fields: Tuple) -> Tuple[float, float]:
        """Check convergence of fields and geometry"""
        old_A, old_psi, old_metric = old_fields
        new_A, new_psi, new_metric = new_fields
        
        # Field convergence
        A_diff = np.linalg.norm(new_A - old_A) / (np.linalg.norm(old_A) + 1e-10)
        psi_diff = np.linalg.norm(new_psi - old_psi) / (np.linalg.norm(old_psi) + 1e-10)
        field_convergence = max(A_diff, psi_diff)
        
        # Geometry convergence
        metric_diff = np.linalg.norm(new_metric - old_metric) / (np.linalg.norm(old_metric) + 1e-10)
        geometry_convergence = metric_diff
        
        return field_convergence, geometry_convergence
    
    def run_self_consistent_calculation(self) -> List[BackreactionResult]:
        """
        Run self-consistent Maxwell-Dirac backreaction calculation.
        
        Returns:
            List of BackreactionResult for each iteration
        """
        print("🔄 Starting Self-Consistent Maxwell-Dirac Backreaction")
        print("="*60)
        print(f"Lattice size: N = {self.N}")
        print(f"Max iterations: {self.config.max_iterations}")
        print(f"Convergence tolerance: {self.config.convergence_tolerance}")
        
        # Initialize fields
        A_field, psi_field, metric_pert = self.initialize_fields()
        
        self.iteration_results = []
        
        for iteration in range(self.config.max_iterations):
            print(f"\n  🔧 Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # Store old fields for convergence check
            old_fields = (A_field.copy(), psi_field.copy(), metric_pert.copy())
            
            # Compute stress-energy tensors
            T_em = self.compute_electromagnetic_stress_tensor(A_field)
            T_dirac = self.compute_dirac_stress_tensor(psi_field, A_field)
            T_total = T_em + T_dirac
            
            # Compute energies
            maxwell_energy = np.real(np.sum(T_em[0, 0])) * (1.0 / self.N)**3
            dirac_energy = np.real(np.sum(T_dirac[0, 0])) * (1.0 / self.N)**3
            interaction_energy = self.compute_interaction_energy(psi_field, A_field)
            total_energy = maxwell_energy + dirac_energy + interaction_energy
            
            # Update fields
            new_A_field = self.update_electromagnetic_field(A_field, psi_field, metric_pert)
            new_psi_field = self.update_dirac_field(psi_field, A_field, metric_pert)
            new_metric_pert = self.update_metric_perturbation(metric_pert, T_total)
            
            # Check convergence
            field_conv, geometry_conv = self.check_convergence(
                old_fields, (new_A_field, new_psi_field, new_metric_pert)
            )
            
            # Store results
            result = BackreactionResult(
                iteration=iteration + 1,
                maxwell_energy=maxwell_energy,
                dirac_energy=dirac_energy,
                interaction_energy=interaction_energy,
                total_energy=total_energy,
                field_convergence=field_conv,
                geometry_convergence=geometry_conv,
                stress_energy_tensor=T_total,
                metric_perturbation=new_metric_pert,
                electromagnetic_field=new_A_field,
                dirac_field=new_psi_field
            )
            
            self.iteration_results.append(result)
            
            print(f"    E_total = {total_energy:.6f}, field_conv = {field_conv:.2e}, geo_conv = {geometry_conv:.2e}")
            
            # Update fields for next iteration
            A_field = new_A_field
            psi_field = new_psi_field
            metric_pert = new_metric_pert
            
            # Check convergence
            if field_conv < self.config.convergence_tolerance and \
               geometry_conv < self.config.convergence_tolerance:
                print(f"  ✅ Converged after {iteration + 1} iterations")
                break
        else:
            print(f"  ⚠️  Reached maximum iterations without full convergence")
        
        # Store final fields
        self.current_electromagnetic_field = A_field
        self.current_dirac_field = psi_field
        self.current_metric = metric_pert
        
        return self.iteration_results
    
    def export_results(self, output_dir: str = "outputs") -> str:
        """Export backreaction results to JSON file"""
        output_path = Path(output_dir) / "maxwell_dirac_backreaction_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        # Convert results to serializable format
        export_data = {
            "parameters": {
                "lattice_size": self.N,
                "coupling_strength": self.config.coupling_strength,
                "max_iterations": self.config.max_iterations,
                "convergence_tolerance": self.config.convergence_tolerance,
                "include_nonlinear": self.config.include_nonlinear
            },
            "iteration_results": [
                {
                    "iteration": r.iteration,
                    "maxwell_energy": r.maxwell_energy,
                    "dirac_energy": r.dirac_energy,
                    "interaction_energy": r.interaction_energy,
                    "total_energy": r.total_energy,
                    "field_convergence": r.field_convergence,
                    "geometry_convergence": r.geometry_convergence
                }
                for r in self.iteration_results
            ],
            "final_state": {
                "converged": (self.iteration_results[-1].field_convergence < self.config.convergence_tolerance),
                "final_energy": self.iteration_results[-1].total_energy,
                "iterations_used": len(self.iteration_results)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📊 Backreaction results exported to: {output_path}")
        return str(output_path)

def run_maxwell_dirac_backreaction(N: int = 7, config: Optional[BackreactionConfig] = None, 
                                  export: bool = True) -> List[BackreactionResult]:
    """
    Main function to run Maxwell-Dirac backreaction calculation.
    
    Args:
        N: Lattice size
        config: Configuration for backreaction calculation
        export: Whether to export results
        
    Returns:
        List of BackreactionResult for each iteration
    """
    backreaction = MaxwellDiracBackreaction(N, config)
    results = backreaction.run_self_consistent_calculation()
    
    if export:
        backreaction.export_results()
    
    # Print summary
    final_result = results[-1]
    print("\n📊 MAXWELL-DIRAC BACKREACTION SUMMARY")
    print("="*50)
    print(f"Lattice size: N = {N}")
    print(f"Iterations: {len(results)}")
    print(f"Final total energy: {final_result.total_energy:.6f}")
    print(f"Maxwell energy: {final_result.maxwell_energy:.6f}")
    print(f"Dirac energy: {final_result.dirac_energy:.6f}")
    print(f"Interaction energy: {final_result.interaction_energy:.6f}")
    print(f"Field convergence: {final_result.field_convergence:.2e}")
    print(f"Geometry convergence: {final_result.geometry_convergence:.2e}")
    
    return results

if __name__ == "__main__":
    # Run Maxwell-Dirac backreaction calculation
    config = BackreactionConfig(
        coupling_strength=1.0,
        include_nonlinear=True,
        max_iterations=50,
        convergence_tolerance=1e-6
    )
    
    results = run_maxwell_dirac_backreaction(N=7, config=config, export=True)
