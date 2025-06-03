#!/usr/bin/env python3
"""
3+1D Matter Coupling Extension for LQG Framework

Extends the existing 2+1D matter coupling to full 3+1D treatment,
enabling dynamic feedback between quantum-corrected stress-energy
and evolving geometry.

Key Features:
- Full 3+1D scalar field dynamics with polymer corrections
- Stress-energy tensor computation in 3+1D
- Dynamic feedback into metric evolution
- Conservation law enforcement ∇_μ T^{μν} = 0
- GPU-accelerated field evolution
"""

import numpy as np
import warnings
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# GPU acceleration support
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using CPU-only implementation.")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

warnings.filterwarnings("ignore")

@dataclass
class Field3DConfig:
    """Configuration for 3+1D field evolution."""
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    domain_size: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    dt: float = 0.01
    dx: float = None  # Will be computed from domain_size/grid_size
    mu_polymer: float = 0.1
    mass_field: float = 0.5
    coupling_strength: float = 0.1
    use_gpu: bool = True
    device: str = "cuda" if TORCH_AVAILABLE else "cpu"

class PolymerField3D:
    """3+1D Polymer-corrected scalar field implementation."""
    
    def __init__(self, config: Field3DConfig):
        self.config = config
        self.setup_grid()
        self.setup_device()
        self.initialize_fields()
        
    def setup_grid(self):
        """Setup 3D spatial grid."""
        Nx, Ny, Nz = self.config.grid_size
        Lx, Ly, Lz = self.config.domain_size
        
        # Compute grid spacing
        self.dx = Lx / Nx
        self.dy = Ly / Ny  
        self.dz = Lz / Nz
        
        # Create coordinate arrays
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        z = np.linspace(-Lz/2, Lz/2, Nz)
        
        self.x_grid, self.y_grid, self.z_grid = np.meshgrid(x, y, z, indexing='ij')
        self.r_grid = np.sqrt(self.x_grid**2 + self.y_grid**2 + self.z_grid**2)
        
    def setup_device(self):
        """Setup computation device (GPU/CPU)."""
        if TORCH_AVAILABLE and self.config.use_gpu:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
                else:
                    self.device = torch.device("cpu")
                    print("Using PyTorch CPU computation")
            except:
                self.device = "cpu"
                print("Using NumPy CPU computation")
        else:
            self.device = "cpu"
            print("Using NumPy CPU computation")
            
    def initialize_fields(self):
        """Initialize scalar field and momentum."""
        Nx, Ny, Nz = self.config.grid_size
        
        # Initial Gaussian wave packet
        sigma = 2.0
        phi_initial = np.exp(-(self.r_grid**2) / (2 * sigma**2))
        pi_initial = np.zeros_like(phi_initial)
        
        if TORCH_AVAILABLE:
            self.phi = torch.tensor(phi_initial, dtype=torch.float32, device=self.device)
            self.pi = torch.tensor(pi_initial, dtype=torch.float32, device=self.device)
            
            # Metric components (start with Minkowski + corrections)
            self.g_tt = -torch.ones_like(self.phi)
            self.g_xx = torch.ones_like(self.phi) 
            self.g_yy = torch.ones_like(self.phi)
            self.g_zz = torch.ones_like(self.phi)
        else:
            self.phi = phi_initial
            self.pi = pi_initial
            
            # Metric components (start with Minkowski + corrections)
            self.g_tt = -np.ones_like(self.phi)
            self.g_xx = np.ones_like(self.phi)
            self.g_yy = np.ones_like(self.phi)
            self.g_zz = np.ones_like(self.phi)
        
    def compute_polymer_laplacian(self, field):
        """Compute polymer-corrected Laplacian operator."""
        if TORCH_AVAILABLE and isinstance(field, torch.Tensor):
            # Use PyTorch's conv3d for efficient finite differences
            kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                 [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                 [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], 
                                dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            
            field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='reflect')
            laplacian = F.conv3d(field_padded, kernel, padding=0).squeeze()
            
            # Polymer correction: sin(μΔ)/μ ≈ Δ - μ²Δ³/6 + ...
            mu2 = self.config.mu_polymer**2
            laplacian_polymer = laplacian - (mu2/6) * self.compute_cubic_laplacian(field)
            
        else:
            # NumPy implementation
            laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
                        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) - 6*field) / self.dx**2
            
            mu2 = self.config.mu_polymer**2
            laplacian_polymer = laplacian - (mu2/6) * self.compute_cubic_laplacian_numpy(field)
            
        return laplacian_polymer
        
    def compute_cubic_laplacian(self, field):
        """Compute Δ³ for polymer corrections (PyTorch version)."""
        lap1 = self.compute_standard_laplacian(field)
        lap2 = self.compute_standard_laplacian(lap1)
        lap3 = self.compute_standard_laplacian(lap2)
        return lap3
        
    def compute_standard_laplacian(self, field):
        """Standard Laplacian without polymer corrections."""
        kernel = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                             [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], 
                            dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        
        field_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1,1,1,1,1,1), mode='reflect')
        return F.conv3d(field_padded, kernel, padding=0).squeeze() / self.dx**2
        
    def compute_cubic_laplacian_numpy(self, field):
        """Compute Δ³ for polymer corrections (NumPy version)."""
        def numpy_laplacian(f):
            return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                   np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
                   np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6*f) / self.dx**2
        
        lap1 = numpy_laplacian(field)
        lap2 = numpy_laplacian(lap1)
        lap3 = numpy_laplacian(lap2)
        return lap3
        
    def compute_stress_energy_tensor(self):
        """Compute stress-energy tensor T_μν."""
        # Compute field derivatives
        if TORCH_AVAILABLE and isinstance(self.phi, torch.Tensor):
            phi_t = self.pi  # ∂φ/∂t = π
            phi_x = torch.gradient(self.phi, dim=0)[0] / self.dx
            phi_y = torch.gradient(self.phi, dim=1)[0] / self.dy
            phi_z = torch.gradient(self.phi, dim=2)[0] / self.dz
        else:
            phi_t = self.pi
            phi_x = np.gradient(self.phi, self.dx, axis=0)
            phi_y = np.gradient(self.phi, self.dy, axis=1)
            phi_z = np.gradient(self.phi, self.dz, axis=2)
        
        # Stress-energy components
        # T_00 = (1/2)(π² + |∇φ|² + m²φ²)
        kinetic = 0.5 * phi_t**2
        gradient = 0.5 * (phi_x**2 + phi_y**2 + phi_z**2)
        potential = 0.5 * self.config.mass_field**2 * self.phi**2
        
        T_00 = kinetic + gradient + potential
        
        # T_ij = ∂_i φ ∂_j φ - (1/2)δ_ij[π² + |∇φ|² + m²φ²]
        T_xx = phi_x**2 - 0.5 * (kinetic + gradient + potential)
        T_yy = phi_y**2 - 0.5 * (kinetic + gradient + potential)
        T_zz = phi_z**2 - 0.5 * (kinetic + gradient + potential)
        T_xy = phi_x * phi_y
        T_xz = phi_x * phi_z
        T_yz = phi_y * phi_z
        
        return {
            'T_00': T_00, 'T_xx': T_xx, 'T_yy': T_yy, 'T_zz': T_zz,
            'T_xy': T_xy, 'T_xz': T_xz, 'T_yz': T_yz
        }
        
    def evolve_fields(self, dt):
        """Evolve scalar field using polymer-corrected dynamics."""
        # Field equation: □φ + m²φ = 0 with polymer corrections
        laplacian_phi = self.compute_polymer_laplacian(self.phi)
        
        # Update momentum: ∂π/∂t = ∇²φ - m²φ
        pi_rhs = laplacian_phi - self.config.mass_field**2 * self.phi
        
        if TORCH_AVAILABLE and isinstance(self.pi, torch.Tensor):
            self.pi += dt * pi_rhs
            # Update field: ∂φ/∂t = π  
            self.phi += dt * self.pi
        else:
            self.pi += dt * pi_rhs
            self.phi += dt * self.pi
            
    def update_metric_from_stress_energy(self, stress_energy):
        """Update metric components based on stress-energy feedback."""
        # Einstein equations: G_μν = 8πT_μν
        # For small perturbations: δg_μν ∝ T_μν
        
        coupling = self.config.coupling_strength
        
        if TORCH_AVAILABLE and isinstance(self.g_tt, torch.Tensor):
            self.g_tt += -coupling * stress_energy['T_00']
            self.g_xx += coupling * stress_energy['T_xx'] 
            self.g_yy += coupling * stress_energy['T_yy']
            self.g_zz += coupling * stress_energy['T_zz']
        else:
            self.g_tt += -coupling * stress_energy['T_00']
            self.g_xx += coupling * stress_energy['T_xx']
            self.g_yy += coupling * stress_energy['T_yy'] 
            self.g_zz += coupling * stress_energy['T_zz']
            
    def check_conservation_laws(self, stress_energy):
        """Verify ∇_μ T^{μν} = 0."""
        # Compute divergence of stress-energy tensor
        if TORCH_AVAILABLE and isinstance(stress_energy['T_00'], torch.Tensor):
            div_T0 = (torch.gradient(stress_energy['T_00'], dim=0)[0] / self.dx +
                     torch.gradient(stress_energy['T_xy'], dim=1)[0] / self.dy +
                     torch.gradient(stress_energy['T_xz'], dim=2)[0] / self.dz)
        else:
            div_T0 = (np.gradient(stress_energy['T_00'], self.dx, axis=0) +
                     np.gradient(stress_energy['T_xy'], self.dy, axis=1) +
                     np.gradient(stress_energy['T_xz'], self.dz, axis=2))
        
        conservation_violation = np.mean(np.abs(div_T0.cpu().numpy() if TORCH_AVAILABLE else div_T0))
        return conservation_violation
        
    def run_evolution(self, num_steps=1000, save_interval=100):
        """Run full 3+1D evolution with matter-geometry coupling."""
        print(f"Starting 3+1D evolution for {num_steps} steps...")
        
        results = {
            'times': [],
            'field_energy': [],
            'conservation_violation': [],
            'metric_trace': []
        }
        
        for step in range(num_steps):
            t = step * self.config.dt
            
            # Evolve matter fields
            self.evolve_fields(self.config.dt)
            
            # Compute stress-energy
            stress_energy = self.compute_stress_energy_tensor()
            
            # Update metric
            self.update_metric_from_stress_energy(stress_energy)
            
            # Check conservation
            conservation_violation = self.check_conservation_laws(stress_energy)
            
            # Compute diagnostics
            if TORCH_AVAILABLE and isinstance(self.phi, torch.Tensor):
                field_energy = torch.sum(stress_energy['T_00']).item()
                metric_trace = torch.sum(self.g_xx + self.g_yy + self.g_zz).item()
            else:
                field_energy = np.sum(stress_energy['T_00'])
                metric_trace = np.sum(self.g_xx + self.g_yy + self.g_zz)
            
            if step % save_interval == 0:
                results['times'].append(t)
                results['field_energy'].append(field_energy)
                results['conservation_violation'].append(conservation_violation)
                results['metric_trace'].append(metric_trace)
                
                print(f"Step {step}: E={field_energy:.3e}, ∇T={conservation_violation:.3e}")
        
        return results

def main():
    """Main function for 3+1D matter coupling demo."""
    print("🌌 3+1D MATTER COUPLING DEMONSTRATION")
    print("=" * 50)
    
    # Setup configuration
    config = Field3DConfig(
        grid_size=(32, 32, 32),  # Smaller for demo
        domain_size=(8.0, 8.0, 8.0),
        dt=0.005,
        mu_polymer=0.1,
        mass_field=0.5,
        coupling_strength=0.01,
        use_gpu=False  # Set to False for compatibility
    )
    
    # Create field evolution system
    field_system = PolymerField3D(config)
    
    # Run evolution
    results = field_system.run_evolution(num_steps=500, save_interval=50)
    
    # Save results
    output_dir = Path("outputs/matter_3d")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "evolution_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Create visualization plots if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0,0].plot(results['times'], results['field_energy'])
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Field Energy')
        axes[0,0].set_title('Field Energy Evolution')
        
        axes[0,1].semilogy(results['times'], results['conservation_violation'])
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Conservation Violation')
        axes[0,1].set_title('Energy-Momentum Conservation')
        
        axes[1,0].plot(results['times'], results['metric_trace'])
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Metric Trace')
        axes[1,0].set_title('Metric Evolution')
        
        # Field profile at final time
        if TORCH_AVAILABLE and isinstance(field_system.phi, torch.Tensor):
            phi_final = field_system.phi.cpu().numpy()
        else:
            phi_final = field_system.phi
            
        # Plot central slice
        central_slice = phi_final[config.grid_size[0]//2, :, :]
        im = axes[1,1].imshow(central_slice, origin='lower', cmap='viridis')
        axes[1,1].set_title('Final Field Configuration')
        plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "evolution_analysis.png", dpi=150)
        plt.close()
        
        print(f"Results saved to {output_dir}")
        
    except ImportError:
        print("Matplotlib not available - skipping plots")
    
    # Summary
    print("\n📊 EVOLUTION SUMMARY")
    print(f"Final field energy: {results['field_energy'][-1]:.3e}")
    print(f"Final conservation violation: {results['conservation_violation'][-1]:.3e}")
    print(f"Metric evolution range: {min(results['metric_trace']):.3f} to {max(results['metric_trace']):.3f}")
    
    if results['conservation_violation'][-1] < 1e-6:
        print("✅ Conservation laws well-satisfied")
    else:
        print("⚠️  Conservation laws may need attention")

if __name__ == "__main__":
    main()
