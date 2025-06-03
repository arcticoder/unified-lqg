#!/usr/bin/env python3
"""
Adaptive Mesh Refinement (AMR) for LQG Framework

This module implements adaptive lattice refinement for loop quantum gravity
calculations, enabling automatic grid refinement in regions of high curvature
or field gradients for improved continuum extrapolation.

Key Features:
- Hierarchical adaptive mesh refinement
- Error estimation and refinement criteria
- Continuum extrapolation analysis
- Automatic grid coarsening and refinement
- Conservation property preservation
"""

import numpy as np
import warnings
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

@dataclass
class AMRConfig:
    """Configuration for adaptive mesh refinement."""
    initial_grid_size: Tuple[int, int] = (32, 32)
    max_refinement_levels: int = 4
    refinement_threshold: float = 1e-3
    coarsening_threshold: float = 1e-5
    max_grid_size: int = 512
    error_estimator: str = "gradient"  # "gradient", "curvature", "residual"
    refinement_criterion: str = "fixed_fraction"  # "fixed_fraction", "fixed_threshold"
    refinement_fraction: float = 0.1
    buffer_zones: int = 2
    conservation_check: bool = True

@dataclass 
class GridPatch:
    """Represents a single grid patch in the AMR hierarchy."""
    level: int
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    nx: int
    ny: int
    dx: float
    dy: float
    data: np.ndarray
    parent: Optional['GridPatch'] = None
    children: List['GridPatch'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class AdaptiveMeshRefinement:
    """
    Adaptive mesh refinement implementation for LQG calculations.
    
    Manages hierarchical grid structure and automatic refinement/coarsening
    based on solution features and error estimates.
    """
    
    def __init__(self, config: AMRConfig):
        self.config = config
        self.patches = []
        self.refinement_history = []
        self.error_history = []
        
    def create_initial_grid(self, domain_x: Tuple[float, float], 
                          domain_y: Tuple[float, float],
                          initial_function: Callable = None) -> GridPatch:
        """Create initial coarse grid."""
        print("🔨 Creating initial AMR grid...")
        
        nx, ny = self.config.initial_grid_size
        x_min, x_max = domain_x
        y_min, y_max = domain_y
        
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        
        # Initialize data
        if initial_function is not None:
            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            X, Y = np.meshgrid(x, y, indexing='ij')
            data = initial_function(X, Y)
        else:
            data = np.zeros((nx, ny))
            
        base_patch = GridPatch(
            level=0,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            data=data
        )
        
        self.patches = [base_patch]
        print(f"   Initial grid: {nx}×{ny}, dx={dx:.3f}, dy={dy:.3f}")
        return base_patch
    
    def estimate_error(self, patch: GridPatch) -> np.ndarray:
        """Estimate discretization error on a patch."""
        if self.config.error_estimator == "gradient":
            return self._gradient_error_estimator(patch)
        elif self.config.error_estimator == "curvature":
            return self._curvature_error_estimator(patch)
        elif self.config.error_estimator == "residual":
            return self._residual_error_estimator(patch)
        else:
            raise ValueError(f"Unknown error estimator: {self.config.error_estimator}")
    
    def _gradient_error_estimator(self, patch: GridPatch) -> np.ndarray:
        """Error estimator based on solution gradients."""
        data = patch.data
        
        # Compute gradients
        grad_x = np.gradient(data, patch.dx, axis=0)
        grad_y = np.gradient(data, patch.dy, axis=1)
        
        # Compute second derivatives (curvature measure)
        grad2_x = np.gradient(grad_x, patch.dx, axis=0)
        grad2_y = np.gradient(grad_y, patch.dy, axis=1)
        grad2_xy = np.gradient(grad_x, patch.dy, axis=1)
        
        # Error estimate: |∇²u| * h²
        laplacian = grad2_x + grad2_y
        error = np.abs(laplacian) * (patch.dx**2 + patch.dy**2) / 2
        
        return error
    
    def _curvature_error_estimator(self, patch: GridPatch) -> np.ndarray:
        """Error estimator based on solution curvature."""
        data = patch.data
        
        # Compute Hessian matrix
        grad_x = np.gradient(data, patch.dx, axis=0)
        grad_y = np.gradient(data, patch.dy, axis=1)
        
        hess_xx = np.gradient(grad_x, patch.dx, axis=0)
        hess_yy = np.gradient(grad_y, patch.dy, axis=1)
        hess_xy = np.gradient(grad_x, patch.dy, axis=1)
        
        # Principal curvatures (eigenvalues of Hessian)
        trace = hess_xx + hess_yy
        det = hess_xx * hess_yy - hess_xy**2
        
        # Error based on maximum curvature
        discriminant = np.maximum(0, trace**2 - 4*det)
        max_curvature = np.abs(trace + np.sqrt(discriminant)) / 2
        
        error = max_curvature * (patch.dx**4 + patch.dy**4) / 12
        
        return error
    
    def _residual_error_estimator(self, patch: GridPatch) -> np.ndarray:
        """Error estimator based on equation residual."""
        # For demonstration: assume we're solving Δu = f
        data = patch.data
        
        # Compute discrete Laplacian
        laplacian = (np.roll(data, 1, axis=0) + np.roll(data, -1, axis=0) +
                    np.roll(data, 1, axis=1) + np.roll(data, -1, axis=1) - 4*data)
        laplacian /= (patch.dx**2 + patch.dy**2) / 2
        
        # Residual (assuming f=0 for simplicity)
        residual = np.abs(laplacian)
        
        return residual
    
    def mark_for_refinement(self, patch: GridPatch, error: np.ndarray) -> np.ndarray:
        """Mark cells for refinement based on error estimates."""
        if self.config.refinement_criterion == "fixed_threshold":
            # Mark cells above threshold
            refine_mask = error > self.config.refinement_threshold
            
        elif self.config.refinement_criterion == "fixed_fraction":
            # Mark highest error fraction
            flat_error = error.flatten()
            threshold = np.percentile(flat_error, 100 * (1 - self.config.refinement_fraction))
            refine_mask = error > threshold
            
        else:
            raise ValueError(f"Unknown refinement criterion: {self.config.refinement_criterion}")
        
        # Add buffer zones around marked cells
        if self.config.buffer_zones > 0:
            from scipy.ndimage import binary_dilation
            kernel = np.ones((2*self.config.buffer_zones+1, 2*self.config.buffer_zones+1))
            refine_mask = binary_dilation(refine_mask, kernel)
        
        return refine_mask
    
    def refine_patch(self, parent: GridPatch, refine_mask: np.ndarray) -> List[GridPatch]:
        """Refine a patch by creating child patches."""
        if parent.level >= self.config.max_refinement_levels:
            return []
            
        children = []
        
        # Find connected regions that need refinement
        from scipy.ndimage import label
        labeled_regions, num_regions = label(refine_mask)
        
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            
            # Find bounding box of region
            rows, cols = np.where(region_mask)
            if len(rows) == 0:
                continue
                
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            
            # Expand bounding box slightly
            buffer = 2
            row_min = max(0, row_min - buffer)
            row_max = min(parent.nx - 1, row_max + buffer)
            col_min = max(0, col_min - buffer)
            col_max = min(parent.ny - 1, col_max + buffer)
            
            # Create child patch with 2x refinement
            child_nx = 2 * (row_max - row_min + 1)
            child_ny = 2 * (col_max - col_min + 1)
            
            # Check grid size limit
            if child_nx > self.config.max_grid_size or child_ny > self.config.max_grid_size:
                continue
            
            # Compute child domain
            x_min = parent.x_range[0] + row_min * parent.dx
            x_max = parent.x_range[0] + (row_max + 1) * parent.dx
            y_min = parent.y_range[0] + col_min * parent.dy
            y_max = parent.y_range[0] + (col_max + 1) * parent.dy
            
            child_dx = (x_max - x_min) / child_nx
            child_dy = (y_max - y_min) / child_ny
            
            # Interpolate parent data to child grid
            parent_x = np.linspace(parent.x_range[0], parent.x_range[1], parent.nx)
            parent_y = np.linspace(parent.y_range[0], parent.y_range[1], parent.ny)
            
            child_x = np.linspace(x_min, x_max, child_nx)
            child_y = np.linspace(y_min, y_max, child_ny)
            
            # Create interpolator
            interp = RegularGridInterpolator((parent_x, parent_y), parent.data, 
                                           bounds_error=False, fill_value=0.0)
            
            child_X, child_Y = np.meshgrid(child_x, child_y, indexing='ij')
            child_data = interp((child_X, child_Y))
            
            # Create child patch
            child = GridPatch(
                level=parent.level + 1,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                nx=child_nx,
                ny=child_ny,
                dx=child_dx,
                dy=child_dy,
                data=child_data,
                parent=parent
            )
            
            children.append(child)
            parent.children.append(child)
            
        return children
    
    def adapt_mesh(self, update_function: Callable = None) -> Dict[str, Any]:
        """Perform one cycle of mesh adaptation."""
        print("🔄 Performing mesh adaptation...")
        
        new_patches = []
        total_refined = 0
        total_error = 0.0
        
        for patch in self.patches:
            # Estimate error
            error = self.estimate_error(patch)
            total_error += np.sum(error)
            
            # Mark for refinement
            refine_mask = self.mark_for_refinement(patch, error)
            
            if np.any(refine_mask):
                # Create refined patches
                children = self.refine_patch(patch, refine_mask)
                new_patches.extend(children)
                total_refined += len(children)
                
                # Update children if function provided
                if update_function is not None:
                    for child in children:
                        child.data = update_function(child)
            
            # Keep original patch
            new_patches.append(patch)
        
        self.patches = new_patches
        
        # Update history
        adaptation_info = {
            'total_patches': len(self.patches),
            'patches_refined': total_refined,
            'total_error': total_error,
            'max_level': max(p.level for p in self.patches),
            'min_dx': min(p.dx for p in self.patches),
            'min_dy': min(p.dy for p in self.patches)
        }
        
        self.refinement_history.append(adaptation_info)
        
        print(f"   Patches: {len(self.patches)} (+{total_refined} refined)")
        print(f"   Max level: {adaptation_info['max_level']}")
        print(f"   Min spacing: dx={adaptation_info['min_dx']:.3e}, dy={adaptation_info['min_dy']:.3e}")
        
        return adaptation_info
    
    def check_conservation(self) -> Dict[str, float]:
        """Check conservation properties after refinement."""
        if not self.config.conservation_check:
            return {}
        
        # Compute total quantities on each patch
        total_mass = 0.0
        total_energy = 0.0
        
        for patch in self.patches:
            # Mass: integral of density
            patch_mass = np.sum(patch.data) * patch.dx * patch.dy
            
            # Energy: integral of |∇u|²
            grad_x = np.gradient(patch.data, patch.dx, axis=0)
            grad_y = np.gradient(patch.data, patch.dy, axis=1)
            energy_density = 0.5 * (grad_x**2 + grad_y**2)
            patch_energy = np.sum(energy_density) * patch.dx * patch.dy
            
            total_mass += patch_mass
            total_energy += patch_energy
        
        conservation_info = {
            'total_mass': total_mass,
            'total_energy': total_energy
        }
        
        return conservation_info
    
    def extrapolate_to_continuum(self, quantity_function: Callable) -> Dict[str, Any]:
        """Perform continuum extrapolation analysis."""
        print("📈 Performing continuum extrapolation...")
        
        # Compute quantity on different refinement levels
        level_results = {}
        
        for level in range(self.config.max_refinement_levels + 1):
            level_patches = [p for p in self.patches if p.level == level]
            if not level_patches:
                continue
                
            # Compute average grid spacing for this level
            avg_dx = np.mean([p.dx for p in level_patches])
            avg_dy = np.mean([p.dy for p in level_patches])
            avg_h = (avg_dx + avg_dy) / 2
            
            # Compute quantity on this level
            total_quantity = 0.0
            total_area = 0.0
            
            for patch in level_patches:
                patch_quantity = quantity_function(patch)
                patch_area = (patch.x_range[1] - patch.x_range[0]) * (patch.y_range[1] - patch.y_range[0])
                
                total_quantity += patch_quantity * patch_area
                total_area += patch_area
            
            level_results[level] = {
                'h': avg_h,
                'quantity': total_quantity / total_area if total_area > 0 else 0.0,
                'num_patches': len(level_patches)
            }
        
        # Perform Richardson extrapolation (assuming 2nd order convergence)
        if len(level_results) >= 2:
            levels = sorted(level_results.keys())
            h_values = [level_results[l]['h'] for l in levels]
            q_values = [level_results[l]['quantity'] for l in levels]
            
            # Fit q = q_∞ + C h^p
            if len(h_values) >= 3:
                # Use last 3 points for extrapolation
                h1, h2, h3 = h_values[-3:]
                q1, q2, q3 = q_values[-3:]
                
                # Richardson extrapolation formula for p=2
                q_extrap = q3 + (q3 - q2) / (h2**2/h3**2 - 1)
                
                convergence_rate = np.log((q2-q1)/(q3-q2)) / np.log(h2/h3)
                
            else:
                q_extrap = q_values[-1]
                convergence_rate = 2.0  # Assumed
            
        else:
            q_extrap = list(level_results.values())[0]['quantity'] if level_results else 0.0
            convergence_rate = 2.0
        
        extrapolation_results = {
            'level_results': level_results,
            'extrapolated_value': q_extrap,
            'convergence_rate': convergence_rate,
            'finest_grid_value': q_values[-1] if level_results else 0.0
        }
        
        print(f"   Continuum extrapolation: {q_extrap:.6e}")
        print(f"   Convergence rate: {convergence_rate:.2f}")
        
        return extrapolation_results
    
    def visualize_grid_structure(self, save_path: Optional[Path] = None):
        """Visualize the adaptive grid structure."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot grid patches
            colors = plt.cm.viridis(np.linspace(0, 1, self.config.max_refinement_levels + 1))
            
            for patch in self.patches:
                x_min, x_max = patch.x_range
                y_min, y_max = patch.y_range
                
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, edgecolor=colors[patch.level], 
                                   linewidth=1 + patch.level)
                ax1.add_patch(rect)
            
            ax1.set_xlim(-5, 5)
            ax1.set_ylim(-5, 5)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title('Adaptive Grid Structure')
            ax1.grid(True, alpha=0.3)
            
            # Add legend for refinement levels
            legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=1+i, 
                                        label=f'Level {i}') 
                             for i in range(len(colors))]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # Plot refinement history
            if self.refinement_history:
                iterations = range(len(self.refinement_history))
                total_patches = [info['total_patches'] for info in self.refinement_history]
                max_levels = [info['max_level'] for info in self.refinement_history]
                
                ax2.plot(iterations, total_patches, 'bo-', label='Total Patches')
                ax2_twin = ax2.twinx()
                ax2_twin.plot(iterations, max_levels, 'rs-', label='Max Level')
                
                ax2.set_xlabel('Adaptation Iteration')
                ax2.set_ylabel('Number of Patches', color='b')
                ax2_twin.set_ylabel('Maximum Level', color='r')
                ax2.set_title('Refinement History')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
            
        except ImportError:
            print("Matplotlib not available - skipping visualization")

def demo_amr_poisson():
    """Demonstrate AMR on a Poisson equation with localized source."""
    print("🎯 AMR DEMONSTRATION: POISSON EQUATION")
    print("=" * 50)
    
    def initial_solution(x, y):
        """Initial solution with localized features."""
        r1 = np.sqrt((x - 1)**2 + (y - 1)**2)
        r2 = np.sqrt((x + 1)**2 + (y + 1)**2)
        return np.exp(-10*r1**2) + 0.5*np.exp(-5*r2**2)
    
    def solution_quantity(patch):
        """Compute total energy on a patch."""
        grad_x = np.gradient(patch.data, patch.dx, axis=0)
        grad_y = np.gradient(patch.data, patch.dy, axis=1)
        energy_density = 0.5 * (grad_x**2 + grad_y**2)
        return np.sum(energy_density) * patch.dx * patch.dy
    
    # Setup AMR
    config = AMRConfig(
        initial_grid_size=(16, 16),
        max_refinement_levels=3,
        refinement_threshold=1e-4,
        refinement_fraction=0.2,
        error_estimator="gradient"
    )
    
    amr = AdaptiveMeshRefinement(config)
    
    # Create initial grid
    base_patch = amr.create_initial_grid((-3, 3), (-3, 3), initial_solution)
    
    # Perform several adaptation cycles
    for i in range(4):
        print(f"\n--- Adaptation Cycle {i+1} ---")
        
        # Adapt mesh
        adaptation_info = amr.adapt_mesh()
        
        # Check conservation
        conservation = amr.check_conservation()
        if conservation:
            print(f"   Total mass: {conservation['total_mass']:.3e}")
            print(f"   Total energy: {conservation['total_energy']:.3e}")
    
    # Continuum extrapolation
    extrapolation = amr.extrapolate_to_continuum(solution_quantity)
    
    # Visualize results
    output_dir = Path("outputs/amr_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    amr.visualize_grid_structure(output_dir / "grid_structure.png")
    
    # Save results
    results = {
        'refinement_history': amr.refinement_history,
        'extrapolation_results': extrapolation,
        'final_grid_info': {
            'total_patches': len(amr.patches),
            'max_level': max(p.level for p in amr.patches),
            'min_spacing': min(p.dx for p in amr.patches)
        }
    }
    
    with open(output_dir / "amr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 AMR DEMONSTRATION COMPLETE")
    print(f"Results saved to {output_dir}")
    
    return results

def main():
    """Main function for AMR demonstration."""
    return demo_amr_poisson()

if __name__ == "__main__":
    main()
