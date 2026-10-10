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

    def nodes(self):
        """Node coordinates (X, Y), shape (nx, ny).  Nodal convention: nodes include both
        endpoints of x_range / y_range and dx = (x_max - x_min) / (nx - 1)."""
        x = self.x_range[0] + np.arange(self.nx) * self.dx
        y = self.y_range[0] + np.arange(self.ny) * self.dy
        return np.meshgrid(x, y, indexing='ij')

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
        
        # nodal grid: both endpoints are nodes, so there are nx-1 cells
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
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
        
        # Bounding box (in parent node indices) of every connected region, buffered.
        boxes = []
        for region_id in range(1, num_regions + 1):
            rows, cols = np.where(labeled_regions == region_id)
            if len(rows) == 0:
                continue
            buffer = 2
            row_min = max(0, rows.min() - buffer)
            row_max = min(parent.nx - 1, rows.max() + buffer)
            col_min = max(0, cols.min() - buffer)
            col_max = min(parent.ny - 1, cols.max() + buffer)
            if row_max <= row_min or col_max <= col_min:
                continue
            boxes.append([row_min, row_max, col_min, col_max])

        # Merge boxes whose interiors overlap, so sibling children never overlap each other
        # (overlapping siblings would be counted twice in every integral).
        merged = True
        while merged:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]:
                        boxes[i] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break

        for row_min, row_max, col_min, col_max in boxes:
            # Child nodes coincide with parent nodes (2x refinement): the child spans
            # (row_max - row_min) parent cells, i.e. 2*(...)+1 nodes.
            child_nx = 2 * (row_max - row_min) + 1
            child_ny = 2 * (col_max - col_min) + 1

            # Check grid size limit
            if child_nx > self.config.max_grid_size or child_ny > self.config.max_grid_size:
                continue
            
            # Compute child domain
            x_min = parent.x_range[0] + row_min * parent.dx
            x_max = parent.x_range[0] + row_max * parent.dx
            y_min = parent.y_range[0] + col_min * parent.dy
            y_max = parent.y_range[0] + col_max * parent.dy
            
            child_dx = parent.dx / 2
            child_dy = parent.dy / 2
            
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
        
        # Only leaf patches are candidates: re-refining a parent that already has
        # children would stack duplicate, overlapping children on it.
        for patch in [p for p in self.patches if not p.children]:
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
            
        self.patches = self.patches + new_patches
        
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
    
    # ------------------------------------------------------------------
    # Composite (non-overlapping) integration
    # ------------------------------------------------------------------
    def _uncovered_cells(self, patch: GridPatch, max_level: Optional[int] = None) -> np.ndarray:
        """Boolean mask over the (nx-1, ny-1) cells of `patch`: True where the cell is not
        covered by a child patch (of level <= max_level).  Child rectangles are aligned with
        parent nodes, so every cell is either fully covered or fully uncovered."""
        keep = np.ones((patch.nx - 1, patch.ny - 1), dtype=bool)
        cx = patch.x_range[0] + (np.arange(patch.nx - 1) + 0.5) * patch.dx
        cy = patch.y_range[0] + (np.arange(patch.ny - 1) + 0.5) * patch.dy
        for child in patch.children:
            if max_level is not None and child.level > max_level:
                continue
            in_x = (cx > child.x_range[0]) & (cx < child.x_range[1])
            in_y = (cy > child.y_range[0]) & (cy < child.y_range[1])
            keep &= ~(in_x[:, None] & in_y[None, :])
        return keep

    def composite_integral(self, density_function: Callable, max_level: Optional[int] = None) -> float:
        """Integral of a node-centred density over the domain, each point of the domain
        counted exactly once (finest available patch wins).  `density_function(patch)` must
        return an array of shape (nx, ny).  Cells are integrated with the 4-node average."""
        total = 0.0
        for patch in self.patches:
            if max_level is not None and patch.level > max_level:
                continue
            rho = np.asarray(density_function(patch))
            cell = 0.25 * (rho[:-1, :-1] + rho[1:, :-1] + rho[:-1, 1:] + rho[1:, 1:])
            keep = self._uncovered_cells(patch, max_level)
            total += float(np.sum(cell[keep]) * patch.dx * patch.dy)
        return total

    @staticmethod
    def mass_density(patch: GridPatch) -> np.ndarray:
        return patch.data

    @staticmethod
    def energy_density(patch: GridPatch) -> np.ndarray:
        gx, gy = np.gradient(patch.data, patch.dx, patch.dy)
        return 0.5 * (gx ** 2 + gy ** 2)

    def count_degrees_of_freedom(self) -> int:
        """Number of nodes that carry the solution: uncovered cells of every patch, counted once."""
        return int(sum(np.sum(self._uncovered_cells(p)) for p in self.patches))

    def check_conservation(self) -> Dict[str, float]:
        """Total mass and energy, with every region counted once.  (Summing over all patches,
        as an earlier version did, counts refined regions twice and makes the 'conserved'
        totals grow with every refinement.)"""
        if not self.config.conservation_check:
            return {}
        return {
            'total_mass': self.composite_integral(self.mass_density),
            'total_energy': self.composite_integral(self.energy_density),
        }

    def extrapolate_to_continuum(self, quantity_function: Callable = None) -> Dict[str, Any]:
        """Convergence of a composite integral with refinement level, and Richardson
        extrapolation.  `quantity_function(patch)` returns a node-centred density (default:
        the energy density).  Q_L is the integral using patches of level <= L only, so all
        Q_L refer to the same domain.  The observed order is reported only when the successive
        differences are monotone; otherwise it is None (not NaN)."""
        print("📈 Performing continuum extrapolation...")
        density = quantity_function or self.energy_density
        levels = sorted({p.level for p in self.patches})
        Q = [self.composite_integral(density, max_level=L) for L in levels]
        h = [max(p.dx for p in self.patches if p.level == L) for L in levels]
        result = {'levels': levels, 'h': h, 'quantity': Q, 'observed_order': None, 'extrapolated': None}
        if len(Q) >= 3:
            d1, d2 = Q[-2] - Q[-3], Q[-1] - Q[-2]
            if d1 != 0 and d2 != 0 and d1 * d2 > 0:
                order = float(np.log2(abs(d1) / abs(d2)))
                result['observed_order'] = order
                if order > 0:
                    result['extrapolated'] = float(Q[-1] + d2 / (2.0 ** order - 1.0))
        print(f"   Observed order: {result['observed_order']}   Extrapolated: {result['extrapolated']}")
        return result

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
    """AMR on a smooth two-Gaussian field with a known answer.

    Exact values over the (effectively whole-plane) domain:
        mass   = pi/10 + 0.5*pi/5          = 0.2*pi
        energy = 0.5*pi*(1 + 0.5**2)       = 0.625*pi   (cross term ~ e^-27, neglected)
    """
    print("🎯 AMR DEMONSTRATION: TWO-GAUSSIAN FIELD")
    print("=" * 50)

    def initial_solution(x, y):
        r1 = np.sqrt((x - 1)**2 + (y - 1)**2)
        r2 = np.sqrt((x + 1)**2 + (y + 1)**2)
        return np.exp(-10*r1**2) + 0.5*np.exp(-5*r2**2)

    def reevaluate(patch):
        """Fill a refined patch from the field itself (without this, refinement only
        interpolates the coarse data and cannot improve accuracy)."""
        return initial_solution(*patch.nodes())

    config = AMRConfig(
        initial_grid_size=(17, 17),
        max_refinement_levels=3,
        refinement_threshold=1e-4,
        refinement_fraction=0.2,
        error_estimator="gradient"
    )
    amr = AdaptiveMeshRefinement(config)
    amr.create_initial_grid((-3, 3), (-3, 3), initial_solution)

    exact = {'total_mass': 0.2 * np.pi, 'total_energy': 0.625 * np.pi}
    for i in range(config.max_refinement_levels):
        print(f"\n--- Adaptation Cycle {i+1} ---")
        amr.adapt_mesh(update_function=reevaluate)
        c = amr.check_conservation()
        print(f"   Total mass:   {c['total_mass']:.6f}  (exact {exact['total_mass']:.6f})")
        print(f"   Total energy: {c['total_energy']:.6f}  (exact {exact['total_energy']:.6f})")

    extrapolation = amr.extrapolate_to_continuum(amr.energy_density)

    output_dir = Path("outputs/amr_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    amr.visualize_grid_structure(output_dir / "grid_structure.png")

    results = {
        'refinement_history': amr.refinement_history,
        'final_conservation': amr.check_conservation(),
        'exact': exact,
        'extrapolation_results': extrapolation,
        'degrees_of_freedom': amr.count_degrees_of_freedom(),
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
