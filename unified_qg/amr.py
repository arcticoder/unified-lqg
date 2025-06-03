"""
Adaptive Mesh Refinement (AMR) module for LQG calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AMRConfig:
    """Configuration for Adaptive Mesh Refinement."""
    initial_grid_size: Tuple[int, int] = (32, 32)
    max_refinement_levels: int = 3
    refinement_threshold: float = 1e-3
    coarsening_threshold: float = 1e-5
    max_grid_size: int = 256
    error_estimator: str = "curvature"  # "gradient", "curvature", "residual"
    refinement_criterion: str = "fixed_fraction"  # "fixed_threshold", "fixed_fraction"
    refinement_fraction: float = 0.1
    buffer_zones: int = 2


@dataclass
class GridPatch:
    """Represents a grid patch in the AMR hierarchy."""
    level: int
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    grid_size: Tuple[int, int]
    data: np.ndarray
    error_map: Optional[np.ndarray] = None
    children: List['GridPatch'] = None
    parent: Optional['GridPatch'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


class AdaptiveMeshRefinement:
    """Adaptive Mesh Refinement framework for LQG calculations."""

    def __init__(self, config: AMRConfig):
        self.config = config
        self.patches = []
        self.error_history = []

    def create_initial_grid(self,
                            domain_x: Tuple[float, float],
                            domain_y: Tuple[float, float],
                            initial_function: callable) -> GridPatch:
        """Create the initial coarse grid."""
        x_min, x_max = domain_x
        y_min, y_max = domain_y
        nx, ny = self.config.initial_grid_size

        # Create coordinate arrays
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Initialize data with the provided function
        data = initial_function(X, Y)

        # Create root patch
        root_patch = GridPatch(
            level=0,
            bounds=(x_min, x_max, y_min, y_max),
            grid_size=(nx, ny),
            data=data
        )

        self.patches = [root_patch]
        return root_patch

    def compute_error_estimator(self, patch: GridPatch) -> np.ndarray:
        """Compute error estimator for the given patch."""
        data = patch.data
        nx, ny = patch.grid_size
        x_min, x_max, y_min, y_max = patch.bounds

        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)

        if self.config.error_estimator == "gradient":
            # Gradient-based estimator
            grad_x = np.gradient(data, dx, axis=0)
            grad_y = np.gradient(data, dy, axis=1)
            error_map = np.sqrt(grad_x**2 + grad_y**2)

        elif self.config.error_estimator == "curvature":
            # Curvature-based estimator (Laplacian)
            laplacian = np.zeros_like(data)
            laplacian[1:-1, 1:-1] = (
                (data[2:, 1:-1] - 2*data[1:-1, 1:-1] + data[:-2, 1:-1]) / dx**2 +
                (data[1:-1, 2:] - 2*data[1:-1, 1:-1] + data[1:-1, :-2]) / dy**2
            )
            error_map = np.abs(laplacian)

        elif self.config.error_estimator == "residual":
            # Residual-based estimator (simplified)
            residual = np.zeros_like(data)
            residual[1:-1, 1:-1] = np.abs(
                data[2:, 1:-1] + data[:-2, 1:-1]
                + data[1:-1, 2:] + data[1:-1, :-2]
                - 4 * data[1:-1, 1:-1]
            )
            error_map = residual

        else:
            raise ValueError(f"Unknown error estimator: {self.config.error_estimator}")

        patch.error_map = error_map
        return error_map

    def refine_or_coarsen(self, patch: GridPatch):
        """Refine or coarsen patches based on error criteria."""
        if patch.error_map is None:
            self.compute_error_estimator(patch)

        error_map = patch.error_map

        if self.config.refinement_criterion == "fixed_threshold":
            refine_mask = error_map > self.config.refinement_threshold
            coarsen_mask = error_map < self.config.coarsening_threshold
        else:  # fixed_fraction
            error_flat = error_map.flatten()
            error_sorted = np.sort(error_flat)[::-1]  # Descending order
            n_refine = int(self.config.refinement_fraction * len(error_flat))
            if n_refine > 0:
                refine_threshold = error_sorted[n_refine - 1]
                refine_mask = error_map >= refine_threshold
                coarsen_mask = error_map < self.config.coarsening_threshold
            else:
                refine_mask = np.zeros_like(error_map, dtype=bool)
                coarsen_mask = error_map < self.config.coarsening_threshold

        # Perform refinement (simplified placeholder)
        if np.any(refine_mask) and patch.level < self.config.max_refinement_levels:
            print(f"Refining patch at level {patch.level}")
            # In a full implementation, create child patches here

        # Process children recursively
        for child in patch.children:
            self.refine_or_coarsen(child)

    def visualize_grid_hierarchy(self, root_patch: GridPatch):
        """Create a visualization of the grid hierarchy."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot data
        x_min, x_max, y_min, y_max = root_patch.bounds
        im1 = ax1.imshow(
            root_patch.data.T,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            aspect='auto'
        )
        ax1.set_title("Initial Data")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1)

        # Plot error map
        if root_patch.error_map is not None:
            im2 = ax2.imshow(
                root_patch.error_map.T,
                extent=[x_min, x_max, y_min, y_max],
                origin='lower',
                aspect='auto'
            )
            ax2.set_title("Error Map")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        return fig, (ax1, ax2)
