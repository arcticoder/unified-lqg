"""
Unified Quantum Gravity Pipeline Package

This package provides tools for:
- Adaptive Mesh Refinement (AMR) for LQG calculations
- Constraint-closure testing for midisuperspace quantization
- 3+1D polymer-quantized matter coupling
- GPU-accelerated Hamiltonian solvers
- Phenomenology generation for quantum-corrected observables
"""

from .amr import AdaptiveMeshRefinement, AMRConfig, GridPatch
from .polymer_field import PolymerField3D, Field3DConfig
from .constraint_closure import run_constraint_closure_scan
from .phenomenology import generate_qc_phenomenology
from .gpu_solver import solve_constraint_gpu, GPUConstraintSolver, is_gpu_available
from .packaging import package_pipeline_as_library

__version__ = "0.1.0"
__author__ = "QG Team"

__all__ = [
    "AdaptiveMeshRefinement",
    "AMRConfig", 
    "GridPatch",
    "PolymerField3D",
    "Field3DConfig",
    "run_constraint_closure_scan",
    "generate_qc_phenomenology",
    "solve_constraint_gpu",
    "GPUConstraintSolver",
    "is_gpu_available",
    "package_pipeline_as_library"
]
