"""
Unified Quantum Gravity Pipeline Package

This package provides tools for:
- Adaptive Mesh Refinement (AMR) for LQG calculations
- Constraint-closure testing for midisuperspace quantization
- 3+1D polymer-quantized matter coupling
- GPU-accelerated Hamiltonian solvers
- Phenomenology generation for quantum-corrected observables
- Complete packaging and deployment utilities
"""

# Core AMR functionality
from .amr import AdaptiveMeshRefinement, AMRConfig, GridPatch

# 3+1D polymer field evolution
from .polymer_field import PolymerField3D, Field3DConfig

# Constraint closure testing
from .constraint_closure import run_constraint_closure_scan

# Phenomenology generation
from .phenomenology import (
    generate_qc_phenomenology,
    compute_qnm_frequency,
    compute_isco_shift,
    compute_horizon_area_spectrum
)

# GPU-accelerated solvers
from .gpu_solver import (
    solve_constraint_gpu,
    GPUConstraintSolver,
    solve_eigenvalue_problem_gpu,
    is_gpu_available,
    get_device_info
)

# Packaging utilities
from .packaging import (
    package_pipeline_as_library,
    create_package_structure,
    create_example_config
)

__version__ = "0.1.0"
__author__ = "Quantum Gravity Research Team"
__email__ = "research@quantumgravity.org"

__all__ = [
    # AMR components
    "AdaptiveMeshRefinement",
    "AMRConfig", 
    "GridPatch",
    
    # Polymer field components
    "PolymerField3D",
    "Field3DConfig",
    
    # Constraint closure
    "run_constraint_closure_scan",
    
    # Phenomenology
    "generate_qc_phenomenology",
    "compute_qnm_frequency",
    "compute_isco_shift", 
    "compute_horizon_area_spectrum",
    
    # GPU solver components
    "solve_constraint_gpu",
    "GPUConstraintSolver",
    "solve_eigenvalue_problem_gpu",
    "is_gpu_available",
    "get_device_info",
    
    # Packaging utilities
    "package_pipeline_as_library",
    "create_package_structure",
    "create_example_config"
]
