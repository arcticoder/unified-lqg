# UNIFIED_QG: Unified Quantum Gravity Pipeline

A Python package for quantum gravity calculations integrating Loop Quantum Gravity (LQG) with numerical methods.

## Features

- **Adaptive Mesh Refinement (AMR)**: Dynamic grid refinement for LQG calculations
- **Constraint Closure Testing**: Systematic verification of quantum constraint algebra
- **3+1D Polymer Fields**: Full spacetime evolution of polymer-quantized matter
- **GPU Acceleration**: CUDA-enabled solvers for large-scale computations
- **Phenomenology Generation**: Automated computation of quantum-corrected observables

## Installation

```bash
pip install unified_qg
```

For GPU support:
```bash
pip install unified_qg[gpu]
```

For MPI parallel computing:
```bash
pip install unified_qg[mpi]
```

## Quick Start

```python
import unified_qg as uqg

# Create AMR configuration
amr_config = uqg.AMRConfig(initial_grid_size=(64, 64))
amr = uqg.AdaptiveMeshRefinement(amr_config)

# Initialize 3D polymer field
field_config = uqg.Field3DConfig(grid_size=(32, 32, 32))
polymer_field = uqg.PolymerField3D(field_config)

# Run phenomenology generation
results = uqg.generate_qc_phenomenology({"masses": [1.0], "spins": [0.5]})
```

## Development

To run tests:
```bash
python -m pytest unified_qg/tests/
```

## License

The Unlicense - see LICENSE file for details.
