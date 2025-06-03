"""
Packaging Utilities Module

This module contains utilities for packaging the quantum gravity pipeline
as a Python library and managing the package structure.
"""

import os
import json
from typing import Dict, Any


def create_package_structure(pkg_name: str = "unified_qg") -> None:
    """
    Create the basic package directory structure.
    
    Args:
        pkg_name: Name of the package to create
    """
    # Main package directory
    os.makedirs(pkg_name, exist_ok=True)
    
    # Tests directory
    os.makedirs(f"{pkg_name}/tests", exist_ok=True)
    
    # Data directory for examples
    os.makedirs(f"{pkg_name}/data", exist_ok=True)
    
    print(f"Created package structure for {pkg_name}")


def create_setup_py(pkg_name: str = "unified_qg") -> None:
    """
    Create setup.py file for the package.
    
    Args:
        pkg_name: Name of the package
    """
    setup_content = f'''"""Setup script for {pkg_name} package."""

from setuptools import setup, find_packages

# Read requirements from requirements.txt if it exists
try:
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ]

# Optional GPU dependencies
gpu_requirements = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
]

# Optional MPI dependencies  
mpi_requirements = [
    "mpi4py>=3.0.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]

setup(
    name="{pkg_name}",
    version="0.1.0",
    author="Quantum Gravity Research Team",
    author_email="research@quantumgravity.org",
    description="Unified Quantum Gravity Pipeline for LQG calculations",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/quantumgravity/unified_qg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={{
        "gpu": gpu_requirements,
        "mpi": mpi_requirements,
        "dev": dev_requirements,
        "all": gpu_requirements + mpi_requirements + dev_requirements,
    }},
    include_package_data=True,
    package_data={{
        "{pkg_name}": ["data/*.json", "data/*.txt"],
    }},
    entry_points={{
        "console_scripts": [
            "{pkg_name}-demo=unified_qg.demo:main",
            "{pkg_name}-test=unified_qg.tests.run_tests:main",
        ],
    }},
    project_urls={{
        "Bug Reports": "https://github.com/quantumgravity/unified_qg/issues",
        "Source": "https://github.com/quantumgravity/unified_qg",
        "Documentation": "https://unified-qg.readthedocs.io/",
    }},
)
'''
    
    with open("setup.py", "w") as f:
        f.write(setup_content)
    
    print("Created setup.py file")


def create_requirements_txt() -> None:
    """Create requirements.txt file with core dependencies."""
    requirements = [
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "matplotlib>=3.3.0",
        "# Optional GPU support",
        "# torch>=1.9.0",
        "# torchvision>=0.10.0",
        "# Optional MPI support",
        "# mpi4py>=3.0.0",
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\\n".join(requirements))
    
    print("Created requirements.txt file")


def create_manifest_in(pkg_name: str = "unified_qg") -> None:
    """Create MANIFEST.in file for including additional files in the package."""
    manifest_content = f'''include README.md
include LICENSE
include requirements.txt
recursive-include {pkg_name}/data *
recursive-include {pkg_name}/tests *.py
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
'''
    
    with open("MANIFEST.in", "w") as f:
        f.write(manifest_content)
    
    print("Created MANIFEST.in file")


def create_license_file() -> None:
    """Create MIT license file."""
    license_content = '''MIT License

Copyright (c) 2024 Quantum Gravity Research Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
    
    with open("LICENSE", "w") as f:
        f.write(license_content)
    
    print("Created LICENSE file")


def create_readme(pkg_name: str = "unified_qg") -> None:
    """Create comprehensive README.md file."""
    readme_content = f'''# {pkg_name.upper()}: Unified Quantum Gravity Pipeline

A comprehensive Python package for quantum gravity calculations integrating Loop Quantum Gravity (LQG) with advanced numerical methods.

## Features

- **Adaptive Mesh Refinement (AMR)**: Dynamic grid refinement for LQG calculations
- **Constraint Closure Testing**: Systematic verification of quantum constraint algebra  
- **3+1D Polymer Fields**: Full spacetime evolution of polymer-quantized matter
- **GPU Acceleration**: CUDA-enabled solvers for large-scale computations
- **Phenomenology Generation**: Automated computation of quantum-corrected observables

## Installation

### Basic Installation

```bash
pip install {pkg_name}
```

### With GPU Support

```bash
pip install {pkg_name}[gpu]
```

### With MPI Parallel Computing

```bash
pip install {pkg_name}[mpi]
```

### Development Installation

```bash
git clone https://github.com/quantumgravity/unified_qg.git
cd unified_qg
pip install -e .[dev]
```

## Quick Start

```python
import {pkg_name} as uqg

# Create AMR configuration
amr_config = uqg.AMRConfig(initial_grid_size=(64, 64))
amr = uqg.AdaptiveMeshRefinement(amr_config)

# Initialize 3D polymer field
field_config = uqg.Field3DConfig(grid_size=(32, 32, 32))
polymer_field = uqg.PolymerField3D(field_config)

# Run phenomenology generation
results = uqg.generate_qc_phenomenology({{"masses": [1.0], "spins": [0.5]}})
```

## Modules

### Core Components

- `amr`: Adaptive Mesh Refinement for dynamic grid optimization
- `constraint_closure`: Quantum constraint algebra verification
- `polymer_field`: 3+1D polymer-quantized field evolution
- `gpu_solver`: GPU-accelerated Hamiltonian solvers
- `phenomenology`: Quantum-corrected observable generation

### Advanced Features

- Multi-scale adaptive grids with automatic refinement
- GPU-accelerated constraint solving using PyTorch
- Parallel MPI computations for large-scale simulations
- Comprehensive phenomenological predictions

## Examples

### Basic AMR Example

```python
import {pkg_name} as uqg
import numpy as np

# Set up adaptive mesh
config = uqg.AMRConfig(
    initial_grid_size=(32, 32),
    max_refinement_level=3,
    refinement_threshold=0.1
)

amr = uqg.AdaptiveMeshRefinement(config)

# Initialize with test data
def test_function(x, y):
    return np.exp(-10.0 * ((x - 0.5)**2 + (y - 0.5)**2))

root_patch = amr.initialize_root_patch(test_function)
amr.refine_or_coarsen(root_patch)

# Visualize results
fig, axes = amr.visualize_grid_hierarchy(root_patch)
```

### GPU Constraint Solving

```python
import {pkg_name} as uqg
import numpy as np

# Create test Hamiltonian
dim = 100
H = np.random.randn(dim, dim)
H = H + H.T  # Make Hermitian

# Solve Wheeler-DeWitt equation
solver = uqg.GPUConstraintSolver()
initial_state = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)

result = solver.solve_wheeler_dewitt(H, initial_state)
print(f"Converged: {{result['converged']}}")
print(f"Final violation: {{result['constraint_violation']:.3e}}")
```

## Development

### Running Tests

```bash
python -m pytest {pkg_name}/tests/
```

### Code Quality

```bash
# Format code
black {pkg_name}/

# Lint code  
flake8 {pkg_name}/

# Type checking
mypy {pkg_name}/
```

### Building Documentation

```bash
cd docs/
make html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{{unified_qg,
    title = {{Unified Quantum Gravity Pipeline}},
    author = {{Quantum Gravity Research Team}},
    url = {{https://github.com/quantumgravity/unified_qg}},
    version = {{0.1.0}},
    year = {{2024}}
}}
```

## Support

- **Documentation**: https://unified-qg.readthedocs.io/
- **Issues**: https://github.com/quantumgravity/unified_qg/issues
- **Discussions**: https://github.com/quantumgravity/unified_qg/discussions

## Acknowledgments

This package builds upon decades of research in Loop Quantum Gravity and computational physics. We acknowledge the contributions of the LQG community and the open-source scientific computing ecosystem.
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("Created README.md file")


def create_pyproject_toml(pkg_name: str = "unified_qg") -> None:
    """Create modern pyproject.toml file."""
    pyproject_content = f'''[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{pkg_name}"
description = "Unified Quantum Gravity Pipeline for LQG calculations"
readme = "README.md"
license = {{text = "MIT"}}
authors = [
    {{name = "Quantum Gravity Research Team", email = "research@quantumgravity.org"}},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research", 
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
]
dynamic = ["version"]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.3.0",
]

[project.optional-dependencies]
gpu = [
    "torch>=1.9.0",
    "torchvision>=0.10.0",
]
mpi = [
    "mpi4py>=3.0.0", 
]
dev = [
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]
all = [
    "torch>=1.9.0",
    "torchvision>=0.10.0", 
    "mpi4py>=3.0.0",
    "pytest>=6.0.0",
    "pytest-cov>=2.0.0",
    "black>=21.0.0",
    "flake8>=3.9.0", 
    "mypy>=0.910",
]

[project.urls]
"Homepage" = "https://github.com/quantumgravity/unified_qg"
"Bug Reports" = "https://github.com/quantumgravity/unified_qg/issues"
"Source" = "https://github.com/quantumgravity/unified_qg"
"Documentation" = "https://unified-qg.readthedocs.io/"

[project.scripts]
{pkg_name}-demo = "{pkg_name}.demo:main"
{pkg_name}-test = "{pkg_name}.tests.run_tests:main"

[tool.setuptools]
package-dir = {{"" = "."}}

[tool.setuptools.packages.find]
where = ["."]
include = ["{pkg_name}*"]

[tool.setuptools.package-data]
{pkg_name} = ["data/*.json", "data/*.txt"]

[tool.setuptools_scm]
write_to = "{pkg_name}/_version.py"

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311"]

[tool.pytest.ini_options]
testpaths = ["{pkg_name}/tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config", 
    "--cov={pkg_name}",
    "--cov-report=term-missing",
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "scipy.*", 
    "torch.*",
    "mpi4py.*",
]
ignore_missing_imports = true
'''
    
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    print("Created pyproject.toml file")


def package_pipeline_as_library(pkg_name: str = "unified_qg") -> None:
    """
    Complete packaging function that creates all necessary files.
    
    Args:
        pkg_name: Name of the package to create
    """
    print(f"📦 Packaging pipeline as Python library: {pkg_name}")
    print("-" * 50)
    
    # Create package structure
    create_package_structure(pkg_name)
    
    # Create all packaging files
    create_setup_py(pkg_name) 
    create_requirements_txt()
    create_manifest_in(pkg_name)
    create_license_file()
    create_readme(pkg_name)
    create_pyproject_toml(pkg_name)
    
    print(f"✅ Package {pkg_name} successfully created!")
    print(f"   - Setup files: setup.py, pyproject.toml, requirements.txt")
    print(f"   - Documentation: README.md, LICENSE, MANIFEST.in") 
    print(f"   - Package structure: {pkg_name}/ with tests/")
    print(f"\\nTo install in development mode:")
    print(f"   pip install -e .")
    print(f"\\nTo build and distribute:")
    print(f"   python -m build")
    print(f"   python -m twine upload dist/*")


def create_example_config(output_file: str = "example_config.json") -> None:
    """Create example configuration file for the pipeline."""
    config = {
        "amr_config": {
            "initial_grid_size": [64, 64],
            "max_refinement_level": 4,
            "refinement_threshold": 0.1,
            "coarsening_threshold": 0.05
        },
        "field_3d_config": {
            "grid_size": [32, 32, 32],
            "dx": 0.05,
            "dt": 0.001,
            "epsilon": 0.01,
            "mass": 1.0,
            "total_time": 0.1
        },
        "gpu_solver_config": {
            "max_iterations": 1000,
            "learning_rate": 0.001,
            "tolerance": 1e-10,
            "device": "auto"
        },
        "phenomenology_config": {
            "masses": [1.0, 2.0, 5.0],
            "spins": [0.0, 0.3, 0.7, 0.9],
            "output_formats": ["json", "csv", "hdf5"]
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created example configuration: {output_file}")
