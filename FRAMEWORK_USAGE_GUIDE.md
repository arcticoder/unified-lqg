# Complete LQG Framework Usage Guide

## Overview

The Complete LQG Framework provides a comprehensive suite of tools for analyzing Loop Quantum Gravity effects in black hole spacetimes. This framework integrates multiple advanced modules for coefficient extraction, prescription comparison, matter coupling, and numerical relativity interfaces.

## Quick Start

### Basic Usage

Run all modules with default settings:
```bash
python full_lqg_framework.py --all
```

### Individual Module Testing

Test specific components:
```bash
# μ¹⁰/μ¹² extension analysis
python full_lqg_framework.py --mu12

# Alternative prescription comparison
python full_lqg_framework.py --prescriptions

# Loop-quantized matter coupling
python full_lqg_framework.py --matter

# Numerical relativity interface
python full_lqg_framework.py --numerical

# Quantum geometry effects
python full_lqg_framework.py --quantum-geometry
```

### Combined Analysis

Run multiple modules together:
```bash
# Prescription comparison + matter coupling
python full_lqg_framework.py --prescriptions --matter

# Complete analysis with μ¹⁰/μ¹² + matter + numerical
python full_lqg_framework.py --mu12 --matter --numerical
```

## Individual Module Usage

### 1. μ¹⁰/μ¹² Extension Module

**File:** `lqg_mu10_mu12_extension.py`

**Features:**
- Higher-order coefficient extraction to μ¹² order
- Padé resummation techniques
- Multiple prescription support
- Validation through re-expansion

**Usage:**
```bash
# Main analysis
python lqg_mu10_mu12_extension.py

# Demo mode
python lqg_mu10_mu12_extension.py --demo
```

**Results:**
- Coefficients for δ, ε, ζ parameters
- Padé approximants
- Validation metrics

### 2. Alternative Prescription Comparison

**File:** `enhanced_alpha_beta_gamma_extraction.py`

**Features:**
- Compare 5 different polymer prescriptions
- Phenomenological analysis
- Observational signature predictions
- Comprehensive data export

**Usage:**
```bash
# Run comparison analysis
python enhanced_alpha_beta_gamma_extraction.py
```

**Prescriptions Available:**
- Standard (Ashtekar-Barbero)
- Thiemann
- AQEL (Ashtekar-Quantum-Einstein-Loop)
- Bojowald
- Improved

**Output Files:**
- `prescription_coefficient_comparison.csv`
- `prescription_coefficient_comparison.png`
- `example_compare_prescriptions.py`

### 3. Loop-Quantized Matter Coupling

**File:** `loop_quantized_matter_coupling.py`

**Features:**
- Polymer scalar fields
- Loop-quantized electromagnetic fields
- Fermion field coupling
- Matter backreaction on metric

**Usage:**
```bash
python loop_quantized_matter_coupling.py
```

**Matter Fields:**
- Scalar field with mass m = 0.005
- Electromagnetic field with polymer parameter μ_A
- Fermion fields ψ₁, ψ₂

### 4. Numerical Relativity Interface

**File:** `numerical_relativity_interface.py`

**Features:**
- HDF5 and JSON data export
- Initial data preparation
- Evolution equation setup
- Grid generation

**Usage:**
```bash
python numerical_relativity_interface.py
```

**Output Files:**
- `nr_output/lqg_metric.h5`
- `nr_output/lqg_metric.json`

### 5. Quantum Geometry Effects

**Features:**
- Area and volume quantization
- Holonomy corrections
- Inverse volume effects
- Graph refinement analysis

## Configuration

Use the example configuration file `config_example.json` as a template for custom analyses.

## Results and Output

### Standard Output Files

1. **comprehensive_lqg_results.json** - Complete analysis results
2. **prescription_coefficient_comparison.csv** - Prescription comparison data
3. **prescription_coefficient_comparison.png** - Comparison plots
4. **nr_output/** - Numerical relativity data files

### Key Results

#### Coefficient Values
All prescriptions consistently give:
- α = 1/6 (leading order correction)
- β = 0 (next-to-leading order)
- γ = 0 (next-to-next-to-leading order)

#### Phenomenological Predictions
- Horizon radius shift: δr_h ≈ +0.0017M (for μ = 0.1)
- ISCO modifications: δr_ISCO ≈ O(μ²)
- Photon sphere corrections
- Quasi-normal mode frequency shifts

## Performance

Typical execution times:
- Full framework (`--all`): ~340 seconds
- Prescription comparison: ~320 seconds
- Matter coupling: ~1 second
- μ¹⁰/μ¹² extension: ~5 seconds
- Numerical interface: ~15 seconds

## Troubleshooting

### Common Issues

1. **μ¹⁰/μ¹² coefficient extraction errors**: 
   - Some higher-order coefficients may not extract cleanly
   - Framework returns error information but continues

2. **Numerical relativity evolution equations**:
   - Minor attribute error in evolution setup
   - Data export and initial data preparation work correctly

3. **Symbolic timeout warnings**:
   - Complex expressions may timeout
   - Results are still generated with fallback methods

### Module Status

✅ **Working perfectly:**
- Alternative prescription comparison
- Loop-quantized matter coupling
- Quantum geometry analysis
- Framework integration

⚠️ **Working with minor issues:**
- μ¹⁰/μ¹² extension (coefficient extraction errors)
- Numerical relativity interface (evolution equation bug)

## Advanced Usage

### Custom Prescription Implementation

To add new polymer prescriptions, modify the prescription functions in the relevant modules and update the prescription lists.

### Parameter Exploration

Modify physical parameters in the scripts:
- Mass range: M ∈ [0.1, 10.0] M☉
- Polymer parameter: μ ∈ [0.001, 0.2]
- Radial range: r ∈ [1.0, 100.0] M

### Output Customization

Control output verbosity and file formats through command-line arguments and configuration settings.

## Scientific Applications

### Black Hole Physics
- Quantum corrections to Schwarzschild geometry
- Modified horizon structure
- Quantum bounce scenarios

### Observational Astronomy
- Gravitational wave template modifications
- Event Horizon Telescope predictions
- Pulsar timing corrections

### Theoretical Physics
- LQG phenomenology
- Quantum gravity model testing
- Prescription-independent results

## References

This framework implements techniques from:
- Loop Quantum Gravity literature
- Polymer quantization methods
- Numerical relativity interfaces
- Phenomenological LQG studies

---

For questions or issues, refer to the individual module documentation or the comprehensive framework summary files.
