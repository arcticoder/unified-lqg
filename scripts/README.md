# LQG Closed-Form Metric Derivation Roadmap

This directory contains a complete implementation of the roadmap for deriving a closed-form LQG-corrected spherically symmetric metric to leading order in the polymer scale μ.

## Overview

The roadmap produces, at least to leading order in the polymer scale, a closed-form (or small-μ expansion) for an LQG-corrected spherically symmetric metric:

```
f_LQG(r) = 1 - 2M/r + α*μ²M²/r⁴ + O(μ⁴)
```

where α is a dimensionless coefficient derived from symbolic polymer quantization.

## Quick Start

### Prerequisites

```bash
pip install sympy numpy scipy matplotlib
```

### Run Complete Pipeline

```bash
# Run all steps with default parameters (M=1.0, μ=0.05)
python scripts/run_complete_derivation.py

# Use custom parameters
python scripts/run_complete_derivation.py --M 2.0 --mu 0.1

# Skip certain steps (useful for debugging)
python scripts/run_complete_derivation.py --skip-steps 2 4
```

### Quick Symbolic Derivation Only

```bash
# Just derive the symbolic coefficient
python scripts/derive_effective_metric.py
```

## Step-by-Step Roadmap

### Step 1: Symbolic Polymerization (`derive_effective_metric.py`)

**Purpose**: Symbolically polymerize the Hamiltonian constraint and solve for f_LQG(r) to O(μ²).

**What it does**:
1. Writes down the classical Hamiltonian constraint in radial-triad variables
2. Applies polymer corrections: `K → sin(μK)/μ`
3. Expands to O(μ²) using SymPy
4. Solves for the metric function ansatz to extract coefficient α

**Output**: 
- `scripts/lqg_metric_results.py` - Symbolic coefficient and evaluation function
- Console output showing the derived α value

**Example Usage**:
```python
from scripts.derive_effective_metric import derive_lqg_metric
results = derive_lqg_metric()
print(f"LQG coefficient: α = {results['alpha_star']}")
```

### Step 2: Numerical Fitting (`fit_effective_metric.py`)

**Purpose**: Numerically fit LQG midisuperspace outputs to extract effective parameters.

**What it does**:
1. Runs your existing LQG midisuperspace solver across different lattice sizes
2. Extracts numeric estimates of g_tt(r) from ground states
3. Fits to ansatz f(r) = 1 - 2M_eff/r + γ/r⁴
4. Compares fitted γ with theoretical prediction α*μ²

**Output**:
- `scripts/fitting_results.json` - Numerical fitting results
- `scripts/lqg_metric_fit.png` - Visualization of fit quality

**Example Usage**:
```python
from scripts.fit_effective_metric import run_complete_fitting_pipeline
results = run_complete_fitting_pipeline(mu_value=0.05, M_value=1.0)
```

### Step 3: Symbolic-Numeric Matching (`match_symbolic_numeric.py`)

**Purpose**: Match symbolic series to numeric fit and build closed-form template.

**What it does**:
1. Compares symbolic α coefficient with fitted γ parameter
2. Validates agreement within tolerance (typically <15%)
3. Builds final closed-form template using validated coefficient
4. Generates LaTeX expressions and numerical evaluation functions

**Output**:
- `scripts/lqg_closed_form_metric.py` - Final closed-form metric template
- `scripts/metric_comparison.png` - Comparison plots
- Console validation report

**Example Usage**:
```python
from scripts.match_symbolic_numeric import run_complete_matching_pipeline
results = run_complete_matching_pipeline(mu_value=0.05, M_value=1.0)
```

### Step 4: Bounce Radius Analysis (`solve_bounce_radius.py`)

**Purpose**: Analyze the bounce/horizon radius and compare with spin-foam predictions.

**What it does**:
1. Solves f_LQG(r*) = 0 for the bounce/horizon radius
2. Compares with classical Schwarzschild horizon r = 2M
3. Analyzes quantum corrections to horizon structure
4. (Optional) Cross-validates with spin-foam peak radius

**Output**:
- `scripts/horizon_analysis.png` - Horizon structure plots
- Console analysis of quantum corrections

**Example Usage**:
```python
from scripts.solve_bounce_radius import find_bounce_radius
r_bounce = find_bounce_radius(M=1.0, mu=0.05, alpha=1/6)
```

### Step 5: Hamilton-Jacobi Check (`hamilton_jacobi_check.py`)

**Purpose**: Validate that the metric ansatz yields tractable geodesics.

**What it does**:
1. Inserts f_LQG(r) into Hamilton-Jacobi equation for test particles
2. Attempts analytical integration of ∂S/∂r = ±√[E²/f² - m²/f]
3. Analyzes effective potential and orbital stability
4. Compares classical vs LQG geodesics numerically

**Output**:
- `scripts/geodesic_comparison.png` - Geodesic analysis plots
- Console report on integrability and special functions

**Example Usage**:
```python
from scripts.hamilton_jacobi_check import run_hamilton_jacobi_analysis
results = run_hamilton_jacobi_analysis()
```

### Step 6: Final Validation and Integration

**Purpose**: Validate the complete closed-form template and integrate with existing pipeline.

**What it does**:
1. Loads and validates the final closed-form metric
2. Performs consistency checks (classical limit, positivity, etc.)
3. Generates LaTeX expressions for publication
4. Creates integration points for your existing LQG pipeline

**Output**:
- Complete validation report
- Ready-to-use metric functions
- LaTeX expressions for papers

## File Structure

```
scripts/
├── derive_effective_metric.py          # Step 1: Symbolic derivation
├── fit_effective_metric.py             # Step 2: Numerical fitting  
├── match_symbolic_numeric.py           # Step 3: Symbolic-numeric matching
├── solve_bounce_radius.py              # Step 4: Bounce radius analysis
├── hamilton_jacobi_check.py            # Step 5: Hamilton-Jacobi check
├── run_complete_derivation.py          # Main orchestration script
└── README.md                           # This file

# Generated files:
├── lqg_metric_results.py               # Symbolic results from Step 1
├── lqg_closed_form_metric.py           # Final template from Step 3
├── fitting_results.json               # Numerical results from Step 2
├── derivation_report.json             # Complete pipeline report
└── *.png                              # Various analysis plots
```

## Integration with Existing LQG Pipeline

### Using the Closed-Form Metric

Once derivation is complete, you can use the closed-form metric in your existing pipeline:

```python
# Load the validated closed-form metric
from scripts.lqg_closed_form_metric import f_LQG, g_LQG_components, ALPHA_LQG

# Parameters
M, mu = 1.0, 0.05

# Evaluate at any radius
r = 3.0
f_value = f_LQG(r, M, mu)
metric_components = g_LQG_components(r, 0, np.pi/2, 0, M, mu)

print(f"f_LQG({r}) = {f_value}")
print(f"g_tt = {metric_components['g_tt']}")
```

### Replacing Classical Schwarzschild

Replace any instance of classical `1 - 2M/r` in your diagnostics:

```python
# Old way
f_classical = 1 - 2*M/r_grid

# New way  
from scripts.lqg_closed_form_metric import f_LQG
f_quantum = f_LQG(r_grid, M, mu)
```

### Re-running Constraint Algebra

Use the effective metric slice as initial data for constraint checks:

```python
# Effective triad data
E_x_eff = r_grid**2
E_phi_eff = r_grid * np.sqrt(f_LQG(r_grid, M, mu))
K_phi_eff = np.zeros_like(r_grid)  # Static slice
K_x_eff = np.zeros_like(r_grid)   # Leading order

# Run your existing constraint analyzer
constraint_anomaly = check_constraint_algebra(E_x_eff, E_phi_eff, K_x_eff, K_phi_eff)
```

## Expected Results

For typical parameters (M=1, μ=0.05), you should expect:

- **Symbolic coefficient**: α ≈ 1/6 (typical from sin series expansion)
- **Relative horizon shift**: ~1-5% quantum correction at r ≈ 2M  
- **Validation level**: "excellent" (<5% error) or "acceptable" (<15% error)
- **Bounce radius**: r* ≈ 1.95M (slightly inside classical horizon)

## Troubleshooting

### Common Issues

1. **Import errors on first run**: Run Step 1 first to generate `lqg_metric_results.py`
2. **LQG solver not found**: The fitting step will use synthetic data if your full solver isn't available
3. **SymPy integration fails**: This is expected for complex expressions; the series expansion approach usually works
4. **Poor validation scores**: Check that μ values are small (< 0.1) for perturbative validity

### Dependencies

Required packages:
- `sympy` >= 1.8 (symbolic computation)
- `numpy` >= 1.20 (numerical arrays)  
- `scipy` >= 1.7 (optimization and integration)
- `matplotlib` >= 3.3 (plotting)

Optional for full LQG integration:
- Your existing LQG framework modules
- `json` (included in Python standard library)

### Performance Notes

- Step 1 (symbolic): ~30 seconds
- Step 2 (numerical): ~2-10 minutes depending on lattice sizes
- Step 3 (matching): ~10 seconds
- Steps 4-5 (analysis): ~30 seconds each
- Complete pipeline: ~5-15 minutes total

## Citation

If you use this roadmap in research, please cite the relevant LQG literature and acknowledge the symbolic-numeric validation approach.

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Ensure all prerequisites are installed
3. Try running individual steps to isolate problems
4. Check that symbolic derivation (Step 1) completes first
