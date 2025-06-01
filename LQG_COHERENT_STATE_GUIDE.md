# LQG Coherent State Construction Guide

## Problem Summary

When attempting to construct LQG coherent states in the flux representation, two common issues arise:

1. **Coherent state normalization fails** (falls back to uniform superposition)
2. **Mismatched expectation values** (⟨E⟩ ≠ E_classical, ⟨K⟩ ≠ K_classical)

This happens when the classical triad and curvature values (E_classical, K_classical) do not align with the discrete spectrum of the quantum operators.

## Root Causes

The LQG coherent state construction in `lqg_fixed_components.py` uses:

```python
# For E-field part:
overlap *= np.exp(-(delta_E_x + delta_E_phi) / (2 * width_E**2))

# For K-field part:
K_x_approx = 0.1 * state.mu_config[site]  # Scaling factor 0.1
K_phi_approx = 0.1 * state.nu_config[site]
overlap *= np.exp(-(delta_K_x + delta_K_phi) / (2 * width_K**2))
```

Where:
- `delta_E_x = (E_x_quantum - classical_E_x[site])**2`
- `E_x_quantum = float(state.mu_config[site])` (integer μ ∈ {-2,-1,0,1,2})

If `classical_E_x` contains values like `[1.0, 0.9, 0.8]`, there's no exact match with any μ ∈ {-2,-1,0,1,2}, resulting in:
- Small Gaussian overlap factors
- Very small (or zero) overall normalization
- Fallback to uniform superposition

## Solutions

### Solution 1: Match Classical Values to Quantum Spectrum

Use classical data that exactly matches the quantum eigenvalues:

```json
"E_x": [2, 1, 0, -1, -2],     # Integers to match μ eigenvalues
"E_phi": [1, 1, 0, -1, -1],   # Integers to match ν eigenvalues
"K_x": [0.2, 0.1, 0.0, -0.1, -0.2],     # Exactly 0.1 * μ
"K_phi": [0.1, 0.05, 0.0, -0.05, -0.1]  # Exactly 0.1 * ν
```

### Solution 2: Increase Coherent State Width

When using the original values, increase the width parameters:

```python
params = LQGParameters(
    mu_max=2,
    nu_max=2,
    coherent_width_E=2.0,  # Much larger than default 0.5
    coherent_width_K=2.0   # Much larger than default 0.5
)
```

### Solution 3: Modify Flux-Curvature Scaling

Change the scaling factor between μ and K_x in the code:

```python
# If classical_K_x = [-0.04, -0.02, 0.0, 0.02, 0.04]
# And μ = [2, 1, 0, -1, -2]
# Then set:
K_x_approx = -0.02 * state.mu_config[site]  # New scaling factor
```

## Fixing Complex Eigenvalues Warning

The `ComplexWarning` when handling eigenvalues can be fixed by:

```python
"eigenvalue": float(np.abs(eigenvals[0])),  # Use absolute value
# OR
"eigenvalue": float(np.real(eigenvals[0])),  # Use real part
```

## Working Example

See `run_lqg_with_integer_values.py` for a complete working example using integer values that match the quantum eigenvalues.
