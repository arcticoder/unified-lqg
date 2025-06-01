#!/usr/bin/env python3
"""
Test script for LQG coherent state construction using integer basis values
that exactly match the discrete μ, ν labels in the flux basis.
"""

from lqg_fixed_components import LQGParameters, run_lqg_quantization, MuBarScheme

# Create customized LQG parameters with wider Gaussian widths
params = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=100,
    # Use wider Gaussians for better overlap with classical values
    coherent_width_E=1.0,  # Increased from default 0.5
    coherent_width_K=1.0,  # Increased from default 0.5
    mu_bar_scheme=MuBarScheme.MINIMAL_AREA
)

# Method 1: Use integer-valued data from new JSON file
print("\n===== Test with integer-valued E and K =====")
result_integer = run_lqg_quantization(
    classical_data_file="examples/lqg_example_integer_values.json",
    output_file="quantum_inputs/T00_quantum_integer_values.json",
    lqg_params=params
)
print("\nResult with integer-valued E and K:")
print(f"  Max |⟨E⟩−E_classical|: {result_integer.get('backreaction_data', {}).get('max_E_error', 'N/A')}")
print(f"  Max |⟨K⟩−K_classical|: {result_integer.get('backreaction_data', {}).get('max_K_error', 'N/A')}")
print(f"  Eigenvalue magnitude: {abs(result_integer.get('eigenvalues', [0])[0]) if result_integer.get('eigenvalues') else 'N/A'}")

# Method 2: Use original data but with wider Gaussians
print("\n===== Test with original data but wider Gaussians =====")
wider_params = LQGParameters(
    mu_max=2,
    nu_max=2,
    basis_truncation=100,
    # Use much wider Gaussians for better overlap with non-integer classical values
    coherent_width_E=2.0,
    coherent_width_K=2.0
)

result_wider = run_lqg_quantization(
    classical_data_file="examples/lqg_demo_classical_data.json",
    output_file="quantum_inputs/T00_quantum_wider_gaussians.json",
    lqg_params=wider_params
)
print("\nResult with original data but wider Gaussians:")
print(f"  Max |⟨E⟩−E_classical|: {result_wider.get('backreaction_data', {}).get('max_E_error', 'N/A')}")
print(f"  Max |⟨K⟩−K_classical|: {result_wider.get('backreaction_data', {}).get('max_K_error', 'N/A')}")
print(f"  Eigenvalue magnitude: {abs(result_wider.get('eigenvalues', [0])[0]) if result_wider.get('eigenvalues') else 'N/A'}")

print("\nTest complete. Check the output files for detailed results.")
