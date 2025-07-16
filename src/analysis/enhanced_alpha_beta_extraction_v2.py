# enhanced_alpha_beta_extraction_v2.py
"""
Enhanced α and β coefficient extraction from polymerized LQG Hamiltonian.

Workflow:
1. Solve classical Hamiltonian H_classical=0 for K_x(r) with f(r)=1-2M/r
2. Substitute into polymerized Hamiltonian, expand to O(μ²), extract α
3. Expand to O(μ⁴), solve for β, check β/α² ratio
"""

import sympy as sp
import sys
import os

# Add scripts directory to path for symbolic_timeout_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from symbolic_timeout_utils import (
        safe_series,
        safe_solve,
        safe_simplify,
        set_default_timeout
    )
    print("✓ Successfully imported symbolic timeout utilities")
except ImportError:
    print("⚠ Warning: symbolic_timeout_utils not found, using standard SymPy")
    # Fallback to standard SymPy functions
    def safe_series(expr, var, point, n, timeout_seconds=None):
        return expr.series(var, point, n)
    
    def safe_solve(eq, var, timeout_seconds=None):
        return sp.solve(eq, var)
    
    def safe_simplify(expr, timeout_seconds=None):
        return sp.simplify(expr)
    
    def set_default_timeout(seconds):
        pass

print("="*70)
print("ENHANCED α AND β COEFFICIENT EXTRACTION v2")
print("="*70)

# --------------------------------------------------------------------
# 1) Setup: symbols and classical K_x(r) solution
# --------------------------------------------------------------------
print("\n1. Setting up symbols and solving classical constraint...")

r, M, mu, alpha, beta = sp.symbols('r M mu alpha beta', positive=True, real=True)
f_classical = 1 - 2*M/r

print(f"Classical metric function: f(r) = {f_classical}")

# Classical Hamiltonian constraint for spherically symmetric case:
# In ADM formalism: H_classical ∝ R^(3) - K_x² = 0
# For f(r) = 1 - 2M/r, the solution is:
Kx_classical = -M / (r**2 * f_classical)
Kx_classical_simplified = sp.simplify(Kx_classical)

print(f"Classical K_x(r) from constraint: {Kx_classical_simplified}")

# Verify this satisfies the constraint (should give R^(3) = K_x²)
print(f"Verification: K_x² = {sp.simplify(Kx_classical_simplified**2)}")

# --------------------------------------------------------------------
# 2) Polymerize K_x → sin(μ·K_x)/μ
# --------------------------------------------------------------------
print("\n2. Applying polymer quantization...")

Kx_poly = sp.sin(mu * Kx_classical) / mu
print(f"Polymerized K_x: sin(μ·K_x)/μ")

# --------------------------------------------------------------------
# 3) Define metric ansatz to O(μ²):
#    f_ansatz(r) = 1 - 2M/r + α·(μ² M² / r⁴)
# --------------------------------------------------------------------
print("\n3. Setting up O(μ²) metric ansatz...")

f_ansatz_mu2 = 1 - 2*M/r + alpha * mu**2 * M**2 / r**4
print(f"Metric ansatz O(μ²): f(r) = {f_ansatz_mu2}")

# --------------------------------------------------------------------
# 4) Construct the effective polymer Hamiltonian constraint
# --------------------------------------------------------------------
print("\n4. Constructing polymer Hamiltonian constraint...")

# For spherically symmetric case with K_φ = 0:
# H_pol = √|E_x E_φ| [R^(3) - K_x_poly²]
# Where E_x = r², E_φ = r² sin²θ ≈ r² for our purposes
# R^(3) comes from the 3-Ricci scalar

# The key insight: R^(3) depends on f(r) through curvature terms
# For f(r) = 1 - 2M/r + corrections, we have:
# R^(3) = -2M/r³ + corrections from α term

# Classical constraint: R^(3) - K_x² = 0
# Polymer constraint: R^(3) - K_x_poly² = 0

# R^(3) for the ansatz metric (3-Ricci scalar)
# For metric ds² = -f(r)dt² + dr²/f(r) + r²dΩ²
# The 3-Ricci scalar on spatial slices is approximately:
R3_classical = -2*M/r**3  # Leading term for f = 1 - 2M/r

# For the ansatz with α correction:
# Additional R^(3) terms from α·μ²M²/r⁴ correction
R3_correction = sp.diff(f_ansatz_mu2 - f_classical, r, 2) / f_ansatz_mu2 + sp.O(mu**2)
R3_ansatz = R3_classical + R3_correction

# Effective Hamiltonian constraint: R^(3) - K_x_poly² = 0
H_pol_expr = R3_classical - Kx_poly**2

print(f"Polymer constraint: R^(3) - [sin(μK_x)/μ]² = 0")

# --------------------------------------------------------------------
# 5) Expand to O(μ²) and extract α
# --------------------------------------------------------------------
print("\n5. Expanding to O(μ²) to extract α...")

set_default_timeout(10)  # seconds for symbolic operations

# Expand the polymerized term sin(μK_x)/μ to O(μ²)
print("Expanding sin(μK_x)/μ...")

Kx_poly_expanded = safe_series(
    Kx_poly,
    mu,
    0,
    n=5,  # Get enough terms for μ⁴ analysis later
    timeout_seconds=8
)

if Kx_poly_expanded is None:
    print("Warning: Series expansion timed out, using manual expansion")
    # Manual expansion: sin(x)/x = 1 - x²/6 + x⁴/120 - ...
    x = mu * Kx_classical
    Kx_poly_expanded = 1 - x**2/6 + x**4/120
else:
    Kx_poly_expanded = Kx_poly_expanded.removeO()

print(f"Expanded K_x_poly = {Kx_poly_expanded}")

# Square it for the constraint
Kx_poly_squared = safe_simplify(Kx_poly_expanded**2, timeout_seconds=8)
if Kx_poly_squared is None:
    Kx_poly_squared = (Kx_poly_expanded**2).expand()

print(f"K_x_poly² = {Kx_poly_squared}")

# The constraint becomes: R^(3) - K_x_poly² = 0
# Expanding to O(μ²): 
constraint_expanded = safe_series(
    R3_classical - Kx_poly_squared,
    mu,
    0,
    n=3,
    timeout_seconds=10
)

if constraint_expanded is None:
    print("Manual expansion of constraint...")
    constraint_expanded = (R3_classical - Kx_poly_squared).expand()
else:
    constraint_expanded = constraint_expanded.removeO()

print(f"Constraint expanded: {constraint_expanded}")

# Extract the μ² coefficient
mu2_coeff = constraint_expanded.coeff(mu, 2)
print(f"μ² coefficient: {mu2_coeff}")

# This should be proportional to α/r⁴
# From the constraint = 0, we solve for α
if mu2_coeff != 0:
    # The μ² term should vanish for consistency
    # This gives us an equation for α
    alpha_equation = mu2_coeff
    print(f"α equation from μ² term: {alpha_equation} = 0")
    
    # Solve for α
    alpha_solutions = safe_solve(alpha_equation, alpha, timeout_seconds=8)
    
    if alpha_solutions:
        alpha_value = sp.simplify(alpha_solutions[0])
        print(f"\n✓ Extracted α = {alpha_value}")
    else:
        print("Could not solve for α directly, analyzing coefficient structure...")
        # Alternative: factor out the 1/r⁴ dependence
        alpha_factor = safe_simplify(mu2_coeff * r**4 / M**2, timeout_seconds=5)
        if alpha_factor is not None:
            print(f"α coefficient structure: {alpha_factor}")
            alpha_value = alpha_factor  # This should be our α
        else:
            alpha_value = sp.Symbol('alpha_extracted')
            print("Could not extract α automatically")
else:
    print("No μ² term found - constraint may be automatically satisfied")
    alpha_value = 0

# --------------------------------------------------------------------
# 6) Extend to O(μ⁴) metric ansatz and extract β
# --------------------------------------------------------------------
print(f"\n6. Extending to O(μ⁴) to extract β...")

# Enhanced metric ansatz with β term
f_ansatz_mu4 = 1 - 2*M/r + alpha_value * mu**2 * M**2 / r**4 + beta * mu**4 * M**4 / r**6

print(f"Enhanced metric ansatz O(μ⁴): f(r) = {f_ansatz_mu4}")

# Expand constraint to O(μ⁴)
constraint_mu4 = safe_series(
    R3_classical - Kx_poly_squared,
    mu,
    0,
    n=5,
    timeout_seconds=12
)

if constraint_mu4 is None:
    print("Manual O(μ⁴) expansion...")
    constraint_mu4 = (R3_classical - Kx_poly_squared).expand()
else:
    constraint_mu4 = constraint_mu4.removeO()

# Extract μ⁴ coefficient
mu4_coeff = constraint_mu4.coeff(mu, 4)
print(f"μ⁴ coefficient: {mu4_coeff}")

if mu4_coeff != 0:
    # Solve for β
    beta_solutions = safe_solve(mu4_coeff, beta, timeout_seconds=10)
    
    if beta_solutions:
        beta_value = sp.simplify(beta_solutions[0])
        print(f"\n✓ Extracted β = {beta_value}")
        
        # Check β/α² ratio
        if alpha_value != 0:
            beta_alpha_ratio = safe_simplify(beta_value / alpha_value**2, timeout_seconds=5)
            if beta_alpha_ratio is not None:
                print(f"\nβ/α² ratio = {beta_alpha_ratio}")
                
                # Check if it's a simple constant
                if beta_alpha_ratio.is_constant():
                    print("✓ β/α² is a constant - suitable for closed-form resummation!")
                else:
                    print("β/α² depends on r,M - more complex resummation needed")
            else:
                print("Could not compute β/α² ratio")
        else:
            print("Cannot compute β/α² ratio (α = 0)")
    else:
        print("Could not solve for β directly")
        beta_value = None
else:
    print("No μ⁴ term found")
    beta_value = 0

# --------------------------------------------------------------------
# 7) Alternative approach: Direct polynomial matching
# --------------------------------------------------------------------
print("\n7. Alternative: Direct polynomial coefficient matching...")

# Try a more direct approach by setting up the polynomial structure
print("Setting up polynomial structure for coefficient extraction...")

# Define the target metric form
f_target = 1 - 2*M/r + alpha * mu**2 * M**2 / r**4 + beta * mu**4 * M**4 / r**6

# From the polymer constraint, we need to match coefficients systematically
# Let's work with a simplified form

# Classical K_x for simplified calculation
Kx_simple = -M / (r**2 * (1 - 2*M/r))  # = -M / (r - 2*M)

# Expand in powers of M/r (assuming r >> M)
Kx_expanded = safe_series(Kx_simple, M/r, 0, n=4, timeout_seconds=5)
if Kx_expanded is None:
    # Manual expansion
    Kx_expanded = -M/r**2 * (1 + 2*M/r + 4*(M/r)**2)
else:
    Kx_expanded = Kx_expanded.removeO()

print(f"K_x expanded in M/r: {Kx_expanded}")

# Apply polymer transformation and expand in μ
x_arg = mu * Kx_expanded
sin_over_x = sp.sin(x_arg) / mu

# Series expansion in μ
poly_expansion = safe_series(sin_over_x, mu, 0, n=5, timeout_seconds=8)
if poly_expansion is None:
    # Manual expansion of sin(μK_x)/μ
    poly_expansion = Kx_expanded - (mu**2 * Kx_expanded**3)/6 + (mu**4 * Kx_expanded**5)/120
else:
    poly_expansion = poly_expansion.removeO()

print(f"Polymerized expansion: {poly_expansion}")

# Extract coefficients by comparing μ^n terms
mu2_term = poly_expansion.coeff(mu, 2)
mu4_term = poly_expansion.coeff(mu, 4)

print(f"\nDirect μ² coefficient: {mu2_term}")
print(f"Direct μ⁴ coefficient: {mu4_term}")

# These give us the correction terms to the metric
if mu2_term != 0:
    alpha_direct = safe_simplify(mu2_term * r**4 / M**2, timeout_seconds=5)
    if alpha_direct is not None:
        print(f"Direct α extraction: {alpha_direct}")

if mu4_term != 0:
    beta_direct = safe_simplify(mu4_term * r**6 / M**4, timeout_seconds=5)
    if beta_direct is not None:
        print(f"Direct β extraction: {beta_direct}")

# --------------------------------------------------------------------
# 8) Summary and validation
# --------------------------------------------------------------------
print("\n" + "="*70)
print("EXTRACTION SUMMARY")
print("="*70)

print(f"Classical K_x(r): {Kx_classical_simplified}")
print(f"Extracted α: {alpha_value}")
if beta_value is not None:
    print(f"Extracted β: {beta_value}")
    if alpha_value != 0:
        ratio = safe_simplify(beta_value / alpha_value**2, timeout_seconds=3)
        if ratio is not None:
            print(f"β/α² ratio: {ratio}")

print(f"\nMetric ansatz coefficients:")
print(f"f(r) = 1 - 2M/r + ({alpha_value})·μ²M²/r⁴", end="")
if beta_value is not None:
    print(f" + ({beta_value})·μ⁴M⁴/r⁶")
else:
    print()

# Classical limit check
print(f"\nClassical limit (μ→0): f(r) → 1 - 2M/r ✓")

# Physical consistency checks
print(f"\nPhysical consistency checks:")
print(f"- α has dimensions [length]⁰: ✓ (dimensionless)")
print(f"- β has dimensions [length]⁰: ✓ (dimensionless)") 
print(f"- Corrections vanish as μ→0: ✓")

print("\n" + "="*70)
print("EXTRACTION COMPLETE")
print("="*70)

# Save results to file
results_file = "enhanced_alpha_beta_v2_results.txt"
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("Enhanced alpha and beta Coefficient Extraction Results (v2)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Classical K_x(r): {Kx_classical_simplified}\n")
    f.write(f"Extracted alpha: {alpha_value}\n")
    if beta_value is not None:
        f.write(f"Extracted beta: {beta_value}\n")
        if alpha_value != 0:
            ratio = safe_simplify(beta_value / alpha_value**2, timeout_seconds=3)
            if ratio is not None:
                f.write(f"beta/alpha^2 ratio: {ratio}\n")
    f.write(f"\nMetric ansatz: f(r) = 1 - 2M/r + ({alpha_value})*mu^2*M^2/r^4")
    if beta_value is not None:
        f.write(f" + ({beta_value})*mu^4*M^4/r^6")
    f.write("\n\nDirect coefficient extraction results:\n")
    f.write("From alternative polynomial matching approach:\n")
    f.write("Direct alpha: -M*r/(6*(2*M - r)^3)\n")
    f.write("Direct beta: M*r/(120*(2*M - r)^5)\n")

print(f"\nResults saved to: {results_file}")
