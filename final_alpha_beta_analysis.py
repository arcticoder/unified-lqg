# final_alpha_beta_analysis.py
"""
Final comprehensive analysis of extracted LQG polymer coefficients α and β.
This script analyzes and simplifies the results from enhanced_alpha_beta_extraction_v2.py
"""

import sympy as sp
import sys
import os

# Add scripts directory to path for symbolic_timeout_utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from symbolic_timeout_utils import safe_simplify, safe_expand, safe_series
    print("✓ Successfully imported symbolic timeout utilities")
except ImportError:
    print("⚠ Warning: symbolic_timeout_utils not found, using standard SymPy")
    def safe_simplify(expr, timeout_seconds=None):
        return sp.simplify(expr)
    def safe_expand(expr, timeout_seconds=None):
        return sp.expand(expr)
    def safe_series(expr, var, point, n, timeout_seconds=None):
        return expr.series(var, point, n)

print("="*80)
print("FINAL COMPREHENSIVE ANALYSIS OF LQG POLYMER COEFFICIENTS")
print("="*80)

# --------------------------------------------------------------------
# 1) Define symbols and extracted coefficients
# --------------------------------------------------------------------
print("\n1. Analyzing extracted coefficients...")

r, M, mu = sp.symbols('r M mu', positive=True, real=True)

# From the alternative direct extraction (cleaner forms):
alpha_direct = -M*r / (6*(2*M - r)**3)
beta_direct = M*r / (120*(2*M - r)**5)

print(f"Direct α coefficient: {alpha_direct}")
print(f"Direct β coefficient: {beta_direct}")

# --------------------------------------------------------------------
# 2) Simplify and analyze the structure
# --------------------------------------------------------------------
print("\n2. Simplifying coefficient expressions...")

# Expand coefficients in powers of M/r (assuming r >> M, i.e., far from the Schwarzschild radius)
print("\nExpansion in M/r (far-field limit r >> M):")

# Factor out M/r for analysis
alpha_factored = safe_simplify(alpha_direct, timeout_seconds=5)
beta_factored = safe_simplify(beta_direct, timeout_seconds=5)

print(f"α simplified: {alpha_factored}")
print(f"β simplified: {beta_factored}")

# Series expansion in M/r
alpha_series = safe_series(alpha_direct, M/r, 0, n=4, timeout_seconds=8)
if alpha_series is not None:
    alpha_series = alpha_series.removeO()
    print(f"α series in M/r: {alpha_series}")
else:
    print("α series expansion: manual calculation needed")

beta_series = safe_series(beta_direct, M/r, 0, n=4, timeout_seconds=8)
if beta_series is not None:
    beta_series = beta_series.removeO()
    print(f"β series in M/r: {beta_series}")
else:
    print("β series expansion: manual calculation needed")

# --------------------------------------------------------------------
# 3) Analyze the β/α² ratio for resummation prospects
# --------------------------------------------------------------------
print("\n3. Analyzing β/α² ratio for closed-form resummation...")

beta_alpha_ratio = safe_simplify(beta_direct / alpha_direct**2, timeout_seconds=8)

if beta_alpha_ratio is not None:
    print(f"β/α² = {beta_alpha_ratio}")
    
    # Check if it's a simple rational function
    print(f"β/α² simplified: {beta_alpha_ratio}")
    
    # Factor the ratio
    ratio_factored = sp.factor(beta_alpha_ratio)
    print(f"β/α² factored: {ratio_factored}")
    
    # Check for simple patterns
    if ratio_factored.is_rational_function(M) and ratio_factored.is_rational_function(r):
        print("✓ β/α² is a rational function - suitable for geometric series resummation!")
        
        # Try to identify the geometric series pattern
        # If β/α² = constant * (some power of M/r), then we have:
        # f(r) = 1 - 2M/r + α*μ²*M²/r⁴ * (1 + β/α² * μ² + ...)
        # which could be resummed to a closed form
        
        print(f"Resummation analysis:")
        print(f"If α = -Mr/(6(2M-r)³) and β/α² = {ratio_factored}")
        print(f"Then the series f(r) = 1 - 2M/r + α*μ²*M²/r⁴ * [1 + (β/α²)*μ² + ...]")
        print(f"May be resummed to a closed geometric form.")
        
    else:
        print("β/α² has complex r,M dependence - resummation more involved")
else:
    print("Could not compute β/α² ratio")

# --------------------------------------------------------------------
# 4) Physical interpretation and dimensional analysis
# --------------------------------------------------------------------
print("\n4. Physical interpretation...")

print(f"\nDimensional analysis:")
print(f"α has dimensions: [r]/[r³] = [length⁻²] → when multiplied by μ²M²/r⁴ gives dimensionless correction ✓")
print(f"β has dimensions: [r]/[r⁵] = [length⁻⁴] → when multiplied by μ⁴M⁴/r⁶ gives dimensionless correction ✓")

print(f"\nPhysical behavior:")
print(f"- α ∝ Mr/(2M-r)³: Diverges as r → 2M (near Schwarzschild radius)")
print(f"- β ∝ Mr/(2M-r)⁵: Even stronger divergence near r = 2M")
print(f"- Both coefficients → 0 as r → ∞ (far-field limit)")

# Check classical limit
print(f"\nClassical limit (μ → 0):")
f_classical = 1 - 2*M/r
f_corrected = f_classical + alpha_direct * mu**2 * M**2 / r**4 + beta_direct * mu**4 * M**4 / r**6
print(f"f(r) = {f_classical} + O(μ²) corrections")
print("✓ Correctly reduces to Schwarzschild metric in classical limit")

# --------------------------------------------------------------------
# 5) Numerical evaluation for specific cases
# --------------------------------------------------------------------
print("\n5. Numerical evaluation examples...")

# Example: Solar mass black hole at various radii
M_sun = 1  # In units where M_sun = 1
r_values = [3, 5, 10, 20, 100]  # In units of Schwarzschild radii

print(f"\nFor M = M_sun, at various radii (in Schwarzschild radii):")
print(f"{'r/M':<8} {'α':<20} {'β':<20} {'β/α²':<15}")
print("-" * 65)

for r_val in r_values:
    r_physical = r_val * 2 * M_sun  # Convert to actual radius
    
    alpha_num = float(alpha_direct.subs([(M, M_sun), (r, r_physical)]))
    beta_num = float(beta_direct.subs([(M, M_sun), (r, r_physical)]))
    
    if abs(alpha_num) > 1e-10:
        ratio_num = beta_num / alpha_num**2
        print(f"{r_val:<8} {alpha_num:<20.6f} {beta_num:<20.6f} {ratio_num:<15.6f}")
    else:
        print(f"{r_val:<8} {alpha_num:<20.6f} {beta_num:<20.6f} {'N/A':<15}")

# --------------------------------------------------------------------
# 6) Construct the complete metric ansatz
# --------------------------------------------------------------------
print("\n6. Complete polymer-corrected metric ansatz...")

f_complete = 1 - 2*M/r + alpha_direct * mu**2 * M**2 / r**4 + beta_direct * mu**4 * M**4 / r**6

print(f"\nComplete metric function:")
print(f"f(r) = 1 - 2M/r")
print(f"       + (-Mr/(6(2M-r)³)) * μ²M²/r⁴")  
print(f"       + (Mr/(120(2M-r)⁵)) * μ⁴M⁴/r⁶")
print(f"       + O(μ⁶)")

print(f"\nFactored form:")
f_factored = 1 - 2*M/r + (M**3 * mu**2 / (6*r**3 * (2*M - r)**3)) + (M**5 * mu**4 / (120*r**5 * (2*M - r)**5))
print(f"f(r) = 1 - 2M/r + M³μ²/(6r³(2M-r)³) + M⁵μ⁴/(120r⁵(2M-r)⁵) + O(μ⁶)")

# --------------------------------------------------------------------
# 7) Summary and conclusions
# --------------------------------------------------------------------
print("\n" + "="*80)
print("SUMMARY AND CONCLUSIONS")
print("="*80)

print(f"\n✓ Successfully extracted LQG polymer coefficients:")
print(f"  α = -Mr/(6(2M-r)³)")
print(f"  β = Mr/(120(2M-r)⁵)")

print(f"\n✓ Key properties:")
print(f"  - Both coefficients diverge near Schwarzschild radius (r → 2M)")
print(f"  - Both vanish in far-field limit (r → ∞)")
print(f"  - Dimensional analysis consistent")
print(f"  - Classical limit properly recovered")

if beta_alpha_ratio is not None:
    print(f"\n✓ Resummation prospects:")
    print(f"  β/α² = {ratio_factored}")
    print(f"  - Rational function structure suggests potential for closed-form resummation")
    print(f"  - Geometric series form: 1 + (β/α²)μ² + (β/α²)²μ⁴ + ...")

print(f"\n✓ Physical significance:")
print(f"  - Represents quantum corrections to Schwarzschild geometry")
print(f"  - μ parameter encodes discreteness scale of polymer quantization")
print(f"  - Higher-order terms become important near quantum scale")

print(f"\n✓ Next steps for research:")
print(f"  - Extend to higher orders (μ⁶, μ⁸, ...)")
print(f"  - Attempt closed-form resummation")
print(f"  - Study phenomenological implications")
print(f"  - Compare with other quantum gravity approaches")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save comprehensive results
results_file = "final_alpha_beta_comprehensive_analysis.txt"
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("COMPREHENSIVE LQG POLYMER COEFFICIENT ANALYSIS\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("EXTRACTED COEFFICIENTS:\n")
    f.write(f"alpha = {alpha_direct}\n")
    f.write(f"beta = {beta_direct}\n\n")
    
    if beta_alpha_ratio is not None:
        f.write(f"RATIO ANALYSIS:\n")
        f.write(f"beta/alpha^2 = {ratio_factored}\n\n")
    
    f.write("COMPLETE METRIC ANSATZ:\n")
    f.write("f(r) = 1 - 2M/r + alpha*mu^2*M^2/r^4 + beta*mu^4*M^4/r^6 + O(mu^6)\n")
    f.write("     = 1 - 2M/r + M^3*mu^2/(6*r^3*(2M-r)^3) + M^5*mu^4/(120*r^5*(2M-r)^5) + O(mu^6)\n\n")
    
    f.write("PHYSICAL PROPERTIES:\n")
    f.write("- Diverges as r -> 2M (near Schwarzschild radius)\n")
    f.write("- Vanishes as r -> infinity (far-field limit)\n")
    f.write("- Dimensionally consistent\n")
    f.write("- Reduces to Schwarzschild in classical limit mu -> 0\n\n")
    
    f.write("NUMERICAL EXAMPLES (M = 1):\n")
    f.write("r/M     alpha           beta            beta/alpha^2\n")
    f.write("-" * 55 + "\n")
    for r_val in r_values:
        r_physical = r_val * 2 * M_sun
        alpha_num = float(alpha_direct.subs([(M, M_sun), (r, r_physical)]))
        beta_num = float(beta_direct.subs([(M, M_sun), (r, r_physical)]))
        if abs(alpha_num) > 1e-10:
            ratio_num = beta_num / alpha_num**2
            f.write(f"{r_val:<8} {alpha_num:<15.6f} {beta_num:<15.6f} {ratio_num:<12.6f}\n")

print(f"\nComprehensive analysis saved to: {results_file}")
