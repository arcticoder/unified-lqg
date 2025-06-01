# Improved compute_negative_energy.py - Implementation Notes

## ✅ **Improvements Implemented**

### **1. Robust b0 Parameter Handling**
- **✅ Dedicated `b0` field**: Script now expects and prefers `"b0": 1.6e-35` in input JSON
- **✅ Graceful fallback**: If `b0` field missing, falls back to parsing from label with warnings
- **✅ Better error handling**: Clear warning messages when parsing fails

**Before (brittle):**
```python
parts = label.split("_refined")[0].split("=")
b0 = float(parts[1])  # Could easily fail
```

**After (robust):**
```python
b0 = entry.get("b0")  # Preferred approach
if b0 is None:
    print(f"Warning: No 'b0' field in {label}, attempting to parse...")
    # Fallback parsing with error handling
```

### **2. Placeholder Ready for Real Integration**
- **✅ Clear integration placeholder**: `integrate_over_volume_placeholder()` function ready for replacement
- **✅ TODO comments**: Specific instructions for implementing actual T^00 integration
- **✅ Scipy integration template**: Ready-to-use code structure for numerical quadrature

**Placeholder structure:**
```python
def integrate_over_volume_placeholder(refined_metric, b0, r_range=None):
    """
    TODO: Replace with actual implementation:
        from scipy.integrate import quad
        
        def T00_integrand(r):
            g = compute_metric_at_r(refined_metric, r, b0)
            T00 = compute_stress_energy_00(g, r)  # From Einstein equations
            vol_element = compute_volume_element(g, r)  # √g d³x
            return abs(T00) * vol_element
        
        integral, error = quad(T00_integrand, r_min, r_max)
        return integral
    """
```

### **3. Enhanced Workflow Integration**
- **✅ Clear workflow documentation**: Step-by-step pipeline instructions
- **✅ Expected input format**: Documented JSON schema for `refined_metrics.ndjson`
- **✅ Richer output format**: Includes `parent_solution`, `b0`, `throat_radius`, `computation_method`
- **✅ Example workflow script**: Complete demo in `examples/example_workflow.py`

**Workflow:**
```
1. metric_refinement.py → refined_metrics.ndjson (with "b0" field)
2. compute_negative_energy.py → negative_energy_integrals.ndjson  
3. design_control_field.py → control_fields.ndjson
```

## **Expected Input Format**
```json
{
    "label": "wormhole_b0=1.60e-35_source=upstream_data_refined",
    "parent_solution": "wormhole_b0=1.60e-35_source=upstream_data", 
    "b0": 1.6e-35,
    "throat_radius": 1.6e-35,
    "refined_metric": {"g_tt": -1.01, "g_rr": 1.02, "g_thth": 2.6e-70, "g_phph": 2.6e-70},
    "refinement_method": "perturbative_correction",
    "convergence_error": 1e-6
}
```

## **Output Format**
```json
{
    "label": "wormhole_b0=1.60e-35_source=upstream_data_refined",
    "parent_solution": "wormhole_b0=1.60e-35_source=upstream_data",
    "b0": 1.6e-35,
    "throat_radius": 1.6e-35,
    "negative_energy_integral": 3.49e-177,
    "computation_method": "placeholder_integration"
}
```

## **Usage Commands**

### **Individual Script:**
```bash
python metric_engineering/compute_negative_energy.py \
  --refined metric_engineering/outputs/refined_metrics.ndjson \
  --out metric_engineering/outputs/negative_energy_integrals.ndjson
```

### **Complete Workflow Demo:**
```bash
python metric_engineering/examples/example_workflow.py
```

## **Next Steps for Real Implementation**

1. **Replace placeholder with actual T^00 calculation:**
   ```python
   negative_integral = integrate_over_volume(abs(T00(r)), r_range)
   ```

2. **Implement stress-energy tensor computation from refined metric**

3. **Add numerical integration using `scipy.integrate.quad`**

4. **Validate against known analytical solutions**

The script is now production-ready as a scaffold and will integrate smoothly into the `metric_engineering/` pipeline.
