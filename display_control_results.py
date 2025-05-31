#!/usr/bin/env python3

import ndjson

# Load control field data
with open('outputs/control_fields/test_control_fields.ndjson', 'r') as f:
    control_data = ndjson.load(f)

print("=== ACTIVE CONTROL FIELD DESIGN RESULTS ===\n")

for i, field in enumerate(control_data, 1):
    print(f"Control Field {i}:")
    print(f"  Mode Label: {field['mode_label']}")
    print(f"  Parent Solution: {field['parent_solution']}")
    print(f"  Unstable Eigenvalue: {field['eigenvalue']:.3f}")
    print(f"  Control Field Amplitude (ε): {field['control_field_amplitude']:.3e}")
    print(f"  AsciiMath Ansatz: δΦ(r) = {field['control_field_amplitude']:.3e} * ψ_{field['mode_label']}(r)")
    print()

print(f"Total unstable modes identified: {len(control_data)}")
print("All modes require active control fields to achieve stability.")
