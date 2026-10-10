#!/usr/bin/env python3
"""
Reproducible AMR convergence study on a field with a known answer.

Field:   u = exp(-10 |x-(1,1)|^2) + 0.5 exp(-5 |x+(1,1)|^2) on [-3,3]^2
Exact:   mass   = int u           = 0.2 pi
         energy = int |grad u|^2/2 = 0.625 pi       (cross term ~ e^-27, neglected)

Compares, at equal numbers of degrees of freedom:
  * uniform grids (observed order should be ~2),
  * AMR with refined patches re-evaluated from the field (the intended use),
  * AMR with refined patches only interpolated from the coarse data (no new
    information, so no accuracy gain is expected).

Deterministic (no random numbers).  Writes outputs/repro/amr_convergence.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adaptive_mesh_refinement import AdaptiveMeshRefinement, AMRConfig  # noqa: E402

EXACT_MASS = 0.2 * np.pi
EXACT_ENERGY = 0.625 * np.pi


def field(x, y):
    r1 = np.sqrt((x - 1) ** 2 + (y - 1) ** 2)
    r2 = np.sqrt((x + 1) ** 2 + (y + 1) ** 2)
    return np.exp(-10 * r1 ** 2) + 0.5 * np.exp(-5 * r2 ** 2)


def quiet(fn, *a, **k):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def build_amr(n0, levels, reevaluate, estimator="gradient", fraction=0.2):
    cfg = AMRConfig(initial_grid_size=(n0, n0), max_refinement_levels=levels,
                    refinement_fraction=fraction, error_estimator=estimator, max_grid_size=4096)
    amr = AdaptiveMeshRefinement(cfg)
    quiet(amr.create_initial_grid, (-3, 3), (-3, 3), field)
    upd = (lambda p: field(*p.nodes())) if reevaluate else None
    for _ in range(levels):
        quiet(amr.adapt_mesh, update_function=upd)
    return amr


def errors(amr):
    c = amr.check_conservation()
    return (abs(c["total_mass"] - EXACT_MASS) / EXACT_MASS,
            abs(c["total_energy"] - EXACT_ENERGY) / EXACT_ENERGY)


def uniform_study(sizes):
    rows = []
    for n in sizes:
        amr = AdaptiveMeshRefinement(AMRConfig(initial_grid_size=(n, n)))
        quiet(amr.create_initial_grid, (-3, 3), (-3, 3), field)
        em, ee = errors(amr)
        rows.append({"n": n, "dof": (n - 1) ** 2, "h": 6 / (n - 1), "mass_err": em, "energy_err": ee})
    for key in ("mass_err", "energy_err"):
        for a, b in zip(rows, rows[1:]):
            if a[key] > 0 and b[key] > 0:
                b[key.replace("_err", "_order")] = float(np.log(a[key] / b[key]) / np.log(a["h"] / b["h"]))
    return rows


def amr_study(n0_list, levels, reevaluate):
    rows = []
    for n0 in n0_list:
        amr = build_amr(n0, levels, reevaluate)
        em, ee = errors(amr)
        rows.append({"n0": n0, "levels": levels, "dof": amr.count_degrees_of_freedom(),
                     "mass_err": em, "energy_err": ee})
    return rows


def main():
    uniform = uniform_study([17, 33, 65, 129, 257])
    amr_re = amr_study([17, 25, 33], 3, True)
    amr_interp = amr_study([17, 25, 33], 3, False)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True).stdout.strip()
    except Exception:
        sha = "unknown"
    out = {"git_commit": sha, "numpy": np.__version__,
           "exact": {"mass": EXACT_MASS, "energy": EXACT_ENERGY},
           "uniform": uniform, "amr_reevaluated": amr_re, "amr_interpolated_only": amr_interp}
    path = ROOT / "outputs" / "repro" / "amr_convergence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    print("uniform:           n   dof    mass_err   energy_err  energy_order")
    for r in uniform:
        print(f"                {r['n']:4d} {r['dof']:6d}  {r['mass_err']:.2e}   {r['energy_err']:.2e}   "
              f"{r.get('energy_order', float('nan')):.2f}")
    for name, rows in (("AMR re-evaluated", amr_re), ("AMR interpolated only", amr_interp)):
        print(f"{name}:")
        for r in rows:
            print(f"   n0={r['n0']:3d} dof={r['dof']:6d}  mass_err={r['mass_err']:.2e}  energy_err={r['energy_err']:.2e}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
