"""Regression tests for adaptive_mesh_refinement.py (known-answer, deterministic)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import amr_convergence_study as study  # noqa: E402


def test_patches_tile_the_domain_exactly():
    """Every point counted once: uncovered cell areas add up to the domain area."""
    amr = study.build_amr(17, 3, reevaluate=True)
    area = sum(np.sum(amr._uncovered_cells(p)) * p.dx * p.dy for p in amr.patches)
    assert abs(area - 36.0) < 1e-9


def test_no_duplicate_children_after_repeated_adaptation():
    amr = study.build_amr(17, 3, reevaluate=True)
    for p in amr.patches:
        rects = {(c.x_range, c.y_range) for c in p.children}
        assert len(rects) == len(p.children)


def test_mass_is_conserved_under_refinement():
    """The old all-patch sum doubled the mass in refined regions."""
    amr = study.build_amr(17, 3, reevaluate=True)
    assert abs(amr.check_conservation()["total_mass"] - study.EXACT_MASS) / study.EXACT_MASS < 1e-3


def test_uniform_grid_is_second_order():
    rows = study.uniform_study([65, 129, 257])
    assert 1.8 < rows[-1]["energy_order"] < 2.2


def test_reevaluated_amr_beats_uniform_at_equal_dof():
    amr = study.build_amr(17, 3, reevaluate=True)
    dof, (_, energy_err) = amr.count_degrees_of_freedom(), study.errors(amr)
    uniform = {r["n"]: r for r in study.uniform_study([65, 129])}
    assert energy_err < uniform[65]["energy_err"]          # 8.5k dof vs 4.1k dof uniform
    assert dof < 1.2 * uniform[129]["dof"] and energy_err < 1.2 * uniform[129]["energy_err"]


def test_interpolation_only_refinement_gives_no_accuracy_gain():
    good = study.errors(study.build_amr(17, 3, reevaluate=True))[1]
    bad = study.errors(study.build_amr(17, 3, reevaluate=False))[1]
    assert bad > 10 * good


def test_extrapolation_reports_none_not_nan_when_not_monotone():
    amr = study.build_amr(17, 1, reevaluate=True)  # only two levels: no order available
    res = amr.extrapolate_to_continuum(amr.energy_density)
    assert res["observed_order"] is None and res["extrapolated"] is None
