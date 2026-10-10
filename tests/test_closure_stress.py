"""Tests that the closure checker in scripts/closure_stress_test.py can both pass and fail."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import closure_stress_test as cst  # noqa: E402


def _defect(n, kind, delta, amp=0.3, kamp=0.3, seed=0):
    a = 1.0 / n
    x = cst.random_state(n, a, amp, kamp, np.random.default_rng(seed))
    return cst.closure_defect(x, a, kind, delta)


def test_classical_algebra_closes_with_second_order_convergence():
    d32, d64 = _defect(32, "classical", 0.0)["undeformed"], _defect(64, "classical", 0.0)["undeformed"]
    assert d64 < 2e-2
    assert d64 < d32 / 3  # ~ a^2


def test_checker_detects_broken_constraint():
    good = _defect(64, "classical", 0.0)["undeformed"]
    bad = _defect(64, "broken", 0.0)["undeformed"]
    assert bad > 100 * good


def test_naive_polymer_anomaly_survives_refinement():
    """The excess over the classical defect must not vanish as the lattice is refined."""
    delta = 0.2377 * 0.5
    excess = []
    for n in (64, 128):
        p = _defect(n, "kphi", delta, kamp=10.0)["undeformed"]
        c = _defect(n, "classical", 0.0, kamp=10.0)["undeformed"]
        excess.append(p - c)
    assert excess[1] > 0.5 * excess[0] > 0


def test_small_polymer_parameter_recovers_classical():
    assert _defect(64, "kphi", 1e-4)["undeformed"] == pytest.approx(
        _defect(64, "classical", 0.0)["undeformed"], rel=1e-2)
