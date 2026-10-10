#!/usr/bin/env python3
"""
Non-vacuous constraint-closure test for spherically symmetric (radially
inhomogeneous) polymer-modified gravity.

Why this exists
---------------
The older closure checks in this repo (constraint_closure_tests.py,
constraint_anomaly_scanner.py, unified_qg/constraint_closure.py) cannot fail:
they evaluate {H,H} (zero by antisymmetry) or build H[N] and H[M] as multiples
of one matrix (they always commute).  This script instead computes the full
lattice Poisson-bracket matrix of the Hamiltonian-constraint densities,

    A_ij = { h_i , h_j },      H[N] = sum_i N_i h_i,

and asks whether it has the form the hypersurface-deformation algebra requires,

    { H[N], H[M] } = D[ W (N M' - N' M) ],

for *some* structure function W(x).  Classically W = E^x / (E^phi)^2.  Closure
holds (possibly with a deformed W, e.g. W = beta * E^x/(E^phi)^2) if the
least-squares fit of A to that form leaves a negligible residual; whatever the
fit cannot explain is the anomaly.

Model: Bojowald-Brahma-Reyes spherically symmetric variables (E^x, E^phi, K_x,
K_phi), brackets {K_x,E^x} = 2G, {K_phi,E^phi} = G.  Polymer modification:
K -> sin(delta K)/delta with delta = gamma * mu (holonomy of A = gamma K).

Limits (stated so results are not over-read): the fields are inhomogeneous in
the radial direction only.  Angular (non-spherical) perturbations are NOT
covered by this model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
G_NEWTON = 1.0


# --------------------------------------------------------------------------
# lattice helpers (periodic, central differences)
# --------------------------------------------------------------------------
def ddx(f, a):
    return (torch.roll(f, -1) - torch.roll(f, 1)) / (2 * a)


def make_modifier(kind, delta):
    """Return (kphi_fn, kx_fn) applied to K inside H.  kind in
    {'classical','kphi','kphi_kx','broken'}."""
    ident = lambda k: k
    if kind in ("classical", "broken") or delta == 0:
        return ident, ident
    poly = lambda k: torch.sin(delta * k) / delta
    if kind == "kphi":
        return poly, ident
    if kind == "kphi_kx":
        return poly, poly
    raise ValueError(kind)


def h_density(x, a, kind="classical", delta=0.0):
    """Hamiltonian-constraint density times a (so H[N] = sum N_i h_i).
    x has shape (4, n): rows Kx, Ex, Kphi, Ephi."""
    Kx, Ex, Kp, Ep = x
    kp_fn, kx_fn = make_modifier(kind, delta)
    Kp_m, Kx_m = kp_fn(Kp), kx_fn(Kx)
    sq = torch.sqrt(Ex)
    Exp = ddx(Ex, a)
    Gam = -Exp / (2 * Ep)
    dens = (
        Ep * Kp_m**2 / sq
        + 2 * sq * Kp_m * Kx_m
        + (1 - Gam**2) * Ep / sq
        + (0.0 if kind == "broken" else 2 * sq * ddx(Gam, a))
    )
    return -a * dens / (2 * G_NEWTON)


def d_density(x, a):
    """Diffeomorphism-constraint density times a."""
    Kx, Ex, Kp, Ep = x
    # the 1/2 on the K_x term makes D generate diffeos given {K_x,E^x} = 2G
    return a * (Ep * ddx(Kp, a) - 0.5 * Kx * ddx(Ex, a)) / G_NEWTON


def poisson_matrix(h_fn, x, a):
    """A_ij = {h_i, h_j} using the lattice Poisson tensor."""
    n = x.shape[1]
    J = torch.autograd.functional.jacobian(lambda y: h_fn(y.reshape(4, n)), x.reshape(-1))
    J = J.reshape(n, 4, n)  # J[i, field, site]
    dKx, dEx, dKp, dEp = J[:, 0], J[:, 1], J[:, 2], J[:, 3]
    # {F,G} = (1/a) sum [2G(dF/dKx dG/dEx - dF/dEx dG/dKx) + G(dF/dKp dG/dEp - dF/dEp dG/dKp)]
    A = (2 * G_NEWTON / a) * (dKx @ dEx.T - dEx @ dKx.T) + (G_NEWTON / a) * (dKp @ dEp.T - dEp @ dKp.T)
    return A


def fourier_basis(n, a, max_mode):
    """Smooth periodic functions on the lattice: 1, cos(2 pi m x/L), sin(2 pi m x/L)."""
    xs = np.arange(n) * a
    L = n * a
    cols = [np.ones(n)]
    for m in range(1, max_mode + 1):
        cols += [np.cos(2 * np.pi * m * xs / L), np.sin(2 * np.pi * m * xs / L)]
    return torch.tensor(np.stack(cols))  # (n_basis, n)


def d_bilinear(x, a, W, Np, Nq):
    """D[W (N M' - N' M)] with N=Np, M=Nq (lattice central differences)."""
    d = d_density(x, a) / a
    return float(torch.sum(a * W * d * (Np * ddx(Nq, a) - ddx(Np, a) * Nq)))


def closure_defect(x, a, kind="classical", delta=0.0, lapse_modes=3):
    """Compare {H[N],H[M]} with D[W (N M' - N' M)] on smooth lapse functions N, M.

    The structure function W is NOT fitted: a free smooth W can absorb almost any
    local term (it even 'closes' a deliberately broken H), so a fit proves
    nothing.  Instead each explicit hypothesis is tested:

      undeformed : W = E^x/(E^phi)^2                      (classical algebra)
      cos2       : W = cos(2 delta K_phi) E^x/(E^phi)^2   (known polymer deformation)

    Returns the relative residual ||lhs - rhs|| / ||lhs|| for each."""
    n = x.shape[1]
    A = poisson_matrix(lambda y: h_density(y, a, kind, delta), x, a)
    Ns = fourier_basis(n, a, lapse_modes)
    nb = Ns.shape[0]
    pairs = [(p, q) for p in range(nb) for q in range(p + 1, nb)]
    lhs = np.array([float(Ns[p] @ A @ Ns[q]) for p, q in pairs])
    W_cl = x[1] / x[3] ** 2
    hyps = {"undeformed": W_cl, "cos2": torch.cos(2 * delta * x[2]) * W_cl}
    nA = np.linalg.norm(lhs)
    out = {"norm_A": float(nA)}
    for name, W in hyps.items():
        rhs = np.array([d_bilinear(x, a, W, Ns[p], Ns[q]) for p, q in pairs])
        out[name] = float(np.linalg.norm(lhs - rhs) / nA) if nA > 0 else 0.0
    return out


def random_state(n, a, amp, kamp, rng):
    """Smooth, radially inhomogeneous, strictly positive E's. amp scales the
    relative E perturbation, kamp the amplitude of K (mu*K ~ O(1) is 'extreme')."""
    xs = np.arange(n) * a
    L = n * a

    def field(base, rel, modes=(1, 2, 3)):
        f = np.ones(n)
        for m in modes:
            f += rel / m * (rng.normal() * np.cos(2 * np.pi * m * xs / L)
                            + rng.normal() * np.sin(2 * np.pi * m * xs / L))
        return base * np.clip(f, 0.2, None)

    Ex = field(1.0, amp)
    Ep = field(1.0, amp)
    Kx = kamp * (field(1.0, 0.5) - 1.0 + rng.normal() * 0.5)
    Kp = kamp * (field(1.0, 0.5) - 1.0 + rng.normal() * 0.5)
    return torch.tensor(np.stack([Kx, Ex, Kp, Ep]))


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------
def selftest(n_list=(32, 64, 128), seed=0):
    """Validate the checker: the classical constraint must close against the
    classical algebra with error -> 0 as the lattice is refined, and a
    deliberately broken constraint (Gamma' term dropped) must not."""
    out = []
    for n in n_list:
        rng = np.random.default_rng(seed)
        a = 1.0 / n
        x = random_state(n, a, amp=0.3, kamp=0.3, rng=rng)
        c = closure_defect(x, a, "classical", 0.0)
        b = closure_defect(x, a, "broken", 0.0)
        out.append({"n": n, "classical": c["undeformed"], "broken": b["undeformed"]})
    return out


def scan(gammas, mus, kinds, n, amps, kamps, n_states, seed):
    rows = []
    a = 1.0 / n
    for kind in kinds:
        for gamma in gammas:
            for mu in mus:
                delta = gamma * mu
                for amp in amps:
                    for kamp in kamps:
                        und, cos2 = [], []
                        for s_ in range(n_states):
                            x = random_state(n, a, amp, kamp, np.random.default_rng(seed + s_))
                            r = closure_defect(x, a, kind, delta)
                            und.append(r["undeformed"])
                            cos2.append(r["cos2"])
                        rows.append({
                            "kind": kind, "gamma": gamma, "mu": mu, "delta": delta,
                            "E_amp": amp, "K_amp": kamp,
                            "defect_undeformed_max": max(und),
                            "defect_cos2_max": max(cos2),
                        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--states", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/constraint_tests/closure_stress_test.json")
    ap.add_argument("--selftest-only", action="store_true")
    args = ap.parse_args()

    print("== self-test (validates the checker itself) ==")
    st = selftest()
    for r in st:
        print(f"  n={r['n']:3d}  classical defect={r['classical']:.2e}  "
              f"broken-H defect={r['broken']:.2e}")
    ok = (st[-1]["classical"] < st[0]["classical"] / 8   # ~a^2 convergence
          and st[-1]["classical"] < 1e-2 and st[-1]["broken"] > 100 * st[-1]["classical"])
    print("  checker validated:", ok)
    if args.selftest_only or not ok:
        return 0 if ok else 1

    gammas = [0.2377]
    mus = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = scan(gammas, mus, ["classical", "kphi", "kphi_kx"], args.n,
                amps=[0.1, 0.5, 0.9], kamps=[0.5, 2.0, 5.0, 10.0],
                n_states=args.states, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"selftest": st, "scan": rows, "n": args.n,
               "note": "radially inhomogeneous spherical symmetry only; no angular perturbations"},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
