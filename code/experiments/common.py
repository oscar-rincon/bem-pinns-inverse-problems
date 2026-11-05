"""Shared set-ups for the experiments."""
import json
import os

import numpy as np
import torch

from bempinn import analytic, geometry as g, ibem

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)


def save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


# --- forward benchmarks of Wu et al. (2026) ---------------------------------------------

TOPOS = {
    # name: (two_region, Gamma_s curve, Gamma_I curve)
    "canyon": (False, g.semicircle(1.0, above=False), None),
    "mountain": (True, g.semicircle(1.0, above=True), g.flat(1.0)),
    "vcanyon": (False, g.v_canyon(1.0, 1.0), None),
}
X0 = np.linspace(-3, 3, 241)
X0 = X0[np.abs(X0) > 1.0 + 1e-9]          # receivers on Gamma_0


TRUNCATED = {"vcanyon"}      # meshed flat surface + full-space G (Wu et al. 2026)


def truncated_setup(topo, n_el, eta, alphas, n_flat=240, n_lam=8, flat_len=None):
    """Wu et al. formulation; rows reordered as [flat |x| <= 3 (sorted), topography].

    The flat surface extends n_lam wavelengths on each side unless flat_len is given.
    """
    Gs = g.discretize(TOPOS[topo][1], n_el, "up")
    S = ibem.assemble_truncated(Gs, eta, alphas, 1.0, flat_len or n_lam * 2.0 / eta, n_flat)
    xr = S["xr"][:, 0]
    flat = (~S["on_topo"]) & (xr.abs() <= 3.0)
    idx = torch.cat([torch.nonzero(flat)[:, 0], torch.nonzero(S["on_topo"])[:, 0]])
    S["Rop"], S["r0"] = S["Rop"][idx], S["r0"][idx]
    S["x0"] = xr[flat].numpy()
    S["zin"] = S["xr"]                       # network input: source coordinates (x0, y0)
    return Gs, None, S


def forward_setup(topo, n_el, eta, alphas):
    if topo in TRUNCATED:
        return truncated_setup(topo, n_el, eta, alphas)
    two, cs, ci = TOPOS[topo]
    Gs = g.discretize(cs, n_el, "up")
    GI = g.discretize(ci, n_el, "down") if two else None
    S = ibem.assemble(Gs, GI, eta, alphas, (1.0, 1.0, 1.0, 1.0), rec_x=X0,
                      rec_s=torch.arange(n_el), two_region=two)
    return Gs, GI, S


def reference(topo, eta, alphas, n_ref=1200):
    """Reference U on Gamma_0 (abscissae x0) and as a function of s on Gamma_s."""
    if topo in TRUNCATED:
        Gs, _, S = truncated_setup(topo, 400, eta, alphas, n_flat=960, n_lam=16)
        _, u = ibem.solve(S)
        u = u.numpy()
        xf, n0, s_ref = S["x0"], len(S["x0"]), Gs.s.numpy()

        def interp(xq, xp, fp):
            return np.interp(xq, xp, fp.real) + 1j * np.interp(xq, xp, fp.imag)

        def ref(s, x0=X0):
            return np.stack([np.concatenate([interp(x0, xf, u[:n0, j]), interp(s, s_ref, u[n0:, j])])
                             for j in range(len(alphas))], 1)
        return ref
    if topo == "canyon":
        k = np.pi * eta

        def ref(s, x0=X0):
            th = -np.pi + np.pi * np.asarray(s)       # semicircle below: angle from -pi to 0
            xs, ys = np.cos(th), np.sin(th)
            return np.stack([np.concatenate([analytic.canyon(x0, 0 * x0, k, a),
                                             analytic.canyon(xs, ys, k, a)]) for a in alphas], 1)
        return ref
    Gs, GI, S = forward_setup(topo, n_ref, eta, alphas)
    _, u = ibem.solve(S)
    u = u.numpy()
    n0 = len(X0)
    s_ref = Gs.s.numpy()

    def ref(s, x0=X0):
        us = np.stack([np.interp(s, s_ref, u[n0:, j].real) + 1j * np.interp(s, s_ref, u[n0:, j].imag)
                       for j in range(len(alphas))], 1)
        return np.concatenate([u[:n0], us], 0)
    return ref


def wu_metrics(u, uref):
    """Wu et al. (2026) Eq. 33 (mean pointwise relative error of |U|) and R^2."""
    a, b = np.abs(u), np.abs(uref)
    eps = np.mean(np.abs(a - b) / b)
    r2 = 1 - np.sum((a - b) ** 2) / np.sum((b - b.mean()) ** 2)
    return float(eps), float(r2)
