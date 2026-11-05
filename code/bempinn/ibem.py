"""Indirect BEM (single-layer potentials) for SH scattering by irregularities.

Conventions: time factor exp(+i w t), y upward, half-space y < 0, a = beta_2 = mu_2 = 1.
Region 1 (irregularity): full-space Green's function, sources on Gamma_s and Gamma_I.
Region 2 (half-space):   half-space Green's function (image source), sources on Gamma_I
(or on Gamma_s when there is no region 1, e.g. an empty canyon).
Gamma_s carries the "up" normal (out of the solid), Gamma_I the "down" normal (out of
region 1, into region 2).
"""
import numpy as np
import torch

from .geometry import DTYPE
from .special import hankel2_0, hankel2_1

CDTYPE = torch.complex128
_US, _UW = np.polynomial.legendre.leggauss(10)
_US = torch.tensor(0.5 * (_US + 1), dtype=DTYPE)     # Gauss points on [0, 1]
_UW = torch.tensor(0.5 * _UW, dtype=DTYPE)


def wavenumbers(eta, mu1, rho1, mu2=1.0, rho2=1.0, a=1.0):
    """eta = w a / (pi beta_2)."""
    beta2 = (mu2 / rho2) ** 0.5
    w = np.pi * eta * beta2 / a
    k1 = w * torch.sqrt(torch.as_tensor(rho1, dtype=DTYPE) / torch.as_tensor(mu1, dtype=DTYPE))
    return k1, w / beta2


def _self_G(L, k, mu):
    """Integral of G over a straight element evaluated at its own midpoint."""
    tq = 0.5 * L[:, None] * _US[None, :] ** 2
    integrand = hankel2_0(k * tq) * (L[:, None] * _US[None, :])
    return 2 * (integrand * _UW[None, :]).sum(-1) / (4j * mu)


def kernels(X, nX, el, k, mu, half=False, self_pairs=None, sigma=1.0):
    """Integrated kernels G[m, l] = int G(X_m, y) dS and T[m, l] = mu dG/dn_X.

    self_pairs: (m_idx, l_idx) where X_m is the midpoint of element l; their entries
    are replaced by the analytic self terms (sigma * 1/2 jump for T).
    """
    def part(Y):
        d = X[:, None, None, :] - Y[None]
        r = torch.linalg.norm(d, dim=-1)
        kr = k * r
        G = (hankel2_0(kr) * el.qw[None]).sum(-1) / (4j * mu)
        dn = (d * nX[:, None, None, :]).sum(-1) / r
        T = (0.25j * k * hankel2_1(kr) * dn * el.qw[None]).sum(-1)
        return G, T

    G, T = part(el.qp)
    if self_pairs is not None:
        mi, li = self_pairs
        G = G.clone()
        T = T.clone()
        G[mi, li] = _self_G(el.L[li], k, mu)
        # jump term plus the principal value over the (curved) element itself
        T[mi, li] = (0.5 * sigma + el.kappa[li] * el.L[li] / (4 * np.pi)).to(T.dtype)
    if half:
        Yi = el.qp * torch.tensor([1.0, -1.0], dtype=DTYPE)
        Gi, Ti = part(Yi)
        if self_pairs is not None:
            # elements lying on y = 0 coincide with their image
            mi, li = self_pairs
            on0 = (el.c[li, 1].abs() < 1e-12) & (el.t[li, 1].abs() < 1e-12)
            if bool(on0.any()):
                Gi, Ti = Gi.clone(), Ti.clone()
                Gi[mi[on0], li[on0]] = _self_G(el.L[li[on0]], k, mu)
                Ti[mi[on0], li[on0]] = torch.full_like(Ti[mi[on0], li[on0]], 0.5 * sigma)
        G, T = G + Gi, T + Ti
    return G, T


def free_field(X, n, k, mu, alphas):
    """Incident + reflected unit plane SH wave; alpha measured from the vertical.

    Returns u (M, nA) and traction mu du/dn (M, nA).
    """
    al = torch.as_tensor(alphas, dtype=DTYPE)[None, :]
    x, y = X[:, :1], X[:, 1:]
    ph = torch.exp(-1j * k * x * torch.sin(al))
    u = 2 * ph * torch.cos(k * y * torch.cos(al))
    ux = -1j * k * torch.sin(al) * u
    uy = -2 * k * torch.cos(al) * ph * torch.sin(k * y * torch.cos(al))
    t = mu * (ux * n[:, :1] + uy * n[:, 1:])
    return u.to(CDTYPE), t.to(CDTYPE)


def assemble(Gs, GI, eta, alphas, mats, rec_x=None, rec_s=None, two_region=True):
    """Collocation system A phi = b for all incidence angles at one frequency.

    Gs, GI: Elements of Gamma_s and Gamma_I (GI unused when two_region=False).
    mats: (mu1, rho1, mu2, rho2). rec_x: receiver abscissae on Gamma_0 (y = 0).
    rec_s: indices of Gamma_s elements used as receivers.
    Returns dict with A (R, N), b (R, nA), Rop (nrec, N), r0 (nrec, nA).
    Traction rows are scaled by 1 / (mu2 k2).
    """
    mu1, rho1, mu2, rho2 = mats
    k1, k2 = wavenumbers(eta, mu1, rho1, mu2, rho2)
    ts = 1.0 / (mu2 * k2)
    out = {}
    if two_region:
        ns, nI = len(Gs), len(GI)
        el1 = _concat(Gs, GI)
        X1 = el1.c
        nX1 = el1.n
        idx = torch.arange(ns + nI)
        G1, T1 = kernels(X1, nX1, el1, k1, mu1, self_pairs=(idx, idx), sigma=1.0)
        iI = torch.arange(nI)
        G2, T2 = kernels(GI.c, GI.n, GI, k2, mu2, half=True, self_pairs=(iI, iI), sigma=-1.0)
        Z = torch.zeros(ns, nI, dtype=CDTYPE)
        A = torch.cat([
            torch.cat([T1[:ns] * ts, Z], 1),
            torch.cat([G1[ns:], -G2], 1),
            torch.cat([T1[ns:] * ts, -T2 * ts], 1),
        ], 0)
        uI, tI = free_field(GI.c, GI.n, k2, mu2, alphas)
        b = torch.cat([torch.zeros(ns, len(alphas), dtype=CDTYPE), uI, tI * ts], 0)
        rops, r0s = [], []
        if rec_x is not None:
            XR = torch.stack([torch.as_tensor(rec_x, dtype=DTYPE),
                              torch.zeros(len(rec_x), dtype=DTYPE)], -1)
            GR, _ = kernels(XR, torch.zeros_like(XR), GI, k2, mu2, half=True)
            rops.append(torch.cat([torch.zeros(len(rec_x), ns + nI, dtype=CDTYPE), GR], 1))
            r0s.append(free_field(XR, torch.zeros_like(XR), k2, mu2, alphas)[0])
        if rec_s is not None:
            rops.append(torch.cat([G1[rec_s], torch.zeros(len(rec_s), nI, dtype=CDTYPE)], 1))
            r0s.append(torch.zeros(len(rec_s), len(alphas), dtype=CDTYPE))
        out["sizes"] = (ns, nI, nI)
    else:
        ns = len(Gs)
        idx = torch.arange(ns)
        G2, T2 = kernels(Gs.c, Gs.n, Gs, k2, mu2, half=True, self_pairs=(idx, idx), sigma=1.0)
        A = T2 * ts
        us, tsf = free_field(Gs.c, Gs.n, k2, mu2, alphas)
        b = -tsf * ts
        rops, r0s = [], []
        if rec_x is not None:
            XR = torch.stack([torch.as_tensor(rec_x, dtype=DTYPE),
                              torch.zeros(len(rec_x), dtype=DTYPE)], -1)
            GR, _ = kernels(XR, torch.zeros_like(XR), Gs, k2, mu2, half=True)
            rops.append(GR)
            r0s.append(free_field(XR, torch.zeros_like(XR), k2, mu2, alphas)[0])
        if rec_s is not None:
            rops.append(G2[rec_s])
            r0s.append(us[rec_s])
        out["sizes"] = (ns,)
    out.update(A=A, b=b)
    if rops:
        out["Rop"] = torch.cat(rops, 0)
        out["r0"] = torch.cat(r0s, 0)
    return out


def solve(sys):
    phi = torch.linalg.solve(sys["A"], sys["b"])
    u = sys["Rop"] @ phi + sys["r0"] if "Rop" in sys else None
    return phi, u


def _concat(e1, e2):
    from .geometry import Elements
    return Elements(*(torch.cat([getattr(e1, f), getattr(e2, f)], 0)
                      for f in ("c", "L", "t", "n", "qp", "qw", "s", "kappa")))


def assemble_truncated(Gs, eta, alphas, half_width, flat_len, n_flat, q=4):
    """Formulation of Wu et al. (2026): full-space Green's function and a meshed flat
    surface of length flat_len on each side of the topography (|x| <= half_width).

    Every element midpoint is a receiver. Returns the same dict as assemble() plus
    'xr' (receiver coordinates) and 'on_topo' (mask of topography elements).
    """
    from .geometry import discretize, flat as flat_curve, Elements
    mu, k = 1.0, np.pi * eta

    def line(x0, x1, n):
        def curve(s):
            return x0 + (x1 - x0) * s, torch.zeros_like(s)
        return discretize(curve, n, "up", q=q, s_nodes=torch.linspace(0, 1, n + 1, dtype=DTYPE))

    left = line(-half_width - flat_len, -half_width, n_flat)
    right = line(half_width, half_width + flat_len, n_flat)
    el = Elements(*(torch.cat([getattr(left, f), getattr(Gs, f), getattr(right, f)], 0)
                    for f in ("c", "L", "t", "n", "qp", "qw", "s", "kappa")))
    idx = torch.arange(len(el))
    G, T = kernels(el.c, el.n, el, k, mu, self_pairs=(idx, idx), sigma=1.0)
    uf, tf = free_field(el.c, el.n, k, mu, alphas)
    on_topo = torch.zeros(len(el), dtype=torch.bool)
    on_topo[n_flat:n_flat + len(Gs)] = True
    return dict(A=T / k, b=-tf / k, Rop=G, r0=uf, xr=el.c, on_topo=on_topo, sizes=(len(el),))
