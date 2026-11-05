"""Boundary curves and their discretization into straight constant elements."""
from dataclasses import dataclass

import numpy as np
import torch

DTYPE = torch.float64


@dataclass
class Elements:
    c: torch.Tensor       # (N, 2) element midpoints (source and collocation points)
    L: torch.Tensor       # (N,) element lengths
    t: torch.Tensor       # (N, 2) unit tangents
    n: torch.Tensor       # (N, 2) unit normals
    qp: torch.Tensor      # (N, q, 2) Gauss points
    qw: torch.Tensor      # (N, q) Gauss weights (including the Jacobian L/2)
    s: torch.Tensor       # (N,) curve coordinate of the midpoints
    kappa: torch.Tensor   # (N,) signed curvature (> 0 when the curve bends towards n)

    def __len__(self):
        return self.c.shape[0]


def gauss(q):
    x, w = np.polynomial.legendre.leggauss(q)
    return torch.tensor(x, dtype=DTYPE), torch.tensor(w, dtype=DTYPE)


def cosine_nodes(n_el):
    """Element end points clustered towards s = 0 and s = 1 (corners)."""
    j = torch.arange(n_el + 1, dtype=DTYPE) / n_el
    return 0.5 * (1 - torch.cos(np.pi * j))


def discretize(curve, n_el, normal="up", q=4, s_nodes=None):
    """Split curve(s) -> (x, y), s in [0, 1], into n_el straight elements.

    By default the nodes are cosine-clustered towards the curve ends.

    normal="up" rotates the tangent counter-clockwise (outward from the solid below a
    free surface traversed left to right); normal="down" rotates it clockwise.
    """
    s = cosine_nodes(n_el) if s_nodes is None else s_nodes
    x, y = curve(s)
    P = torch.stack([x, y], dim=-1)
    d = P[1:] - P[:-1]
    L = torch.linalg.norm(d, dim=-1)
    t = d / L[:, None]
    n = torch.stack([-t[:, 1], t[:, 0]], -1) if normal == "up" else torch.stack([t[:, 1], -t[:, 0]], -1)
    c = 0.5 * (P[1:] + P[:-1])
    gx, gw = gauss(q)
    qp = c[:, None, :] + 0.5 * L[:, None, None] * gx[None, :, None] * t[:, None, :]
    qw = 0.5 * L[:, None] * gw[None, :]
    return Elements(c, L, t, n, qp, qw, 0.5 * (s[1:] + s[:-1]), curvature(t, n, L))


def curvature(t, n, L, corner=0.5):
    """Signed curvature at element midpoints from the turning of neighbouring tangents.

    Nodes where the tangent turns by more than `corner` radians are geometric corners
    (e.g. the bottom of a V-shaped canyon); they are excluded and the estimate becomes
    one-sided there.
    """
    if len(L) < 3:
        return torch.zeros_like(L)
    tp = torch.cat([t[:1], t[:-1]], 0)
    tn = torch.cat([t[1:], t[-1:]], 0)
    Lp = torch.cat([L[:1] * 0, L[:-1]], 0)
    Ln = torch.cat([L[1:], L[-1:] * 0], 0)
    ds = 0.5 * (Lp + Ln) + L * torch.cat([L[:1] * 0 + 0.5, torch.ones_like(L[1:-1]), L[:1] * 0 + 0.5])
    kappa = ((tn - tp) * n).sum(-1) / ds
    turn = torch.arccos(torch.clamp((t[1:] * t[:-1]).sum(-1), -1.0, 1.0))   # at interior nodes
    if not bool((turn > corner).any()):
        return kappa
    # one-sided estimates, ignoring corner nodes
    left = ((t - tp) * n).sum(-1) / (0.5 * (Lp + L))
    right = ((tn - t) * n).sum(-1) / (0.5 * (L + Ln))
    okl = torch.cat([torch.zeros(1, dtype=torch.bool), turn <= corner])
    okr = torch.cat([turn <= corner, torch.zeros(1, dtype=torch.bool)])
    num = torch.where(okl, left, 0 * left) + torch.where(okr, right, 0 * right)
    den = okl.to(L.dtype) + okr.to(L.dtype)
    return torch.where(den > 0, num / torch.clamp(den, min=1.0), 0 * num)


# --- curve families --------------------------------------------------------------------

def semicircle(a=1.0, above=True):
    """Semicircle of radius a centred at the origin, traversed from x=-a to x=+a."""
    sign = 1.0 if above else -1.0

    def curve(s):
        return -a * torch.cos(np.pi * s), sign * a * torch.sin(np.pi * s)
    return curve


def flat(a=1.0, y=0.0):
    def curve(s):
        return a * (2 * s - 1), torch.full_like(s, y)
    return curve


def sine_series(xi, a=1.0, sign=1.0):
    """x = a(2s-1), y = sign * sum_k xi_k sin(k pi s); xi may require grad."""
    def curve(s):
        k = torch.arange(1, len(xi) + 1, dtype=DTYPE)
        y = sign * (torch.sin(np.pi * k[None, :] * s[:, None]) * xi[None, :]).sum(-1)
        return a * (2 * s - 1), y
    return curve


def v_canyon(b=1.0, d=1.0):
    def curve(s):
        x = b * (2 * s - 1)
        return x, -d * (1 - torch.abs(x) / b)
    return curve
