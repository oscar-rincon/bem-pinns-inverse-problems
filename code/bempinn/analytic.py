"""Exact solutions for SH waves: semicircular canyon (Trifunac 1973) and semicircular
alluvial valley (Trifunac 1971), obtained as full-space cylinder problems by symmetry.

Time factor exp(+i w t); y upward; alpha from the vertical; a = 1.
"""
import numpy as np
from scipy.special import hankel2, h2vp, jv, jvp


def _plane_dirs(alpha):
    # incident direction (sin a, cos a) and its mirror image (sin a, -cos a)
    return [np.pi / 2 - alpha, -(np.pi / 2 - alpha)]


def canyon(x, y, k, alpha, a=1.0, nmax=None):
    """Total displacement outside a traction-free semicircular canyon of radius a."""
    nmax = nmax or int(k * a + 30)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    n = np.arange(-nmax, nmax + 1)[:, None]
    u = np.zeros(np.shape(x), complex)
    for psi in _plane_dirs(alpha):
        inc = (-1j) ** n * jv(n, k * r) * np.exp(1j * n * (th - psi))
        c = -(-1j) ** n * jvp(n, k * a) / h2vp(n, k * a)
        u += (inc + c * hankel2(n, k * r) * np.exp(1j * n * (th - psi))).sum(0)
    return u


def valley(x, y, k1, k2, mu1, mu2, alpha, a=1.0, nmax=None):
    """Total displacement for a semicircular inclusion (mu1, k1) of radius a."""
    nmax = nmax or int(max(k1, k2) * a + 30)
    r, th = np.hypot(x, y), np.arctan2(y, x)
    n = np.arange(-nmax, nmax + 1)[:, None]
    inside = r < a
    u = np.zeros(np.shape(x), complex)
    for psi in _plane_dirs(alpha):
        f = (-1j) ** n
        # f J(k2 a) + c H(k2 a) = b J(k1 a);  mu2 k2 (f J' + c H') = mu1 k1 b J'
        M11, M12 = hankel2(n, k2 * a), -jv(n, k1 * a)
        M21, M22 = mu2 * k2 * h2vp(n, k2 * a), -mu1 * k1 * jvp(n, k1 * a)
        r1, r2 = -f * jv(n, k2 * a), -mu2 * k2 * f * jvp(n, k2 * a)
        det = M11 * M22 - M12 * M21
        c = (r1 * M22 - M12 * r2) / det
        b = (M11 * r2 - M21 * r1) / det
        e = np.exp(1j * n * (th - psi))
        out = (f * jv(n, k2 * r) + c * hankel2(n, k2 * r)) * e
        ins = b * jv(n, k1 * r) * e
        u += np.where(inside, ins.sum(0), out.sum(0))
    return u
