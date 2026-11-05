"""Three-dimensional scalar indirect BEM (Helmholtz) for irregularities in a half-space.

Scalar analogue of the SH formulation: mu (lap u + k^2 u) = 0 in the solid z < 0, traction-free
(Neumann) flat surface z = 0 handled by the half-space Green's function
    G = [exp(-ik r)/r + exp(-ik r')/r'] / (4 pi mu),
a traction-free irregular surface Gamma_s discretized with flat triangles (constant densities,
collocation at the centroids). Time factor exp(+i w t), z upward.
"""
import math

import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from .geometry import DTYPE

CDTYPE = torch.complex128

# --- triangle quadrature rules (barycentric coordinates, weights summing to 1) -------------
_Q3 = (np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]), np.full(3, 1 / 3))
_a1, _b1, _a2, _b2 = 0.059715871789770, 0.470142064105115, 0.797426985353087, 0.101286507323456
_Q7 = (np.array([[1 / 3, 1 / 3, 1 / 3], [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
                 [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2]]),
       np.array([0.225, *[0.132394152788506] * 3, *[0.125939180544827] * 3]))


def _subdivided(rule, levels=1):
    """Rule on the 4^levels congruent sub-triangles of the reference triangle."""
    bary, w = rule
    tris = [np.eye(3)]
    for _ in range(levels):
        new = []
        for T in tris:
            m01, m12, m20 = (T[0] + T[1]) / 2, (T[1] + T[2]) / 2, (T[2] + T[0]) / 2
            new += [np.array([T[0], m01, m20]), np.array([m01, T[1], m12]),
                    np.array([m20, m12, T[2]]), np.array([m01, m12, m20])]
        tris = new
    pts = np.concatenate([bary @ T for T in tris])
    return pts, np.concatenate([w / len(tris)] * len(tris))


_Q28 = _subdivided(_Q7, 1)
_Q12 = _subdivided(_Q3, 1)          # no point at the centroid (used for self terms)


# --- parametric surfaces and meshes ------------------------------------------------------

def disk_mesh(n_rings, grading=True):
    """Points in the unit disk on rings (graded towards the rim) and a ccw triangulation."""
    pts = [[0.0, 0.0]]
    for j in range(1, n_rings + 1):
        rho = math.sin(0.5 * math.pi * j / n_rings) if grading else j / n_rings
        n = 6 * j
        th = 2 * math.pi * (np.arange(n) + 0.5 * (j % 2)) / n
        pts += list(np.stack([rho * np.cos(th), rho * np.sin(th)], 1))
    P = np.array(pts)
    T = Delaunay(P).simplices
    a = P[T[:, 1]] - P[T[:, 0]]
    b = P[T[:, 2]] - P[T[:, 0]]
    neg = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0
    T[neg] = T[neg][:, [0, 2, 1]]
    return torch.tensor(P, dtype=DTYPE), torch.tensor(T)


def hemisphere(a=1.0):
    """Hemispherical canyon of radius a: polar angle psi = (pi/2) rho from the bottom."""
    def surf(P):
        rho = torch.linalg.norm(P, dim=-1)
        psi = 0.5 * math.pi * rho
        s = torch.where(rho > 0, torch.sin(psi) / torch.clamp(rho, min=1e-30),
                        torch.full_like(rho, 0.5 * math.pi))
        return torch.stack([a * s * P[:, 0], a * s * P[:, 1], -a * torch.cos(psi)], -1)
    return surf


def canyon_basis(P):
    """Shape functions psi_k(u, v), vanishing on the rim rho = 1 (K = 6)."""
    u, v = P[:, 0], P[:, 1]
    r2 = u * u + v * v
    w = 1 - r2
    return torch.stack([w, w * u, w * v, w * (u * u - v * v), w * 2 * u * v, w * r2], -1)


def canyon(xi, a=1.0):
    """Depression z = -sum_k xi_k psi_k over the disk of radius a; xi may require grad."""
    def surf(P):
        return torch.stack([a * P[:, 0], a * P[:, 1], -(canyon_basis(P) @ xi)], -1)
    return surf


class Mesh3D:
    def __init__(self, P, T, surf):
        X = surf(P)                                    # (Nv, 3)
        V = X[T]                                       # (N, 3, 3)
        e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
        cr = torch.cross(e1, e2, dim=-1)
        nrm = torch.linalg.norm(cr, dim=-1)
        self.V, self.T, self.uv = V, T, P[T].mean(1)  # uv: centroid in parameter space
        self.n = cr / nrm[:, None]                     # out of the solid (into the air)
        self.area = 0.5 * nrm
        self.c = V.mean(1)
        self.kappa = self._curvature()

    def _curvature(self):
        """Sum of principal curvatures (> 0 when the surface bends towards n), from the
        variation of the normal across the edge-adjacent triangles."""
        T = self.T.numpy()
        edges = {}
        for m, tri in enumerate(T):
            for a, b in ((0, 1), (1, 2), (2, 0)):
                edges.setdefault(tuple(sorted((tri[a], tri[b]))), []).append(m)
        pairs = np.array([p for p in edges.values() if len(p) == 2])
        i, j = torch.tensor(pairs[:, 0]), torch.tensor(pairs[:, 1])
        dc = self.c[j] - self.c[i]
        dn = self.n[j] - self.n[i]
        k = (dn * dc).sum(-1) / (dc * dc).sum(-1)      # directional curvature (sign below)
        tot = torch.zeros(len(T), dtype=DTYPE).index_add(0, i, k).index_add(0, j, k)
        cnt = torch.zeros(len(T), dtype=DTYPE).index_add(0, i, torch.ones_like(k)).index_add(
            0, j, torch.ones_like(k))
        return -2 * tot / torch.clamp(cnt, min=1)

    def __len__(self):
        return self.c.shape[0]

    def quad(self, rule):
        bary, w = (torch.tensor(r, dtype=DTYPE) for r in rule)
        Y = torch.einsum("qk,nkd->nqd", bary, self.V)
        return Y, self.area[:, None] * w[None, :]


def _flat_inv_r(V, x0):
    """Integral of 1/|x - x0| over the flat triangles V (N, 3, 3) for x0 (N, 3) in their plane."""
    out = 0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p1, p2 = V[:, a], V[:, b]
        L = torch.linalg.norm(p2 - p1, dim=-1)
        t = (p2 - p1) / L[:, None]
        t1 = ((p1 - x0) * t).sum(-1)
        t2 = ((p2 - x0) * t).sum(-1)
        d = torch.linalg.norm((p1 - x0) - t1[:, None] * t, dim=-1)
        out = out + d * (torch.asinh(t2 / d) - torch.asinh(t1 / d))
    return out


def _kernel(X, nX, Y, W, k, mu, chunk=None):
    """G and mu dG/dn_X (one source term) for field points X (M, 3) and sources Y (N, q, 3)."""
    d = X[:, None, None, :] - Y[None]
    r = torch.linalg.norm(d, dim=-1)
    e = torch.exp(-1j * k * r)
    G = (e / r * W[None]).sum(-1) / (4 * math.pi * mu)
    dn = (d * nX[:, None, None, :]).sum(-1)
    T = (-(e * (1 + 1j * k * r)) / r ** 3 * dn * W[None]).sum(-1) / (4 * math.pi)
    return G, T


def kernels(X, nX, mesh, k, mu, half=True, self_idx=True, sigma=1.0, near=2.5):
    """Integrated kernels G[m, l], T[m, l] (rows: field points, columns: triangles)."""
    Y3, W3 = mesh.quad(_Q3)
    images = [False, True] if half else [False]
    flip = torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE)
    h = torch.sqrt(mesh.area).detach()
    parts = []
    for img in images:
        Yr = Y3 * flip if img else Y3
        Gp, Tp = _kernel(X, nX, Yr, W3, k, mu)
        # refined quadrature (4 x 7 points) for near pairs, including the image of the self term
        cl = mesh.c * flip if img else mesh.c
        dist = torch.cdist(X.detach(), cl.detach())
        mi, li = torch.nonzero(dist < near * h[None, :], as_tuple=True)
        if self_idx and not img:
            keep = mi != li
            mi, li = mi[keep], li[keep]
        if len(mi):
            Yq, Wq = mesh.quad(_Q28)
            Yq = Yq * flip if img else Yq
            d = X[mi][:, None, :] - Yq[li]
            r = torch.linalg.norm(d, dim=-1)
            e = torch.exp(-1j * k * r)
            g = (e / r * Wq[li]).sum(-1) / (4 * math.pi * mu)
            t = (-(e * (1 + 1j * k * r)) / r ** 3 * (d * nX[mi][:, None, :]).sum(-1) * Wq[li]).sum(-1)
            Gp, Tp = Gp.clone(), Tp.clone()
            Gp[mi, li] = g
            Tp[mi, li] = t / (4 * math.pi)
        parts.append((Gp, Tp))
    if self_idx:
        # direct self term: analytic integral of 1/r plus the regular part (e^{-ikr} - 1)/r,
        # jump term and principal value over the curved element
        idx = torch.arange(len(mesh))
        Ys, Ws = mesh.quad(_Q12)
        r = torch.linalg.norm(mesh.c[:, None, :] - Ys, dim=-1)
        reg = ((torch.exp(-1j * k * r) - 1) / r * Ws).sum(-1)
        Gd, Td = parts[0][0].clone(), parts[0][1].clone()
        Gd[idx, idx] = (_flat_inv_r(mesh.V, mesh.c) + reg) / (4 * math.pi * mu)
        Td[idx, idx] = (0.5 * sigma + mesh.kappa * torch.sqrt(mesh.area / math.pi) / 8).to(Td.dtype)
        parts[0] = (Gd, Td)
    G = sum(p[0] for p in parts)
    T = sum(p[1] for p in parts)
    return G, T


def free_field(X, n, k, mu, alpha, phi):
    """Incident + reflected unit plane wave (direction angle alpha from the vertical,
    azimuth phi); returns u (M,) and traction mu du/dn (M,)."""
    sa, ca = math.sin(alpha), math.cos(alpha)
    kx, ky = k * sa * math.cos(phi), k * sa * math.sin(phi)
    ph = torch.exp(-1j * (kx * X[:, 0] + ky * X[:, 1]))
    u = 2 * ph * torch.cos(k * ca * X[:, 2])
    ux, uy = -1j * kx * u, -1j * ky * u
    uz = -2 * k * ca * ph * torch.sin(k * ca * X[:, 2])
    return u, mu * (ux * n[:, 0] + uy * n[:, 1] + uz * n[:, 2])


def assemble(mesh, eta, excitations, rec=None, mu=1.0):
    """Single-region system (empty canyon): T phi = -t_ff, rows scaled by 1 / (mu k).

    excitations: list of (alpha, phi); rec: receiver points (R, 3) on z = 0.
    eta = k a / pi with a = 1.
    """
    k = math.pi * eta
    G, T = kernels(mesh.c, mesh.n, mesh, k, mu)
    tf = torch.stack([free_field(mesh.c, mesh.n, k, mu, a, p)[1] for a, p in excitations], 1)
    out = dict(A=T / (mu * k), b=-tf / (mu * k), sizes=(len(mesh),))
    if rec is not None:
        GR, _ = kernels(rec, torch.zeros_like(rec), mesh, k, mu, self_idx=False)
        out["Rop"] = GR
        out["r0"] = torch.stack([free_field(rec, torch.zeros_like(rec), k, mu, a, p)[0]
                                 for a, p in excitations], 1)
    return out


# --- exact solution: rigid (Neumann) hemispherical canyon --------------------------------

def hemisphere_exact(X, k, alpha, phi, a=1.0, nmax=None):
    """Total field for a sound-hard hemispherical canyon (sphere + mirror image waves)."""
    X = np.asarray(X)
    nmax = nmax or int(k * a + 25)
    r = np.linalg.norm(X, axis=-1)
    u = np.zeros(len(X), complex)
    sa, ca = math.sin(alpha), math.cos(alpha)
    for dz in (ca, -ca):
        dvec = np.array([sa * math.cos(phi), sa * math.sin(phi), dz])
        cg = X @ dvec / r
        for n in range(nmax + 1):
            jn, jp = spherical_jn(n, k * r), spherical_jn(n, k * a, derivative=True)
            hn = spherical_jn(n, k * r) - 1j * spherical_yn(n, k * r)
            hp = spherical_jn(n, k * a, derivative=True) - 1j * spherical_yn(n, k * a, derivative=True)
            u += (2 * n + 1) * (-1j) ** n * (jn - jp / hp * hn) * eval_legendre(n, cg)
    return u
