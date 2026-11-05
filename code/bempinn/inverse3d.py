"""3D inverse problem: reconstruction of the free surface of a canyon (scalar model).

The case object exposes the interface used by the classical solvers of inverse.py
(unknown, init_vec, true_vec, bounds, unpack, predict, regularization, reg_residual,
shape_error), and bem_pinn3d implements the hybrid method.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import bem3d, pinn
from .geometry import DTYPE

DEG = torch.tensor([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], dtype=DTYPE)   # polynomial degree + 1


@dataclass
class Case3D:
    name: str = "surface3d"
    is3d: bool = True
    xi: list = field(default_factory=lambda: [0.55, 0.12, -0.08, 0.06, -0.04, 0.10])
    xi0: list = field(default_factory=lambda: [0.30, 0.0, 0.0, 0.0, 0.0, 0.0])
    etas: tuple = (0.25, 0.5, 0.75)
    excitations: tuple = ((0.0, 0.0), (math.radians(30), 0.0), (math.radians(30), math.pi / 2))
    n_rings: int = 8               # inversion mesh (384 triangles)
    n_fine: int = 16               # data mesh (1536 triangles)
    rec_r: tuple = (1.25, 1.75, 2.5)
    n_az: int = 12
    lam_r: float = 1e-5
    unknown: tuple = ("xi",)

    @property
    def alphas_deg(self):           # number of excitations per frequency (used for counting)
        return tuple(range(len(self.excitations)))

    @property
    def n_el(self):
        return self.n_rings

    @n_el.setter
    def n_el(self, v):
        self.n_rings = v

    def receivers(self):
        th = np.linspace(0, 2 * np.pi, self.n_az, endpoint=False)
        R = [[r * np.cos(t), r * np.sin(t), 0.0] for r in self.rec_r for t in th]
        return torch.tensor(R, dtype=DTYPE)

    def true_vec(self):
        return np.array(self.xi, float)

    def init_vec(self):
        return np.array(self.xi0, float)

    def bounds(self):
        return [(0.05, 1.5)] + [(-0.3, 0.3)] * (len(self.xi) - 1)

    def unpack(self, v):
        return dict(xi=torch.as_tensor(v, dtype=DTYPE))

    def mesh(self, prm, n_rings=None):
        P, T = bem3d.disk_mesh(n_rings or self.n_rings)
        return bem3d.Mesh3D(P, T, bem3d.canyon(prm["xi"]))

    def systems(self, prm, n_rings=None, etas=None):
        m = self.mesh(prm, n_rings)
        R = self.receivers()
        return m, [bem3d.assemble(m, e, list(self.excitations), rec=R) for e in (etas or self.etas)]

    def predict(self, prm, n_el, solver=None):
        _, S = self.systems(prm, n_el)
        out = []
        for i, s in enumerate(S):
            phi = torch.linalg.solve(s["A"], s["b"]) if solver is None else solver(i, s["A"], s["b"])
            out.append(s["Rop"] @ phi + s["r0"])
        return torch.stack(out, 0)

    def regularization(self, prm):
        return (DEG ** 2 * prm["xi"] ** 2).sum()

    def reg_residual(self, prm):
        return DEG * prm["xi"]

    def shape_error(self, v):
        P, _ = bem3d.disk_mesh(30)
        B = bem3d.canyon_basis(P)
        z = B @ torch.as_tensor(v, dtype=DTYPE)
        zt = B @ torch.tensor(self.xi, dtype=DTYPE)
        return {"shape": float(torch.linalg.norm(z - zt) / torch.linalg.norm(zt))}


def make_data(case, noise=0.0, seed=0):
    with torch.no_grad():
        u = case.predict(case.unpack(case.true_vec()), case.n_fine)
    if noise > 0:
        rng = np.random.default_rng(seed)
        e = (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / math.sqrt(2)
        u = u * (1 + noise * torch.tensor(e))
    return u


def bem_pinn3d(case, data, n_pre=1000, n_pre_lbfgs=100, n_joint=1000, n_lbfgs=100, width=150,
               depth=4, m=64, sigma=1.0, lr=1e-3, lr_v=3e-3, lam_d=1.0, seed=0, log=print):
    """Hybrid inversion in 3D: the network maps the centroid parameter coordinates (u, v) of
    each triangle to the densities of every excitation (one output head per excitation)."""
    torch.manual_seed(seed)
    v = torch.tensor(case.init_vec(), dtype=DTYPE, requires_grad=True)
    lo, hi = (torch.tensor(b, dtype=DTYPE) for b in np.array(case.bounds()).T)

    def project():
        with torch.no_grad():
            v.copy_(torch.maximum(torch.minimum(v, hi), lo))

    etas, na = list(case.etas), len(case.excitations)
    C = len(etas) * na
    m0, S0 = case.systems(case.unpack(v.detach()))
    N = len(m0)
    zin = 0.5 * (m0.uv + 1)                      # parameter coordinates mapped to [0, 1]^2
    net = pinn.FourierNet(2, 2 * C, m=m, sigma=sigma, width=width, depth=depth, seed=seed)
    scale = pinn.jacobi_scale(S0)
    t0 = time.perf_counter()
    hist, cnt = [], {"assemblies": 1, "evals": 0, "matvec": 0, "rmatvec": 0}

    def phi_all():
        o = net(zin).reshape(N, C, 2)
        P = torch.complex(o[..., 0], o[..., 1]) * scale
        return [P[:, i * na:(i + 1) * na] for i in range(len(etas))]

    def record(it, stage, L, Lbc, Ld):
        rec = dict(it=it, stage=stage, t=time.perf_counter() - t0, loss=float(L), bc=float(Lbc),
                   data=float(Ld), **cnt, **case.shape_error(v.detach().numpy()))
        hist.append(rec)
        if log:
            log({k: (round(x, 6) if isinstance(x, float) else x) for k, x in rec.items()})

    def lbfgs(params, fun, n, stage, it0):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-14,
                                tolerance_change=1e-16)
        st = {}

        def closure():
            opt.zero_grad()
            L, Lbc, Ld = fun()
            L.backward()
            st.update(L=L.item(), bc=Lbc.item(), d=Ld.item())
            return L

        for j in range(n):
            opt.step(closure)
            project()
            if j % 10 == 0 or j == n - 1:
                record(it0 + j, stage, st["L"], st["bc"], st["d"])
        return st

    def loss_pre():
        cnt["evals"] += 1
        cnt["matvec"] += C
        cnt["rmatvec"] += C
        L = pinn.bc_loss(S0, phi_all())
        return L, L, torch.tensor(float("nan"))

    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for it in range(n_pre):
        opt.zero_grad()
        L, _, _ = loss_pre()
        L.backward()
        opt.step()
        if it % 250 == 0:
            record(it, 1, L.item(), L.item(), float("nan"))
    lbfgs(list(net.parameters()), loss_pre, n_pre_lbfgs, 1, n_pre)

    def loss_joint(n_active=len(etas)):
        cnt["assemblies"] += 1
        cnt["evals"] += 1
        cnt["matvec"] += n_active * na
        cnt["rmatvec"] += n_active * na
        prm = case.unpack(v)
        _, S = case.systems(prm, etas=etas[:n_active])
        phis = phi_all()[:n_active]
        Lbc = pinn.bc_loss(S, phis)
        u = torch.stack([St["Rop"] @ p + St["r0"] for St, p in zip(S, phis)], 0)
        d = data[:n_active]
        Ld = ((u - d).abs() ** 2).sum() / (d.abs() ** 2).sum()
        return Lbc + lam_d * Ld + case.lam_r * case.regularization(prm), Lbc, Ld

    it0 = n_pre + n_pre_lbfgs
    opt = torch.optim.AdamW([dict(params=net.parameters(), lr=lr),
                             dict(params=[v], lr=lr_v, weight_decay=0.0)])
    for it in range(n_joint):
        n_active = min(len(etas), 2 + (it * (len(etas) - 1)) // max(n_joint // 2, 1))
        opt.zero_grad()
        L, Lbc, Ld = loss_joint(n_active)
        L.backward()
        opt.step()
        project()
        if it % 100 == 0 or it == n_joint - 1:
            record(it0 + it, 2, L.item(), Lbc.item(), Ld.item())
    st = lbfgs(list(net.parameters()) + [v], loss_joint, n_lbfgs, 3, it0 + n_joint)
    return dict(v=v.detach().numpy(), time=time.perf_counter() - t0, hist=hist,
                J=st["d"], bc=st["bc"], **cnt)
