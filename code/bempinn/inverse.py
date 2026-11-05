"""Inverse problems: interface identification, surface reconstruction, joint identification.

Unknowns (see methodology): sine coefficients xi_s, xi_I and log(mu1), log(rho1).
Three solvers share the same discretization, data and objective:
  lm_fd     Levenberg-Marquardt, central finite-difference Jacobian (Bignardi et al. 2012 type)
  lbfgs_adj L-BFGS with exact gradients by reverse-mode differentiation through the BEM
            solve, i.e. one forward + one adjoint solve (Bonnet & Guzina 2009 type)
  bem_pinn  joint training of the density network and the unknowns (this work)
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from . import geometry as g, ibem, pinn
from .geometry import DTYPE

CDTYPE = torch.complex128


@dataclass
class Case:
    name: str                       # "interface" | "surface" | "joint"
    xi_s: list = field(default_factory=lambda: [0.70, 0.10, -0.06, 0.04])
    xi_I: list = field(default_factory=lambda: [0.50, 0.15, 0.08, -0.03])
    mu1: float = 0.5
    rho1: float = 0.8
    xi_s0: list = field(default_factory=lambda: [0.35, 0.0, 0.0, 0.0])
    xi_I0: list = field(default_factory=lambda: [0.25, 0.0, 0.0, 0.0])
    mu10: float = 1.0
    rho10: float = 1.0
    etas: tuple = (0.25, 0.5, 0.75, 1.0)
    alphas_deg: tuple = (-30.0, 0.0, 30.0)
    n_el: int = 40
    n_fine: int = 120                # data mesh (3x refinement, no inverse crime)
    rec_x: tuple = tuple(np.concatenate([-np.linspace(3.0, 1.15, 10), np.linspace(1.15, 3.0, 10)]))
    n_rec_s: int = 9                 # receivers on Gamma_s (interface case only)
    lam_r: float = 1e-5
    p_reg: int = 1
    data_kind: str = "complex"       # "complex" displacements or "amplitude" |u| only

    @property
    def alphas(self):
        return [math.radians(a) for a in self.alphas_deg]

    @property
    def unknown(self):
        return {"interface": ("xi_I",), "surface": ("xi_s",),
                "joint": ("xi_s", "log_mu1", "log_rho1")}[self.name]

    def true_vec(self):
        return self.pack(dict(xi_s=self.xi_s, xi_I=self.xi_I, log_mu1=math.log(self.mu1),
                              log_rho1=math.log(self.rho1)))

    def init_vec(self):
        return self.pack(dict(xi_s=self.xi_s0, xi_I=self.xi_I0, log_mu1=math.log(self.mu10),
                              log_rho1=math.log(self.rho10)))

    def pack(self, d):
        return np.concatenate([np.atleast_1d(np.asarray(d[k], float)) for k in self.unknown])

    def unpack(self, v):
        """Unknown vector (tensor) -> full parameter dict of tensors."""
        out = dict(xi_s=torch.tensor(self.xi_s, dtype=DTYPE), xi_I=torch.tensor(self.xi_I, dtype=DTYPE),
                   log_mu1=torch.tensor(math.log(self.mu1), dtype=DTYPE),
                   log_rho1=torch.tensor(math.log(self.rho1), dtype=DTYPE))
        i = 0
        for k in self.unknown:
            n = len(self.xi_s) if k.startswith("xi") else 1
            out[k] = v[i:i + n] if n > 1 else v[i]
            i += n
        return out

    def bounds(self):
        """Box constraints: Gamma_s above Gamma_I, bounded higher modes, material ranges."""
        b = {"xi_s": [(0.05, 1.5)] + [(-0.3, 0.3)] * (len(self.xi_s) - 1),
             "xi_I": [(0.05, 1.5)] + [(-0.3, 0.3)] * (len(self.xi_I) - 1),
             "log_mu1": [(math.log(0.1), math.log(3.0))], "log_rho1": [(math.log(0.3), math.log(3.0))]}
        return [x for k in self.unknown for x in b[k]]

    def rec_s_idx(self, n_el):
        """Gamma_s receivers (interface case): elements at s ~ uniform in (0, 1)."""
        if self.name != "interface":
            return None
        f = n_el // self.n_el
        coarse = np.linspace(0, self.n_el - 1, self.n_rec_s + 2)[1:-1].round().astype(int)
        return torch.tensor(coarse * f + f // 2)


def systems(case, prm, n_el, etas=None, q=4):
    """BEM systems (one per frequency) for the parameter dict prm."""
    Gs = g.discretize(g.sine_series(prm["xi_s"], 1.0, +1.0), n_el, "up", q=q)
    GI = g.discretize(g.sine_series(prm["xi_I"], 1.0, -1.0), n_el, "down", q=q)
    mats = (torch.exp(prm["log_mu1"]), torch.exp(prm["log_rho1"]), 1.0, 1.0)
    rec_s = case.rec_s_idx(n_el)
    out = [ibem.assemble(Gs, GI, eta, case.alphas, mats, rec_x=np.array(case.rec_x), rec_s=rec_s)
           for eta in (etas or case.etas)]
    return Gs, GI, out


def predict(case, prm, n_el, q=4, solver=None):
    """Receiver displacements; solver(i, A, b) replaces the LU solve when given."""
    if getattr(case, "is3d", False):
        return case.predict(prm, n_el, solver)
    _, _, S = systems(case, prm, n_el, q=q)
    if solver is None:
        return torch.stack([ibem.solve(s)[1] for s in S], 0)    # (n_eta, n_rec, n_alpha)
    return torch.stack([s["Rop"] @ solver(i, s["A"], s["b"]) + s["r0"] for i, s in enumerate(S)], 0)


class IterativeSolver:
    """GMRES forward solves (adjoint solves in backward) with warm start, counting products."""

    def __init__(self, tol=1e-10, warm=True):
        from .krylov import Counter
        self.counter, self.tol, self.warm, self.prev = Counter(), tol, warm, {}

    def __call__(self, i, A, b):
        from .krylov import gmres_solve
        X = gmres_solve(A, b, self.prev.get(i) if self.warm else None, self.tol, self.counter)
        self.prev[i] = X.detach().numpy().copy()
        return X


def make_data(case, noise=0.0, seed=0):
    with torch.no_grad():
        u = predict(case, case.unpack(torch.tensor(case.true_vec())), case.n_fine, q=8)
    if noise > 0:
        rng = np.random.default_rng(seed)
        e = (rng.standard_normal(u.shape) + 1j * rng.standard_normal(u.shape)) / math.sqrt(2)
        u = u * (1 + noise * torch.tensor(e))
    return u


def regularization(case, prm):
    if getattr(case, "is3d", False):
        return case.regularization(prm)
    k = torch.arange(1, len(case.xi_s) + 1, dtype=DTYPE) ** (2 * case.p_reg)
    r = 0.0
    for name in ("xi_s", "xi_I"):
        if name in case.unknown:
            r = r + (k * prm[name] ** 2).sum()
    return r


def data_residual(u, d, kind="complex"):
    """Normalized data residual (complex or real); the only place where the data type enters."""
    if kind == "amplitude":
        return (u.abs() - d.abs()) / torch.sqrt((d.abs() ** 2).sum())
    return (u - d) / torch.sqrt((d.abs() ** 2).sum())


def misfit(u, d, kind="complex"):
    return (data_residual(u, d, kind).abs() ** 2).sum()


# --- error measures ----------------------------------------------------------------------

def shape_error(case, v):
    """Relative L2 error of the unknown curves on a fine s-grid, and material errors."""
    if getattr(case, "is3d", False):
        return case.shape_error(v)
    s = torch.linspace(0, 1, 401, dtype=DTYPE)
    prm, tru = case.unpack(torch.as_tensor(v)), case.unpack(torch.tensor(case.true_vec()))
    out = {}
    for name, sign in (("xi_s", 1.0), ("xi_I", -1.0)):
        if name in case.unknown:
            y = g.sine_series(prm[name], 1.0, sign)(s)[1]
            yt = g.sine_series(tru[name], 1.0, sign)(s)[1]
            out["shape"] = float(torch.linalg.norm(y - yt) / torch.linalg.norm(yt))
    if "log_mu1" in case.unknown:
        out["mu1"] = float(abs(torch.exp(prm["log_mu1"]) / case.mu1 - 1))
        out["rho1"] = float(abs(torch.exp(prm["log_rho1"]) / case.rho1 - 1))
    return out


# --- classical solvers -------------------------------------------------------------------

def _objective_np(case, data, solver=None):
    """Numpy objective/residual wrappers counting forward solves."""
    cnt = {"forward": 0, "adjoint": 0}

    def J_and_grad(v):
        vt = torch.tensor(v, dtype=DTYPE, requires_grad=True)
        prm = case.unpack(vt)
        J = misfit(predict(case, prm, case.n_el, solver=solver), data, case.data_kind) + case.lam_r * regularization(case, prm)
        J.backward()
        cnt["forward"] += 1
        cnt["adjoint"] += 1
        return float(J), vt.grad.numpy().copy()

    def residual(v):
        with torch.no_grad():
            prm = case.unpack(torch.tensor(v, dtype=DTYPE))
            u = predict(case, prm, case.n_el, solver=solver)
            cnt["forward"] += 1
            r = data_residual(u, data, case.data_kind).reshape(-1)
            rr = torch.cat([r.real, r.imag]) if r.is_complex() else r
            reg = math.sqrt(case.lam_r) * _reg_residual(case, prm)
            return torch.cat([rr, reg]).numpy()
    return J_and_grad, residual, cnt


def _reg_residual(case, prm):
    if getattr(case, "is3d", False):
        return case.reg_residual(prm)
    k = torch.arange(1, len(case.xi_s) + 1, dtype=DTYPE) ** case.p_reg
    parts = [k * prm[n] for n in ("xi_s", "xi_I") if n in case.unknown]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=DTYPE)


def _products(solver):
    c = solver.counter if solver is not None else None
    return dict(matvec=c.matvec, rmatvec=c.rmatvec) if c else {}


def lbfgs_adj(case, data, max_iter=300, log=None, solver=None):
    J_and_grad, _, cnt = _objective_np(case, data, solver)
    hist, t0 = [], time.perf_counter()

    def cb(v):
        hist.append(dict(t=time.perf_counter() - t0, forward=cnt["forward"], **_products(solver),
                         **shape_error(case, v)))

    v0 = case.init_vec()
    cb(v0)
    res = minimize(J_and_grad, v0, jac=True, method="L-BFGS-B", callback=cb, bounds=case.bounds(),
                   options=dict(maxiter=max_iter, ftol=1e-14, gtol=1e-10, maxcor=20))
    return dict(v=res.x, time=time.perf_counter() - t0, forward=cnt["forward"],
                adjoint=cnt["adjoint"], iters=res.nit, J=float(res.fun), hist=hist, **_products(solver))


def lbfgs_adj_cont(case, data, max_iter=300):
    """L-BFGS-adj with the frequency continuation used by the BEM-PINN (control run):
    successive solves with the 2, 3 and 4 lowest frequencies, each warm-started."""
    import copy
    t0, hist, tot, v = time.perf_counter(), [], dict(forward=0, adjoint=0, iters=0), case.init_vec()
    for n in range(2, len(case.etas) + 1):
        sub = copy.copy(case)
        sub.etas = tuple(case.etas[:n])
        sub.init_vec = (lambda vv: (lambda: vv))(v.copy())
        r = lbfgs_adj(sub, data[:n], max_iter=max_iter)
        for h in r["hist"]:
            hist.append(dict(h, t=h["t"] + time.perf_counter() - t0 - r["time"],
                             forward=h["forward"] + tot["forward"]))
        v = r["v"]
        for k in ("forward", "adjoint", "iters"):
            tot[k] += r[k]
    J, _ = _objective_np(case, data)[0](v)
    return dict(v=v, time=time.perf_counter() - t0, J=float(J), hist=hist, **tot)


def lm_fd(case, data, max_iter=60, h=1e-5, log=None, solver=None):
    """Levenberg-Marquardt with central finite differences (2P forward solves per Jacobian)."""
    _, residual, cnt = _objective_np(case, data, solver)
    lo, hi = np.array(case.bounds()).T
    t0 = time.perf_counter()
    v = case.init_vec()
    r = residual(v)
    J = 0.5 * r @ r
    lam = 1e-2
    hist = [dict(t=time.perf_counter() - t0, forward=cnt["forward"], **_products(solver),
                 **shape_error(case, v))]
    for it in range(max_iter):
        Jac = np.stack([(residual(v + h * e) - residual(v - h * e)) / (2 * h)
                        for e in np.eye(len(v))], 1)
        A, gvec = Jac.T @ Jac, Jac.T @ r
        improved = False
        for _ in range(10):
            dv = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), gvec)
            dv = np.clip(v + dv, lo, hi) - v                      # projection onto the bounds
            r_new = residual(v + dv)
            J_new = 0.5 * r_new @ r_new
            if J_new < J:
                v, r, lam, improved = v + dv, r_new, max(lam / 3, 1e-9), True
                break
            lam *= 10
        hist.append(dict(t=time.perf_counter() - t0, forward=cnt["forward"], **_products(solver),
                         **shape_error(case, v)))
        if not improved or abs(J - J_new) < 1e-12 * max(J, 1e-30) or np.max(np.abs(dv)) < 1e-8:
            J = min(J, J_new)
            break
        J = J_new
    return dict(v=v, time=time.perf_counter() - t0, forward=cnt["forward"], iters=it + 1,
                J=float(2 * J), hist=hist, **_products(solver))


# --- BEM-PINN ------------------------------------------------------------------------------

def bem_pinn(case, data, n_pre=1000, n_pre_lbfgs=100, n_joint=1000, n_lbfgs=100, width=150,
             depth=4, m=64, sigma=3.0, lr=1e-3, lr_v=3e-3, lam_d=1.0, seed=0, log=print):
    """Joint training of the density network (theta) and the unknowns (vartheta).

    The network maps the uniform boundary coordinate s_hat = arccos(1 - 2 s) / pi to the
    densities (phi_s, phi_I^(1), phi_I^(2)) of every excitation (eta_f, alpha_j), one
    output head per excitation. No linear system is ever solved.
    """
    torch.manual_seed(seed)
    v = torch.tensor(case.init_vec(), dtype=DTYPE, requires_grad=True)
    lo, hi = (torch.tensor(b, dtype=DTYPE) for b in np.array(case.bounds()).T)

    def project():
        with torch.no_grad():
            v.copy_(torch.maximum(torch.minimum(v, hi), lo))

    etas, na = list(case.etas), len(case.alphas_deg)
    C = len(etas) * na
    net = pinn.FourierNet(1, 6 * C, m=m, sigma=sigma, width=width, depth=depth, seed=seed)
    Gs0, _, S0 = systems(case, case.unpack(v.detach()), case.n_el)
    scale = pinn.jacobi_scale(S0)
    shat = (torch.arccos(1 - 2 * Gs0.s) / math.pi)[:, None]
    N = len(Gs0.s)
    t0 = time.perf_counter()
    hist, cnt = [], {"assemblies": 0, "evals": 0, "matvec": 0, "rmatvec": 0}

    def phi_all():
        o = net(shat).reshape(N, C, 3, 2)
        P = (torch.complex(o[..., 0], o[..., 1]) * scale).permute(2, 0, 1).reshape(3 * N, C)
        return [P[:, i * na:(i + 1) * na] for i in range(len(etas))]

    def record(it, stage, L, Lbc, Ld):
        rec = dict(it=it, stage=stage, t=time.perf_counter() - t0, loss=float(L), bc=float(Lbc),
                   data=float(Ld), **cnt, **shape_error(case, v.detach().numpy()))
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

    # stage 1: forward solution at the initial guess (systems assembled once)
    def loss_pre():
        cnt["evals"] += 1
        cnt["matvec"] += C
        cnt["rmatvec"] += C
        L = pinn.bc_loss(S0, phi_all())
        return L, L, torch.tensor(float("nan"))

    cnt["assemblies"] += 1
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    for it in range(n_pre):
        opt.zero_grad()
        L, _, _ = loss_pre()
        L.backward()
        opt.step()
        if it % 250 == 0:
            record(it, 1, L.item(), L.item(), float("nan"))
    lbfgs(list(net.parameters()), loss_pre, n_pre_lbfgs, 1, n_pre)

    # stage 2: joint training (frequency continuation), stage 3: joint L-BFGS
    def loss_joint(n_active=len(etas)):
        cnt["assemblies"] += 1
        cnt["evals"] += 1
        cnt["matvec"] += n_active * na
        cnt["rmatvec"] += n_active * na
        prm = case.unpack(v)
        _, _, S = systems(case, prm, case.n_el, etas=etas[:n_active])
        phis = phi_all()[:n_active]
        Lbc = pinn.bc_loss(S, phis)
        u = torch.stack([St["Rop"] @ p + St["r0"] for St, p in zip(S, phis)], 0)
        d = data[:n_active]
        Ld = misfit(u, d, case.data_kind)
        return Lbc + lam_d * Ld + case.lam_r * regularization(case, prm), Lbc, Ld

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
