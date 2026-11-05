"""Experiment 3: cost per iteration vs. number of elements (surface-reconstruction set-up).

Times (best of several repeats) for one objective evaluation of each solver:
  forward      assembly + LU solve for all frequencies (no gradient)
  adjoint      forward + reverse-mode gradient (L-BFGS iteration)
  fd_jacobian  2P forward evaluations (one LM iteration)
  pinn_fixed   BEM-PINN step at fixed geometry (stage 1)
  pinn_joint   BEM-PINN joint step: re-assembly + network + backward (stages 2-3)
"""
import math
import os
import time

import numpy as np
import torch

import common
from bempinn import inverse as inv, pinn

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))


def best(fn, rep):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


case = inv.Case("surface")
data = inv.make_data(case, 0.0)
v0 = torch.tensor(case.init_vec(), dtype=torch.float64)
P = len(v0)
rows = []
for n in (20, 40, 80, 160, 320, 640):
    rep = 5 if n <= 160 else 2
    case.n_el = n

    def forward():
        with torch.no_grad():
            inv.predict(case, case.unpack(v0), n)

    def assembly_only():
        with torch.no_grad():
            inv.systems(case, case.unpack(v0), n)

    def adjoint():
        v = v0.clone().requires_grad_(True)
        J = inv.misfit(inv.predict(case, case.unpack(v), n), data)
        J.backward()

    C = len(case.etas) * len(case.alphas_deg)
    net = pinn.FourierNet(1, 6 * C, m=64, sigma=3.0, width=150, depth=4)
    Gs, _, S0 = inv.systems(case, case.unpack(v0), n)
    shat = (torch.arccos(1 - 2 * Gs.s) / math.pi)[:, None]
    na = len(case.alphas_deg)

    def phis():
        o = net(shat).reshape(n, C, 3, 2)
        Pm = torch.complex(o[..., 0], o[..., 1]).permute(2, 0, 1).reshape(3 * n, C)
        return [Pm[:, i * na:(i + 1) * na] for i in range(len(case.etas))]

    def pinn_fixed():
        L = pinn.bc_loss(S0, phis())
        L.backward()

    def pinn_joint():
        v = v0.clone().requires_grad_(True)
        _, _, S = inv.systems(case, case.unpack(v), n)
        ph = phis()
        L = pinn.bc_loss(S, ph) + sum(((St["Rop"] @ p + St["r0"] - d).abs() ** 2).sum()
                                      for St, p, d in zip(S, ph, data))
        L.backward()

    r = dict(n_el=n, n_unknowns=3 * n, forward=best(forward, rep), assembly=best(assembly_only, rep),
             adjoint=best(adjoint, rep), pinn_fixed=best(pinn_fixed, rep), pinn_joint=best(pinn_joint, rep))
    r["solve"] = r["forward"] - r["assembly"]
    r["fd_jacobian"] = 2 * P * r["forward"]
    rows.append(r)
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
common.save("exp3_scaling.json", dict(rows=rows, P=P, threads=torch.get_num_threads(),
                                      n_freq=len(case.etas), n_alpha=len(case.alphas_deg)))
