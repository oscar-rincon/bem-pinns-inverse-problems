"""Experiment 7: cost per iteration in 3D vs number of triangles (canyon at the initial guess,
three frequencies, three excitations), GMRES iterations per solve and dense-matrix memory."""
import math
import os
import time

import numpy as np
import scipy.sparse.linalg as sla
import torch

import common
from bempinn import inverse3d as i3, pinn
from bempinn.krylov import Counter, gmres_solve

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))


def best(fn, rep=2):
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


case = i3.Case3D()
data = i3.make_data(case, 0.0)
v0 = torch.tensor(case.init_vec(), dtype=torch.float64)
rows = []
for M in (6, 8, 12, 16, 20, 24):
    case.n_rings = M
    with torch.no_grad():
        m, S = case.systems(case.unpack(v0))
    N = len(m)
    r = dict(n_rings=M, n_tri=N, matrix_MB=3 * N * N * 16 / 2 ** 20)

    def assembly():
        with torch.no_grad():
            case.systems(case.unpack(v0))

    def lu():
        for s in S:
            torch.linalg.solve(s["A"], s["b"])

    r["assembly"] = best(assembly)
    r["solve"] = best(lu)
    r["forward"] = r["assembly"] + r["solve"]
    cnt = Counter()
    for s in S:
        gmres_solve(s["A"], s["b"], tol=1e-10, counter=cnt)
    r["gmres_its_per_solve"] = cnt.matvec / cnt.solves
    if M <= 16:
        def adjoint():
            v = v0.clone().requires_grad_(True)
            J = ((case.predict(case.unpack(v), M) - data) .abs() ** 2).sum()
            J.backward()

        C = len(case.etas) * len(case.excitations)
        net = pinn.FourierNet(2, 2 * C, m=64, sigma=1.0, width=150, depth=4)
        zin = 0.5 * (m.uv + 1)

        def pinn_joint():
            v = v0.clone().requires_grad_(True)
            _, Sg = case.systems(case.unpack(v))
            o = net(zin).reshape(N, C, 2)
            P = torch.complex(o[..., 0], o[..., 1])
            L = sum(((s["A"] @ P[:, 3 * i:3 * i + 3] - s["b"]).abs() ** 2).sum() for i, s in enumerate(Sg))
            L.backward()

        def pinn_fixed():
            o = net(zin).reshape(N, C, 2)
            P = torch.complex(o[..., 0], o[..., 1])
            L = sum(((s["A"] @ P[:, 3 * i:3 * i + 3] - s["b"]).abs() ** 2).sum() for i, s in enumerate(S))
            L.backward()

        r["adjoint"] = best(adjoint)
        r["pinn_joint"] = best(pinn_joint)
        r["pinn_fixed"] = best(pinn_fixed)
    rows.append(r)
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
common.save("exp7_scaling3d.json", dict(rows=rows, threads=torch.get_num_threads()))
