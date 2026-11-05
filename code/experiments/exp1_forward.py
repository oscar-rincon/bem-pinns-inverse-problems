"""Experiment 1: reproduction of the forward benchmarks of Wu et al. (2026), Table 2.

Usage: python exp1_forward.py <topo> <alpha_deg> [n_adam] [n_lbfgs]
"""
import sys
import time

import numpy as np
import torch

import common
from bempinn import ibem, pinn

torch.set_num_threads(int(__import__("os").environ.get("NTHREADS", "2")))

topo, adeg = sys.argv[1], float(sys.argv[2])
n_adam = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
n_lbfgs = int(sys.argv[4]) if len(sys.argv) > 4 else 200
ETA, N_EL = 2.0, 100
alpha = np.radians(adeg)

ref = common.reference(topo, ETA, [alpha])

# conventional IBEM: assembly + direct solve (best of 5 timings)
times = []
for _ in range(5):
    t0 = time.perf_counter()
    Gs, GI, S = common.forward_setup(topo, N_EL, ETA, [alpha])
    phi_d, u_d = ibem.solve(S)
    times.append(time.perf_counter() - t0)
x0 = S.get("x0", common.X0)
uref = ref(Gs.s.numpy(), x0)
eps_bem, r2_bem = common.wu_metrics(u_d.numpy(), uref)

# FF-BEM-PINN (Wu et al. Sec. 3 settings: 6 x 150, m = 64, sigma = 1, AdamW + L-BFGS)
t0 = time.perf_counter()
Gs, GI, S = common.forward_setup(topo, N_EL, ETA, [alpha])
t_kernel = time.perf_counter() - t0
n_sets = 3 if GI is not None else 1
scale = pinn.jacobi_scale([S])
# input: curve coordinate s (half-space formulation) or source coordinates (x0, y0)
zin = S["zin"] if "zin" in S else Gs.s[:, None]
net = pinn.FourierNet(zin.shape[1], 2 * n_sets, m=64, sigma=1.0, width=150, depth=6, seed=0)
cases = torch.zeros(1, 0, dtype=torch.float64)


def dens():
    o = net(zin).reshape(len(zin), n_sets, 2)
    return (torch.complex(o[..., 0], o[..., 1]) * scale).T.reshape(-1, 1)


def parts(it):
    return pinn.bc_loss([S], [dens()]), {}


hist = []


def cb(rec):
    with torch.no_grad():
        u = (S["Rop"] @ dens() + S["r0"]).numpy()
    rec["eps"], rec["r2"] = common.wu_metrics(u, uref)
    hist.append(rec)


pinn.train(net, parts, net.parameters(), n_adam=n_adam, lr=1e-3, n_lbfgs=n_lbfgs,
           log_every=500, callback=cb)
t_train = hist[-1]["t"]
with torch.no_grad():
    u_p = (S["Rop"] @ dens() + S["r0"]).numpy()
eps_p, r2_p = common.wu_metrics(u_p, uref)

out = dict(topo=topo, alpha=adeg, eta=ETA, n_el=N_EL, n_adam=n_adam, n_lbfgs=n_lbfgs,
           bem=dict(eps=eps_bem, r2=r2_bem, time=min(times)),
           pinn=dict(eps=eps_p, r2=r2_p, time=t_train + t_kernel, t_kernel=t_kernel),
           hist=[dict(it=h["it"], t=h["t"], loss=h["loss"], eps=h["eps"]) for h in hist],
           x0=x0, n_dens=int(S["A"].shape[1]), xs=Gs.c[:, 0].numpy(), ys=Gs.c[:, 1].numpy(),
           u_ref=np.abs(uref[:, 0]), u_bem=np.abs(u_d.numpy()[:, 0]), u_pinn=np.abs(u_p[:, 0]))
common.save(f"exp1_{topo}_{int(adeg)}.json", out)
print(topo, adeg, "BEM eps %.2e (%.3fs)  PINN eps %.2e (%.0fs)" % (eps_bem, min(times), eps_p, out["pinn"]["time"]))
