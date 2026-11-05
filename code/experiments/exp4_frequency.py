"""Experiment 4: frequency as a network input (Wu et al. 2026, Sec. 4.1.2 / Table 3 analogue).

V-shaped canyon, alpha = 0, eta in [1.0, 1.5], formulation of Wu et al. (flat surface meshed
over 8 wavelengths of the lowest frequency, 240 elements per side, 100 on the canyon). The
FF-BEM-PINN, with input (x0, y0, eta), is trained at three frequencies and evaluated at 11;
the IBEM solves each of the 11 frequencies directly.
"""
import math
import os
import time

import numpy as np
import torch

import common
from bempinn import ibem, pinn

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
N_EL, ALPHA = 100, 0.0
E0, E1 = float(os.environ.get("ETA_MIN", "1.0")), float(os.environ.get("ETA_MAX", "1.5"))
ETA_TRAIN = [E0, 0.5 * (E0 + E1), E1]
ETA_TEST = list(np.linspace(E0, E1, 11))
TAG = os.environ.get("TAG", "")
n_adam = int(os.environ.get("N_ADAM", "20000"))

refs = {e: common.reference("vcanyon", e, [ALPHA]) for e in ETA_TEST}

FLAT = 8 * 2.0 / min(ETA_TEST)


def setup(e):
    return common.truncated_setup("vcanyon", N_EL, e, [ALPHA], flat_len=FLAT)


# conventional IBEM, one solve per test frequency
t0 = time.perf_counter()
u_bem = {}
for e in ETA_TEST:
    Gs, _, S = setup(e)
    u_bem[e] = ibem.solve(S)[1].numpy()
t_bem = time.perf_counter() - t0
x0 = S["x0"]
eps_bem = [common.wu_metrics(u_bem[e], refs[e](Gs.s.numpy(), x0))[0] for e in ETA_TEST]

# FF-BEM-PINN with input (x0, y0, eta), trained at ETA_TRAIN only
t0 = time.perf_counter()
sys_train = [setup(e)[2] for e in ETA_TRAIN]
sys_test = {e: setup(e)[2] for e in ETA_TEST}
t_kernels = time.perf_counter() - t0
zin = sys_train[0]["zin"]
net = pinn.FourierNet(3, 2, m=128, sigma=1.0, width=200, depth=8, seed=0)
scale = pinn.jacobi_scale(sys_train)


def dens(e):
    z = torch.cat([zin, torch.full_like(zin[:, :1], (e - E0) / (E1 - E0))], -1)
    o = net(z)
    return (torch.complex(o[:, 0], o[:, 1]) * scale)[:, None]


def parts(it):
    return pinn.bc_loss(sys_train, [dens(e) for e in ETA_TRAIN]), {}


WD = float(os.environ.get("WD", "0"))
hist = pinn.train(net, parts, net.parameters(), n_adam=n_adam, lr=1e-3, n_lbfgs=100, log_every=1000,
                  weight_decay=WD)
t_train = hist[-1]["t"]
t0 = time.perf_counter()
with torch.no_grad():
    u_p = {e: (sys_test[e]["Rop"] @ dens(e) + sys_test[e]["r0"]).numpy() for e in ETA_TEST}
t_pred = time.perf_counter() - t0
eps_p = [common.wu_metrics(u_p[e], refs[e](Gs.s.numpy(), x0))[0] for e in ETA_TEST]
out = dict(eta_test=ETA_TEST, eta_train=ETA_TRAIN, eps_bem=eps_bem, eps_pinn=eps_p, t_bem=t_bem,
           t_pinn=t_train + t_kernels + t_pred, t_train=t_train, t_kernels=t_kernels,
           t_pred=t_pred, n_adam=n_adam, weight_decay=WD, threads=torch.get_num_threads())
common.save(f"exp4_frequency{TAG}.json", out)
print("BEM: mean eps %.2e, %.2fs | PINN: mean eps %.2e, %.0fs" % (np.mean(eps_bem), t_bem, np.mean(eps_p), out["t_pinn"]))
