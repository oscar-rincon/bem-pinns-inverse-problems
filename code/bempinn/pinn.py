"""Fourier-feature network for the IBEM source densities (BEM-PINN, after Wu et al. 2026)."""
import math
import time

import numpy as np
import torch
from torch import nn

from .geometry import DTYPE


class FourierNet(nn.Module):
    """z in [0, 1]^d -> [cos(2 pi B z), sin(2 pi B z)] -> MLP -> real outputs."""

    def __init__(self, d_in, d_out, m=64, sigma=1.0, width=150, depth=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(d_in, m, generator=g, dtype=DTYPE) * sigma)
        layers, w_in = [], 2 * m
        for _ in range(depth):
            layers += [nn.Linear(w_in, width, dtype=DTYPE), nn.GELU()]
            w_in = width
        layers.append(nn.Linear(w_in, d_out, dtype=DTYPE))
        self.mlp = nn.Sequential(*layers)
        torch.manual_seed(seed)
        for mod in self.mlp:
            if isinstance(mod, nn.Linear):
                nn.init.xavier_normal_(mod.weight)
                nn.init.zeros_(mod.bias)

    def forward(self, z):
        f = 2 * math.pi * z @ self.B
        return self.mlp(torch.cat([torch.cos(f), torch.sin(f)], -1))


def densities(net, s, cases, n_sets, scale):
    """Densities for every (eta, alpha) case, stacked as columns: (n_sets * N, n_cases).

    s: (N,) curve coordinates; cases: (C, 2) normalized (eta, alpha) pairs.
    The network output is [Re, Im] for each density set.
    """
    N, C = len(s), len(cases)
    z = torch.cat([s[None, :, None].expand(C, N, 1),
                   cases[:, None, :].expand(C, N, cases.shape[1])], -1).reshape(C * N, -1)
    o = net(z).reshape(C, N, n_sets, 2)
    phi = torch.complex(o[..., 0], o[..., 1]) * scale            # (C, N, n_sets)
    return phi.permute(2, 1, 0).reshape(n_sets * N, C)


def jacobi_scale(systems):
    """Density scale from the diagonal of the BIE system (no linear solve)."""
    vals = [S["b"] / torch.diagonal(S["A"])[:, None] for S in systems]
    return torch.cat([v.reshape(-1) for v in vals]).abs().pow(2).mean().sqrt().item()


def bc_loss(systems, phi_by_freq):
    """Relative BIE residual sum_f ||A_f phi_f - b_f||^2 / sum_f ||b_f||^2."""
    num = sum(((S["A"] @ p - S["b"]).abs() ** 2).sum() for S, p in zip(systems, phi_by_freq))
    den = sum((S["b"].abs() ** 2).sum() for S in systems)
    return num / den


def train(net, closure_parts, params_adam, n_adam, lr=1e-3, n_lbfgs=0, params_lbfgs=None,
          log_every=500, callback=None, weight_decay=0.0, time_limit=None):
    """Generic AdamW (+ optional L-BFGS) loop. closure_parts() returns (loss, info dict)."""
    hist = []
    params_adam = list(params_adam)
    params_lbfgs = list(params_lbfgs) if params_lbfgs is not None else params_adam
    opt = torch.optim.AdamW(params_adam, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(n_adam, 1), eta_min=lr * 0.05)
    t0 = time.perf_counter()
    for it in range(n_adam):
        opt.zero_grad()
        loss, info = closure_parts(it)
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == n_adam - 1:
            rec = dict(it=it, t=time.perf_counter() - t0, loss=loss.item(), **info)
            hist.append(rec)
            if callback:
                callback(rec)
        if time_limit and time.perf_counter() - t0 > time_limit:
            break
    if n_lbfgs:
        opt2 = torch.optim.LBFGS(params_lbfgs, lr=1.0, max_iter=20,
                                 history_size=50, line_search_fn="strong_wolfe",
                                 tolerance_grad=1e-12, tolerance_change=1e-14)
        state = {}

        def closure():
            opt2.zero_grad()
            loss, info = closure_parts(n_adam)
            loss.backward()
            state.update(loss=loss.item(), info=info)
            return loss

        for j in range(n_lbfgs):
            opt2.step(closure)
            if j % max(1, log_every // 20) == 0 or j == n_lbfgs - 1:
                rec = dict(it=n_adam + j, t=time.perf_counter() - t0, loss=state["loss"],
                           **state["info"])
                hist.append(rec)
                if callback:
                    callback(rec)
    return hist
