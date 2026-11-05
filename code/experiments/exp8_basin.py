"""Experiment 8: basin of attraction (surface reconstruction, 1% noise).

The first sine coefficient of the initial guess is varied (true value 0.70, higher
coefficients start at zero) for two frequency bands, and every solver is run from each start.

Usage: python exp8_basin.py <low|mid|high> <h0> <lm_fd|lbfgs_adj|lbfgs_adj_cont|bem_pinn> [seed]
"""
import os
import sys

import torch

import common
from bempinn import inverse as inv

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))

BANDS = {"low": (0.25, 0.5, 0.75, 1.0), "mid": (0.5, 1.0, 1.5, 2.0), "high": (1.5, 1.75, 2.0, 2.25)}
H0 = (0.1, 0.2, 0.35, 0.5, 0.9, 1.1, 1.3, 1.5)

band, h0, method = sys.argv[1], float(sys.argv[2]), sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
case = inv.Case("surface", etas=BANDS[band], xi_s0=[h0, 0.0, 0.0, 0.0])
data = inv.make_data(case, 0.01, seed=1)

if method == "bem_pinn":
    res = inv.bem_pinn(case, data, seed=seed, log=None)
else:
    res = getattr(inv, method)(case, data)

err = inv.shape_error(case, res["v"])
out = dict(band=band, etas=case.etas, h0=h0, method=method, seed=seed, err=err, v=res["v"],
           time=res["time"], J=res.get("J"), bc=res.get("bc"), forward=res.get("forward"),
           assemblies=res.get("assemblies"), hist=res["hist"], threads=torch.get_num_threads())
tag = f"_s{seed}" if method == "bem_pinn" and seed else ""
common.save(f"exp8_{band}_{h0:g}_{method}{tag}.json", out)
print(band, h0, method, seed, err, "t %.1fs" % res["time"])
