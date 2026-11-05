"""Experiment 2: inversion benchmark (3 problems x 3 noise levels x 3 solvers).

Usage: python exp2_inverse.py <interface|surface|joint> <noise> <lm_fd|lbfgs_adj|bem_pinn>
"""
import os
import sys

import numpy as np
import torch

import common
from bempinn import geometry as g
from bempinn import inverse as inv

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))

name, noise, method = sys.argv[1], float(sys.argv[2]), sys.argv[3]
case = inv.Case(name)
data = inv.make_data(case, noise, seed=1)

if method == "lm_fd":
    res = inv.lm_fd(case, data)
elif method == "lbfgs_adj":
    res = inv.lbfgs_adj(case, data)
elif method == "lbfgs_adj_cont":
    res = inv.lbfgs_adj_cont(case, data)
else:
    res = inv.bem_pinn(case, data, log=None)

err = inv.shape_error(case, res["v"])
s = torch.linspace(0, 1, 201, dtype=torch.float64)
curves = {}
for label, vec in (("true", case.true_vec()), ("init", case.init_vec()), ("est", res["v"])):
    prm = case.unpack(torch.as_tensor(vec, dtype=torch.float64))
    xs, ys = g.sine_series(prm["xi_s"], 1.0, 1.0)(s)
    _, yi = g.sine_series(prm["xi_I"], 1.0, -1.0)(s)
    curves[label] = dict(x=xs.numpy(), ys=ys.numpy(), yi=yi.numpy())

out = dict(case=name, noise=noise, method=method, err=err, v=res["v"], true=case.true_vec(),
           time=res["time"], J=res.get("J"), forward=res.get("forward"),
           assemblies=res.get("assemblies"), iters=res.get("iters"), hist=res["hist"],
           curves=curves, threads=torch.get_num_threads())
common.save(f"exp2_{name}_{noise:g}_{method}.json", out)
print(name, noise, method, err, "t %.1fs" % res["time"])
