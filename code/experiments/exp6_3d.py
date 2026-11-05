"""Experiment 6: 3D canyon surface reconstruction (scalar model).

Usage: python exp6_3d.py <noise> <lm_lu|lbfgs_lu|lbfgs_gmres|bem_pinn>
"""
import os
import sys

import numpy as np
import torch

import common
from bempinn import bem3d, inverse as inv, inverse3d as i3

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
noise, method = float(sys.argv[1]), sys.argv[2]
case = i3.Case3D()
data = i3.make_data(case, noise, seed=1)
if method == "lm_lu":
    res = inv.lm_fd(case, data)
elif method == "lbfgs_lu":
    res = inv.lbfgs_adj(case, data)
elif method == "lbfgs_gmres":
    res = inv.lbfgs_adj(case, data, solver=inv.IterativeSolver(warm=False))
else:
    res = i3.bem_pinn3d(case, data, log=None)

# depth maps on a polar grid for the figure
P, _ = bem3d.disk_mesh(30)
B = bem3d.canyon_basis(P)
maps = {k: (B @ torch.as_tensor(v, dtype=torch.float64)).numpy()
        for k, v in (("true", case.true_vec()), ("init", case.init_vec()), ("est", res["v"]))}
out = dict(noise=noise, method=method, err=case.shape_error(res["v"]), v=res["v"],
           true=case.true_vec(), time=res["time"], forward=res.get("forward"),
           evals=res.get("evals"), matvec=res.get("matvec"), rmatvec=res.get("rmatvec"),
           hist=res["hist"], uv=P.numpy(), maps=maps, n_tri=len(case.mesh(case.unpack(case.init_vec()))),
           threads=torch.get_num_threads())
common.save(f"exp6_{noise:g}_{method}.json", out)
print(noise, method, out["err"], "t %.1fs" % res["time"])
