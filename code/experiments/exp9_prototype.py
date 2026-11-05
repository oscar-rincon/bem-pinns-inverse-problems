"""Experiment 9 (prototyping): inversion from amplitude-only records |u|.

The only change with respect to experiment 2 is the data residual (inverse.data_residual),
shared by every solver through Case.data_kind.

Usage: python exp9_prototype.py <interface|surface|joint> <noise> <lm_fd|lbfgs_adj|lbfgs_adj_cont|bem_pinn>
"""
import os
import sys

import torch

import common
from bempinn import inverse as inv

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))

name, noise, method = sys.argv[1], float(sys.argv[2]), sys.argv[3]
case = inv.Case(name, data_kind="amplitude")
data = inv.make_data(case, noise, seed=1)

if method == "bem_pinn":
    res = inv.bem_pinn(case, data, log=None)
else:
    res = getattr(inv, method)(case, data)

err = inv.shape_error(case, res["v"])
out = dict(case=name, noise=noise, method=method, data_kind=case.data_kind, err=err, v=res["v"],
           true=case.true_vec(), time=res["time"], J=res.get("J"), bc=res.get("bc"),
           forward=res.get("forward"), assemblies=res.get("assemblies"), hist=res["hist"],
           threads=torch.get_num_threads())
common.save(f"exp9_{name}_{noise:g}_{method}.json", out)
print(name, noise, method, err, "t %.1fs" % res["time"])
