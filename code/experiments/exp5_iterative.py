"""Experiment 5: iterative (matrix-free) regime in 2D, surface reconstruction, 1% noise.

Classical solvers use Jacobi-preconditioned GMRES for every forward and adjoint solve, as they
must when the dense matrix cannot be factorized; the cost is counted in matrix-vector products
(products with A or A^H, one per right-hand side). The BEM-PINN performs one product with A and
one with A^H per right-hand side and loss evaluation.

Usage: python exp5_iterative.py <n_el> <lbfgs_gmres|lm_gmres|bem_pinn|lbfgs_lu>
"""
import os
import sys

import torch

import common
from bempinn import inverse as inv

torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
n_el, method = int(sys.argv[1]), sys.argv[2]
case = inv.Case("surface", n_el=n_el, n_fine=3 * n_el)
data = inv.make_data(case, 0.01, seed=1)
if method == "lbfgs_gmres":
    res = inv.lbfgs_adj(case, data, solver=inv.IterativeSolver(warm=False))
elif method == "lm_gmres":
    res = inv.lm_fd(case, data, solver=inv.IterativeSolver(warm=True))
elif method == "lbfgs_lu":
    res = inv.lbfgs_adj(case, data)
else:
    res = inv.bem_pinn(case, data, log=None)
out = dict(n_el=n_el, n_unknowns=3 * n_el, method=method, err=inv.shape_error(case, res["v"]),
           time=res["time"], matvec=res.get("matvec"), rmatvec=res.get("rmatvec"),
           forward=res.get("forward"), evals=res.get("evals"), hist=res["hist"],
           threads=torch.get_num_threads())
common.save(f"exp5_{n_el}_{method}.json", out)
print(n_el, method, out["err"], "t %.1fs" % res["time"], "products", (out["matvec"] or 0) + (out["rmatvec"] or 0))
