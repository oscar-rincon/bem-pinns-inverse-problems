# Hybrid BEM–PINN inversion for SH waves in irregular topographies

Code for the paper in `../paper/`. Everything runs on CPU with PyTorch (double precision).

## Package `bempinn/`

| Module | Content |
|---|---|
| `special.py` | Hankel functions H0^(2), H1^(2) with analytical derivatives (autograd) |
| `geometry.py` | Curves (semicircle, flat, V-canyon, sine series), cosine-graded straight elements, curvature |
| `ibem.py` | Indirect BEM: integrated kernels with self terms, half-space Green's function, two-region assembly, Wu et al. (2026) truncated formulation |
| `analytic.py` | Exact solutions: semicircular canyon (Trifunac 1973), alluvial valley (Trifunac 1971) |
| `pinn.py` | Fourier-feature network and training loop |
| `inverse.py` | Inverse problems; solvers `lm_fd`, `lbfgs_adj`, `lbfgs_adj_cont`, `bem_pinn`; `IterativeSolver` (GMRES) |
| `krylov.py` | Jacobi-preconditioned GMRES solve with adjoint backward pass and product counting |
| `bem3d.py` | 3D scalar (Helmholtz) indirect BEM on triangulated surfaces, exact rigid-hemisphere solution |
| `inverse3d.py` | 3D canyon reconstruction case and `bem_pinn3d` |

## Experiments (`experiments/`, run from that directory with `PYTHONPATH=..`)

| Script | Paper section | Output |
|---|---|---|
| `exp1_forward.py <topo> <alpha>` | Forward reproduction of Wu et al. (2026), Table 2 / Figs. 4–6 | `results/exp1_*.json` |
| `exp4_frequency.py` | Frequency as network input (Wu et al. Table 3 analogue) | `results/exp4_frequency.json` |
| `exp2_inverse.py <problem> <noise> <solver>` | Inversion benchmark | `results/exp2_*.json` |
| `exp3_scaling.py` | Cost per iteration vs. number of elements | `results/exp3_scaling.json` |
| `exp5_iterative.py <n_el> <solver>` | Iterative (GMRES) regime vs N, 2D | `results/exp5_*.json` |
| `exp6_3d.py <noise> <solver>` | 3D canyon reconstruction | `results/exp6_*.json` |
| `exp7_scaling3d.py` | 3D cost per iteration, GMRES iterations, memory | `results/exp7_scaling3d.json` |
| `exp8_basin.py <band> <h0> <solver> [seed]` | Basin of attraction: 8 initial guesses x 3 frequency bands, surface reconstruction | `results/exp8_*.json` |
| `exp9_prototype.py <problem> <noise> <solver>` | Prototyping example: inversion from amplitude-only records (`Case.data_kind="amplitude"`) | `results/exp9_*.json` |
| `run_all.sh`, `run_3d.sh`, `run_basin.sh`, `run_prototype.sh` | Experiments 3, 2, 4 / 7, 6 / 8 / 9 in sequence | |
| `figures.py`, `make_tables.py` | Figures and tables written to `../paper/figures` and `../paper/tables` | |

Each process uses `NTHREADS` CPU threads (default 4). Timings in the paper were measured on an
Intel Core i9-13900H; the classical inversions and the scaling study ran alone on the machine,
the BEM–PINN inversions three at a time and the forward reproductions five at a time.

Typical run times: forward reproduction 4–10 min per case, BEM–PINN inversion 3–5 min per case,
classical inversions about 1 s each.

## Notes

* The installed PyTorch build (CUDA 13) does not run on the machine's GPU driver (CUDA 12.8), so
  no GPU results were produced.
* The half-space single-layer formulation has irregular frequencies when the free surface and
  its image enclose a region (e.g. the V-shaped canyon at eta = 1, 1.58, 2); the V-canyon
  benchmarks therefore use the truncated formulation of Wu et al. (2026).
