"""Generate LaTeX tables (paper/tables/*.tex) from the result files."""
import os

import numpy as np

from figures import PROBLEM, SOLVER, load

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "tables")
os.makedirs(OUT, exist_ok=True)

# Wu et al. (2026), Table 2: FF PINN error eps at eta = 2.0
WU = {"canyon": [5.85e-3, 1.12e-2, 1.57e-2, 2.61e-2],
      "mountain": [1.80e-2, 1.60e-2, 1.25e-2, 1.96e-2],
      "vcanyon": [2.51e-3, 5.59e-3, 2.15e-2, 2.27e-2]}
TOPO = {"canyon": "Semicircular canyon", "mountain": "Semicircular hill", "vcanyon": "V-shaped canyon"}


def sci(x, d=1):
    if x is None or not np.isfinite(x):
        return "--"
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def table_forward():
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r" & & \multicolumn{2}{c}{Wu et al.\ (2026)} & \multicolumn{2}{c}{IBEM (this work)} & \multicolumn{2}{c}{FF-BEM--PINN (this work)}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Topography & $\alpha$ & $\varepsilon$ & $R^2$ & $\varepsilon$ & time (s) & $\varepsilon$ & time (s)\\",
             r"\midrule"]
    WU_R2 = {"canyon": [0.997, 0.993, 0.994, 0.994], "mountain": [0.991, 0.984, 0.992, 0.993],
             "vcanyon": [0.999, 0.999, 0.994, 0.997]}
    for topo in WU:
        for j, a in enumerate((0, 30, 60, 90)):
            r = load(f"exp1_{topo}_{a}.json")
            name = TOPO[topo] if j == 0 else ""
            lines.append(f"{name} & {a}$^\\circ$ & {sci(WU[topo][j])} & {WU_R2[topo][j]:.3f} & "
                         f"{sci(r['bem']['eps'])} & {r['bem']['time']:.3f} & {sci(r['pinn']['eps'])} & "
                         f"{r['pinn']['time']:.0f}\\\\")
        if topo != "vcanyon":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "forward.tex"), "w").write("\n".join(lines) + "\n")


def table_inverse():
    lines = [r"\begin{tabular}{llccccccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{BEM + LM (FD Jacobian)} & \multicolumn{3}{c}{BEM + L-BFGS (adjoint)} & \multicolumn{3}{c}{BEM--PINN (this work)}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             r"Problem & noise & error & time (s) & solves & error & time (s) & solves & error & time (s) & assemblies\\",
             r"\midrule"]
    for prob in PROBLEM:
        for k, nz in enumerate((0.0, 0.01, 0.03)):
            row = [PROBLEM[prob] if k == 0 else "", f"{int(nz * 100)}\\%"]
            for m in SOLVER:
                r = load(f"exp2_{prob}_{nz:g}_{m}.json")
                e = r["err"]
                err = max(e["shape"], e.get("mu1", 0), e.get("rho1", 0))
                cnt = r["forward"] if m != "bem_pinn" else r["assemblies"]
                row += [f"{100 * err:.2f}\\%", f"{r['time']:.3g}", f"{cnt}"]
            lines.append(" & ".join(row) + r"\\")
        if prob != "joint":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "inverse.tex"), "w").write("\n".join(lines) + "\n")


def table_joint_detail():
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Solver & noise & shape & $\mu_1$ & $\rho_1$ & time (s)\\", r"\midrule"]
    labels = {m: st["label"] for m, st in SOLVER.items()}
    labels["lbfgs_adj_cont"] = "BEM + L-BFGS (adjoint), frequency continuation"
    order = ["lm_fd", "lbfgs_adj", "lbfgs_adj_cont", "bem_pinn"]
    for m in order:
        for k, nz in enumerate((0.0, 0.01, 0.03)):
            r = load(f"exp2_joint_{nz:g}_{m}.json")
            e = r["err"]
            name = labels[m].replace("–", "--") if k == 0 else ""
            lines.append(f"{name} & {int(nz * 100)}\\% & {100 * e['shape']:.2f}\\% & "
                         f"{100 * e['mu1']:.2f}\\% & {100 * e['rho1']:.2f}\\% & {r['time']:.3g}\\\\")
        if m != "bem_pinn":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "joint.tex"), "w").write("\n".join(lines) + "\n")


def table_three_d():
    labels = {"lm_lu": "BEM + LM (FD Jacobian), LU", "lbfgs_lu": "BEM + L-BFGS (adjoint), LU",
              "lbfgs_gmres": "BEM + L-BFGS (adjoint), GMRES", "bem_pinn": "BEM--PINN (this work)"}
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Solver & noise & error & time (s) & evaluations & products\\", r"\midrule"]
    for m, lab in labels.items():
        for k, nz in enumerate((0.0, 0.01, 0.03)):
            fn = f"exp6_{nz:g}_{m}.json"
            if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "results", fn)):
                continue
            r = load(fn)
            ev = r["evals"] if m == "bem_pinn" else r["forward"]
            pr = (r["matvec"] or 0) + (r["rmatvec"] or 0)
            lines.append(f"{lab if k == 0 else ''} & {int(nz * 100)}\\% & {100 * r['err']['shape']:.2f}\\% & "
                         f"{r['time']:.3g} & {ev} & {pr if pr else '--'}\\\\")
        if m != "bem_pinn":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "three_d.tex"), "w").write("\n".join(lines) + "\n")


def table_basin():
    """Number of starts (out of 8) reaching shape error < 2%, and mean time per run."""
    from figures import BASIN_BANDS, BASIN_H0, BASIN_TOL
    res = os.path.join(os.path.dirname(__file__), "..", "results")
    methods = [("lm_fd", "BEM + LM (FD)"), ("lbfgs_adj", "BEM + L-BFGS (adjoint)"),
               ("lbfgs_adj_cont", "BEM + L-BFGS (adjoint) + continuation"), ("bem_pinn", "BEM--PINN")]
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{$\eta\in\{0.25,\dots,1\}$} & \multicolumn{2}{c}{$\eta\in\{0.5,\dots,2\}$}"
             r" & \multicolumn{2}{c}{$\eta\in\{1.5,\dots,2.25\}$}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"Solver & converged & time (s) & converged & time (s) & converged & time (s)\\", r"\midrule"]
    for m, lab in methods:
        cells = []
        for band, _ in BASIN_BANDS:
            rows = [load(f"exp8_{band}_{h:g}_{m}.json") for h in BASIN_H0
                    if os.path.exists(os.path.join(res, f"exp8_{band}_{h:g}_{m}.json"))]
            if not rows:
                cells += ["--", "--"]
                continue
            ok = sum(r["err"]["shape"] < BASIN_TOL for r in rows)
            cells += [f"{ok}/{len(rows)}", f"{np.mean([r['time'] for r in rows]):.3g}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "basin.tex"), "w").write("\n".join(lines) + "\n")


def table_prototype():
    """Amplitude-only inversion (experiment 9): shape error (%) per noise level and mean time."""
    methods = [("lm_fd", "BEM + LM (FD)"), ("lbfgs_adj", "BEM + L-BFGS (adjoint)"),
               ("lbfgs_adj_cont", "BEM + L-BFGS (adjoint) + cont."), ("bem_pinn", "BEM--PINN")]
    probs = [("interface", "Interface"), ("surface", "Free surface"), ("joint", "Joint")]
    noises = [0.0, 0.01, 0.03]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{shape error (\%)} & \\", r"\cmidrule(lr){3-5}",
             r"Problem & Solver & 0\% & 1\% & 3\% & time (s)\\", r"\midrule"]
    for i, (p, plab) in enumerate(probs):
        if i:
            lines.append(r"\midrule")
        for j, (m, lab) in enumerate(methods):
            rows = [load(f"exp9_{p}_{n:g}_{m}.json") for n in noises]
            errs = [f"{100 * r['err']['shape']:.2f}" for r in rows]
            t = np.mean([r["time"] for r in rows])
            lines.append(f"{plab if j == 0 else ''} & {lab} & " + " & ".join(errs) + f" & {t:.3g}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "prototype.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    table_prototype()
    table_basin()
    table_forward()
    table_inverse()
    table_joint_detail()
    table_three_d()
    print("tables written to", os.path.abspath(OUT))
