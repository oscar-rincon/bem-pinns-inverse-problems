"""Figures for the paper (PDF, 7 in wide, Open Sans 6-8 pt)."""
import glob
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "..", "paper", "figures")
for f in glob.glob("/usr/share/fonts/truetype/open-sans/OpenSans-*.ttf"):
    font_manager.fontManager.addfont(f)
# Palette shared by all figures (blue-grey schematic style): ink for text and axes, navy and
# blue for the solvers, light steel and panel grey for fills, one copper tone for signed maps.
PAL = dict(ink="#2b3138", navy="#14324d", blue="#2f6690", slate="#4e6479", grey="#8a8f94",
           steel="#b9cbe0", mist="#dce5ef", panel="#e9e9e9", copper="#a8683a")
mpl.rcParams.update({
    "font.family": "Open Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "axes.titleweight": "bold", "axes.titlecolor": PAL["navy"],
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Open Sans", "mathtext.it": "Open Sans:italic",
    "mathtext.bf": "Open Sans:bold", "axes.linewidth": 0.7, "axes.edgecolor": PAL["ink"],
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "xtick.color": PAL["ink"], "ytick.color": PAL["ink"],
    "axes.labelcolor": PAL["ink"], "text.color": PAL["ink"], "axes.grid": False,
    "grid.color": "#edf0f3", "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.solid_capstyle": "round",
    "axes.axisbelow": True,
})
INK, GREY = PAL["ink"], PAL["grey"]
SEQ = mpl.colors.LinearSegmentedColormap.from_list("bp_seq", ["#ffffff", PAL["steel"], PAL["blue"], PAL["navy"]])
DIV = mpl.colors.LinearSegmentedColormap.from_list(
    "bp_div", [PAL["navy"], PAL["blue"], "#f4f6f8", "#d9b99b", PAL["copper"]])
# solver identity (fixed order); markers give a second, colour-free cue
SOLVER = {
    "lm_fd": dict(c=PAL["grey"], m="s", label="BEM + LM (finite-difference Jacobian)"),
    "lbfgs_adj": dict(c=PAL["navy"], m="o", label="BEM + L-BFGS (adjoint gradient)"),
    "bem_pinn": dict(c=PAL["blue"], m="^", label="BEM–PINN (this work)"),
}
CONTROL = dict(c=PAL["navy"], m="D", label="BEM + L-BFGS (adjoint) + frequency continuation")
PROBLEM = {"interface": "Interface identification", "surface": "Surface reconstruction",
           "joint": "Joint identification"}


def load(name):
    with open(os.path.join(RES, name)) as fh:
        return json.load(fh)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(RES, name + ".png"), dpi=200)
    plt.close(fig)


# --- Fig.: forward reproduction (Wu et al. 2026, Figs. 4-6) -------------------------------

def fig_forward():
    topos = [("canyon", "Semicircular canyon", "Trifunac (1973)"),
             ("mountain", "Semicircular hill", "IBEM, 1200 el."),
             ("vcanyon", "V-shaped canyon", "IBEM, 1200 el.")]
    angles = [0, 30, 60, 90]
    fig, axes = plt.subplots(3, 4, figsize=(7, 4.6), sharex=True)
    for i, (topo, title, reflabel) in enumerate(topos):
        for j, a in enumerate(angles):
            ax = axes[i, j]
            r = load(f"exp1_{topo}_{a}.json")
            x0, xs = np.array(r["x0"]), np.array(r["xs"])
            n0 = len(x0)
            x = np.concatenate([x0, xs])
            o = np.argsort(x)
            uref, ub, up = (np.array(r[k])[o] for k in ("u_ref", "u_bem", "u_pinn"))
            ax.plot(x[o], uref, color=INK, lw=1.4, label="Reference")
            ax.plot(x[o], ub, color=PAL["navy"], lw=1.0, ls=(0, (4, 2)), label="IBEM (100 el.)")
            ax.plot(x[o][::6], up[::6], ls="none", marker="^", ms=3.2, mfc="none", mew=0.7,
                    color=PAL["blue"], label="FF-BEM–PINN")
            ax.set_xlim(-3, 3)
            ax.set_ylim(0, max(4.6, uref.max() * 1.12))
            ax.text(0.03, 0.95, f"$\\alpha$ = {a}°", transform=ax.transAxes, va="top", fontsize=6.5)
            ax.text(0.97, 0.95, f"$\\varepsilon$ = {r['pinn']['eps']:.1e}", transform=ax.transAxes,
                    va="top", ha="right", fontsize=6, color=PAL["blue"])
            if j == 0:
                ax.set_ylabel(f"{title}\n|U|")
            if i == 2:
                ax.set_xlabel("x/a")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, ["Reference (canyon: analytical; hill: IBEM, 1200 el.; V-canyon: refined truncated IBEM)",
                   "IBEM (100 elements)", "FF-BEM–PINN (100 elements)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.035), handlelength=2.4)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, "forward_reproduction")


# --- Fig.: reconstructed geometries --------------------------------------------------------

def fig_reconstruction(noise=0.03):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.05), sharey=True)
    for ax, prob in zip(axes, PROBLEM):
        r0 = load(f"exp2_{prob}_{noise:g}_lm_fd.json")
        cv = r0["curves"]
        key = "yi" if prob == "interface" else "ys"
        other = "ys" if prob == "interface" else "yi"
        x = np.array(cv["true"]["x"])
        ax.fill_between(x, cv["true"]["yi"], cv["true"]["ys"], color=PAL["steel"], lw=0)
        ax.axhline(0, color=INK, lw=0.8, zorder=1)
        ax.plot(x, cv["true"][other], color=INK, lw=0.9)
        ax.plot(x, cv["true"][key], color=INK, lw=1.6, label="True")
        ax.plot(x, cv["init"][key], color=GREY, lw=0.9, ls=(0, (1.5, 1.5)), label="Initial guess")
        for m, st in SOLVER.items():
            r = load(f"exp2_{prob}_{noise:g}_{m}.json")
            xe = np.array(r["curves"]["est"]["x"])
            ye = np.array(r["curves"]["est"][key])
            ax.plot(xe, ye, color=st["c"], lw=1.0, ls=(0, (4, 2)))
            ax.plot(xe[::20], ye[::20], ls="none", marker=st["m"], ms=3.2, mfc="white",
                    mew=0.8, color=st["c"], label=st["label"])
        ax.set_title(PROBLEM[prob])
        ax.set_xlabel("x/a")
        ax.set_xlim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_ylim(-0.95, 0.95)
    axes[0].set_ylabel("y/a")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.07), handlelength=2.0,
               columnspacing=1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "inverse_reconstruction")


# --- Fig.: accuracy vs computational time ------------------------------------------------

NOISE_FILL = {0.0: "full", 0.01: "left", 0.03: "none"}


def fig_accuracy_time():
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True)
    for ax, prob in zip(axes, PROBLEM):
        for m, st in SOLVER.items():
            pts = []
            for nz in (0.0, 0.01, 0.03):
                r = load(f"exp2_{prob}_{nz:g}_{m}.json")
                e = r["err"]
                err = max(e["shape"], e.get("mu1", 0), e.get("rho1", 0))
                pts.append((r["time"], err))
                ax.plot(r["time"], err, marker=st["m"], ms=5, color=st["c"], ls="none",
                        fillstyle=NOISE_FILL[nz], mew=0.9, markerfacecoloralt="white")
            t, e = np.array(pts).T
            ax.plot(t, e, color=st["c"], lw=0.5, alpha=0.6)
        pts = []
        for nz in (0.0, 0.01, 0.03):
            r = load(f"exp2_{prob}_{nz:g}_lbfgs_adj_cont.json")
            e = r["err"]
            pts.append((r["time"], max(e["shape"], e.get("mu1", 0), e.get("rho1", 0))))
            ax.plot(*pts[-1], marker="D", ms=4, color=CONTROL["c"], ls="none", fillstyle=NOISE_FILL[nz],
                    mew=0.8, markerfacecoloralt="white")
        t, e = np.array(pts).T
        ax.plot(t, e, color=CONTROL["c"], lw=0.6, ls=(0, (2, 1.5)), alpha=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(PROBLEM[prob])
        ax.set_xlabel("Wall-clock time (s)")
        ax.set_xlim(0.1, 3000)
    axes[0].set_ylabel("Relative error\n(max of shape and material)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=st["c"], marker=st["m"], ls="none", ms=5, label=st["label"])
         for st in SOLVER.values()]
    h.insert(2, Line2D([], [], color=CONTROL["c"], marker="D", ls=(0, (2, 1.5)), ms=4, lw=0.6,
                       label=CONTROL["label"]))
    h += [Line2D([], [], color=GREY, marker="o", ls="none", ms=5, fillstyle=f,
                 markerfacecoloralt="white", label=f"noise {int(nz * 100)}%")
          for nz, f in NOISE_FILL.items()]
    fig.legend(handles=h, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), columnspacing=1.0,
               handletextpad=0.3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "inverse_accuracy_time")


# --- Fig.: convergence histories ----------------------------------------------------------

def fig_convergence(noise=0.01):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.1), sharey=True)
    for ax, prob in zip(axes, PROBLEM):
        for m, st in SOLVER.items():
            r = load(f"exp2_{prob}_{noise:g}_{m}.json")
            t = np.array([h["t"] for h in r["hist"]])
            e = np.array([max(h["shape"], h.get("mu1", 0), h.get("rho1", 0)) for h in r["hist"]])
            t = np.maximum(t, 1e-2)
            ax.plot(t, e, color=st["c"], lw=1.1, label=st["label"])
            ax.plot(t[-1], e[-1], marker=st["m"], color=st["c"], ms=4.5, ls="none")
            if m == "bem_pinn":
                st1 = [h["t"] for h in r["hist"] if h["stage"] == 1]
                if st1:
                    ax.axvspan(1e-2, max(st1), color=PAL["steel"], alpha=0.45, lw=0)
        r = load(f"exp2_{prob}_{noise:g}_lbfgs_adj_cont.json")
        t = np.maximum(np.array([h["t"] for h in r["hist"]]), 1e-2)
        e = np.array([max(h["shape"], h.get("mu1", 0), h.get("rho1", 0)) for h in r["hist"]])
        ax.plot(t, e, color=CONTROL["c"], lw=1.0, ls=(0, (2, 1.5)), label=CONTROL["label"])
        ax.plot(t[-1], e[-1], marker="D", color=CONTROL["c"], ms=3.8, ls="none", mfc="white", mew=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 3000)
        ax.set_title(PROBLEM[prob])
        ax.set_xlabel("Wall-clock time (s)")
    axes[0].set_ylabel("Relative error")
    axes[0].text(0.02, 0.04, "shaded: BEM–PINN pre-training\nat the initial guess",
                 transform=axes[0].transAxes, fontsize=5.8, color=PAL["blue"])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=0.6)
    save(fig, "inverse_convergence")


# --- Fig.: cost per iteration vs number of elements -------------------------------------

def fig_scaling():
    d = load("exp3_scaling.json")
    rows = d["rows"]
    n = np.array([r["n_unknowns"] for r in rows])
    series = [("fd_jacobian", "LM iteration (2P forward solves)", PAL["grey"], "s", "-"),
              ("adjoint", "L-BFGS iteration (forward + adjoint)", PAL["navy"], "o", "-"),
              ("pinn_joint", "BEM–PINN joint step (assembly + network)", PAL["blue"], "^", "-"),
              ("pinn_fixed", "BEM–PINN step at fixed geometry", PAL["blue"], "^", (0, (3, 2))),
              ("forward", "Forward solve (assembly + LU)", INK, "D", "-"),
              ("solve", "LU solve only", PAL["slate"], "D", (0, (3, 2)))]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    for key, lab, c, mk, ls in series:
        y = np.array([r[key] for r in rows])
        ax.plot(n, y, color=c, marker=mk, ms=3.5, lw=1.0, ls=ls, label=lab,
                mfc="white" if ls != "-" else c, mew=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of unknown densities N")
    ax.set_ylabel(f"Time per iteration (s), {d['threads']} CPU threads")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=2.4)
    ax.set_title("Cost of one iteration")
    # total cost = iterations x cost per iteration, for the 1% noise runs
    ax = axes[1]
    probs = list(PROBLEM)
    w = 0.26
    for k, (m, st) in enumerate(SOLVER.items()):
        tt = [load(f"exp2_{p}_0.01_{m}.json")["time"] for p in probs]
        xs = np.arange(len(probs)) + (k - 1) * w
        ax.bar(xs, tt, width=w - 0.03, color=st["c"], label=st["label"])
        for xx, t in zip(xs, tt):
            ax.text(xx, t * 1.15, f"{t:.3g}", ha="center", va="bottom", fontsize=5.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(probs)))
    ax.set_xticklabels([PROBLEM[p].replace(" ", "\n", 1) for p in probs])
    ax.set_ylabel("Total inversion time (s)")
    ax.set_title("Total time, noise 1%")
    ax.set_ylim(0.1, 1e4)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "computational_cost")


# --- Fig.: frequency as network input ------------------------------------------------------

def fig_frequency():
    runs = [("exp4_frequency.json", None, r"$\eta\in[1.0, 1.5]$ (relative band 50%)"),
            ("exp4_frequency_narrow.json", "exp4_frequency_narrow_wd.json",
             r"$\eta\in[1.0, 1.125]$ (relative band 12.5%, as Wu et al.)")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2), sharey=True)
    for ax, (fn, fn_wd, title) in zip(axes, runs):
        d = load(fn)
        e = np.array(d["eta_test"])
        ax.plot(e, d["eps_bem"], color=PAL["navy"], marker="o", ms=3.5, lw=1.0, label="IBEM, one solve per frequency")
        ax.plot(e, d["eps_pinn"], color=PAL["blue"], marker="^", ms=4, lw=1.0,
                label=r"FF-BEM–PINN, input $(x_0, y_0, \eta)$")
        txt = f"IBEM {d['t_bem']:.1f} s   ·   BEM–PINN {d['t_pinn'] / 60:.0f} min"
        if fn_wd and os.path.exists(os.path.join(RES, fn_wd)):
            w = load(fn_wd)
            ax.plot(e, w["eps_pinn"], color=PAL["blue"], marker="^", ms=4, lw=0.9, ls=(0, (3, 1.5)),
                    mfc="white", label=f"FF-BEM–PINN with weight decay {w['weight_decay']:g}")
            txt += f"   ·   with weight decay {w['t_pinn'] / 60:.0f} min"
        for et in d["eta_train"]:
            ax.axvline(et, color=GREY, lw=0.6, ls=(0, (2, 2)), zorder=0)
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 3)
        ax.set_xlabel(r"$\eta$  (dashed: training frequencies)")
        ax.set_title(title)
        ax.text(0.5, 0.95, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=INK)
    axes[0].set_ylabel(r"Error $\varepsilon$")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(w_pad=0.6)
    save(fig, "frequency_input")


# --- Fig.: iterative (matrix-free) regime, 2D -------------------------------------------

ITER = {"lbfgs_gmres": dict(c=PAL["navy"], m="o", label="BEM + L-BFGS (adjoint), GMRES solves"),
        "lm_gmres": dict(c=PAL["grey"], m="s", label="BEM + LM (FD Jacobian), GMRES solves"),
        "bem_pinn": dict(c=PAL["blue"], m="^", label="BEM–PINN (this work)"),
        "lbfgs_lu": dict(c=INK, m="D", label="BEM + L-BFGS (adjoint), LU solves")}


def fig_iterative():
    ns = [40, 80, 160, 320]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2))
    for m, st in ITER.items():
        rows = [load(f"exp5_{n}_{m}.json") for n in ns
                if os.path.exists(os.path.join(RES, f"exp5_{n}_{m}.json"))]
        if not rows:
            continue
        N = np.array([r["n_unknowns"] for r in rows])
        kw = dict(color=st["c"], marker=st["m"], ms=4, lw=1.0, label=st["label"])
        if m != "lbfgs_lu":
            axes[0].plot(N, [(r["matvec"] or 0) + (r["rmatvec"] or 0) for r in rows], **kw)
        axes[1].plot(N, [r["time"] for r in rows], **kw)
        axes[2].plot(N, [r["err"]["shape"] for r in rows], **kw)
    for ax, yl, t in zip(axes, ["Products with A and A$^H$", "Wall-clock time (s)", "Shape error"],
                         ["Matrix–vector products", "Time", "Accuracy"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([120, 240, 480, 960])
        ax.set_xticklabels(["120", "240", "480", "960"])
        ax.minorticks_off()
        ax.set_xlabel("Number of unknown densities N")
        ax.set_ylabel(yl)
        ax.set_title(t)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout(w_pad=0.8)
    save(fig, "iterative_regime")


# --- Fig.: 3D canyon reconstruction -----------------------------------------------------

def fig_three_d(noise=0.01):
    import matplotlib.tri as mtri
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    sys.path.insert(0, os.path.join(HERE, ".."))
    from bempinn import bem3d, inverse3d as i3
    import torch
    case = i3.Case3D()
    fig = plt.figure(figsize=(7, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.35)
    # perspective view of the true canyon (inversion mesh) and receivers
    ax = fig.add_subplot(gs[0], projection="3d")
    P, T = bem3d.disk_mesh(case.n_rings)
    mesh = bem3d.Mesh3D(P, T, bem3d.canyon(torch.tensor(case.xi, dtype=torch.float64)))
    X = bem3d.canyon(torch.tensor(case.xi, dtype=torch.float64))(P).numpy()
    # flat ground around the canyon, then the canyon surface
    rg, tg = np.meshgrid(np.linspace(1.0, 2.8, 8), np.linspace(0, 2 * np.pi, 49))
    ax.plot_surface(rg * np.cos(tg), rg * np.sin(tg), 0 * rg, color=PAL["panel"], linewidth=0,
                    alpha=0.45, shade=False, zorder=0)
    ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=T.numpy(), color=PAL["steel"],
                    edgecolor=PAL["slate"], linewidth=0.15, alpha=1.0, shade=True)
    R = case.receivers().numpy()
    ax.scatter(R[:, 0], R[:, 1], R[:, 2] + 0.04, marker="v", s=9, color=PAL["navy"], depthshade=False,
               zorder=10)
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(th), np.sin(th), 0 * th, color=INK, lw=0.6)
    ax.set_box_aspect((1, 1, 0.12))
    ax.view_init(elev=38, azim=-60)
    ax.set_axis_off()
    ax.set_title(f"Canyon mesh ({len(mesh)} triangles)\nand 36 receivers", fontsize=6.5)
    # depth maps
    r0 = load(f"exp6_{noise:g}_lbfgs_lu.json")
    uv = np.array(r0["uv"])
    tri = mtri.Triangulation(uv[:, 0], uv[:, 1])
    ztrue = np.array(r0["maps"]["true"])
    lev = np.arange(0, 0.66, 0.06)
    ax = fig.add_subplot(gs[1])
    cs = ax.tricontourf(tri, ztrue, levels=lev, cmap=SEQ)
    ax.set_title("True depth / a", fontsize=7)
    panels = [("lbfgs_lu", "Error, BEM + L-BFGS"), ("bem_pinn", "Error, BEM–PINN")]
    emax = 0
    errs = []
    for m, _ in panels:
        fn = f"exp6_{noise:g}_{m}.json"
        if os.path.exists(os.path.join(RES, fn)):
            r = load(fn)
            e = np.array(r["maps"]["est"]) - ztrue
            errs.append((r, e))
            emax = max(emax, np.abs(e).max())
        else:
            errs.append((None, None))
    for k, ((m, title), (r, e)) in enumerate(zip(panels, errs)):
        ax = fig.add_subplot(gs[2 + k])
        if e is not None:
            ce = ax.tricontourf(tri, e, levels=np.linspace(-0.05, 0.05, 11), cmap=DIV)
            ax.set_title(f"{title}\n{100 * r['err']['shape']:.1f}%, {r['time']:.0f} s", fontsize=6.5)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels([])
        ax.grid(False)
    for ax in fig.axes[1:]:
        ax.set_aspect("equal")
        ax.set_xlabel("x/a")
        ax.grid(False)
    fig.axes[1].set_ylabel("y/a")
    fig.axes[1].set_xticks([-1, 0, 1])
    fig.axes[1].set_yticks([-1, 0, 1])
    cb = fig.colorbar(cs, ax=fig.axes[1], fraction=0.046, pad=0.03, ticks=[0, 0.2, 0.4, 0.6])
    cb.ax.tick_params(labelsize=5.5)
    if emax:
        cb2 = fig.colorbar(ce, ax=fig.axes[2:], fraction=0.03, pad=0.03, ticks=[-0.05, -0.025, 0, 0.025, 0.05])
        cb2.ax.tick_params(labelsize=5.5)
        cb2.set_label("depth error / a", fontsize=6)
    save(fig, "three_d_reconstruction")


def fig_scaling3d():
    d = load("exp7_scaling3d.json")
    rows = d["rows"]
    n = np.array([r["n_tri"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axes[0]
    for key, lab, c, mk, ls in (("adjoint", "L-BFGS iteration (forward + adjoint)", PAL["navy"], "o", "-"),
                                ("pinn_joint", "BEM–PINN joint step", PAL["blue"], "^", "-"),
                                ("pinn_fixed", "BEM–PINN step at fixed geometry", PAL["blue"], "^", (0, (3, 2))),
                                ("assembly", "Kernel assembly", INK, "D", "-"),
                                ("solve", "LU solve", PAL["slate"], "D", (0, (3, 2)))):
        sel = [(r["n_tri"], r[key]) for r in rows if key in r]
        x, y = np.array(sel).T
        ax.plot(x, y, color=c, marker=mk, ms=3.5, lw=1.0, ls=ls, label=lab,
                mfc="white" if ls != "-" else c, mew=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of triangles N")
    ax.set_ylabel(f"Time per iteration (s), {d['threads']} threads")
    ax.legend(fontsize=5.8, loc="upper left")
    ax.set_title("Cost of one iteration (3D)")
    ax = axes[1]
    ax.plot(n, [r["gmres_its_per_solve"] for r in rows], color=PAL["navy"], marker="o", ms=3.5, lw=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0, 16)
    ax.set_xlabel("Number of triangles N")
    ax.set_ylabel("GMRES iterations per solve")
    ax.set_title("Preconditioned GMRES")
    for r in rows:
        ax.annotate(f"{r['matrix_MB']:.0f} MB", (r["n_tri"], r["gmres_its_per_solve"]),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=5.5, color=GREY)
    fig.tight_layout(w_pad=1.0)
    save(fig, "scaling_3d")


# --- Fig.: basin of attraction -----------------------------------------------------------

BASIN_BANDS = [("low", r"$\eta \in \{0.25, 0.5, 0.75, 1\}$"), ("mid", r"$\eta \in \{0.5, 1, 1.5, 2\}$"),
               ("high", r"$\eta \in \{1.5, 1.75, 2, 2.25\}$")]
BASIN_H0 = [0.1, 0.2, 0.35, 0.5, 0.9, 1.1, 1.3, 1.5]
BASIN_TOL = 0.02


def fig_basin():
    styles = [("lm_fd", SOLVER["lm_fd"], -0.024), ("lbfgs_adj", SOLVER["lbfgs_adj"], -0.008),
              ("lbfgs_adj_cont", CONTROL, 0.008), ("bem_pinn", SOLVER["bem_pinn"], 0.024)]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.2), sharey=True)
    for ax, (band, title) in zip(axes, BASIN_BANDS):
        for m, st, dx in styles:
            pts = [(h, load(f"exp8_{band}_{h:g}_{m}.json")["err"]["shape"]) for h in BASIN_H0
                   if os.path.exists(os.path.join(RES, f"exp8_{band}_{h:g}_{m}.json"))]
            if pts:
                h, e = np.array(pts).T
                ax.plot(h + dx, e, ls="none", marker=st["m"], ms=4, color=st["c"],
                        mfc=st["c"] if m != "lbfgs_adj_cont" else "white", label=st["label"])
        ax.axhline(BASIN_TOL, color=GREY, lw=0.6, ls="--")
        ax.axvline(0.70, color=GREY, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel(r"Initial guess $\xi_{s,1}^0/a$ (true value 0.70)")
        ax.set_title(title)
    axes[0].set_ylabel("Final shape error")
    leg = {}
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            leg.setdefault(ll, hh)
    l, h = list(leg), list(leg.values())
    fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout(w_pad=0.6)
    save(fig, "basin_of_attraction")


# --- Fig.: prototyping workflow (schematic) ----------------------------------------------

def _box(ax, x, y, w, h, text, kind="num", fs=6.5):
    from matplotlib.patches import FancyBboxPatch
    st = {"net": dict(fc=PAL["steel"], ec="none", lw=0), "num": dict(fc=PAL["panel"], ec="none", lw=0),
          "plug": dict(fc="white", ec=PAL["blue"], lw=0.9, ls=(0, (3, 2))),
          "loss": dict(fc="white", ec=PAL["ink"], lw=0.8)}[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0", **st))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=PAL["ink"], linespacing=1.15)


def _circ(ax, x, y, text, r=0.27):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), r, fc="white", ec=PAL["navy"], lw=0.9, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=7, color=PAL["ink"], zorder=4)


def _path(ax, pts, dashed=False, head=True):
    col = PAL["blue"] if dashed else PAL["ink"]
    ls = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=col, lw=0.8, ls=ls, zorder=1, solid_capstyle="butt")
    if head:
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))


def _net_icon(ax, x, y, w, h):
    layers = [3, 4, 4, 2]
    xs = np.linspace(x - 0.36 * w, x + 0.36 * w, len(layers))
    nodes = [[(xi, yi) for yi in np.linspace(y - 0.28 * h, y + 0.28 * h, n)] for xi, n in zip(xs, layers)]
    for a, b in zip(nodes[:-1], nodes[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="white", lw=0.35, zorder=2)
    for layer in nodes:
        ax.scatter(*zip(*layer), s=7, color="white", zorder=3, lw=0)


def fig_prototyping():
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    W, H = 7.0, 5.0
    fig = plt.figure(figsize=(W, H))
    xl = (0.0, 13.4)
    sc = W / (xl[1] - xl[0])                         # inches per data unit (equal aspect)
    ya, yb = (-0.4, 4.85), (-0.05, 3.3)
    hb, ha = (yb[1] - yb[0]) * sc, (ya[1] - ya[0]) * sc
    axb = fig.add_axes([0, 0.32 / H, 1, hb / H])
    axa = fig.add_axes([0, (0.32 + hb + 0.32) / H, 1, ha / H])
    for ax, yl in ((axa, ya), (axb, yb)):
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect("equal")
        ax.axis("off")

    # (a) BEM-PINN: one loss, no linear solve
    ax, yt, yb_ = axa, 3.6, 1.0
    ax.set_title("(a) BEM–PINN: one loss, no linear solve", pad=4, loc="left")
    _circ(ax, 0.5, yt, r"$\hat s$", r=0.3)
    _box(ax, 2.3, yt, 1.8, 1.2, "", "net")
    _net_icon(ax, 2.3, yt, 1.8, 1.2)
    ax.text(2.3, yt - 0.7, r"network NN$_\theta$", ha="center", va="top", fontsize=6.5)
    _circ(ax, 4.4, yt, r"$\hat\phi$", r=0.3)
    _circ(ax, 0.5, yb_, r"$\vartheta$", r=0.3)
    _box(ax, 2.3, yb_, 2.0, 0.8, "geometry and\nmaterial parameters", "plug")
    _box(ax, 4.9, yb_, 1.9, 1.05, "differentiable\nBEM assembly\n$\\mathbf{A},\\ \\mathbf{b}$", "num")
    _box(ax, 8.4, yt, 2.4, 0.7,
         r"$L_{BC}=\|\mathbf{A}\hat\phi-\mathbf{b}\|^2/\|\mathbf{b}\|^2$", "loss", fs=6.0)
    _box(ax, 8.4, 2.2, 2.4, 0.8, "data misfit\n$\\lambda_d L_d(\\mathbf{u},d)$", "plug")
    _box(ax, 8.4, 0.8, 2.4, 0.55, r"penalty $\lambda_r R(\vartheta)$", "plug")
    _circ(ax, 9.15, 1.47, r"$d$", r=0.2)
    _circ(ax, 11.6, 2.2, r"$\Sigma$", r=0.3)
    _path(ax, [(0.8, yt), (1.4, yt)])
    _path(ax, [(3.2, yt), (4.1, yt)])
    _path(ax, [(4.7, yt), (7.2, yt)])
    _path(ax, [(4.4, yt - 0.3), (4.4, 2.38), (7.2, 2.38)])
    _path(ax, [(0.8, yb_), (1.3, yb_)])
    _path(ax, [(3.3, yb_), (3.95, yb_)])
    _path(ax, [(5.85, yb_), (6.6, yb_), (6.6, 3.4), (7.2, 3.4)])
    _path(ax, [(6.6, 2.02), (7.2, 2.02)])
    ax.scatter([6.6], [2.02], s=6, color=PAL["ink"], zorder=3, lw=0)
    _path(ax, [(2.3, yb_ - 0.4), (2.3, 0.3), (8.4, 0.3), (8.4, 0.525)])
    _path(ax, [(9.15, 1.67), (9.15, 1.8)])
    _path(ax, [(9.6, yt), (11.6, yt), (11.6, 2.5)])
    _path(ax, [(9.6, 2.2), (11.3, 2.2)])
    _path(ax, [(9.6, 0.8), (11.6, 0.8), (11.6, 1.9)])
    _path(ax, [(11.9, 2.2), (12.7, 2.2), (12.7, 4.55), (2.3, 4.55), (2.3, 4.2)], dashed=True)
    _path(ax, [(12.7, 2.2), (12.7, -0.05), (0.5, -0.05), (0.5, 0.7)], dashed=True)
    ax.text(7.5, 4.6, r"joint optimization of network weights $\theta$ and unknowns $\vartheta$ (AdamW, then L-BFGS)",
            ha="center", va="bottom", fontsize=6.5, color=PAL["blue"])
    ax.text(7.5, -0.1, "gradients by automatic differentiation through the assembly",
            ha="center", va="top", fontsize=6.5, color=PAL["blue"])

    # (b) classical: reduced objective, adjoint gradient
    ax, y = axb, 1.6
    ax.set_title("(b) BEM + L-BFGS: reduced objective, adjoint gradient", pad=4, loc="left")
    _circ(ax, 0.5, y, r"$\vartheta$", r=0.3)
    _box(ax, 2.3, y, 2.0, 0.8, "geometry and\nmaterial parameters", "plug")
    _box(ax, 4.75, y, 1.9, 1.05, "differentiable\nBEM assembly\n$\\mathbf{A},\\ \\mathbf{b}$", "num")
    _box(ax, 7.0, y, 1.9, 1.05, "linear solve\n$\\mathbf{A}\\boldsymbol{\\phi}=\\mathbf{b}$\n(LU or GMRES)", "num")
    _circ(ax, 8.7, y, r"$\phi$", r=0.3)
    _box(ax, 10.4, y, 2.0, 0.8, "objective\n$L_d(\\mathbf{u},d)+\\lambda_r R(\\vartheta)$", "plug", fs=6.0)
    _circ(ax, 10.4, 2.55, r"$d$", r=0.2)
    _circ(ax, 12.3, y, r"$J$", r=0.3)
    _path(ax, [(0.8, y), (1.3, y)])
    _path(ax, [(3.3, y), (3.8, y)])
    _path(ax, [(5.7, y), (6.05, y)])
    _path(ax, [(7.95, y), (8.4, y)])
    _path(ax, [(9.0, y), (9.4, y)])
    _path(ax, [(11.4, y), (12.0, y)])
    _path(ax, [(10.4, 2.35), (10.4, 2.0)])
    _path(ax, [(2.3, 2.0), (2.3, 2.44), (10.0, 2.44), (10.0, 2.0)])
    _path(ax, [(12.3, 1.9), (12.3, 3.0), (0.5, 3.0), (0.5, 1.9)], dashed=True)
    ax.text(6.4, 3.05, r"L-BFGS-B on $\vartheta$ only (tens of iterations)", ha="center", va="bottom",
            fontsize=6.5, color=PAL["blue"])
    _box(ax, 7.0, 0.42, 5.6, 0.62, "", "plug")
    ax.text(7.0, 0.42, "adjoint solve $\\mathbf{A}^H\\boldsymbol{\\lambda}=\\partial L_d/\\partial\\boldsymbol{\\phi}$\n"
            "automatic for LU; custom backward for GMRES in this implementation", ha="center",
            va="center", fontsize=6.2, color=PAL["ink"], linespacing=1.2)
    _path(ax, [(10.4, y - 0.4), (10.4, 0.42), (9.8, 0.42)], dashed=True)
    _path(ax, [(7.0, 0.73), (7.0, y - 0.525)], dashed=True)

    handles = [Patch(fc=PAL["steel"], ec="none", label="trainable network"),
               Patch(fc=PAL["panel"], ec="none", label="boundary element computations"),
               Patch(fc="white", ec=PAL["blue"], ls=(0, (3, 2)), lw=0.9,
                     label="component changed when prototyping"),
               Line2D([], [], color=PAL["blue"], lw=0.8, ls=(0, (3, 2)), label="gradient / optimization")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.4)
    save(fig, "prototyping_workflow")


if __name__ == "__main__":
    which = sys.argv[1:] or ["forward", "reconstruction", "accuracy_time", "convergence", "scaling"]
    for w in which:
        globals()["fig_" + w]()
