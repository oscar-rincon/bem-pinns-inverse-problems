"""Reproduce the guide figures using definitions from the original notebook."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parent / "bigardi.ipynb"
FIGURES = ROOT / "figuras"
DATA = ROOT / "datos"
SOURCE_CELLS = {
    "cell-02", "cell-06", "cell-08", "cell-11",
    "cell-13", "cell-15", "cell-19",
}
ASSIGNMENTS = {
    "R", "RHO", "MU", "NU", "BETA", "P", "mat", "m0", "x1", "OMEGA1", "GX", "GW",
}


def load_notebook():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    statements = []
    found = set()
    for cell in notebook["cells"]:
        if cell.get("id") not in SOURCE_CELLS:
            continue
        found.add(cell["id"])
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
                statements.append(node)
            elif isinstance(node, ast.Assign):
                names = {
                    item.id
                    for target in node.targets
                    for item in ast.walk(target)
                    if isinstance(item, ast.Name)
                }
                if names and names <= ASSIGNMENTS:
                    statements.append(node)
    if found != SOURCE_CELLS:
        raise RuntimeError(f"Notebook source cells missing: {sorted(SOURCE_CELLS - found)}")
    namespace = {"__name__": "notebook_bem_definitions"}
    module = ast.Module(body=statements, type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def solve_fields(bem, omega, pressure):
    H, G = bem.matrices(omega)
    normal = bem.nodes / bem.R
    traction = (-pressure * normal).ravel()
    rhs = G @ traction
    displacement = np.linalg.solve(H, rhs)
    relative_residual = np.linalg.norm(H @ displacement - rhs) / (
        np.linalg.norm(H) * np.linalg.norm(displacement) + np.linalg.norm(rhs)
    )
    vector = displacement.reshape(-1, 2)
    radial = np.sum(vector * normal, axis=1)
    tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
    tangential = np.sum(vector * tangent, axis=1)
    return radial, tangential, relative_residual, H


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    FIGURES.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    ns = load_notebook()
    R, pressure = ns["R"], ns["P"]
    material, analytical = ns["material"], ns["u_r_analitico"]
    shape, dshape = ns["shape"], ns["dshape"]
    quadrature = ns["puntos_cuadratura"]
    Bem = ns["BEMCilindro"]
    omega1 = ns["OMEGA1"]
    mat = ns["mat"]
    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "figure.dpi": 120, "savefig.dpi": 160,
    })
    summary = {
        "notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "omega1": float(omega1),
        "x1": float(ns["x1"]),
        "static_undamped_magnitude": float(pressure * R / (2 * (ns["MU"] + ns["m0"]["lam"].real))),
        "static_damped_magnitude": float(abs(-pressure * R / (2 * (mat["lam"] + mat["mu"])))),
        "checks": {},
    }

    eta = np.linspace(-1, 1, 401)
    np.testing.assert_allclose(shape(eta).sum(axis=-1), 1, atol=1e-14)
    np.testing.assert_allclose(dshape(eta).sum(axis=-1), 0, atol=1e-14)
    np.testing.assert_allclose(shape(np.array([-1, 0, 1])), np.eye(3), atol=1e-14)
    for source in (None, -1.0, 0.0, 1.0):
        points, weights = quadrature(source)
        np.testing.assert_allclose(weights.sum(), 2, atol=1e-14)
        if source is not None:
            check(np.all(points != source), "Quadrature sampled the singular point")
    bems = {ne: Bem(R, ne, mat) for ne in (4, 8, 16, 32, 64)}
    for ne, bem in bems.items():
        check(bem.nodes.shape == (2 * ne, 2), "Unexpected mesh shape")
        for direction in np.eye(2):
            rigid = np.tile(direction, bem.N)
            np.testing.assert_allclose(bem.Hs @ rigid, 0, atol=1e-13)
    summary["checks"]["shape_functions_quadrature_static_translation"] = "passed"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    theta = np.linspace(0, 2 * np.pi, 600)
    for ax, ne in zip(axes, (8, 32)):
        bem = bems[ne]
        ax.plot(R * np.cos(theta), R * np.sin(theta), "--", color="0.7", label="Circulo exacto")
        for k, conn in enumerate(bem.conn):
            points = shape(eta) @ bem.nodes[conn]
            ax.plot(*points.T, color=f"C{k % 2}", lw=1.5)
        ax.plot(*bem.nodes[::2].T, "ks", ms=3, label="Extremos")
        ax.plot(*bem.nodes[1::2].T, "o", ms=3, mfc="white", mec="C0", label="Intermedios")
        ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]",
               title=f"{ne} elementos; {bem.N} nodos")
    axes[0].legend(fontsize=8, loc="lower left")
    save_figure(fig, "mallas")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for local in range(3):
        axes[0].plot(eta, shape(eta)[:, local], label=f"P{local + 1}")
        axes[1].plot(eta, dshape(eta)[:, local], label=f"P{local + 1}'")
    for ax in axes:
        ax.set(xlabel="Coordenada local eta", xlim=(-1, 1))
        ax.legend()
    axes[0].set_ylabel("Funcion de forma")
    axes[1].set_ylabel("Derivada")
    save_figure(fig, "funciones_forma")

    fig, ax = plt.subplots(figsize=(9, 3))
    for row, source in enumerate((None, -1.0, 0.0, 1.0)):
        points, _ = quadrature(source)
        ax.plot(points, np.full_like(points, -row), "|", ms=12)
        if source is not None:
            ax.plot(source, -row, "r*", ms=10)
    ax.set(yticks=[0, -1, -2, -3],
           yticklabels=["Regular", "Fuente en -1", "Fuente en 0", "Fuente en +1"],
           xlabel="Coordenada local eta", ylim=(-3.5, 0.5))
    save_figure(fig, "cuadratura")

    rows = []
    nodal_at_resonance = {}
    for omega in (10.0, 40.0, omega1, 100.0, 150.0):
        exact = analytical(omega)
        row = {"omega": float(omega), "analytical_amplitude": float(abs(exact))}
        for ne in (8, 32):
            radial, tangential, residual, H = solve_fields(bems[ne], omega, pressure)
            amplitude = np.abs(radial).mean()
            row[f"bem{ne}_amplitude"] = float(amplitude)
            row[f"bem{ne}_amplitude_error"] = float(abs(amplitude - abs(exact)) / abs(exact))
            row[f"bem{ne}_complex_rms_error"] = float(
                np.linalg.norm(radial - exact) / (np.sqrt(radial.size) * abs(exact))
            )
            row[f"bem{ne}_tangential_rms"] = float(
                np.linalg.norm(tangential) / (np.sqrt(radial.size) * abs(exact))
            )
            row[f"bem{ne}_backward_residual"] = float(residual)
            check(residual < 1e-13, "Linear system backward residual too large")
            check(row[f"bem{ne}_complex_rms_error"] < 0.02, "Complex field validation failed")
            if omega == omega1:
                nodal_at_resonance[ne] = radial
                if ne == 8:
                    fig, ax = plt.subplots(figsize=(5.4, 4.5))
                    image = ax.imshow(np.log10(np.maximum(np.abs(H), 1e-16)), cmap="viridis")
                    fig.colorbar(image, ax=ax, label="log10(|H_ij|)")
                    ax.set(xlabel="Columna: nodo/componente", ylabel="Fila: colocacion/componente",
                           title="H dinamica, 8 elementos, primera resonancia")
                    save_figure(fig, "matriz_h")
        rows.append(row)
    summary["verification"] = rows
    table = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\omega$ & $|u_r^{\rm an}|$ [m] & BEM 8 [m] & $e_{A,8}$ & BEM 32 [m] & $e_{A,32}$\\",
        r"\midrule",
    ]
    for row in rows:
        table.append(
            f"{row['omega']:.2f} & {row['analytical_amplitude']:.5e} & "
            f"{row['bem8_amplitude']:.5e} & {row['bem8_amplitude_error']:.2e} & "
            f"{row['bem32_amplitude']:.5e} & {row['bem32_amplitude_error']:.2e}" + r"\\"
        )
    table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "tabla_verificacion.tex").write_text("\n".join(table) + "\n", encoding="utf-8")

    fine = np.linspace(10, 150, 2001)
    sweep = {}
    for ne, count in ((8, 141), (32, 71)):
        frequencies = np.linspace(10, 150, count)
        amplitudes = np.array([np.abs(bems[ne].resolver(w)).mean() for w in frequencies])
        errors = abs(amplitudes - abs(analytical(frequencies))) / abs(analytical(frequencies))
        check(np.max(errors) < 0.01, f"Amplitude sweep error exceeds 1 percent: mesh {ne}")
        sweep[ne] = (frequencies, amplitudes, errors)
        summary[f"sweep{ne}_max_amplitude_error"] = float(errors.max())
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(fine / omega1, abs(analytical(fine)) * 1e4, "k-", label="Analitica")
    for ne, (frequencies, amplitudes, errors) in sweep.items():
        style = "--" if ne == 8 else "o"
        axes[0].plot(frequencies / omega1, amplitudes * 1e4, style,
                     ms=3, mfc="none", label=f"BEM {ne}")
        axes[1].semilogy(frequencies / omega1, errors, label=f"BEM {ne}")
    axes[0].set(ylabel="Amplitud radial [10^-4 m]")
    axes[1].set(xlabel="Frecuencia normalizada omega/omega1", ylabel="Error relativo de amplitud")
    for ax in axes:
        ax.axvline(1, color="0.6", lw=1)
        ax.legend()
    save_figure(fig, "respuesta_frecuencia")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for beta in (0.01, 0.03, 0.05, 0.10):
        displacement = analytical(fine, m=material(beta=beta))
        axes[0].plot(fine / omega1, abs(displacement) * 1e4, label=f"beta={beta}")
    axes[0].set(xlabel="omega/omega1", ylabel="Amplitud radial [10^-4 m]")
    axes[0].legend()
    inward_response = -analytical(fine)
    axes[1].plot(fine / omega1, np.unwrap(np.angle(inward_response)) * 180 / np.pi)
    axes[1].set(xlabel="omega/omega1", ylabel="Fase de -u_r respecto de p [grados]")
    save_figure(fig, "amortiguamiento_fase")

    convergence = []
    fig, ax = plt.subplots(figsize=(6, 4.2))
    elements = np.array(sorted(bems))
    for omega in (30.0, omega1, 120.0):
        exact = abs(analytical(omega))
        errors = np.array([
            abs(np.abs(bems[int(ne)].resolver(omega)).mean() - exact) / exact
            for ne in elements
        ])
        slopes = -np.diff(np.log(errors)) / np.diff(np.log(elements))
        convergence.append({"omega": float(omega), "elements": elements.tolist(),
                            "errors": errors.tolist(), "slopes": slopes.tolist()})
        ax.loglog(elements, errors, "o-", label=f"omega={omega:.2f}")
    ax.loglog(elements, 0.5 * elements.astype(float) ** -4, "k--", label="Referencia Ne^-4")
    ax.set(xlabel="Numero de elementos", ylabel="Error relativo de amplitud",
           xticks=elements, xticklabels=elements)
    ax.legend()
    save_figure(fig, "convergencia")
    summary["convergence"] = convergence

    quadrature_results = []
    original_gx, original_gw = ns["GX"], ns["GW"]
    try:
        for count in (12, 16, 24, 32):
            ns["GX"], ns["GW"] = np.polynomial.legendre.leggauss(count)
            for ne in (32, 64):
                amplitude = np.abs(Bem(R, ne, mat).resolver(30.0)).mean()
                error = abs(amplitude - abs(analytical(30.0))) / abs(analytical(30.0))
                quadrature_results.append({"gauss": count, "elements": ne, "error": float(error)})
    finally:
        ns["GX"], ns["GW"] = original_gx, original_gw
    fig, ax = plt.subplots(figsize=(6, 4))
    for ne in (32, 64):
        selected = [row for row in quadrature_results if row["elements"] == ne]
        ax.semilogy([row["gauss"] for row in selected], [row["error"] for row in selected],
                    "o-", label=f"{ne} elementos")
    ax.set(xlabel="Puntos de Gauss por intervalo/subintervalo", ylabel="Error relativo de amplitud",
           title="Separar refinamiento de malla y cuadratura; omega=30 rad/s")
    ax.legend()
    save_figure(fig, "sensibilidad_cuadratura")
    summary["quadrature_study"] = quadrature_results
    quadrature_table = [
        r"\begin{tabular}{rrr}", r"\toprule",
        r"Puntos de Gauss & $e_A$, 32 elementos & $e_A$, 64 elementos\\", r"\midrule",
    ]
    for count in (12, 16, 24, 32):
        errors = [row["error"] for row in quadrature_results if row["gauss"] == count]
        quadrature_table.append(f"{count} & {errors[0]:.3e} & {errors[1]:.3e}" + r"\\")
    quadrature_table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "tabla_cuadratura.tex").write_text(
        "\n".join(quadrature_table) + "\n", encoding="utf-8"
    )
    complex_table = [
        r"\begin{tabular}{rrrrr}", r"\toprule",
        r"$\omega$ & $e_{\mathbb C,8}$ & $e_{\mathbb C,32}$ & $e_{\theta,8}$ & $e_{\theta,32}$\\",
        r"\midrule",
    ]
    for row in rows:
        complex_table.append(
            f"{row['omega']:.2f} & {row['bem8_complex_rms_error']:.3e} & "
            f"{row['bem32_complex_rms_error']:.3e} & {row['bem8_tangential_rms']:.3e} & "
            f"{row['bem32_tangential_rms']:.3e}" + r"\\"
        )
    complex_table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "tabla_error_complejo.tex").write_text(
        "\n".join(complex_table) + "\n", encoding="utf-8"
    )
    cancellation = []
    cancellation_table = [
        r"\begin{tabular}{rrr}", r"\toprule",
        r"$d$ [m] & $\operatorname{Re}\chi$ & $\operatorname{Re}\chi'$ [1/m]\\",
        r"\midrule",
    ]
    for distance in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        _, chi, _, dchi = ns["psi_chi"](np.array(distance), 30.0, mat)
        cancellation.append({"distance": distance, "chi_real": float(chi.real),
                             "dchi_real": float(dchi.real)})
        cancellation_table.append(
            f"{distance:.0e} & {chi.real:.9f} & {dchi.real:.3e}" + r"\\"
        )
    cancellation_table += [r"\bottomrule", r"\end{tabular}"]
    (DATA / "tabla_cancelacion.tex").write_text(
        "\n".join(cancellation_table) + "\n", encoding="utf-8"
    )
    summary["small_distance_cancellation"] = cancellation

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ne, radial in nodal_at_resonance.items():
        angle = np.arange(radial.size) * 360 / radial.size
        exact = analytical(omega1)
        axes[0].plot(angle, (abs(radial) / abs(exact) - 1) * 100, ".-", label=f"BEM {ne}")
        axes[1].plot(angle, np.angle(radial / exact, deg=True), ".-", label=f"BEM {ne}")
    axes[0].set(xlabel="Angulo nodal [grados]", ylabel="Error nodal de amplitud [%]")
    axes[1].set(xlabel="Angulo nodal [grados]", ylabel="Error nodal de fase [grados]")
    for ax in axes:
        ax.legend()
    save_figure(fig, "errores_nodales")

    radius = np.linspace(0, R, 301)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for omega in (30.0, omega1, 120.0):
        x = omega * R / mat["cp"]
        denominator = (mat["lam"] + 2 * mat["mu"]) * x * ns["jv"](0, x) - 2 * mat["mu"] * ns["jv"](1, x)
        field = -pressure * R * ns["jv"](1, omega * radius / mat["cp"]) / denominator
        axes[0].plot(radius / R, abs(field) * 1e4, label=f"omega={omega:.2f}")
        axes[1].plot(radius / R, field.real * 1e4, label=f"omega={omega:.2f}")
    axes[0].set(xlabel="r/R", ylabel="|u_r(r)| [10^-4 m]")
    axes[1].set(xlabel="r/R", ylabel="Re(u_r(r)) [10^-4 m], instante t=0")
    for ax in axes:
        ax.legend()
    save_figure(fig, "campo_interior_analitico")

    kernels, static_kernel = ns["kernels"], ns["kernel_Ts"]
    xi = np.array([0.1, -0.2])
    point = np.array([[1.3, 0.7]])
    normal = np.array([[0.6, 0.8]])
    U, T = kernels(point, xi, normal, 40.0, mat)
    def fundamental(position):
        return kernels(position[None], xi, normal, 40.0, mat)[0][0]
    h = 1e-6
    gradient = np.stack([
        (fundamental(point[0] + h * direction) - fundamental(point[0] - h * direction)) / (2 * h)
        for direction in np.eye(2)
    ], axis=-1)
    finite_difference_T = np.array([
        (mat["lam"] * np.trace(gradient[i]) * np.eye(2)
         + mat["mu"] * (gradient[i] + gradient[i].T)) @ normal[0]
        for i in range(2)
    ])
    traction_error = float(np.max(abs(finite_difference_T - T[0])) / np.max(abs(T)))
    check(traction_error < 1e-6, "Fundamental traction finite-difference check failed")
    h = 1e-3
    hessian = np.zeros((2, 2, 2, 2), complex)
    for k in range(2):
        for ell in range(2):
            ek, el = np.eye(2)[k] * h, np.eye(2)[ell] * h
            hessian[..., k, ell] = (
                fundamental(point[0] + ek + el) - fundamental(point[0] + ek - el)
                - fundamental(point[0] - ek + el) + fundamental(point[0] - ek - el)
            ) / (4 * h * h)
    laplacian_term = mat["mu"] * (hessian[..., 0, 0] + hessian[..., 1, 1])
    navier = (laplacian_term
              + (mat["lam"] + mat["mu"]) * np.einsum("ijjl->il", hessian)
              + mat["rho"] * 40.0**2 * U[0])
    navier_error = float(np.max(abs(navier)) / np.max(abs(laplacian_term)))
    check(navier_error < 1e-3, "Fundamental Navier finite-difference check failed")
    distances = np.logspace(-3, -1, 80)
    points = xi + distances[:, None] * normal
    normals = np.broadcast_to(normal, points.shape)
    _, dynamic = kernels(points, xi, normals, 40.0, mat)
    static = static_kernel(points, xi, normals, ns["NU"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(distances, np.max(abs(dynamic), axis=(1, 2)), label="max |T_ij|")
    ax.loglog(distances, np.max(abs(dynamic - static), axis=(1, 2)), label="max |T_ij - Ts_ij|")
    ax.set(xlabel="Distancia a la fuente [m]", ylabel="Magnitud del nucleo de traccion [1/m]")
    ax.legend()
    save_figure(fig, "regularizacion")
    summary["checks"]["traction_finite_difference_relative_error"] = traction_error
    summary["checks"]["navier_finite_difference_relative_error"] = navier_error

    (DATA / "resultados.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "figures": len(list(FIGURES.glob("*.pdf"))),
        "omega1": omega1,
        "sweep8_max_error": summary["sweep8_max_amplitude_error"],
        "sweep32_max_error": summary["sweep32_max_amplitude_error"],
        "traction_fd_error": traction_error,
        "navier_fd_error": navier_error,
    }, indent=2))


if __name__ == "__main__":
    main()
