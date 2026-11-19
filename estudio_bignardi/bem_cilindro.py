"""BEM directo para el cilindro del notebook ``bigardi.ipynb``.

El modulo es independiente del notebook y necesita NumPy y SciPy.
Ejemplo desde la raiz del repositorio::

    python estudio_bignardi/bem_cilindro.py

La clase devuelve un desplazamiento radial complejo por nodo; su magnitud
es la amplitud. Para una explicacion detallada, consultar el documento
``explicacion_bem_cilindro/explicacion_bem_cilindro.pdf``.
El modelo dinamico del notebook se usa para frecuencias positivas.
"""

import numpy as np
from scipy.special import jv, kv, kvp


R = 6.0       # radio [m]
RHO = 100.0   # densidad [kg/m^3]
MU = 1.0e6    # modulo de corte [Pa]
NU = 0.25     # coeficiente de Poisson
BETA = 0.05   # amortiguamiento histeretico
P = 100.0     # amplitud de presion [Pa]


def material(rho=RHO, mu=MU, nu=NU, beta=BETA):
    """Construir los modulos complejos y las velocidades de ondas S y P."""
    mu_c = mu * (1 + 2j * beta)
    lam_c = 2 * mu_c * nu / (1 - 2 * nu)
    return dict(
        rho=rho,
        mu=mu_c,
        lam=lam_c,
        nu=nu,
        cs=np.sqrt(mu_c / rho),
        cp=np.sqrt((lam_c + 2 * mu_c) / rho),
    )


def u_r_analitico(omega, m, R=R, p=P):
    """Desplazamiento radial exacto en la superficie para comparacion."""
    x = omega / m["cp"] * R
    lam, mu = m["lam"], m["mu"]
    return -p * R * jv(1, x) / (
        (lam + 2 * mu) * x * jv(0, x) - 2 * mu * jv(1, x)
    )


def shape(eta):
    """Funciones de forma cuadraticas en los nodos locales -1, 0 y 1."""
    eta = np.asarray(eta, float)
    return np.stack(
        [0.5 * eta * (eta - 1), 1 - eta**2, 0.5 * eta * (eta + 1)], -1
    )


def dshape(eta):
    """Derivadas de las funciones de forma respecto a eta."""
    eta = np.asarray(eta, float)
    return np.stack([eta - 0.5, -2 * eta, eta + 0.5], -1)


def malla_circulo(R, ne):
    """Crear nodos antihorarios y conectividad de elementos de tres nodos."""
    nn = 2 * ne
    th = np.linspace(0, 2 * np.pi, nn, endpoint=False)
    nodes = R * np.stack([np.cos(th), np.sin(th)], -1)
    conn = np.array([[2 * e, 2 * e + 1, (2 * e + 2) % nn] for e in range(ne)])
    return nodes, conn


GX, GW = np.polynomial.legendre.leggauss(16)


def puntos_cuadratura(eta_s=None):
    """Cuadratura de Gauss, adaptada si el punto fuente esta en el elemento."""
    if eta_s is None:
        return GX.copy(), GW.copy()

    # Partir en eta_s y concentrar puntos cerca de la fuente con un mapa cuadratico.
    t, wt = (GX + 1) / 2, GW / 2
    pts, wts = [], []
    for a, b in [(-1.0, eta_s), (eta_s, 1.0)]:
        if b - a < 1e-12:
            continue
        if a == eta_s:
            s, ds = t**2, 2 * t
        else:
            s, ds = 1 - (1 - t)**2, 2 * (1 - t)
        pts.append(a + (b - a) * s)
        wts.append(wt * ds * (b - a))
    return np.concatenate(pts), np.concatenate(wts)


def psi_chi(r, w, m):
    """Funciones radiales de la solucion fundamental y sus derivadas."""
    cs, cp = m["cs"], m["cp"]
    zs, zp = 1j * w * r / cs, 1j * w * r / cp
    dzs, dzp = 1j * w / cs, 1j * w / cp
    g = cs / cp
    psi = kv(0, zs) + (kv(1, zs) - g * kv(1, zp)) / zs
    chi = kv(2, zs) - g**2 * kv(2, zp)
    dpsi = (
        kvp(0, zs) * dzs
        + (kvp(1, zs) * dzs - g * kvp(1, zp) * dzp) / zs
        - (kv(1, zs) - g * kv(1, zp)) * dzs / zs**2
    )
    dchi = kvp(2, zs) * dzs - g**2 * kvp(2, zp) * dzp
    return psi, chi, dpsi, dchi


def kernels(x, xi, n, w, m):
    """Evaluar los nucleos dinamicos U y T en los puntos de integracion."""
    d = x - xi
    r = np.linalg.norm(d, axis=-1)
    ri = d / r[..., None]
    drdn = (ri * n).sum(-1)
    psi, chi, dpsi, dchi = psi_chi(r, w, m)
    I = np.eye(2)
    rr = ri[..., :, None] * ri[..., None, :]
    rj_ni = ri[..., None, :] * n[..., :, None]
    ri_nj = ri[..., :, None] * n[..., None, :]

    def a(s):
        return s[..., None, None]

    U = (a(psi) * I - a(chi) * rr) / (2 * np.pi * m["mu"])
    T = (
        a(dpsi - chi / r) * (I * a(drdn) + rj_ni)
        - a(2 * chi / r) * (ri_nj - 2 * rr * a(drdn))
        - a(2 * dchi * drdn) * rr
        + (m["lam"] / m["mu"]) * a(dpsi - dchi - chi / r) * ri_nj
    ) / (2 * np.pi)
    return U, T


def kernel_Ts(x, xi, n, nu):
    """Nucleo de traccion estatico de Kelvin en deformacion plana."""
    d = x - xi
    r = np.linalg.norm(d, axis=-1)
    ri = d / r[..., None]
    drdn = (ri * n).sum(-1)
    I = np.eye(2)
    rr = ri[..., :, None] * ri[..., None, :]
    c = (-1 / (4 * np.pi * (1 - nu) * r))[..., None, None]
    return c * (
        drdn[..., None, None] * ((1 - 2 * nu) * I + 2 * rr)
        - (1 - 2 * nu)
        * (ri[..., :, None] * n[..., None, :] - ri[..., None, :] * n[..., :, None])
    )


class BEMCilindro:
    """Preparar la frontera y resolver H(w) u = G(w) t para un cilindro."""

    def __init__(self, R, ne, m):
        self.R, self.ne, self.m = R, ne, m
        self.nodes, self.conn = malla_circulo(R, ne)
        self.N = len(self.nodes)
        self.quad = []
        Hs = np.zeros((2 * self.N, 2 * self.N))

        # Para cada nodo de colocacion se integran todos los elementos.
        for ic, xi in enumerate(self.nodes):
            fila = []
            for cn in self.conn:
                loc = np.where(cn == ic)[0]
                eta_s = [-1.0, 0.0, 1.0][loc[0]] if len(loc) else None
                eta, wq = puntos_cuadratura(eta_s)
                Pn, dP = shape(eta), dshape(eta)
                X, dX = Pn @ self.nodes[cn], dP @ self.nodes[cn]
                J = np.linalg.norm(dX, axis=-1)
                nrm = np.stack([dX[:, 1], -dX[:, 0]], -1) / J[:, None]
                W = Pn * (J * wq)[:, None]
                Ts = kernel_Ts(X, xi, nrm, m["nu"])

                # Cada nodo aporta un bloque 2x2; la diagonal se calcula despues.
                for a, g in enumerate(cn):
                    if g != ic:
                        Hs[2 * ic:2 * ic + 2, 2 * g:2 * g + 2] += np.einsum(
                            "q,qij->ij", W[:, a], Ts
                        )
                fila.append((cn, X, nrm, W, Ts))
            self.quad.append(fila)

        # Una traslacion rigida tiene traccion nula: cada fila de bloques suma cero.
        for ic in range(self.N):
            Hs[2 * ic:2 * ic + 2, 2 * ic:2 * ic + 2] = -sum(
                Hs[2 * ic:2 * ic + 2, 2 * g:2 * g + 2]
                for g in range(self.N)
                if g != ic
            )
        self.Hs = Hs

    def matrices(self, w):
        """Ensamblar matrices densas complejas para la frecuencia angular w."""
        N = self.N
        G = np.zeros((2 * N, 2 * N), complex)
        H = self.Hs.astype(complex)
        for ic, xi in enumerate(self.nodes):
            for cn, X, nrm, W, Ts in self.quad[ic]:
                U, T = kernels(X, xi, nrm, w, self.m)
                Gb = np.einsum("qa,qij->aij", W, U)
                Hb = np.einsum("qa,qij->aij", W, T - Ts)
                for a, g in enumerate(cn):
                    G[2 * ic:2 * ic + 2, 2 * g:2 * g + 2] += Gb[a]
                    H[2 * ic:2 * ic + 2, 2 * g:2 * g + 2] += Hb[a]
        return H, G

    def resolver(self, w, p=P):
        """Resolver la presion uniforme y devolver u radial complejo por nodo."""
        H, G = self.matrices(w)
        nrm = self.nodes / self.R
        t = (-p * nrm).ravel()
        u = np.linalg.solve(H, G @ t).reshape(-1, 2)
        return (u * nrm).sum(-1)


if __name__ == "__main__":
    mat = material()
    bem = BEMCilindro(R, ne=8, m=mat)
    omega = 40.0
    ur = bem.resolver(omega, p=P)
    referencia = u_r_analitico(omega, mat, R=R, p=P)

    print(f"Elementos: {bem.ne}; nodos: {bem.N}; omega: {omega:.1f} rad/s")
    print(f"Desplazamiento radial complejo en el nodo 0: {ur[0]:.6e} m")
    print(f"Amplitud radial media BEM: {np.abs(ur).mean():.6e} m")
    print(f"Amplitud radial analitica: {abs(referencia):.6e} m")
