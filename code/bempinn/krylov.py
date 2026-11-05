"""Matrix-free style linear solves: GMRES with warm start and an adjoint (GMRES) backward pass.

Every product with A or A^H is counted, so the cost of the classical solvers can be compared
with the BEM-PINN in the regime where the dense matrix cannot be factorized (large 3D models).
"""
import numpy as np
import scipy.sparse.linalg as sla
import torch


class Counter:
    def __init__(self):
        self.matvec = 0          # products with A (forward solves)
        self.rmatvec = 0         # products with A^H (adjoint solves)
        self.solves = 0

    def total(self):
        return self.matvec + self.rmatvec


def _gmres(A, B, X0, tol, counter, adjoint=False):
    """Column-wise restarted GMRES with a Jacobi (diagonal) preconditioner; returns the
    solution and adds the iterations to counter. The diagonal holds the analytic self terms,
    so the preconditioner is available without storing the matrix."""
    M = A.conj().T if adjoint else A
    d = np.diag(M).copy()
    P = sla.LinearOperator(M.shape, matvec=lambda x: x / d, dtype=M.dtype)
    X = np.zeros_like(B)
    for j in range(B.shape[1]):
        it = [0]

        def cb(_):
            it[0] += 1

        x0 = None if X0 is None else X0[:, j]
        X[:, j], info = sla.gmres(M, B[:, j], x0=x0, rtol=tol, atol=0.0, restart=200, maxiter=50,
                                  M=P, callback=cb, callback_type="pr_norm")
        if adjoint:
            counter.rmatvec += it[0]
        else:
            counter.matvec += it[0]
        counter.solves += 1
    return X


class GMRESSolve(torch.autograd.Function):
    """X = A^{-1} B by GMRES; backward solves the adjoint system A^H Y = dX by GMRES."""

    @staticmethod
    def forward(ctx, A, B, X0, tol, counter):
        An = A.detach().numpy()
        X = _gmres(An, B.detach().numpy(), X0, tol, counter)
        Xt = torch.from_numpy(X)
        ctx.save_for_backward(A, Xt)
        ctx.tol, ctx.counter = tol, counter
        return Xt

    @staticmethod
    def backward(ctx, gX):
        A, X = ctx.saved_tensors
        Y = torch.from_numpy(_gmres(A.detach().numpy(), gX.numpy(), None, ctx.tol, ctx.counter,
                                    adjoint=True))
        gA = -Y @ X.conj().T
        return gA, Y, None, None, None


def gmres_solve(A, B, X0=None, tol=1e-10, counter=None):
    return GMRESSolve.apply(A, B, X0, tol, counter if counter is not None else Counter())
