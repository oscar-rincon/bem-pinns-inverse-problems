"""Hankel functions H^(2)_0 and H^(2)_1 with first-order autograd support."""
import torch
from torch.special import bessel_j0, bessel_j1, bessel_y0, bessel_y1


class _J0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return bessel_j0(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g * bessel_j1(x)


class _Y0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return bessel_y0(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g * bessel_y1(x)


class _J1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return bessel_j1(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (bessel_j0(x) - bessel_j1(x) / x)


class _Y1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return bessel_y1(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (bessel_y0(x) - bessel_y1(x) / x)


def hankel2_0(z):
    """H^(2)_0(z) = J0(z) - i Y0(z) for real z > 0."""
    return torch.complex(_J0.apply(z), -_Y0.apply(z))


def hankel2_1(z):
    """H^(2)_1(z) = J1(z) - i Y1(z) for real z > 0."""
    return torch.complex(_J1.apply(z), -_Y1.apply(z))
