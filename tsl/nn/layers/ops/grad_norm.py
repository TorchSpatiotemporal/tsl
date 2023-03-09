import torch


class GradNorm(torch.autograd.Function):
    """Scales the gradient in back-propagation. In the forward pass is an
    identity operation."""

    @staticmethod
    def forward(ctx, x, norm):
        """"""
        ctx.save_for_backward(x)
        ctx.norm = norm  # save normalization coefficient
        return x  # identity

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        norm = ctx.norm
        return grad_output / norm, None  # return the normalized gradient
