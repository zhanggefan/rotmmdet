import torch
from torch.autograd import Function
from .gwd_utils_ext import matsqrt2x2sym_fwd, matsqrt2x2sym_bwd


class _matsqrt2x2sym(Function):

    @staticmethod
    def forward(ctx, input):
        output = matsqrt2x2sym_fwd(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = matsqrt2x2sym_bwd(output, grad_output)
        return grad_input


matsqrt2x2sym = _matsqrt2x2sym.apply
