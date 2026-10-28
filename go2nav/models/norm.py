"""LayerNorm with its own backward pass.

torch before 2.13 on RDNA (the R9700, gfx1201) never writes the odd-indexed elements of the fused LayerNorm (and
RMSNorm) backward's weight and bias gradients from 128 rows up: they hold whatever the allocator's memory held, while
the even elements, the input gradient and the forward pass are right (pytorch/pytorch#199265, fixed by #183864 in
2.13). Half of every LayerNorm gain and bias trained on torch 2.12.0+rocm7.2 took a random walk near its initial value
instead of learning.

`LayerNorm` normalises with the plain kernel's forward and computes all three gradients itself, in elementwise ops and
sums, from the saved input and weight: the memory of the fused op (writing the affine out as autograd ops would keep an
fp32 copy of every normalised activation), and torch.compile traces it. The parameters and their names are
nn.LayerNorm's, so checkpoints load unchanged."""
import torch
from torch import nn
from torch.nn import functional as F


class _AffineLayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps, ctx.has_bias = eps, bias is not None
        y = F.layer_norm(x, weight.shape, None, None, eps)*weight
        return y if bias is None else y+bias

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        dims = tuple(range(x.dim()-weight.dim(), x.dim()))
        dtype = torch.promote_types(x.dtype, torch.float32)
        xf, grad = x.to(dtype), grad.to(dtype)
        mean = xf.mean(dims, keepdim=True)
        rstd = torch.rsqrt(xf.var(dims, unbiased=False, keepdim=True)+ctx.eps)
        normalised = (xf-mean)*rstd
        g = grad*weight.to(dtype)
        dx = rstd*(g-g.mean(dims, keepdim=True)-normalised*(g*normalised).mean(dims, keepdim=True))
        rows = tuple(range(x.dim()-weight.dim()))
        dw = (grad*normalised).sum(rows).to(weight.dtype)
        db = grad.sum(rows).to(weight.dtype) if ctx.has_bias else None
        return dx.to(x.dtype), dw, db, None


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        if self.weight is None:
            return F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if torch.is_autocast_enabled(x.device.type):  # layer_norm runs in fp32 under autocast, as nn.LayerNorm does
            x = x.float()
        return _AffineLayerNorm.apply(x, self.weight, self.bias, self.eps)
