"""LayerNorm with its affine written out.

torch 2.12.0+rocm7.2 on the R9700 (gfx1201) returns wrong weight and bias gradients from the fused LayerNorm (and
RMSNorm) backward kernel from 128 rows up: errors as large as the gradients themselves, varying from run to run at a few
thousand rows, while the input gradient and the forward pass are right (BatchNorm, GroupNorm and Linear are right too).
Every LayerNorm's gain and bias trained on that build took a random walk near its initial value instead of learning.
Normalising without the affine and applying it as plain elementwise ops gives the right gradients. The parameters and
their names are nn.LayerNorm's, so checkpoints load unchanged."""
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        y = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        if self.weight is not None:
            y = y*self.weight
        return y if self.bias is None else y+self.bias
