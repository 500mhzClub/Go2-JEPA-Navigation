import pytest
import torch
from torch import nn

from go2nav.models.norm import LayerNorm


def test_layer_norm_matches_nn_and_loads_its_checkpoints():
    reference, ours = nn.LayerNorm(16), LayerNorm(16)
    with torch.no_grad():
        reference.weight.uniform_(.5, 1.5)
        reference.bias.uniform_(-.5, .5)
    ours.load_state_dict(reference.state_dict())
    x = torch.randn(300, 16)
    assert torch.allclose(ours(x), reference(x), atol=1e-6)
    assert LayerNorm(16, elementwise_affine=False)(x).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_layer_norm_affine_gradients_are_right_on_the_gpu():
    """The fused kernel's weight and bias gradients are wrong from 128 rows on torch 2.12+rocm7.2 (gfx1201)."""
    x, grad = torch.randn(4096, 192, dtype=torch.float64), torch.randn(4096, 192, dtype=torch.float64)
    normalised = (x-x.mean(-1, keepdim=True))/torch.sqrt(x.var(-1, unbiased=False, keepdim=True)+1e-5)
    norm = LayerNorm(192).cuda()
    norm(x.float().cuda()).backward(grad.float().cuda())
    assert torch.allclose(norm.weight.grad.cpu().double(), (grad*normalised).sum(0), rtol=1e-3, atol=1e-2)
    assert torch.allclose(norm.bias.grad.cpu().double(), grad.sum(0), rtol=1e-3, atol=1e-2)
