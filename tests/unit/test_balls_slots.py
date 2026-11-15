import torch

from go2nav.models.balls_slots import ObjectMaskedPredictor, RGBSlotConfig, RGBSlots, object_mask, predictor_loss


def predictor():
    return ObjectMaskedPredictor(dim=16, depth=2, heads=2, head_dim=8, mlp=32)


def test_masked_history_only_exposes_earliest_identity_anchor():
    model = predictor().eval()
    history, actions = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 2)
    mask = torch.tensor([[True, False, False, False, False]]).expand(2, -1)
    first, selected = model.tokens(history, actions, mask)
    changed = history.clone()
    changed[:, 1:, 0] += 999
    second, _ = model.tokens(changed, actions, mask)
    assert torch.equal(first, second)
    changed[:, 0, 0] += 1
    third, _ = model.tokens(changed, actions, mask)
    assert not torch.equal(first, third)
    assert not selected[:, 0].any() and selected[:, 1:3, 0].all() and selected[:, 3].all()


def test_predictor_is_equivariant_under_joint_slot_permutation():
    model = predictor().eval()
    history, actions = torch.randn(2, 3, 5, 16), torch.randn(2, 3, 2)
    mask = torch.tensor([[True, False, True, False, False]]).expand(2, -1)
    order = [4, 2, 3, 1, 0]
    original, _ = model(history, actions, mask)
    permuted, _ = model(history[:, :, order], actions, mask[:, order])
    torch.testing.assert_close(permuted, original[:, :, order], rtol=1e-4, atol=1e-5)


def test_targets_have_no_gradient_or_forward_path_into_prediction():
    model = predictor()
    states = torch.randn(2, 4, 5, 16, requires_grad=True)
    actions = torch.randn(2, 3, 2)
    value, _ = predictor_loss(model, states, actions, None)
    value.backward()
    assert torch.equal(states.grad[:, 3], torch.zeros_like(states.grad[:, 3]))
    assert states.grad[:, :3].abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_mask_stream_is_independent_of_initialization_and_data_order():
    torch.manual_seed(1)
    o0 = predictor()
    torch.manual_seed(1)
    o1 = predictor()
    mask = object_mask(100, 5, torch.Generator().manual_seed(7), 'cpu')
    assert set(mask.sum(-1).tolist()) == {0, 1, 2}
    assert all(torch.equal(p, o1.state_dict()[k]) for k, p in o0.state_dict().items())


def test_rgb_slots_reconstruct_shape_and_have_deterministic_batch_independent_initialization():
    model = RGBSlots(RGBSlotConfig(slots=3, dim=16, channels=8)).eval()
    pixels = torch.randn(2, 2, 3, 64, 64)
    with torch.no_grad():
        states = model(pixels)
        recon, masks = model.decode(states)
        single = torch.cat([model(p[None]) for p in pixels])
    assert states.shape == (2, 2, 3, 16) and recon.shape == pixels.shape
    torch.testing.assert_close(masks.sum(2), torch.ones(2, 2, 64, 64))
    torch.testing.assert_close(states, single, rtol=1e-4, atol=1e-5)
