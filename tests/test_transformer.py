import pytest

torch = pytest.importorskip("torch")

from my_project_1.transformer import FeedForward, TransformerBlock


def test_ffn_does_not_mix_positions():
    ffn = FeedForward(8, 32, dropout=0).eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 2:] = 100
    assert ffn(x).shape == x.shape
    torch.testing.assert_close(ffn(x)[:, :2], ffn(changed)[:, :2])


def test_layer_norm_is_per_token_and_pre_attention():
    block = TransformerBlock(8, 2, 32, 8, dropout=0).eval()
    x = torch.randn(2, 4, 8)
    normalized = []
    handle = block.attention.register_forward_pre_hook(lambda module, args: normalized.append(args[0].detach()))
    block(x)
    handle.remove()
    torch.testing.assert_close(normalized[0], block.norm1(x))
    torch.testing.assert_close(normalized[0].mean(-1), torch.zeros(2, 4), atol=1e-6, rtol=0)
    changed = x.clone()
    changed[:, 2:] = 100
    torch.testing.assert_close(block.norm1(x)[:, :2], block.norm1(changed)[:, :2])


def test_zero_branches_preserve_residual_and_identity_gradient():
    block = TransformerBlock(8, 2, 32, 8, dropout=0)
    with torch.no_grad():
        block.attention.out_proj.weight.zero_()
        block.attention.out_proj.bias.zero_()
        block.ffn.fc2.weight.zero_()
        block.ffn.fc2.bias.zero_()
    x = torch.randn(1, 4, 8, requires_grad=True)
    output = block(x)
    torch.testing.assert_close(output, x)
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_block_and_stacked_blocks_remain_causal():
    torch.manual_seed(42)
    blocks = [TransformerBlock(8, 2, 32, 8, dropout=0).eval() for _ in range(4)]
    x = torch.randn(1, 6, 8)
    changed = x.clone()
    changed[:, 3:] += 10
    for block in blocks:
        x, weights = block(x, return_attention=True)
        changed = block(changed)
        torch.testing.assert_close(x[:, :3], changed[:, :3])
        assert (weights.triu(1) == 0).all()


def test_padding_zero_with_nonzero_biases_and_backward():
    torch.manual_seed(42)
    block = TransformerBlock(8, 2, 32, 8, dropout=0).eval()
    with torch.no_grad():
        block.norm1.bias.fill_(.3)
        block.norm2.bias.fill_(.3)
        block.ffn.fc2.bias.fill_(.5)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
    output = block(x, mask)
    assert output.isfinite().all()
    assert (output[~mask.bool()] == 0).all()
    torch.testing.assert_close(output[0, :3], block(x[:1, :3])[0])
    changed = x.detach().clone()
    changed[~mask.bool()] = 100
    torch.testing.assert_close(output, block(changed, mask))
    output.square().mean().backward()
    assert (x.grad[~mask.bool()] == 0).all()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
        assert parameter.grad.abs().sum() > 0


def test_eval_disables_dropout():
    block = TransformerBlock(8, 2, 32, 8, dropout=.5).eval()
    x = torch.randn(1, 4, 8)
    torch.testing.assert_close(block(x), block(x))


@pytest.mark.parametrize("kwargs", [{"d_ff": 0}, {"layer_norm_eps": 0},
                                    {"layer_norm_eps": float("nan")}])
def test_invalid_block_configuration(kwargs):
    with pytest.raises(ValueError):
        TransformerBlock(**kwargs)
