import math

import pytest

torch = pytest.importorskip("torch")

from my_project_1.attention import CausalSelfAttention, TokenPositionEmbedding, LearnedPositionEmbedding


def test_embedding_adds_position_and_zeros_padding():
    model = TokenPositionEmbedding(20, 8, 8, dropout=0).eval()
    ids = torch.tensor([[5, 5, 0]])
    actual = model(ids)
    expected = model.token_embedding(ids) + model.position_embedding(3).unsqueeze(0)
    torch.testing.assert_close(actual[:, :2], expected[:, :2])
    assert not torch.equal(actual[:, 0], actual[:, 1])
    assert (actual[:, 2] == 0).all()
    actual.sum().backward()
    assert (model.token_embedding.weight.grad[0] == 0).all()
    assert model.position_embedding.embedding.weight.grad[:2].abs().sum() > 0


def test_single_head_matches_hand_calculation():
    model = CausalSelfAttention(2, 1, 4, dropout=0).eval()
    with torch.no_grad():
        for layer in (model.q_proj, model.k_proj, model.v_proj, model.out_proj):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    x = torch.tensor([[[1., 0.], [0., 1.]]])
    output, weights = model(x, return_attention=True)
    expected_weights = torch.softmax(torch.tensor([0., 1 / math.sqrt(2)]), dim=0)
    torch.testing.assert_close(output[0, 0], x[0, 0])
    torch.testing.assert_close(output[0, 1], expected_weights)
    assert weights[0, 0, 0, 1] == 0


def test_future_token_changes_do_not_affect_prefix():
    torch.manual_seed(42)
    model = CausalSelfAttention(16, 4, 8, dropout=0).eval()
    x = torch.randn(2, 6, 16)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    before, weights = model(x, return_attention=True)
    after = model(changed)
    torch.testing.assert_close(before[:, :3], after[:, :3])
    assert (weights.triu(1) == 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 6))


def test_padding_invariance_and_gradients():
    torch.manual_seed(42)
    model = CausalSelfAttention(8, 2, 8, dropout=0).eval()
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
    output, weights = model(x, mask, return_attention=True)
    assert output.isfinite().all()
    assert (output[~mask.bool()] == 0).all()
    assert (weights[0, :, :, 2:] == 0).all()
    assert (weights[1] == 0).all()
    torch.testing.assert_close(output[0, :2], model(x[:1, :2])[0])
    changed = x.detach().clone()
    changed[~mask.bool()] = 100
    torch.testing.assert_close(output, model(changed, mask))
    output.square().sum().backward()
    assert (x.grad[~mask.bool()] == 0).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
    assert model.q_proj.weight.grad.abs().sum() > 0


def test_same_output_with_or_without_attention_weights():
    model = CausalSelfAttention(8, 2, 8, dropout=.1).eval()
    x = torch.randn(1, 3, 8)
    torch.testing.assert_close(model(x), model(x, return_attention=True)[0])


@pytest.mark.parametrize("kwargs", [{"d_model": 7, "n_heads": 2}, {"n_heads": 0},
                                    {"max_length": 0}, {"dropout": 1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CausalSelfAttention(**kwargs)


def test_sequence_length_and_mask_checks():
    with pytest.raises(ValueError, match="sequence length"):
        LearnedPositionEmbedding(3, 8)(4)
    model = CausalSelfAttention(8, 2, 3, dropout=0)
    with pytest.raises(ValueError, match="max_length"):
        model(torch.randn(1, 4, 8))
    with pytest.raises(ValueError, match="right-padding"):
        model(torch.randn(1, 3, 8), torch.tensor([[0, 1, 1]]))
    with pytest.raises(ValueError, match="only 0 and 1"):
        model(torch.randn(1, 3, 8), torch.tensor([[1, 2, 0]]))
