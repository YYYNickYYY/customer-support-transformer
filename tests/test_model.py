import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from my_project_1.model import TinyDecoderLM


def small_model():
    torch.manual_seed(42)
    return TinyDecoderLM(vocab_size=20, d_model=8, n_layers=4, n_heads=2,
                         d_ff=32, max_length=12, dropout=0)


def test_shapes_independent_blocks_and_final_norm():
    model = small_model().eval()
    assert len({id(block) for block in model.blocks}) == 4
    assert len({block.attention.q_proj.weight.data_ptr() for block in model.blocks}) == 4
    assert model.embedding.token_embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    captured = []
    handle = model.lm_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    output = model(torch.tensor([[1, 2, 3, 4]]))
    handle.remove()
    assert output["logits"].shape == (1, 4, 20)
    assert output["loss"] is None
    torch.testing.assert_close(captured[0].mean(-1), torch.zeros(1, 4), atol=1e-6, rtol=0)


def test_complete_model_is_causal():
    model = small_model().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = ids.clone()
    changed[:, 3:] = torch.tensor([7, 8, 9])
    torch.testing.assert_close(model(ids)["logits"][:, :3], model(changed)["logits"][:, :3])


def test_padding_invariance_and_all_padding_finite():
    model = small_model().eval()
    with torch.no_grad():
        model.final_norm.bias.fill_(0.5)
    ids = torch.tensor([[1, 2, 3, 0, 0], [0, 0, 0, 0, 0]])
    mask = ids.ne(0)
    output = model(ids, mask)["logits"]
    assert output.isfinite().all()
    assert (output[~mask] == 0).all()
    torch.testing.assert_close(output[0, :3], model(ids[:1, :3])["logits"][0])
    changed = ids.clone()
    changed[~mask] = 15
    torch.testing.assert_close(output, model(changed, mask)["logits"])


def test_loss_directly_aligned_and_ignored_logits_have_zero_gradient():
    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 5, 0, 0]])
    # Already shifted: the first reply token (7) is predicted at position 1.
    labels = torch.tensor([[-100, 7, 8, 4, -100], [-100, 9, 4, -100, -100]])
    result = model(ids, labels=labels)
    supervised = labels.ne(-100)
    expected = F.cross_entropy(result["logits"][supervised], labels[supervised])
    torch.testing.assert_close(result["loss"], expected)
    result["logits"].retain_grad()
    result["loss"].backward()
    assert (result["logits"].grad[~supervised] == 0).all()
    assert result["logits"].grad[0, 1].abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
    assert (model.embedding.token_embedding.weight.grad[0] == 0).all()
    for block in model.blocks:
        assert block.ffn.fc1.weight.grad.abs().sum() > 0
        assert block.attention.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("labels", [
    torch.tensor([[-100, -100, -100]]),
    torch.tensor([[-100, 4, 2]]),  # padding supervised
    torch.tensor([[-100, 20, -100]]),
    torch.tensor([[-100, -2, -100]]),
    torch.tensor([[1, 2]]),
    torch.tensor([[-100., 4., -100.]]),
])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError):
        small_model()(torch.tensor([[1, 2, 0]]), labels=labels)


def test_invalid_layers_ids_and_length():
    with pytest.raises(ValueError):
        TinyDecoderLM(n_layers=0)
    model = small_model()
    for ids in (torch.tensor([[20]]), torch.tensor([[-1]]), torch.ones(1, 13, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(ids)


def test_tiny_batch_can_overfit():
    model = small_model()
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 7, 8, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(ids, labels=labels)["loss"].item()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        model(ids, labels=labels)["loss"].backward()
        optimizer.step()
    final = model(ids, labels=labels)["loss"].item()
    assert final < initial * 0.2
