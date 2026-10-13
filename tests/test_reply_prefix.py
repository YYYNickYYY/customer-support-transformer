import math
import pytest
import torch
from my_project_1.reply_prefix import branch_nll, prefix_reply, trajectory_prefixes, checked_prefix
from my_project_1.generate import generate_reply


CFG = dict(max_length=32, pad_token_id=0, bos_token_id=1, user_token_id=2, assistant_token_id=3, eos_token_id=4)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoded", (), {"ids": [5 + ord(c) % 10 for c in text]})()
    def token_to_id(self, name):
        return {"<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3, "<EOS>": 4}[name]
    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(i) for i in ids if i >= 5)


class Model(torch.nn.Module):
    def __init__(self, token=4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.token = token
    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert (attention_mask == 1).all()
        logits = torch.zeros(*input_ids.shape, 16) + self.weight
        if self.token is not None:
            logits[..., self.token] = 4
            logits[..., 0] = 5  # role must be masked in generation, not NLL
        return {"logits": logits}


def test_nll_only_branch_not_prefix_or_prompt():
    model = Model(token=None).train()
    result = branch_nll(model, Tokenizer(), CFG, "q", [5, 6], "xyz")
    assert result["token_count"] == 3
    assert len(result["per_token_nll"]) == 3
    assert result["mean_nll"] == pytest.approx(math.log(16))
    assert result["total_nll"] == pytest.approx(3 * math.log(16))
    assert model.training and model.weight.grad is None


def test_no_eos_fabrication_and_reject_truncation():
    with pytest.raises(ValueError, match="does not fit"):
        branch_nll(Model(), Tokenizer(), dict(CFG, max_length=6), "q", [5, 6], "xyz")
    with pytest.raises(ValueError, match="trajectory"):
        branch_nll(Model(), Tokenizer(), CFG, "q", [4], "xyz")
    assert 4 not in branch_nll(Model(token=None), Tokenizer(), CFG, "q", [], "abc")["target_ids"]


def test_empty_prefix_exact_existing_greedy_and_eos():
    model = Model().train()
    old = generate_reply(model, Tokenizer(), CFG, "q", max_new_tokens=6)
    new = prefix_reply(model, Tokenizer(), CFG, "q", "", total_reply_budget=6)
    for field in ("response", "generated_token_ids", "generated_tokens", "prompt_tokens", "stop_reason"):
        assert old[field] == new[field]
    assert new["continuation_tokens"] == 1 and new["stop_reason"] == "eos"
    assert model.training and model.weight.grad is None


def test_budget_counts_forced_prefix_and_context():
    row = prefix_reply(Model(token=5), Tokenizer(), CFG, "q", "hi", total_reply_budget=5)
    assert row["prefix_tokens"] == 2 and row["continuation_tokens"] == 3
    assert row["generated_tokens"] == 5 and row["stop_reason"] == "max_new_tokens"
    row = prefix_reply(Model(token=5), Tokenizer(), dict(CFG, max_length=8), "q", "hi", total_reply_budget=10)
    assert row["generated_tokens"] == 4 and row["stop_reason"] == "context_limit"
    with pytest.raises(ValueError, match="budget"):
        prefix_reply(Model(), Tokenizer(), CFG, "q", "hi", total_reply_budget=2)


@pytest.mark.parametrize("prefix", ["<EOS>", "<USER> hello", "<PAD>"])
def test_reserved_prefix_rejected(prefix):
    with pytest.raises(ValueError, match="Invalid"):
        checked_prefix(Tokenizer(), prefix, CFG)


def test_sentence_stages_and_duplicate_boundaries_not_future():
    class Decode:
        def decode(self, ids, skip_special_tokens=True):
            return "".join("." if i == 9 else "a" for i in ids if i != 4)
    ids = [5] * 69 + [9] + [5] * 30 + [9, 4]
    stages = trajectory_prefixes(Decode(), ids, 4)
    assert [v["prefix_tokens"] for v in stages] == [0, 70, 70, 101]
    for row in stages:
        assert row["prefix_ids"] == ids[:row["prefix_tokens"]]
    assert not trajectory_prefixes(Decode(), [5, 9, 4], 4)[1]["available"]
