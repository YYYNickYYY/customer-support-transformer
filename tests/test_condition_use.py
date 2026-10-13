import pytest
import torch
from my_project_1.condition_use import POOLS, build_plan, matched_target_batches, condition_nll


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoded", (), {"ids": [5 + ord(c) % 10 for c in text]})()

    def token_to_id(self, name):
        return {"<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3, "<EOS>": 4}[name]


def test_same_target_prefix_under_different_prompt_lengths():
    batches, meta = matched_target_batches(Tokenizer(), "a", "abcd", "response", 10)
    labels = [[t for t in b["labels"] if t != -100] for b in batches]
    assert labels[0] == labels[1] and len(labels[0]) == 3
    assert meta["eos_retained"] is False
    assert meta["lost_response_tokens"] == 5
    assert all(len(b["labels"]) == len(b["input_ids"]) <= 9 for b in batches)


def test_eos_only_when_whole_target_fits_and_first_reply_label():
    batches, meta = matched_target_batches(Tokenizer(), "a", "ab", "hi", 20)
    assert meta["eos_retained"]
    for b in batches:
        positions = [i for i, token in enumerate(b["labels"]) if token != -100]
        assert b["input_ids"][positions[0]] == 3
        assert b["labels"][positions[-1]] == 4
    with pytest.raises(ValueError, match="No shared"):
        matched_target_batches(Tokenizer(), "abcdef", "abc", "hi", 5)


def test_selection_frozen_and_input_only_generations():
    samples = [dict(id="train:" + s, instruction="original " + s, response="target", intent=f)
               for f, ids in POOLS.items() for s in ids]
    reviews = [dict(id=r["id"], task_alignment="aligned") for r in samples]
    plan = build_plan(samples, reviews)
    assert plan == build_plan(list(reversed(samples)), list(reversed(reviews)))
    assert len(plan["cases"]) == 17 and len(plan["source_rows"]) == 8 and len(plan["pairs"]) == 4
    assert plan["cases"][0]["instruction"] == plan["cases"][-1]["instruction"]
    assert all("response" not in c for c in plan["cases"])
    reviews[0]["task_alignment"] = "ambiguous"
    with pytest.raises(ValueError, match="not confirmed"):
        build_plan(samples, reviews)


def test_nll_sign_and_eval_no_grad():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
        def forward(self, input_ids, labels, attention_mask):
            assert not self.training and not torch.is_grad_enabled()
            assert (attention_mask == 1).all()
            return {"loss": input_ids.shape[1] * 0.1 + self.weight.sum()}
    model = Model().train()
    result = condition_nll(model, Tokenizer(), "a", "abc", "hi", 20)
    assert result["swapped_minus_matched"] == pytest.approx(.2)
    assert model.training and model.weight.grad is None
