"""Independent nonuniform, causal oracle for branch labels; no fifth checkpoint load."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from my_project_1.reply_prefix import branch_nll

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(max_length=64, pad_token_id=0, bos_token_id=1, user_token_id=2, assistant_token_id=3, eos_token_id=4)


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return type("Encoded", (), {"ids": [5 + ord(c) % 7 for c in text]})()
    def token_to_id(self, text):
        return {"<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3, "<EOS>": 4}[text]


class CausalOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert (attention_mask == 1).all()
        running = input_ids.cumsum(-1).float()
        positions = torch.arange(input_ids.shape[1]).float() + 1
        vocabulary = torch.arange(16).float()
        logits = torch.sin(running.unsqueeze(-1) * .13 + positions[None, :, None] * vocabulary[None, None, :] * .09)
        return {"logits": logits + self.weight}


def manual_losses(instruction, prefix, branch):
    # Independent prompt/target construction and scalar log-sum-exp, no production encode_parts/CE.
    tok = Tokenizer()
    prompt = [1, 2] + tok.encode(instruction).ids + [3]
    targets = tok.encode(branch).ids
    history = prompt + prefix
    losses = []
    for token in targets:
        position = len(history)
        logits = [math.sin(sum(history) * .13 + position * candidate * .09) for candidate in range(16)]
        losses.append(math.log(sum(math.exp(v) for v in logits)) - logits[token])
        history.append(token)
    return losses


@pytest.mark.parametrize("question,prefix,branch", [("a", [], "xyz"), ("abcd", [6, 7], "pqrs"), ("xy", [5, 10, 11], "z")])
def test_position_sensitive_oracle_detects_shift_and_supervision(question, prefix, branch):
    model = CausalOracle().train()
    measured = branch_nll(model, Tokenizer(), CFG, question, prefix, branch)
    expected = manual_losses(question, prefix, branch)
    assert measured["per_token_nll"] == pytest.approx(expected, abs=2e-6)
    assert measured["total_nll"] == pytest.approx(sum(expected), abs=3e-6)
    assert measured["mean_nll"] == pytest.approx(sum(expected) / len(expected), abs=2e-6)
    assert model.training and model.weight.grad is None


def test_future_branch_changes_do_not_change_first_target_probability():
    first = branch_nll(CausalOracle(), Tokenizer(), CFG, "question", [6, 7], "abc")
    changed = branch_nll(CausalOracle(), Tokenizer(), CFG, "question", [6, 7], "azz")
    assert first["per_token_nll"][0] == changed["per_token_nll"][0]
    assert first["per_token_nll"][1:] != changed["per_token_nll"][1:]


def test_arithmetic_verifier_checks_missing_not_zero():
    path = ROOT / "scripts/accept_reply_prefix_measurement.py"
    spec = importlib.util.spec_from_file_location("prefix_acceptance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.recompute_probe({"available": False, "question_scores": None, "condition_separation": None})
    with pytest.raises(AssertionError):
        module.recompute_probe({"available": False, "question_scores": None, "condition_separation": 0})
