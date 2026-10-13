from types import SimpleNamespace

import pytest
import torch

from my_project_1.decoding_comparison import banned_fourgram_tokens, choose_token, constrained_reply
from my_project_1.generate import generate_reply


class Tokenizer:
    def token_to_id(self, token):
        return {"<PAD>": 0, "<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3, "<EOS>": 4}[token]

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[6, 7, 8, 9, 6, 7, 8] if text else [])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i > 4)


class Model(torch.nn.Module):
    def __init__(self, answers, invalid=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.answers, self.inputs, self.invalid = answers, [], invalid

    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert (attention_mask == 1).all()
        answer = self.answers[len(self.inputs)]
        self.inputs.append(input_ids.tolist()[0])
        logits = torch.zeros(1, input_ids.shape[1], 12)
        logits[0, -1, :4] = 100
        logits[0, -1, 4] = -10
        logits[0, -1, 10] = 5
        logits[0, -1, answer] = 10
        if self.invalid:
            logits[0, -1, 11] = float("nan")
        return {"logits": logits}


def config(length=60):
    return {"max_length": length, "pad_token_id": 0, "bos_token_id": 1,
            "user_token_id": 2, "assistant_token_id": 3, "eos_token_id": 4}


@pytest.mark.parametrize("history, expected", [([], []), ([6, 7], []), ([6, 7, 8], []),
    ([6, 6, 6, 6, 6], [6]), ([6, 7, 8, 9, 6, 7, 8], [9]),
    ([6, 7, 8, 4, 6, 7, 8], []), ([6, 7, 8, 9, 6, 7, 8, 10, 6, 7, 8], [9, 10])])
def test_short_overlapping_multiple_and_eos_exempt(history, expected):
    assert banned_fourgram_tokens(history, 4) == expected


def test_size_zero_exactly_matches_original_and_no_parameter_update():
    a, b = Model([6, 7, 4]), Model([6, 7, 4])
    before = b.weight.detach().clone()
    old = generate_reply(a, Tokenizer(), config(), "question")
    new = constrained_reply(b, Tokenizer(), config(), "question")
    assert all(new[key] == value for key, value in old.items())
    assert a.inputs == b.inputs
    assert b.training and b.weight.grad is None
    assert torch.equal(before, b.weight)


def test_prompt_does_not_trigger_reply_constraint():
    result = constrained_reply(Model([6, 7, 8, 9, 4]), Tokenizer(), config(), "question",
                               no_repeat_ngram_size=4)
    assert result["generated_token_ids"] == [6, 7, 8, 9, 4]
    assert all(not step["ngram_blocked_token_ids"] for step in result["steps"])


def test_constraint_changes_choice_and_does_not_force_eos():
    result = constrained_reply(Model([6, 7, 8, 9, 6, 7, 8, 9, 4]), Tokenizer(), config(), "question",
                               no_repeat_ngram_size=4)
    assert result["generated_token_ids"] == [6, 7, 8, 9, 6, 7, 8, 10, 4]
    step = result["steps"][7]
    assert step["ngram_blocked_token_ids"] == [9]
    assert step["baseline_greedy_token_id"] == 9 and step["constraint_changed_choice"]


def test_legitimate_entity_reuse_is_not_silently_exempted():
    # Four token IDs represent a legitimate placeholder; its second completion is blocked.
    entity_ids = [6, 7, 8, 9]
    assert banned_fourgram_tokens(entity_ids + entity_ids[:3], 4) == [9]
    logits = torch.zeros(12)
    logits[9], logits[10] = 9, 8
    chosen, _ = choose_token(logits, entity_ids + entity_ids[:3], [0, 1, 2, 3], 4, 4)
    assert chosen == 10  # Negative safety case, NOT a claim of semantic improvement.


def test_roles_tie_and_early_eos():
    chosen, _ = choose_token(torch.zeros(12), [], [0, 1, 2, 3], 4, 4)
    assert chosen == 4
    result = constrained_reply(Model([4]), Tokenizer(), config(), "question", no_repeat_ngram_size=4)
    assert result["stop_reason"] == "eos" and result["generated_token_ids"] == [4]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_logits_rejected(invalid):
    scores = torch.zeros(12)
    scores[0] = invalid  # Even normally suppressed role logits must be finite before masking.
    with pytest.raises(ValueError, match="Non-finite"):
        choose_token(scores, [], [0, 1, 2, 3], 4, 4)


def test_exception_restores_mode():
    model = Model([4], invalid=True)
    with pytest.raises(ValueError, match="Non-finite"):
        constrained_reply(model, Tokenizer(), config(), "question")
    assert model.training and model.weight.grad is None


@pytest.mark.parametrize("answer,reason", [(6, "context_limit"), (4, "eos")])
def test_context_boundary_eos_has_priority(answer, reason):
    result = constrained_reply(Model([answer]), Tokenizer(), config(11), "question", no_repeat_ngram_size=4)
    assert result["stop_reason"] == reason
    assert result["prompt_tokens"] + result["generated_tokens"] == 11


def test_max_tokens_no_fabricated_eos_and_eval_mode_preserved():
    model = Model([6, 7])
    model.eval()
    result = constrained_reply(model, Tokenizer(), config(), "question", max_new_tokens=2)
    assert result["stop_reason"] == "max_new_tokens"
    assert result["generated_token_ids"] == [6, 7] and not model.training


@pytest.mark.parametrize("size", [1, 3, True, 4.0, -1])
def test_unregistered_sizes_rejected(size):
    with pytest.raises(ValueError):
        constrained_reply(Model([4]), Tokenizer(), config(), "question", no_repeat_ngram_size=size)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_max_tokens(limit):
    with pytest.raises(ValueError):
        constrained_reply(Model([4]), Tokenizer(), config(), "question", max_new_tokens=limit)


@pytest.mark.parametrize("question", ["", "   ", None, "hello <EOS>"])
def test_invalid_question(question):
    with pytest.raises(ValueError):
        constrained_reply(Model([4]), Tokenizer(), config(), question)


def test_no_room_and_eos_role_contract():
    with pytest.raises(ValueError, match="no room"):
        constrained_reply(Model([4]), Tokenizer(), config(10), "question")
    with pytest.raises(ValueError, match="EOS"):
        choose_token(torch.zeros(12), [], [0, 1, 2, 4], 4, 4)
