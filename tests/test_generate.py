from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from my_project_1.generate import generate_reply


class TokenizerStub:
    special = {"<PAD>": 0, "<BOS>": 1, "<USER>": 2, "<ASSISTANT>": 3, "<EOS>": 4}

    def token_to_id(self, token):
        return self.special[token]

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[5] if text else [])

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(token) for token in tokens if token > 4)


class ScriptedModel(torch.nn.Module):
    def __init__(self, answers, *, invalid=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.answers = answers
        self.invalid = invalid
        self.inputs = []

    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert (attention_mask == 1).all()
        answer = self.answers[len(self.inputs)]
        self.inputs.append(input_ids.tolist()[0])
        logits = torch.zeros(1, input_ids.shape[1], 10)
        logits[0, -1, :4] = 100  # Forbidden prompt-only controls must be suppressed.
        logits[0, -1, answer] = 10
        if self.invalid:
            logits[0, -1, 6] = float("nan")
        return {"logits": logits}


def config(max_length=10):
    return {"max_length": max_length, "pad_token_id": 0, "bos_token_id": 1,
            "user_token_id": 2, "assistant_token_id": 3, "eos_token_id": 4}


def test_autoregressive_feedback_and_eos_no_teacher_forcing():
    model = ScriptedModel([6, 7, 4])
    output = generate_reply(model, TokenizerStub(), config(), "question")
    assert model.inputs == [[1, 2, 5, 3], [1, 2, 5, 3, 6], [1, 2, 5, 3, 6, 7]]
    assert output["generated_token_ids"] == [6, 7, 4]
    assert output["response"] == "6 7"
    assert output["stop_reason"] == "eos"
    assert model.training
    assert model.anchor.grad is None


def test_token_limit_without_fabricated_eos():
    model = ScriptedModel([6, 7])
    model.eval()
    output = generate_reply(model, TokenizerStub(), config(), "question", max_new_tokens=2)
    assert output["stop_reason"] == "max_new_tokens"
    assert output["generated_token_ids"] == [6, 7]
    assert not model.training


def test_context_budget_and_eos_boundary():
    output = generate_reply(ScriptedModel([6, 7]), TokenizerStub(), config(6), "question")
    assert output["stop_reason"] == "context_limit"
    assert output["prompt_tokens"] + output["generated_tokens"] == 6
    output = generate_reply(ScriptedModel([6, 4]), TokenizerStub(), config(6), "question")
    assert output["stop_reason"] == "eos"


@pytest.mark.parametrize("question", ["", "   ", None, "hi <EOS>"])
def test_invalid_questions(question):
    with pytest.raises(ValueError):
        generate_reply(ScriptedModel([4]), TokenizerStub(), config(), question)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_new_token_limit(limit):
    with pytest.raises(ValueError):
        generate_reply(ScriptedModel([4]), TokenizerStub(), config(), "question", max_new_tokens=limit)


def test_long_prompt_rejected_without_truncation():
    with pytest.raises(ValueError, match="no room"):
        generate_reply(ScriptedModel([4]), TokenizerStub(), config(4), "question")


def test_nonfinite_scores_raise_and_restore_mode():
    model = ScriptedModel([4], invalid=True)
    with pytest.raises(ValueError, match="Non-finite"):
        generate_reply(model, TokenizerStub(), config(), "question")
    assert model.training
