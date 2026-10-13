"""Method-only v2 acceptance: frozen stimuli + synthetic oracle, no fifth inference."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from my_project_1.reply_prefix import branch_nll, checked_prefix
from my_project_1.tokenization import SPECIAL_TOKENS, encode_parts

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/condition-use-measurement-v2-protocol.md"


def plan():
    text = PROTOCOL.read_text(encoding="utf-8")
    return json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


PLAN = plan()
TOK = Tokenizer.from_file(str(ROOT / PLAN["tokenizer"]))
CFG = json.loads((ROOT / PLAN["config"]).read_text(encoding="utf-8"))
STIMULI = [(case, prefix, wording) for case in PLAN["cases"]
           for prefix in PLAN["prefixes"] for wording in range(3)]


@pytest.fixture(autouse=True)
def no_checkpoint_load(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Method acceptance must not load checkpoints")
    monkeypatch.setattr(torch, "load", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def classify(values, total_values, tolerance=1e-6):
    """Executable protocol specification, NOT a production scoring runner."""
    both = values + total_values
    if len(values) != 3 or len(total_values) != 3 or any(
            v is None or not math.isfinite(v) for v in both):
        return "unavailable"
    if all(abs(v) <= tolerance for v in both):
        return "flat"
    if all(v > tolerance for v in both):
        return "positive"
    if all(v < -tolerance for v in both):
        return "negative"
    return "discordant"


def test_provenance_and_fixed_original_questions():
    for relative, expected in PLAN["dependencies"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    source = json.loads((ROOT / "outputs/condition-use-20261006-v1/plan.json").read_text(encoding="utf-8"))
    originals = {int(row["source_id"].rsplit(":", 1)[-1]): row["instruction"]
                 for row in source["cases"] if row["variant"] == "original"}
    assert [case["source_rows"] for case in PLAN["cases"]] == [
        [1469, 824], [1256, 969], [16867, 6461], [16597, 6688]]
    for case in PLAN["cases"]:
        assert case["questions"] == [originals[row] for row in case["source_rows"]]
    assert PLAN["training"] is False and PLAN["fifth_inference"] is False
    assert len(STIMULI) * 4 == PLAN["expected"]["branch_scores"] == 144


@pytest.mark.parametrize("prefix", PLAN["prefixes"])
def test_explicit_complete_prefixes_not_punctuation_search(prefix):
    ids = checked_prefix(TOK, prefix, CFG)
    assert TOK.decode(ids, skip_special_tokens=False) == prefix
    assert not prefix or (prefix.endswith(".") and prefix == prefix.strip())
    assert not ids or not TOK.decode([ids[-1]], skip_special_tokens=False).isspace()
    # Fixed whole sentences, never arbitrary numbered-list/decimal/abbreviation cuts.
    assert prefix in ("", "Thank you for your message.",
                      "Thank you for your message. Let me address your question.")


@pytest.mark.parametrize("case,prefix,wording", STIMULI)
def test_all_stimulus_compositions_fit_and_share_exact_history(case, prefix, wording):
    candidates = PLAN["families"][case["family"]]["branches"][wording]
    histories = [checked_prefix(TOK, prefix, CFG) for question in case["questions"]]
    assert histories[0] == histories[1]
    lengths = []
    for candidate in candidates:
        assert candidate.startswith(" ") and not candidate.endswith(" ")
        assert candidate.strip().endswith(".")
        for text in [prefix, candidate, *case["questions"]]:
            assert not any(s in text for s in SPECIAL_TOKENS)
        target = checked_prefix(TOK, candidate, CFG)
        lengths.append(len(target))
        assert target and not TOK.decode([target[0]], skip_special_tokens=False).isspace()
        assert TOK.encode(prefix + candidate, add_special_tokens=False).ids == histories[0] + target
        assert TOK.decode(histories[0] + target, skip_special_tokens=False) == prefix + candidate
        for question in case["questions"]:
            prompt = encode_parts(TOK, question, "")[2][:-1]
            assert len(prompt + histories[0] + target) <= 512
            # Independently state shifted-label contract including the first candidate token.
            full = prompt + histories[0] + target
            boundary = len(prompt) + len(histories[0])
            labels = ([-100] * boundary + target)[1:]
            assert labels[boundary - 1] == target[0]
            assert labels[:boundary - 1] == [-100] * (boundary - 1)
            assert labels[boundary - 1:] == target
            assert len(labels) == len(full[:-1])
            assert CFG["eos_token_id"] not in target
    assert abs(lengths[0] - lengths[1]) <= PLAN["candidate_length_difference_max"]


class CausalOracle(torch.nn.Module):
    """Nonuniform, history-/position-sensitive logits; no learned model weights."""
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert attention_mask.eq(1).all()
        running = input_ids.double().cumsum(-1)
        position = torch.arange(1, input_ids.shape[1] + 1, dtype=torch.float64)
        vocab = torch.arange(CFG["vocab_size"], dtype=torch.float64)
        logits = torch.sin(running[..., None] * .001
                           + position[None, :, None] * (vocab[None, None, :] + 1) * .0001)
        return {"logits": logits + self.marker}


def independent_losses(question, history, candidate):
    # No encode_parts, production shift/CE or oracle.forward reuse.
    prompt = [TOK.token_to_id("<BOS>"), TOK.token_to_id("<USER>")]
    prompt += TOK.encode(question, add_special_tokens=False).ids
    prompt += [TOK.token_to_id("<ASSISTANT>")] + history
    target = TOK.encode(candidate, add_special_tokens=False).ids
    losses = []
    for token in target:
        logits = [math.sin(sum(prompt) * .001 + len(prompt) * (v + 1) * .0001)
                  for v in range(CFG["vocab_size"])]
        top = max(logits)
        losses.append(top + math.log(sum(math.exp(v - top) for v in logits)) - logits[token])
        prompt.append(token)
    return losses


@pytest.mark.parametrize("case,prefix", list(itertools.product(PLAN["cases"], PLAN["prefixes"])))
def test_real_tokenizer_nonuniform_oracle_and_supervision(case, prefix):
    model = CausalOracle().train()
    history = checked_prefix(TOK, prefix, CFG)
    candidate = PLAN["families"][case["family"]]["branches"][0][0]
    actual = branch_nll(model, TOK, CFG, case["questions"][0], history, candidate)
    expected = independent_losses(case["questions"][0], history, candidate)
    assert actual["per_token_nll"] == pytest.approx(expected, abs=1e-10)
    assert actual["total_nll"] == pytest.approx(sum(expected), abs=1e-9)
    assert actual["mean_nll"] == pytest.approx(sum(expected) / len(expected), abs=1e-10)
    assert actual["token_count"] == len(expected)
    assert model.training and model.marker.grad is None


def test_future_candidate_tokens_cannot_affect_first_prediction_and_repeat_is_stable():
    question = PLAN["cases"][0]["questions"][0]
    history = checked_prefix(TOK, PLAN["prefixes"][1], CFG)
    model = CausalOracle()
    candidate = PLAN["families"]["order"]["branches"][0][0]
    first = branch_nll(model, TOK, CFG, question, history, candidate)
    repeat = branch_nll(model, TOK, CFG, question, history, candidate)
    longer = branch_nll(model, TOK, CFG, question, history, candidate + " More text.")
    assert first == repeat
    assert first["per_token_nll"][0] == longer["per_token_nll"][0]
    assert model.training and model.marker.grad is None


def test_invalid_control_prefix_and_overflow_refused_without_truncation():
    model = CausalOracle()
    args = (model, TOK, CFG, PLAN["cases"][0]["questions"][0])
    with pytest.raises(ValueError, match="Invalid trajectory"):
        branch_nll(*args, [CFG["eos_token_id"]], " next")
    with pytest.raises(ValueError, match="refuse truncation"):
        branch_nll(*args, [], " next" * 600)
    with pytest.raises(ValueError, match="Invalid prefix"):
        checked_prefix(TOK, "<EOS>", CFG)
    assert model.training and model.marker.grad is None


def test_eval_mode_restored_on_forward_failure():
    class Broken(CausalOracle):
        def forward(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")
    model = Broken().train()
    with pytest.raises(RuntimeError, match="synthetic failure"):
        branch_nll(model, TOK, CFG, "question", [], " next")
    assert model.training and model.marker.grad is None


@pytest.mark.parametrize("mean,total,expected", [
    ([.1, .2, .3], [.2, .3, .4], "positive"),
    ([-.1, -.2, -.3], [-.2, -.3, -.4], "negative"),
    ([0, 1e-7, -1e-7], [0, 0, 0], "flat"),
    ([.1, .2, .3], [-.1, .2, .3], "discordant"),
    ([.1, -.2, .3], [.1, .2, .3], "discordant"),
    ([.1, None, .3], [.1, .2, .3], "unavailable"),
    ([.1, float("nan"), .3], [.1, .2, .3], "unavailable"),
    ([.1, .2], [.1, .2, .3], "unavailable"),
])
def test_missing_near_zero_wording_and_normalization_contract(mean, total, expected):
    assert classify(mean, total) == expected


def test_cross_difference_is_not_absolute_candidate_preference():
    # Both questions can prefer a; q_a simply has a stronger relative preference.
    margin_a, margin_b = 3.0 - 1.0, 2.0 - 1.5
    assert margin_a > 0 and margin_b > 0
    assert margin_a - margin_b > 0
    # This cannot be labelled "both tasks correctly chosen" from signs.

