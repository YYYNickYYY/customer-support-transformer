import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

from my_project_1.tokenization import SPECIAL_TOKENS, encode_parts, prepare_example, profile


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["Hello there", "Refund please"], trainers.BpeTrainer(
        vocab_size=300, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    return tok


def test_roundtrip_unicode_and_fixed_ids(tokenizer):
    assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(5))
    text = "退款 Hello!\n {{Order Number}}"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_shift_masks_prompt_but_includes_first_response_and_eos(tokenizer):
    inst, resp, full = encode_parts(tokenizer, "Hello", "Refund please")
    example = prepare_example(tokenizer, "Hello", "Refund please", len(full))
    assert example["input_ids"] == full[:-1]
    assert example["labels"][:len(inst)+2] == [-100] * (len(inst)+2)
    assert example["labels"][len(inst)+2:] == resp + [4]
    assert not example["truncated"]


def test_truncation_does_not_insert_false_eos(tokenizer):
    inst, resp, full = encode_parts(tokenizer, "Hello", "Refund please")
    limit = len(inst)+4
    example = prepare_example(tokenizer, "Hello", "Refund please", limit)
    assert example["labels"][-1] == resp[0]
    assert example["supervised_tokens"] == 1
    assert example["truncated"]
    assert prepare_example(tokenizer, "Hello", "Refund please", len(inst)+3)["supervised_tokens"] == 0


def test_statistic_boundary_and_lost_tokens(tokenizer):
    rows = [{"instruction": "Hello", "response": "Refund please"}]
    _, resp, full = encode_parts(tokenizer, **rows[0])
    stats = profile(tokenizer, rows, [2, len(full)-1, len(full)])
    assert stats["truncation"][str(len(full))]["over_limit"]["full_sequence"]["rate"] == 0
    assert stats["truncation"][str(len(full)-1)]["over_limit"]["full_sequence"]["rate"] == 1
    assert stats["truncation"][str(len(full)-1)]["lost_response_tokens"] == 0
    assert stats["truncation"]["2"]["lost_response_tokens"] == len(resp)


def test_reserved_control_tokens_rejected(tokenizer):
    with pytest.raises(ValueError, match="Reserved"):
        encode_parts(tokenizer, "Hello <EOS>", "Reply")
