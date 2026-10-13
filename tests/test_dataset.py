import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader
from my_project_1.dataset import CustomerSupportDataset, DynamicPaddingCollator
from my_project_1.tokenization import encode_parts, load_frozen, prepare_example


@pytest.fixture
def tokenizer():
    return load_frozen("artifacts/tokenizer/bitext-bpe-v1")[0]


def test_padding_and_supervision(tokenizer):
    examples = [prepare_example(tokenizer, "Hello", "Yes.", 512),
                prepare_example(tokenizer, "Where is my order?", "Please provide your order number.", 512)]
    batch = DynamicPaddingCollator()(examples)
    assert batch["input_ids"].shape == (2, max(len(e["input_ids"]) for e in examples))
    for index, example in enumerate(examples):
        n = len(example["input_ids"])
        assert batch["input_ids"][index, :n].tolist() == example["input_ids"]
        assert batch["labels"][index, :n].tolist() == example["labels"]
        assert batch["attention_mask"][index, :n].tolist() == [1] * n
        assert (batch["input_ids"][index, n:] == 0).all()
        assert (batch["labels"][index, n:] == -100).all()
        assert (batch["attention_mask"][index, n:] == 0).all()
    assert all(t.dtype == torch.long for t in batch.values())


def test_reply_first_token_and_eos(tokenizer):
    instruction, response = "Where is my order?", "Please provide your order number."
    inst, resp, full = encode_parts(tokenizer, instruction, response)
    example = prepare_example(tokenizer, instruction, response, 512)
    first = len(inst) + 2
    assert example["input_ids"][first] == tokenizer.token_to_id("<ASSISTANT>")
    assert example["labels"][first] == resp[0]
    assert example["labels"][:first] == [-100] * first
    assert example["labels"][first:] == resp + [tokenizer.token_to_id("<EOS>")]


def test_truncation_retains_real_response_targets(tokenizer):
    inst, resp, full = encode_parts(tokenizer, "Hello", "Please provide your order number.")
    limit = len(inst) + 4
    example = prepare_example(tokenizer, "Hello", "Please provide your order number.", limit)
    batch = DynamicPaddingCollator()([example])
    assert batch["input_ids"].shape == (1, limit-1)
    assert example["truncated"]
    assert batch["labels"][0, -1].item() == resp[0]
    assert (batch["labels"] != -100).sum().item() == 1


def test_prompt_padding_logits_do_not_affect_loss(tokenizer):
    examples = [prepare_example(tokenizer, "Hello", "Yes.", 512),
                prepare_example(tokenizer, "Where is my order?", "Please provide your order number.", 512)]
    labels = DynamicPaddingCollator()(examples)["labels"]
    logits = torch.zeros(*labels.shape, 4000, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 4000), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    assert (logits.grad[labels == -100] == 0).all()
    assert logits.grad[labels != -100].abs().sum() > 0


@pytest.mark.parametrize("split,expected", [("train", 21528), ("dev", 2688)])
def test_actual_split_and_batch(split, expected):
    dataset = CustomerSupportDataset(split)
    assert len(dataset) == expected
    loader = DataLoader(dataset, batch_size=4, collate_fn=DynamicPaddingCollator(dataset.pad_token_id))
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] <= 511
    assert ((batch["labels"] != -100).sum(dim=1) > 0).all()


def test_sealed_test_is_rejected_before_file_access():
    with pytest.raises(ValueError, match="only train or dev"):
        CustomerSupportDataset("sealed_test", split_dir="not-a-directory")


def test_modified_split_is_rejected(tmp_path):
    (tmp_path / "train.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        CustomerSupportDataset(split_dir=tmp_path)


@pytest.mark.parametrize("examples", [[], [{"input_ids": [1], "labels": []}],
                                      [{"input_ids": [1], "labels": [-100]}]])
def test_invalid_batch_rejected(examples):
    with pytest.raises(ValueError):
        DynamicPaddingCollator()(examples)
