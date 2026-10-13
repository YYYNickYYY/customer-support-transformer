"""Verified train/dev datasets and dynamic right-padding for causal training."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenization import file_hash, load_frozen, prepare_example, read_rows


class CustomerSupportDataset(Dataset):
    """Keep raw rows in memory and tokenize on access; return CPU Python lists.

    Only train/dev are supported here. The frozen tokenizer configuration binds
    each split to its exact file hash. Full-sequence budget includes EOS before
    shifting, so model inputs have at most max_length - 1 positions.
    """

    def __init__(self, split="train", *, split_dir="data/splits/bitext-v1",
                 tokenizer_dir="artifacts/tokenizer/bitext-bpe-v1", train_revision=None):
        if split not in {"train", "dev"}:
            raise ValueError("Training dataset supports only train or dev")
        if train_revision is not None and split != "train":
            raise ValueError("Target revisions are only allowed for train")
        self.tokenizer, self.config = load_frozen(tokenizer_dir)
        self.max_length = self.config["max_length"]
        self.pad_token_id = self.config["special_token_ids"]["<PAD>"]
        path = Path(split_dir) / f"{split}.jsonl"
        if file_hash(path) != self.config["split_sha256"][path.name]:
            raise ValueError(f"Frozen split hash mismatch: {split}")
        self.rows = list(read_rows(path))
        self.revision = None
        if train_revision is not None:
            from .revisions import load_revision
            self.rows, self.revision = load_revision(train_revision, self.rows)
        if not self.rows:
            raise ValueError(f"Empty split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        example = prepare_example(self.tokenizer, row["instruction"], row["response"],
                                  self.max_length)
        if example["supervised_tokens"] == 0:
            raise ValueError(f"No supervised tokens in sample {row['id']}")
        return {**example, "id": row["id"]}


@dataclass(frozen=True)
class DynamicPaddingCollator:
    """Create int64 CPU tensors [batch, longest input] with right-padding.

    attention_mask is 1 for real input positions and 0 for padding. It is a
    padding mask; the model must ALSO apply causal masking. Labels are already
    shifted by the dataset: loss code must not shift them a second time.
    """

    pad_token_id: int = 0

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        for example in examples:
            if not example["input_ids"] or len(example["input_ids"]) != len(example["labels"]):
                raise ValueError("Nonempty inputs and labels must have matching lengths")
            if all(label == -100 for label in example["labels"]):
                raise ValueError("Every sample must have supervised tokens")
        shape = (len(examples), max(len(e["input_ids"]) for e in examples))
        inputs = torch.full(shape, self.pad_token_id, dtype=torch.long)
        labels = torch.full(shape, -100, dtype=torch.long)
        mask = torch.zeros(shape, dtype=torch.long)
        for index, example in enumerate(examples):
            length = len(example["input_ids"])
            inputs[index, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
            labels[index, :length] = torch.tensor(example["labels"], dtype=torch.long)
            mask[index, :length] = 1
        return {"input_ids": inputs, "labels": labels, "attention_mask": mask}


def main():
    parser = argparse.ArgumentParser(description="Inspect a real train/dev batch")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    dataset = CustomerSupportDataset(args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=DynamicPaddingCollator(dataset.pad_token_id))
    batch = next(iter(loader))
    print(f"{args.split}: {len(dataset)} samples; max_length={dataset.max_length}")
    for name, tensor in batch.items():
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print("Input lengths:", batch["attention_mask"].sum(dim=1).tolist())
    print("Supervised tokens:", (batch["labels"] != -100).sum(dim=1).tolist())


if __name__ == "__main__":
    main()
