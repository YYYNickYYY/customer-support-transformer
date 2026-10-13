"""Readable embedding and causal attention building blocks for a tiny decoder."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn


def _positive(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _dropout(value):
    if not 0 <= value < 1:
        raise ValueError("dropout must be in [0, 1)")


def _padding_mask(mask, batch, length, device):
    if mask is None:
        return torch.ones(batch, length, device=device, dtype=torch.bool)
    if mask.shape != (batch, length) or mask.device != device:
        raise ValueError("attention_mask must match batch/length and input device")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("attention_mask must contain only 0 and 1")
    mask = mask.bool()
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ValueError("Only right-padding is supported")
    return mask


class LearnedPositionEmbedding(nn.Module):
    """Positions 0..T-1 map to learned vectors; output shape [T, d_model]."""

    def __init__(self, max_length=512, d_model=128):
        super().__init__()
        _positive("max_length", max_length)
        _positive("d_model", d_model)
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, length):
        if type(length) is not int or not 1 <= length <= self.max_length:
            raise ValueError(f"sequence length must be in [1, {self.max_length}]")
        positions = torch.arange(length, device=self.embedding.weight.device)
        return self.embedding(positions)


class TokenPositionEmbedding(nn.Module):
    """Token lookup + learned position lookup + dropout: [B,T] -> [B,T,C].

    PAD's token vector has no gradient. Position vectors still exist there, so
    the combined padding representations are explicitly set to zero.
    """

    def __init__(self, vocab_size=4000, d_model=128, max_length=512,
                 pad_token_id=0, dropout=0.1):
        super().__init__()
        _positive("vocab_size", vocab_size)
        _positive("d_model", d_model)
        _dropout(dropout)
        if type(pad_token_id) is not int or not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be a valid vocabulary ID")
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = LearnedPositionEmbedding(max_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be an int64 tensor [batch, length]")
        batch, length = input_ids.shape
        if batch == 0:
            raise ValueError("Empty batch")
        mask = _padding_mask(attention_mask if attention_mask is not None
                             else input_ids.ne(self.pad_token_id), batch, length, input_ids.device)
        positions = self.position_embedding(length)
        x = self.token_embedding(input_ids) + positions.unsqueeze(0)
        return self.dropout(x).masked_fill(~mask.unsqueeze(-1), 0)


class CausalSelfAttention(nn.Module):
    """Explicit multi-head Q/K/V attention, with causal AND padding masking.

    x: [B,T,C]; mask: [B,T], 1=real input. Output: [B,T,C].
    Optional returned weights are BEFORE dropout, shape [B,H,T,T].
    Real queries sum to 1 over allowed keys; padding queries have zero weights.
    """

    def __init__(self, d_model=128, n_heads=4, max_length=512, dropout=0.1):
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads), ("max_length", max_length)):
            _positive(name, value)
        _dropout(dropout)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim, self.max_length = d_model // n_heads, max_length
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", torch.ones(max_length, max_length,
                             dtype=torch.bool).tril(), persistent=False)

    def forward(self, x, attention_mask=None, *, return_attention=False):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [batch, length, d_model]")
        batch, length, _ = x.shape
        if batch == 0 or not 1 <= length <= self.max_length:
            raise ValueError("Empty batch or sequence exceeds max_length")
        mask = _padding_mask(attention_mask, batch, length, x.device)

        def split_heads(projection):
            return projection(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(self.q_proj), split_heads(self.k_proj), split_heads(self.v_proj)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = (self.causal_mask[:length, :length][None, None]
                   & mask[:, None, None, :] & mask[:, None, :, None])
        scores = scores.masked_fill(~allowed, float("-inf"))
        # All-masked queries would produce softmax(-inf, ...) = NaN.
        # Temporarily use zeros there, then force all their weights to zero.
        scores = torch.where(allowed.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0)
        context = self.attention_dropout(weights) @ v
        context = context.transpose(1, 2).contiguous().reshape(batch, length, self.d_model)
        output = self.output_dropout(self.out_proj(context))
        output = output.masked_fill(~mask.unsqueeze(-1), 0)
        return (output, weights) if return_attention else output


def main():
    from torch.utils.data import DataLoader
    from .dataset import CustomerSupportDataset, DynamicPaddingCollator
    parser = argparse.ArgumentParser(description="Inspect embeddings and attention on real train data")
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = CustomerSupportDataset(tokenizer_dir=config["tokenizer_artifact"])
    if config["tokenizer_revision"] != dataset.config["tokenizer_sha256"]:
        raise ValueError("Model tokenizer revision mismatch")
    if (config["vocab_size"] != dataset.tokenizer.get_vocab_size()
            or config["pad_token_id"] != dataset.pad_token_id
            or config["max_length"] != dataset.max_length):
        raise ValueError("Model configuration does not match frozen tokenizer")
    batch = next(iter(DataLoader(dataset, batch_size=4,
                      collate_fn=DynamicPaddingCollator(dataset.pad_token_id))))
    torch.manual_seed(42)
    embedding = TokenPositionEmbedding(config["vocab_size"], config["d_model"],
                                       config["max_length"], config["pad_token_id"], config["dropout"])
    attention = CausalSelfAttention(config["d_model"], config["n_heads"],
                                   config["max_length"], config["dropout"])
    embedding.eval()
    attention.eval()
    with torch.no_grad():
        x = embedding(batch["input_ids"], batch["attention_mask"])
        output, weights = attention(x, batch["attention_mask"], return_attention=True)
    print("input_ids:", tuple(batch["input_ids"].shape))
    print("embedding:", tuple(x.shape))
    print("attention output:", tuple(output.shape))
    print("attention weights:", tuple(weights.shape))
    print("All outputs finite:", bool(output.isfinite().all()))
    print("Future weights all zero:", bool((weights.triu(1) == 0).all()))


if __name__ == "__main__":
    main()
