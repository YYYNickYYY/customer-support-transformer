"""Pre-LayerNorm decoder block: causal attention and a position-wise FFN."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .attention import CausalSelfAttention, TokenPositionEmbedding, _positive, _dropout, _padding_mask


class FeedForward(nn.Module):
    """Apply the same 128 -> 512 -> 128 network independently at each position."""

    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        _positive("d_model", d_model)
        _positive("d_ff", d_ff)
        _dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN: x += Attention(LN(x)); x += FFN(LN(x)).

    LayerNorm acts on d_model for each token independently. Branch modules own
    their dropout, so residual connections add their outputs directly. Padding
    is zeroed before computation and after each addition to suppress biases.
    This is one block, not yet a vocabulary prediction model.
    """

    def __init__(self, d_model=128, n_heads=4, d_ff=512, max_length=512,
                 dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        if not 0 < layer_norm_eps < float("inf"):
            raise ValueError("layer_norm_eps must be finite and positive")
        self.attention = CausalSelfAttention(d_model, n_heads, max_length, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.d_model = d_model
        self.max_length = max_length

    def forward(self, x, attention_mask=None, *, return_attention=False):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("x must have shape [batch, length, d_model]")
        batch, length, _ = x.shape
        if batch == 0 or not 1 <= length <= self.max_length:
            raise ValueError("Empty batch or sequence exceeds max_length")
        mask = _padding_mask(attention_mask, batch, length, x.device)

        def zero_padding(value):
            return value.masked_fill(~mask.unsqueeze(-1), 0)

        x = zero_padding(x)
        result = self.attention(self.norm1(x), mask, return_attention=return_attention)
        if return_attention:
            attention_output, weights = result
        else:
            attention_output = result
        x = zero_padding(x + attention_output)
        x = zero_padding(x + self.ffn(self.norm2(x)))
        return (x, weights) if return_attention else x


def main():
    from torch.utils.data import DataLoader
    from .dataset import CustomerSupportDataset, DynamicPaddingCollator

    parser = argparse.ArgumentParser(description="Inspect one Transformer Block on real train data")
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = CustomerSupportDataset(tokenizer_dir=config["tokenizer_artifact"])
    if (config["tokenizer_revision"] != dataset.config["tokenizer_sha256"]
            or config["vocab_size"] != dataset.tokenizer.get_vocab_size()
            or config["pad_token_id"] != dataset.pad_token_id
            or config["max_length"] != dataset.max_length):
        raise ValueError("Model configuration does not match frozen tokenizer")
    batch = next(iter(DataLoader(dataset, batch_size=4,
                      collate_fn=DynamicPaddingCollator(dataset.pad_token_id))))
    torch.manual_seed(42)
    embedding = TokenPositionEmbedding(config["vocab_size"], config["d_model"],
                                       config["max_length"], config["pad_token_id"], config["dropout"])
    block = TransformerBlock(config["d_model"], config["n_heads"], config["d_ff"],
                             config["max_length"], config["dropout"])
    embedding.eval()
    block.eval()
    x = embedding(batch["input_ids"], batch["attention_mask"])
    output, weights = block(x, batch["attention_mask"], return_attention=True)
    # Diagnostic scalar for checking gradient flow, NOT a language-model loss.
    valid = batch["attention_mask"].bool()
    output[valid].square().mean().backward()
    print("Block input/output:", tuple(x.shape), tuple(output.shape))
    print("Block parameters:", sum(p.numel() for p in block.parameters()))
    print("Future weights all zero:", bool((weights.triu(1) == 0).all()))
    print("Padding outputs all zero:", bool((output[~valid] == 0).all()))
    print("All gradients finite:", all(p.grad is not None and bool(p.grad.isfinite().all())
                                      for p in block.parameters()))


if __name__ == "__main__":
    main()
