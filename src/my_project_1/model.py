"""Small decoder-only language model built from our explicit Transformer blocks."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .attention import TokenPositionEmbedding, _padding_mask, _positive
from .transformer import TransformerBlock


class TinyDecoderLM(nn.Module):
    """[B,T] token IDs -> [B,T,V] next-token logits.

    Labels from prepare_example are ALREADY shifted. Only labels != -100
    contribute to the token-averaged cross entropy; never shift them again.
    Input embeddings and the output projection intentionally have separate weights.
    """

    def __init__(self, vocab_size=4000, d_model=128, n_layers=4, n_heads=4,
                 d_ff=512, max_length=512, pad_token_id=0, dropout=0.1):
        super().__init__()
        _positive("n_layers", n_layers)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.embedding = TokenPositionEmbedding(vocab_size, d_model, max_length,
                                                pad_token_id, dropout)
        # Construct each layer separately: do NOT repeat one shared block.
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_length, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @classmethod
    def from_config(cls, config):
        """Select architecture fields from the frozen project configuration."""
        fields = ("vocab_size", "d_model", "n_layers", "n_heads", "d_ff",
                  "max_length", "pad_token_id", "dropout")
        return cls(**{name: config[name] for name in fields})

    def forward(self, input_ids, attention_mask=None, labels=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be an int64 tensor [batch, length]")
        if ((input_ids < 0) | (input_ids >= self.vocab_size)).any():
            raise ValueError("input_ids must contain valid vocabulary IDs")
        batch, length = input_ids.shape
        mask = _padding_mask(attention_mask if attention_mask is not None
                             else input_ids.ne(self.pad_token_id), batch, length, input_ids.device)
        if labels is not None:
            if (labels.shape != input_ids.shape or labels.dtype != torch.long
                    or labels.device != input_ids.device):
                raise ValueError("labels must match input shape/device and be int64")
            supervised = labels.ne(-100)
            if (supervised & ~mask).any():
                raise ValueError("Padding labels must be -100")
            if not supervised.any():
                raise ValueError("At least one supervised token is required for loss")
            if ((labels[supervised] < 0) | (labels[supervised] >= self.vocab_size)).any():
                raise ValueError("Supervised labels must be valid vocabulary IDs")

        x = self.embedding(input_ids, mask)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_norm(x).masked_fill(~mask.unsqueeze(-1), 0)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # Cross entropy accepts raw logits, NOT softmax probabilities.
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                                   labels.reshape(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}


def main():
    from torch.utils.data import DataLoader
    from utils.device import get_device
    from .dataset import CustomerSupportDataset, DynamicPaddingCollator

    parser = argparse.ArgumentParser(description="Check one real batch through the complete decoder")
    parser.add_argument("--config", type=Path, default=Path("configs/model.json"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = CustomerSupportDataset(tokenizer_dir=config["tokenizer_artifact"])
    if (config["tokenizer_revision"] != dataset.config["tokenizer_sha256"]
            or config["vocab_size"] != dataset.tokenizer.get_vocab_size()
            or config["max_length"] != dataset.max_length):
        raise ValueError("Model configuration does not match frozen tokenizer")
    for name, token in (("pad", "<PAD>"), ("bos", "<BOS>"), ("user", "<USER>"),
                        ("assistant", "<ASSISTANT>"), ("eos", "<EOS>")):
        if config[f"{name}_token_id"] != dataset.config["special_token_ids"][token]:
            raise ValueError(f"Model {name} token ID mismatch")
    batch = next(iter(DataLoader(dataset, batch_size=4,
                      collate_fn=DynamicPaddingCollator(dataset.pad_token_id))))
    device = get_device(args.device)
    batch = {name: value.to(device) for name, value in batch.items()}
    torch.manual_seed(42)
    model = TinyDecoderLM.from_config(config).to(device).eval()
    result = model(**batch)
    result["loss"].backward()
    print("Device:", device)
    print("Input:", tuple(batch["input_ids"].shape))
    print("Logits:", tuple(result["logits"].shape))
    print("Layers:", len(model.blocks))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Supervised tokens:", int(batch["labels"].ne(-100).sum()))
    print("Reply cross entropy:", float(result["loss"].detach()))
    print("All gradients finite:", all(p.grad is not None and bool(p.grad.isfinite().all())
                                      for p in model.parameters()))
    print("Diagnostic only: no optimizer update or checkpoint has been made.")


if __name__ == "__main__":
    main()
