"""Train repeatedly on one fixed real train batch; a learning diagnostic only."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import torch

from utils.device import get_device
from .dataset import CustomerSupportDataset, DynamicPaddingCollator
from .model import TinyDecoderLM
from .train import TrainConfig, _save_json, run_training


@dataclass(frozen=True)
class OverfitConfig:
    samples: int = 4
    steps: int = 120
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    cpu_threads: int = 2
    print_every: int = 10
    model_config: str = "configs/model.json"
    output_dir: str = "runs"

    def __post_init__(self):
        for name in ("samples", "steps", "cpu_threads", "print_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("weight_decay", "grad_clip"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.grad_clip == 0:
            raise ValueError("grad_clip must be positive")
        self.train_config()  # Reuse the scaffold's remaining validation.

    def train_config(self):
        return TrainConfig(steps_per_epoch=self.steps, learning_rate=self.learning_rate,
                           seed=self.seed, device=self.device, output_dir=self.output_dir)


def validate_tokenizer(config, dataset):
    if (config["tokenizer_revision"] != dataset.config["tokenizer_sha256"]
            or config["vocab_size"] != dataset.tokenizer.get_vocab_size()
            or config["max_length"] != dataset.max_length):
        raise ValueError("Model configuration does not match frozen tokenizer")
    for name, token in (("pad", "<PAD>"), ("bos", "<BOS>"), ("user", "<USER>"),
                        ("assistant", "<ASSISTANT>"), ("eos", "<EOS>")):
        if config[f"{name}_token_id"] != dataset.config["special_token_ids"][token]:
            raise ValueError(f"Model {name} token ID mismatch")


@torch.no_grad()
def evaluate_batch(model, batch):
    """Same TRAIN examples, dropout off; not dev/test generalization metrics."""
    was_training = model.training
    model.eval()
    try:
        output = model(**batch)
        supervised = batch["labels"].ne(-100)
        accuracy = (output["logits"].argmax(-1)[supervised]
                    == batch["labels"][supervised]).float().mean().item()
        return {"loss": output["loss"].item(), "teacher_forced_accuracy": accuracy}
    finally:
        model.train(was_training)


def run_overfit(config):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        return _run_overfit(config)
    finally:
        torch.set_num_threads(previous_threads)


def _run_overfit(config):
    device = get_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model_config = json.loads(Path(config.model_config).read_text(encoding="utf-8"))
    dataset = CustomerSupportDataset("train", tokenizer_dir=model_config["tokenizer_artifact"])
    validate_tokenizer(model_config, dataset)
    if config.samples > len(dataset):
        raise ValueError("samples exceeds available train rows")
    # First N rows, in frozen file order: no validation/test selection or tuning.
    examples = [dataset[index] for index in range(config.samples)]
    batch = DynamicPaddingCollator(dataset.pad_token_id)(examples)
    batch = {key: value.to(device) for key, value in batch.items()}
    model = TinyDecoderLM.from_config(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    initial = evaluate_batch(model, batch)
    if not math.isfinite(initial["loss"]):
        raise ValueError("Non-finite initial loss")
    print(f"Device: {device}; fixed train samples: {config.samples}; steps: {config.steps}", flush=True)
    print(f"Initial train-batch eval loss: {initial['loss']:.6f}", flush=True)

    def train_step(epoch, step, resolved_device):
        model.train()  # Configured dropout remains enabled during updates.
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch)["loss"]
        if not loss.isfinite():
            raise ValueError(f"Non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        value = loss.detach().item()
        if step == 1 or step % config.print_every == 0 or step == config.steps:
            print(f"Step {step:3d}/{config.steps}: train loss={value:.6f}", flush=True)
        return value

    run_dir = run_training(config.train_config(), train_step, run_metadata={
        "kind": "fixed_train_batch_overfit", "overfit_config": asdict(config),
        "model_config": model_config, "sample_ids": [example["id"] for example in examples],
        "selection": "first N train rows in frozen file order",
        "split_sha256": dataset.config["split_sha256"]["train.jsonl"],
        "input_shape": list(batch["input_ids"].shape),
        "supervised_tokens": int(batch["labels"].ne(-100).sum()),
        "initial_eval": initial, "torch_version": torch.__version__,
        "resolved_device": device,
    })
    final = evaluate_batch(model, batch)
    reduction = 1 - final["loss"] / initial["loss"]
    summary = {"initial_eval": initial, "final_eval": final,
               "loss_reduction_fraction": reduction,
               "clear_decrease": reduction >= 0.5,
               "completed_steps": config.steps,
               "scope": "same train batch; NOT dev/test or generation quality"}
    _save_json(run_dir / "summary.json", summary)
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config, "overfit_config": asdict(config),
                "step": config.steps}, run_dir / "checkpoint.pt")
    print(f"Final train-batch eval loss: {final['loss']:.6f}", flush=True)
    print(f"Loss reduction: {reduction:.1%}; >=50% decrease: {summary['clear_decrease']}", flush=True)
    print(f"Saved config, sample IDs, losses, summary and checkpoint: {run_dir}", flush=True)
    print("This measures memorization of the train batch, not generalization.", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/overfit.json"))
    args = parser.parse_args()
    config = OverfitConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    run_overfit(config)


if __name__ == "__main__":
    main()
