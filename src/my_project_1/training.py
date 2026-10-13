"""Full train minibatches, token-weighted dev loss and best/last checkpoints."""

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, Subset

from utils.device import get_device
from .dataset import CustomerSupportDataset, DynamicPaddingCollator
from .model import TinyDecoderLM
from .overfit import validate_tokenizer
from .train import _save_json


@dataclass(frozen=True)
class FullTrainConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    cpu_threads: int = 2
    print_every: int = 25
    checkpoint_every: int = 200
    model_config: str = "configs/model.json"
    output_dir: str = "runs"
    train_limit: int | None = None
    dev_limit: int | None = None
    train_revision: str | None = None

    def __post_init__(self):
        for name in ("epochs", "batch_size", "cpu_threads", "print_every", "checkpoint_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("train_limit", "dev_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be None or a positive integer")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if name != "weight_decay" and value == 0:
                raise ValueError(f"{name} must be positive")
        if type(self.seed) is not int or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid seed or device")
        for name in ("model_config", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path")
        if self.train_revision is not None and (not isinstance(self.train_revision, str) or not self.train_revision.strip()):
            raise ValueError("train_revision must be None or a nonempty manifest path")


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Sum token NLLs / total supervised tokens, NOT mean of batch means."""
    was_training = model.training
    model.eval()
    total_nll, tokens, samples = 0.0, 0, 0
    try:
        for batch in loader:
            batch = to_device(batch, device)
            count = int(batch["labels"].ne(-100).sum())
            value = model(**batch)["loss"].item()
            if count == 0 or not math.isfinite(value):
                raise ValueError("Invalid dev loss or no supervised tokens")
            total_nll += value * count
            tokens += count
            samples += batch["input_ids"].shape[0]
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError("Empty evaluation loader")
    return {"loss": total_nll / tokens, "supervised_tokens": tokens, "samples": samples}


def _select(dataset, limit):
    if limit is None:
        return dataset
    if limit > len(dataset):
        raise ValueError("Subset limit exceeds split size")
    return Subset(dataset, range(limit))


def _checkpoint(path, payload):
    """Replace only this run's checkpoint after a successful complete write."""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_full_training(config):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        return _run_full_training(config)
    finally:
        torch.set_num_threads(previous_threads)


def _run_full_training(config):
    device = get_device(config.device)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model_config = json.loads(Path(config.model_config).read_text(encoding="utf-8"))
    train_options = {"tokenizer_dir": model_config["tokenizer_artifact"]}
    if config.train_revision is not None:
        train_options["train_revision"] = config.train_revision
    train = CustomerSupportDataset("train", **train_options)
    dev = CustomerSupportDataset("dev", tokenizer_dir=model_config["tokenizer_artifact"])
    validate_tokenizer(model_config, train)
    validate_tokenizer(model_config, dev)
    selected_train, selected_dev = _select(train, config.train_limit), _select(dev, config.dev_limit)
    collator = DynamicPaddingCollator(train.pad_token_id)
    # Dedicated generator: validation does not change the training shuffle RNG.
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(selected_train, batch_size=config.batch_size, shuffle=True,
                              generator=generator, collate_fn=collator, num_workers=0, drop_last=False)
    dev_loader = DataLoader(selected_dev, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=0, drop_last=False)
    model = TinyDecoderLM.from_config(model_config).to(device)  # Fresh initialization, not the 4-row model.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-full-" + uuid4().hex
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    subset_run = config.train_limit is not None or config.dev_limit is not None
    _save_json(run_dir / "config.json", {**asdict(config), "resolved_device": device,
        "run_id": run_id, "subset_run": subset_run, "torch_version": torch.__version__})
    _save_json(run_dir / "experiment.json", {"kind": "full_train_dev",
        "model_config": model_config, "train_samples": len(selected_train),
        "dev_samples": len(selected_dev), "subset_run": subset_run,
        "split_sha256": train.config["split_sha256"]["train.jsonl"],
        "split_hashes": train.config["split_sha256"],
        "train_revision": train.revision,
        "actual_train_sha256": (train.revision["actual_train_sha256"] if train.revision
                                else train.config["split_sha256"]["train.jsonl"]),
        "sample_ids": [row["id"] for row in train.rows[:4]],
        "selection": "all rows unless explicit prefix limits; train shuffled each epoch",
        "loss_scope": "response and retained EOS; ignore -100; already shifted",
        "checkpoint_selection": "lowest completed dev loss; sealed_test untouched"})
    status = {"state": "running", "completed_steps": 0, "completed_epochs": 0,
              "subset_run": subset_run, "phase": "baseline_dev"}
    _save_json(run_dir / "status.json", status)
    best_loss, best_epoch = float("inf"), None

    def save_checkpoint(name, epoch):
        _checkpoint(run_dir / name, {"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "model_config": model_config,
            "training_config": asdict(config), "epoch": epoch,
            "train_revision": train.revision,
            "step": status["completed_steps"], "best_dev_loss": best_loss,
            "best_epoch": best_epoch, "subset_run": subset_run})

    print(f"Run: {run_dir}\nDevice: {device}; train: {len(selected_train)}; dev: {len(selected_dev)}; "
          f"batches/epoch: {len(train_loader)}; subset_run: {subset_run}", flush=True)
    try:
        with (run_dir / "loss.jsonl").open("x", encoding="utf-8") as losses, \
                (run_dir / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
            baseline = evaluate(model, dev_loader, device)
            metrics.write(json.dumps({"epoch": 0, "step": 0, "dev": baseline}, allow_nan=False) + "\n")
            metrics.flush()
            print(f"Initial dev loss: {baseline['loss']:.6f}", flush=True)
            for epoch in range(1, config.epochs + 1):
                model.train()
                status["phase"] = "train"
                train_nll, train_tokens, train_samples = 0.0, 0, 0
                for batch_index, batch in enumerate(train_loader, 1):
                    batch = to_device(batch, device)
                    count = int(batch["labels"].ne(-100).sum())
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**batch)["loss"]
                    if not loss.isfinite():
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    status["completed_steps"] += 1
                    value = loss.detach().item()
                    train_nll += value * count
                    train_tokens += count
                    train_samples += batch["input_ids"].shape[0]
                    losses.write(json.dumps({"epoch": epoch, "step": status["completed_steps"],
                        "batch": batch_index, "loss": value, "supervised_tokens": count,
                        "samples": batch["input_ids"].shape[0]}, allow_nan=False) + "\n")
                    losses.flush()
                    if batch_index == 1 or batch_index % config.print_every == 0 or batch_index == len(train_loader):
                        print(f"Epoch {epoch}/{config.epochs}; batch {batch_index}/{len(train_loader)}; "
                              f"loss={value:.6f}", flush=True)
                        _save_json(run_dir / "status.json", status)
                    if status["completed_steps"] % config.checkpoint_every == 0:
                        save_checkpoint("last.pt", epoch)
                status["phase"] = "dev"
                _save_json(run_dir / "status.json", status)
                print(f"Epoch {epoch}: evaluating dev...", flush=True)
                dev_metrics = evaluate(model, dev_loader, device)
                if dev_metrics["loss"] < best_loss:
                    best_loss, best_epoch = dev_metrics["loss"], epoch
                    save_checkpoint("best.pt", epoch)
                save_checkpoint("last.pt", epoch)
                record = {"epoch": epoch, "step": status["completed_steps"],
                    "train": {"loss": train_nll / train_tokens, "supervised_tokens": train_tokens,
                              "samples": train_samples}, "dev": dev_metrics,
                    "best_dev_loss": best_loss, "best_epoch": best_epoch}
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()
                status["completed_epochs"] = epoch
                _save_json(run_dir / "status.json", status)
                print(f"Epoch {epoch}: train loss={record['train']['loss']:.6f}; "
                      f"dev loss={dev_metrics['loss']:.6f}; best epoch={best_epoch}", flush=True)
            _save_json(run_dir / "summary.json", {"initial_dev": baseline,
                "final_dev": dev_metrics, "best_dev_loss": best_loss, "best_epoch": best_epoch,
                "completed_steps": status["completed_steps"], "subset_run": subset_run})
        status.update(state="completed", phase="finished")
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(run_dir / "status.json", status)
    print(f"Finished. Best checkpoint: {run_dir / 'best.pt'}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--smoke-test", action="store_true", help="Only 16 train/16 dev rows, one epoch")
    args = parser.parse_args()
    config = FullTrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.smoke_test:
        config = replace(config, epochs=1, train_limit=16, dev_limit=16, print_every=1)
    run_full_training(config)


if __name__ == "__main__":
    main()
