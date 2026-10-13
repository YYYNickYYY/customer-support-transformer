"""Training scaffold with per-run configuration and incremental loss logs."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Callable
from uuid import uuid4

from utils.device import get_device


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    steps_per_epoch: int = 10
    learning_rate: float = 0.001
    seed: int = 42
    device: str = "auto"
    output_dir: str = "runs"

    def __post_init__(self):
        for name in ("epochs", "steps_per_epoch"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if (type(self.learning_rate) not in (int, float)
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def run_training(
    config: TrainConfig,
    train_step: Callable[[int, int, str], float] | None = None,
    *,
    dry_run: bool = False,
    run_metadata: dict | None = None,
) -> Path:
    """Run a loop and return its unique output directory.

    train_step(epoch, global_step, device) must perform forward/backward/update
    and return a scalar loss (e.g. loss.item()). Epochs and steps are one-based.
    The caller owns data/model/optimizer setup and PyTorch/NumPy seeding BEFORE
    model initialization. This scaffold seeds only Python's random module.
    Dry-run emits demonstration values, not model training measurements.
    """
    if train_step is None and not dry_run:
        raise ValueError("Provide train_step for training, or use --dry-run")
    if dry_run and train_step is not None:
        raise ValueError("dry_run cannot be combined with train_step")
    device = get_device(config.device)
    random.seed(config.seed)
    started = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
    run_dir = Path(config.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _save_json(run_dir / "config.json", {
        **asdict(config), "resolved_device": device, "dry_run": dry_run,
        "run_id": run_id, "started_at": started,
    })
    if run_metadata is not None:
        _save_json(run_dir / "experiment.json", run_metadata)
    status = {"state": "running", "completed_steps": 0, "dry_run": dry_run}
    _save_json(run_dir / "status.json", status)
    try:
        with (run_dir / "loss.jsonl").open("x", encoding="utf-8") as stream:
            for epoch in range(1, config.epochs + 1):
                for _ in range(config.steps_per_epoch):
                    step = status["completed_steps"] + 1
                    loss = float(1.0 / step if dry_run else train_step(epoch, step, device))
                    if not math.isfinite(loss):
                        raise ValueError(f"Non-finite loss at step {step}: {loss}")
                    record = {"epoch": epoch, "step": step, "loss": loss,
                              "dry_run": dry_run}
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    status["completed_steps"] = step
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt)
                      else "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        status["state"] = "completed"
    finally:
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(run_dir / "status.json", status)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON file containing TrainConfig fields")
    parser.add_argument("--dry-run", action="store_true", help="Emit demo loss; no model training")
    args = parser.parse_args()
    config = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else TrainConfig()
    print(run_training(config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
