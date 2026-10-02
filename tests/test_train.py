import json

import pytest

from my_project_1.train import TrainConfig, run_training


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_records_actual_losses_and_config(tmp_path):
    calls = []

    def train_step(epoch, step, device):
        calls.append((epoch, step, device))
        return step * 0.25

    config = TrainConfig(epochs=2, steps_per_epoch=2, device="cpu", output_dir=str(tmp_path))
    directory = run_training(config, train_step)
    records = [json.loads(line) for line in (directory / "loss.jsonl").read_text().splitlines()]
    assert calls == [(1, 1, "cpu"), (1, 2, "cpu"), (2, 3, "cpu"), (2, 4, "cpu")]
    assert [r["loss"] for r in records] == [0.25, 0.5, 0.75, 1.0]
    assert all(r["dry_run"] is False for r in records)
    assert read(directory / "config.json")["resolved_device"] == "cpu"
    assert read(directory / "status.json")["completed_steps"] == 4
    assert read(directory / "status.json")["state"] == "completed"


def test_dry_run_creates_separate_directories(tmp_path):
    config = TrainConfig(device="cpu", output_dir=str(tmp_path))
    first = run_training(config, dry_run=True)
    before = (first / "loss.jsonl").read_bytes()
    second = run_training(config, dry_run=True)
    assert first != second
    assert (first / "loss.jsonl").read_bytes() == before
    assert read(second / "config.json")["dry_run"] is True


@pytest.mark.parametrize("failure", [ValueError("bad batch"), KeyboardInterrupt()])
def test_partial_loss_survives_failure(tmp_path, failure):
    def train_step(epoch, step, device):
        if step == 2:
            raise failure
        return 0.5

    with pytest.raises(type(failure)):
        run_training(TrainConfig(device="cpu", output_dir=str(tmp_path)), train_step)
    directory, = tmp_path.iterdir()
    assert len((directory / "loss.jsonl").read_text().splitlines()) == 1
    status = read(directory / "status.json")
    assert status["completed_steps"] == 1
    assert status["state"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")


def test_nonfinite_loss_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Non-finite"):
        run_training(TrainConfig(device="cpu", output_dir=str(tmp_path)), lambda *args: float("nan"))
    directory, = tmp_path.iterdir()
    assert (directory / "loss.jsonl").read_text() == ""
    assert read(directory / "status.json")["state"] == "failed"


def test_missing_training_implementation_does_not_create_run(tmp_path):
    with pytest.raises(ValueError, match="train_step"):
        run_training(TrainConfig(output_dir=str(tmp_path)))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"steps_per_epoch": -1},
                                    {"learning_rate": float("nan")}, {"seed": 1.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
