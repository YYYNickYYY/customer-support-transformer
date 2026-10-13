import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from my_project_1 import training
from my_project_1.dataset import CustomerSupportDataset
from my_project_1.model import TinyDecoderLM
from my_project_1.training import FullTrainConfig, evaluate, run_full_training


def test_five_epoch_experiment_configuration_matches_defaults():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "training.json"
    configured = FullTrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    assert configured == FullTrainConfig()
    assert configured.epochs == 5
    assert configured.learning_rate == 0.0003
    assert configured.batch_size == 8
    assert configured.model_config == "configs/model.json"


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0}, {"batch_size": 0}, {"learning_rate": 0},
    {"weight_decay": -1}, {"grad_clip": 0}, {"cpu_threads": 0},
    {"checkpoint_every": 0}, {"train_limit": 0}, {"dev_limit": False},
    {"device": "bad"}, {"seed": True}, {"learning_rate": float("nan")},
])
def test_invalid_training_configuration(kwargs):
    with pytest.raises(ValueError):
        FullTrainConfig(**kwargs)


def test_evaluation_is_token_weighted_and_restores_mode():
    class ConstantLoss(torch.nn.Module):
        def forward(self, **batch):
            assert not self.training and not torch.is_grad_enabled()
            return {"loss": torch.tensor(float(batch["input_ids"][0, 0]))}

    batches = [{"input_ids": torch.tensor([[2, 0, 0]]),
                "labels": torch.tensor([[1, -100, -100]])},
               {"input_ids": torch.tensor([[4, 0, 0]]),
                "labels": torch.tensor([[1, 1, 1]])}]
    model = ConstantLoss()
    result = evaluate(model, batches, "cpu")
    assert result == {"loss": 3.5, "supervised_tokens": 4, "samples": 2}
    assert model.training
    with pytest.raises(ValueError, match="Empty"):
        evaluate(model, [], "cpu")
    assert model.training


def test_best_checkpoint_not_overwritten_by_worse_epoch(tmp_path, monkeypatch):
    splits = []

    def dataset(split, **kwargs):
        splits.append(split)
        assert split in {"train", "dev"}
        return CustomerSupportDataset(split, **kwargs)

    monkeypatch.setattr(training, "CustomerSupportDataset", dataset)
    values = iter([9.0, 3.0, 4.0])
    monkeypatch.setattr(training, "evaluate", lambda *args: {
        "loss": next(values), "supervised_tokens": 100, "samples": 5})
    before = torch.get_num_threads()
    directory = run_full_training(FullTrainConfig(epochs=2, batch_size=4,
        train_limit=6, dev_limit=5, checkpoint_every=1, output_dir=str(tmp_path), device="cpu"))
    assert torch.get_num_threads() == before
    assert splits == ["train", "dev"]
    logs = [json.loads(line) for line in (directory / "loss.jsonl").read_text().splitlines()]
    assert len(logs) == 4
    assert [record["samples"] for record in logs] == [4, 2, 4, 2]  # Last short batch kept.
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert [record["dev"]["loss"] for record in metrics] == [9, 3, 4]
    assert metrics[-1]["best_epoch"] == 1
    best = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    last = torch.load(directory / "last.pt", weights_only=True, map_location="cpu")
    assert best["epoch"] == 1 and best["step"] == 2
    assert last["epoch"] == 2 and last["step"] == 4
    assert best["subset_run"] is True
    model = TinyDecoderLM.from_config(best["model_config"])
    model.load_state_dict(best["model_state_dict"])
    assert json.loads((directory / "status.json").read_text())["state"] == "completed"


def test_eval_failure_preserves_loss_and_checkpoint(tmp_path, monkeypatch):
    calls = 0

    def eval_with_failure(*args):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise KeyboardInterrupt()
        return {"loss": 8.0, "supervised_tokens": 10, "samples": 2}

    monkeypatch.setattr(training, "evaluate", eval_with_failure)
    with pytest.raises(KeyboardInterrupt):
        run_full_training(FullTrainConfig(epochs=1, batch_size=2, train_limit=2,
            dev_limit=2, checkpoint_every=1, output_dir=str(tmp_path), device="cpu"))
    directory, = tmp_path.iterdir()
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "interrupted" and status["completed_steps"] == 1
    assert status["completed_epochs"] == 0 and status["phase"] == "dev"
    assert (directory / "last.pt").is_file()
    assert not (directory / "best.pt").exists()
    assert len((directory / "loss.jsonl").read_text().splitlines()) == 1


def test_default_is_full_splits_and_model_configuration_unchanged():
    config = FullTrainConfig(**json.loads(Path("configs/training.json").read_text()))
    assert config.train_limit is None and config.dev_limit is None
    assert config.batch_size == 8
    assert len(CustomerSupportDataset("train")) == 21528
    assert len(CustomerSupportDataset("dev")) == 2688
