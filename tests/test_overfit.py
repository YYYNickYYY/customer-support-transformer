import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from my_project_1.dataset import CustomerSupportDataset, DynamicPaddingCollator
from my_project_1.model import TinyDecoderLM
from my_project_1.overfit import OverfitConfig, evaluate_batch, run_overfit, validate_tokenizer
from my_project_1.train import TrainConfig, run_training


@pytest.mark.parametrize("kwargs", [
    {"samples": 0}, {"steps": 0}, {"cpu_threads": 0}, {"print_every": 0},
    {"learning_rate": 0}, {"grad_clip": 0}, {"weight_decay": -1},
    {"grad_clip": float("nan")}, {"device": "invalid"},
])
def test_invalid_overfit_config(kwargs):
    with pytest.raises(ValueError):
        OverfitConfig(**kwargs)


def test_real_training_logs_checkpoint_and_restores_threads(tmp_path, monkeypatch):
    from my_project_1 import overfit

    original = CustomerSupportDataset
    splits = []

    def dataset_factory(split, **kwargs):
        splits.append(split)
        assert split == "train"
        return original(split, **kwargs)

    monkeypatch.setattr(overfit, "CustomerSupportDataset", dataset_factory)
    before = torch.get_num_threads()
    directory = run_overfit(OverfitConfig(samples=2, steps=3, output_dir=str(tmp_path)))
    assert torch.get_num_threads() == before
    assert splits == ["train"]
    metadata = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (directory / "loss.jsonl").read_text().splitlines()]
    assert len(records) == 3
    assert all(record["dry_run"] is False for record in records)
    assert summary["final_eval"]["loss"] < summary["initial_eval"]["loss"]
    assert summary["completed_steps"] == 3
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True, map_location="cpu")
    assert checkpoint["step"] == 3
    model = TinyDecoderLM.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = original("train")
    assert metadata["sample_ids"] == [dataset[index]["id"] for index in range(2)]
    batch = DynamicPaddingCollator(dataset.pad_token_id)([dataset[0], dataset[1]])
    torch.set_num_threads(2)
    try:
        reproduced = evaluate_batch(model, batch)
    finally:
        torch.set_num_threads(before)
    assert model.training  # Evaluation restores the original mode.
    assert reproduced["loss"] == pytest.approx(summary["final_eval"]["loss"], abs=1e-6)


def test_tokenizer_mismatch_rejected():
    dataset = CustomerSupportDataset("train")
    config = json.loads(Path("configs/model.json").read_text(encoding="utf-8"))
    config["tokenizer_revision"] = "wrong"
    with pytest.raises(ValueError, match="frozen tokenizer"):
        validate_tokenizer(config, dataset)


def test_threads_restored_on_failure(tmp_path):
    before = torch.get_num_threads()
    with pytest.raises(FileNotFoundError):
        run_overfit(OverfitConfig(model_config=str(tmp_path / "missing.json")))
    assert torch.get_num_threads() == before


def test_metadata_and_loss_survive_interrupted_training(tmp_path):
    def step(epoch, global_step, device):
        if global_step == 2:
            raise KeyboardInterrupt()
        return 1.5

    with pytest.raises(KeyboardInterrupt):
        run_training(TrainConfig(steps_per_epoch=3, output_dir=str(tmp_path), device="cpu"),
                     step, run_metadata={"sample_ids": ["fixed-id"]})
    directory, = tmp_path.iterdir()
    assert json.loads((directory / "experiment.json").read_text())["sample_ids"] == ["fixed-id"]
    assert len((directory / "loss.jsonl").read_text().splitlines()) == 1
    assert json.loads((directory / "status.json").read_text())["state"] == "interrupted"
