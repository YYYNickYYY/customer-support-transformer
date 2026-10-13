import importlib.util
from pathlib import Path

import pytest

from test_decoding_comparison import Model, Tokenizer, config
from my_project_1.decoding_comparison import constrained_reply


spec = importlib.util.spec_from_file_location("comparison_runner", Path(__file__).resolve().parents[1] / "scripts/run_decoding_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_history_gate_checks_every_contract_field():
    r = {"response": "hello", "generated_token_ids": [6, 4], "generated_tokens": 2, "stop_reason": "eos"}
    old = {"generated_response": "hello", **{k: r[k] for k in r if k != "response"}}
    assert runner.matches_history(r, old)
    for key, value in [("generated_response", "other"), ("generated_token_ids", [7, 4]),
                       ("generated_tokens", 3), ("stop_reason", "max_new_tokens")]:
        assert not runner.matches_history(r, {**old, key: value})


def test_create_once_refuses_conflicts_and_malformed_resume(tmp_path):
    path = tmp_path / "record.json"
    runner.save(path, {"a": 1})
    before = path.read_bytes()
    runner.save(path, {"a": 1})
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        runner.save(path, {"a": 2})
    bad = tmp_path / "bad.json"
    bad.write_text('{"incomplete":', encoding="utf-8")
    with pytest.raises(ValueError):
        runner.save(bad, {"a": 1})


def record():
    sample = {"id": "test:1", "instruction": "question", "intent": "test", "cohort": "new108"}
    result = constrained_reply(Model([6, 4]), Tokenizer(), config(), "question", no_repeat_ngram_size=4)
    body = {**sample, **result, "arm": "candidate", "execution_freeze_sha256": "frozen"}
    return sample, {"payload": body, "payload_sha256": runner.payload_hash(body)}


def test_valid_resume_trace():
    sample, item = record()
    assert runner.validate_trace(item, sample, "candidate", "frozen", Tokenizer(), config()) == item["payload"]


@pytest.mark.parametrize("field,value", [("arm", "control"), ("execution_freeze_sha256", "other"),
    ("generated_tokens", 99), ("stop_reason", "max_new_tokens"), ("instruction", "other")])
def test_conflicting_resume_rejected_even_with_recomputed_checksum(field, value):
    sample, item = record()
    item["payload"][field] = value
    item["payload_sha256"] = runner.payload_hash(item["payload"])
    with pytest.raises(ValueError):
        runner.validate_trace(item, sample, "candidate", "frozen", Tokenizer(), config())


def test_resume_checksum_corruption_rejected():
    sample, item = record()
    item["payload"]["response"] = "corrupt"
    with pytest.raises(ValueError, match="Corrupt"):
        runner.validate_trace(item, sample, "candidate", "frozen", Tokenizer(), config())


def test_failed_control_never_creates_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUT", tmp_path)
    sample = {"id": "test:1", "instruction": "question", "intent": "test", "cohort": "new108"}
    old = {"test:1": {"generated_response": "different", "generated_token_ids": [4],
                      "generated_tokens": 1, "stop_reason": "eos"}}
    with pytest.raises(ValueError, match="blocked"):
        runner.run_arm("control", [sample], old, Model([4]), Tokenizer(),
                       {"model_config": config(), "max_new_tokens": 320}, "frozen")
    assert not (tmp_path / "traces/candidate").exists()
    assert not (tmp_path / "control-parity.json").exists()
    assert (tmp_path / "failures/1.json").exists()
