"""Runner gates, create-only recovery, controls and generation tracing; no real checkpoint."""
from copy import deepcopy
import json

import pytest
import torch
from tokenizers import Tokenizer

from my_project_1 import condition_measurement_v2 as r
from my_project_1.reply_prefix import branch_nll, prefix_reply


@pytest.fixture
def setup(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Unit tests must not load checkpoint")
    monkeypatch.setattr(torch, "load", blocked)
    p = r.load_protocol()
    cfg = r.read(r.ROOT / p["config"])
    tok = Tokenizer.from_file(str(r.ROOT / p["tokenizer"]))
    return p, cfg, tok, r.expand_jobs(p, tok, cfg)


class EOSModel(torch.nn.Module):
    def __init__(self, size=4000):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.size = size

    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        logits = torch.zeros((*input_ids.shape, self.size)) + self.marker
        logits[..., 4] += 5
        return {"logits": logits}


def test_fixed_counts_tokens_and_controls(setup):
    p, cfg, tok, jobs = setup
    assert len(jobs["generations"]) == 26 and len(jobs["scores"]) == 148
    for kind, expected in (("generations", 2), ("scores", 4)):
        assert len({j["id"] for j in jobs[kind]}) == len(jobs[kind])
        assert sum(bool(j["control_of"]) for j in jobs[kind]) == expected
        index = {j["id"]: j for j in jobs[kind]}
        for j in jobs[kind]:
            if j["control_of"]:
                assert {k: v for k, v in j.items() if k not in ("id", "control_of")} == {
                    k: v for k, v in index[j["control_of"]].items() if k not in ("id", "control_of")}


def test_atomic_publication_preserves_equal_and_refuses_conflict(tmp_path):
    path = tmp_path / "item.json"
    r.save_once(path, {"value": 1})
    before = path.read_bytes(), path.stat().st_mtime_ns
    r.save_once(path, {"value": 1})
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="overwrite"):
        r.save_once(path, {"value": 2})
    assert path.read_bytes() == before[0]
    assert not list(tmp_path.glob(".partial-*"))


def test_resume_computes_missing_only_and_rejects_corruption(tmp_path):
    job = {"id": "test", "x": 1}
    calls = []
    compute = lambda: calls.append(1) or {"answer": 1}
    validate = lambda j, v: None
    first = r.record(tmp_path, "scores", job, "fixed", compute, validate)
    assert r.record(tmp_path, "scores", job, "fixed", compute, validate) == first
    assert calls == [1]
    with pytest.raises(ValueError, match="Conflicting"):
        r.record(tmp_path, "scores", job, "different", compute, validate)
    path = tmp_path / "scores/test.json"
    value = r.read(path)
    value["payload"]["result"]["answer"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt"):
        r.record(tmp_path, "scores", job, "fixed", compute, validate)
    assert calls == [1]


@pytest.mark.parametrize("history", [0, 1, 2])
def test_generation_validation_and_model_unchanged(setup, history):
    _, cfg, tok, jobs = setup
    job = next(j for j in jobs["generations"] if j["history_index"] == history)
    model = EOSModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    value = prefix_reply(model, tok, cfg, job["instruction"], job["prefix_text"])
    r.validate_generation(job, value, tok, cfg)
    assert value["stop_reason"] == "eos" and value["continuation_token_ids"] == [4]
    r.assert_unchanged(model, before)
    assert model.training
    bad = deepcopy(value)
    bad["stop_reason"] = "max_new_tokens"
    with pytest.raises(ValueError, match="stop reason"):
        r.validate_generation(job, bad, tok, cfg)
    bad = deepcopy(value)
    bad["steps"][0]["input_length"] += 1
    with pytest.raises(ValueError, match="alignment"):
        r.validate_generation(job, bad, tok, cfg)


def test_score_validation_controls_and_mutation_detection(setup):
    _, cfg, tok, jobs = setup
    job = jobs["scores"][0]
    model = EOSModel()
    value = branch_nll(model, tok, cfg, job["instruction"], job["prefix_ids"], job["candidate"])
    r.validate_score(job, value)
    repeat = dict(job, id="repeat", control_of=job["id"])
    r.check_controls([job, repeat], {job["id"]: value, "repeat": deepcopy(value)}, "scores")
    bad = deepcopy(value)
    bad["per_token_nll"][0] += .01
    with pytest.raises(ValueError, match="control failed"):
        r.check_controls([job, repeat], {job["id"]: value, "repeat": bad}, "scores")
    with pytest.raises(ValueError, match="NLL total"):
        r.validate_score(job, bad)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.marker.add_(1)
    with pytest.raises(ValueError, match="state changed"):
        r.assert_unchanged(model, before)


def test_generation_control_failure(setup):
    _, cfg, tok, jobs = setup
    job, repeat = jobs["generations"][2], jobs["generations"][-2]
    assert repeat["control_of"] == job["id"]
    value = prefix_reply(EOSModel(), tok, cfg, job["instruction"], job["prefix_text"])
    bad = deepcopy(value)
    bad["response"] += "changed"
    with pytest.raises(ValueError, match="control failed"):
        r.check_controls([job, repeat], {job["id"]: value, repeat["id"]: bad}, "generations")


def test_scoring_requires_review_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "verify", lambda out: {"config": {}})
    monkeypatch.setattr(r, "load_model", lambda x: pytest.fail("Model must not load before review"))
    with pytest.raises(FileNotFoundError):
        r.run_stage("scores", tmp_path)


def test_semantic_reviews_require_evidence_and_control_consistency():
    jobs = [{"id": "a", "control_of": None}, {"id": "b", "control_of": "a"}]
    results = {k: {"continuation": "Please check the order options."} for k in ("a", "b")}
    row = dict(id="a", first_action="unavailable", full_task="unavailable",
               first_action_quote="", full_quote="", reason="Generic direction cannot distinguish scope",
               help="Limited", risk="None seen", entities="None", repetition="None", fluency="Clear")
    rows = [row, dict(row, id="b")]
    r.validate_reviews(rows, jobs, results)
    rows[1]["first_action_quote"] = "made up evidence"
    with pytest.raises(ValueError, match="evidence"):
        r.validate_reviews(rows, jobs, results)
    rows[1] = dict(row, id="b", full_task="matching", full_quote="Please check")
    with pytest.raises(ValueError, match="control semantic"):
        r.validate_reviews(rows, jobs, results)


@pytest.mark.parametrize("means,totals,expected", [
    ([.1]*3, [.2]*3, "positive"), ([-.1]*3, [-.2]*3, "negative"),
    ([0]*3, [1e-7]*3, "flat"), ([.1]*3, [-.1]*3, "discordant"),
    ([.1, None, .1], [.1]*3, "unavailable")])
def test_classification_matches_preregistration(means, totals, expected):
    assert r.classify(means, totals) == expected


def test_stage_interruption_resume_and_completion_are_read_only(setup, tmp_path, monkeypatch):
    p, cfg, tok, jobs = setup
    out = tmp_path / "outputs/run"
    frozen = dict(jobs=jobs, config=cfg, generation=p["future_generation"], dependencies_sha256={})
    r.save_once(out / "freeze.json", frozen)
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "verify", lambda x: r.read(out / "freeze.json"))
    monkeypatch.setattr(r, "load_frozen", lambda x: (tok, {}))
    model = EOSModel()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(r, "load_model", lambda x: (model, tok, before))
    original = r.prefix_reply
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("simulated outage")
        return original(*args, **kwargs)
    monkeypatch.setattr(r, "prefix_reply", interrupted)
    with pytest.raises(RuntimeError, match="outage"):
        r.run_stage("generations", out)
    existing = {str(f): (f.read_bytes(), f.stat().st_mtime_ns) for f in (out / "generations").glob("*.json")}
    assert len(existing) == 2
    resumed_calls = []
    def resumed(*args, **kwargs):
        resumed_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(r, "prefix_reply", resumed)
    r.run_stage("generations", out)
    assert len(resumed_calls) == 24
    for f, expected in existing.items():
        from pathlib import Path
        assert (Path(f).read_bytes(), Path(f).stat().st_mtime_ns) == expected
    monkeypatch.setattr(r, "load_model", lambda x: pytest.fail("Completed stage must not load weights"))
    r.run_stage("generations", out)
    # Completion freeze catches tampering; no inference tries to repair it.
    item = next((out / "generations").glob("*.json"))
    item.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        r.run_stage("generations", out)
