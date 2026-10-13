"""Independent cloze/probability oracle and create-only runner gates; no fifth weights."""
from copy import deepcopy
import json
import math

import pytest
import torch

from my_project_1 import opening_scope_control as r
from my_project_1.reply_prefix import branch_nll, prefix_reply


@pytest.fixture(autouse=True)
def no_real_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Method acceptance cannot load checkpoints")
    monkeypatch.setattr(torch, "load", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def setup():
    return r.check_method()


def test_jobs_have_separate_provenance_and_equal_scope_positions(setup):
    p, cfg, tok, jobs = setup
    assert {k: len(v) for k, v in jobs.items()} == {"generations": 14, "scores": 36, "probes": 18}
    assert [sum(bool(j["control_of"]) for j in jobs[k]) for k in ("generations", "scores", "probes")] == [2, 4, 2]
    for kind in jobs:
        assert len({j["id"] for j in jobs[kind]}) == len(jobs[kind])
        for j in jobs[kind]:
            assert j["prefix_ids"] == j["history_ids"] + j["stem_ids"]
            assert j["prefix_text"] == j["history_text"] + j["stem_text"]
            assert len(j["prompt_ids"]) == 11 and len(j["history_ids"]) == 12
            assert len(j["stem_ids"]) == (0 if j["opening"] == "free" else 2)
            if kind == "scores":
                assert tok.encode(j["prefix_text"] + j["candidate"]).ids == j["prefix_ids"] + j["target_ids"]
            elif kind == "probes":
                assert j["context_ids"] == j["prompt_ids"] + j["history_ids"] + j["stem_ids"] + j["suffix_ids"]
                assert len(j["context_ids"]) == (26 if j["probe"] == "after-the" else 25)
    assert max(len(j["prompt_ids"] + j["prefix_ids"] + j["target_ids"]) for j in jobs["scores"]) == 28


@pytest.mark.parametrize("bad", ["history", "question", "stem", "candidate", "mask"])
def test_modified_input_contracts_refused(setup, bad):
    p, cfg, tok, _ = setup
    p = deepcopy(p)
    if bad == "history":
        p["history"] += " "
    elif bad == "question":
        p["questions"][0]["instruction"] = "Please help me delete all selected items."
    elif bad == "stem":
        p["openings"][1]["stem"] += " "
    elif bad == "candidate":
        p["branches"][0]["whole"] = " the entire order immediately"
    else:
        p["generation"]["blocked_tokens"].append("<EOS>")
    with pytest.raises(ValueError):
        r.expand_jobs(p, tok, cfg)


class Oracle(torch.nn.Module):
    def __init__(self, size=4000):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.size = size

    def forward(self, ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert attention_mask.eq(1).all()
        cumulative = ids.double().cumsum(-1)
        positions = torch.arange(1, ids.shape[1] + 1, dtype=torch.float64)
        vocabulary = torch.arange(self.size, dtype=torch.float64)
        logits = torch.sin(cumulative[..., None] * .001 + positions[None, :, None] * (vocabulary + 1) * .0001)
        logits[..., :4] += 100  # Controls would dominate unless genuinely masked.
        logits[..., 5:10] += 20  # Ensure designated target tokens are outside top5.
        return {"logits": logits + self.weight}


def scalar_logits(history, size):
    return [math.sin(sum(history) * .001 + len(history) * (v + 1) * .0001)
            + (100 if v < 4 else 20 if 5 <= v < 10 else 0) for v in range(size)]


def scalar_lse(values):
    top = max(values)
    return top + math.log(sum(math.exp(x - top) for x in values))


@pytest.mark.parametrize("question,opening", [(q, s) for q in ("delete-partial", "delete-whole", "remove-partial", "remove-whole")
                                            for s in ("delete", "remove")])
def test_branch_labels_by_independent_real_token_scalar_oracle(setup, question, opening):
    _, cfg, tok, jobs = setup
    j = next(j for j in jobs["scores"] if j["question_id"] == question and j["opening"] == opening and j["candidate_scope"] == "partial")
    model = Oracle()
    value = branch_nll(model, tok, cfg, j["instruction"], j["prefix_ids"], j["candidate"])
    # Construct independently: no production encode_parts/shift/CE or forward reuse.
    history = [1, 2] + tok.encode(j["instruction"]).ids + [3]
    history += tok.encode(j["history_text"]).ids + tok.encode(j["stem_text"]).ids
    expected = []
    for target in tok.encode(j["candidate"]).ids:
        logits = scalar_logits(history, cfg["vocab_size"])
        expected.append(scalar_lse(logits) - logits[target])
        history.append(target)
    assert value["per_token_nll"] == pytest.approx(expected, abs=1e-10)
    assert value["total_nll"] == pytest.approx(sum(expected), abs=1e-9)
    r.old.validate_score(j, value)
    assert model.training and model.weight.grad is None


@pytest.mark.parametrize("probe", ["after-the", "after-verb"])
def test_masked_probe_direct_probability_not_top5_imputation(setup, probe):
    _, cfg, tok, jobs = setup
    j = next(j for j in jobs["probes"] if j["probe"] == probe)
    model = Oracle()
    result = r.probe_distribution(model, j, cfg)
    logits = scalar_logits(j["context_ids"], cfg["vocab_size"])
    norm = scalar_lse(logits[4:])
    expected = [logits[t] - norm for t in j["target_ids"]]
    assert [v["log_probability"] for v in result["targets"]] == pytest.approx(expected, abs=1e-11)
    assert result["log_odds"] == pytest.approx(expected[0] - expected[1], abs=1e-11)
    assert result["eos_probability"] == pytest.approx(math.exp(logits[4] - norm), abs=1e-12)
    assert not {v["token_id"] for v in result["top5"]} & set(j["target_ids"])
    assert all(v["probability"] > 0 for v in result["targets"])
    r.validate_probe(j, result, cfg)
    assert model.training and model.weight.grad is None


def test_future_target_does_not_change_probe_context_prediction(setup):
    _, cfg, tok, jobs = setup
    j = next(j for j in jobs["probes"] if j["probe"] == "after-the")
    model = Oracle()
    short = r.probe_distribution(model, j, cfg)
    # A causal forward with a future appended still has the same earlier fork logits.
    model.eval()
    with torch.no_grad():
        position = len(j["context_ids"]) - 1
        logits_a = model(torch.tensor([j["context_ids"] + [101, 102]]), torch.ones(1, position + 3))["logits"][0, position]
        logits_b = model(torch.tensor([j["context_ids"] + [201, 202]]), torch.ones(1, position + 3))["logits"][0, position]
    assert torch.equal(logits_a, logits_b)
    delta = float(logits_a[j["target_ids"][0]] - logits_a[j["target_ids"][1]])
    assert short["log_odds"] == pytest.approx(delta, abs=1e-11)


def test_probe_failure_restores_mode_and_refuses_overflow(setup):
    _, cfg, _, jobs = setup
    job = jobs["probes"][0]
    class Broken(Oracle):
        def forward(self, *args, **kwargs):
            raise RuntimeError("synthetic failure")
    model = Broken()
    with pytest.raises(RuntimeError, match="synthetic"):
        r.probe_distribution(model, job, cfg)
    assert model.training and model.weight.grad is None
    with pytest.raises(ValueError, match="overflow"):
        r.probe_distribution(Oracle(), dict(job, context_ids=[5]*512), cfg)
    with pytest.raises(ValueError, match="targets"):
        r.probe_distribution(Oracle(), dict(job, target_ids=[4, 5]), cfg)


def test_nonfinite_probe_and_corrupt_probability_rejected(setup):
    _, cfg, _, jobs = setup
    job = jobs["probes"][0]
    class Nonfinite(Oracle):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            output["logits"][..., 20] = float("nan")
            return output
    model = Nonfinite()
    with pytest.raises(ValueError, match="Nonfinite"):
        r.probe_distribution(model, job, cfg)
    assert model.training
    value = r.probe_distribution(Oracle(), job, cfg)
    value["targets"][0]["probability"] = .9
    with pytest.raises(ValueError, match="mismatch"):
        r.validate_probe(job, value, cfg)


def test_probe_repeat_contract_detects_changed_probability(setup):
    _, cfg, _, jobs = setup
    job = jobs["probes"][0]
    repeat = dict(job, id="repeat", control_of=job["id"])
    value = r.probe_distribution(Oracle(), job, cfg)
    r.check_controls([job, repeat], {job["id"]: value, "repeat": deepcopy(value)}, "probes")
    changed = deepcopy(value)
    changed["targets"][0]["probability"] += .01
    with pytest.raises(ValueError, match="repeat control"):
        r.check_controls([job, repeat], {job["id"]: value, "repeat": changed}, "probes")


@pytest.mark.parametrize("probe,wording,position", [("after-the", "explicit", 1), ("after-verb", "alternative", 0)])
def test_probe_matches_same_branch_fork_log_odds_not_mismatched_position(setup, probe, wording, position):
    _, cfg, tok, jobs = setup
    j = next(j for j in jobs["probes"] if j["probe"] == probe)
    model = Oracle()
    probabilities = r.probe_distribution(model, j, cfg)
    scores = []
    for scope in ("partial", "whole"):
        branch = next(b for b in jobs["scores"] if b["question_id"] == j["question_id"]
                      and b["opening"] == j["opening"] and b["wording"] == wording and b["candidate_scope"] == scope)
        scores.append(branch_nll(model, tok, cfg, branch["instruction"], branch["prefix_ids"], branch["candidate"]))
    if position == 1:
        assert scores[0]["per_token_nll"][0] == scores[1]["per_token_nll"][0]
    # Masking changes the shared normalizer, but not odds of the same two allowed tokens.
    delta = scores[1]["per_token_nll"][position] - scores[0]["per_token_nll"][position]
    assert delta == pytest.approx(probabilities["log_odds"], abs=1e-10)


def test_prepare_requires_acceptance_without_creating_output(tmp_path, monkeypatch):
    def no_receipt():
        raise ValueError("No valid method acceptance")
    monkeypatch.setattr(r, "accepted_receipt", no_receipt)
    with pytest.raises(ValueError, match="acceptance"):
        r.prepare(tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_corrupt_partial_trace_rejected_before_model_load(setup, tmp_path, monkeypatch):
    p, cfg, tok, jobs = setup
    out = tmp_path / "outputs/test"
    frozen = dict(config=cfg, jobs=jobs, generation=p["generation"], dependencies_sha256={})
    r.old.save_once(out / "freeze.json", frozen)
    first = jobs["generations"][0]
    r.old.save_once(out / "generations" / (first["id"] + ".json"), {"payload": {}, "sha256": "corrupt"})
    monkeypatch.setattr(r, "verify", lambda x: frozen)
    monkeypatch.setattr(r, "load_frozen", lambda x: (tok, {}))
    monkeypatch.setattr(r.old, "load_model", lambda x: pytest.fail("Corrupt existing records must fail before loading weights"))
    with pytest.raises(ValueError, match="Corrupt"):
        r.run_stage("generations", out)


def test_prepare_freezes_all_jobs_without_weights_and_is_read_only_on_repeat(tmp_path, monkeypatch):
    # Use the existing protocol as a fixture receipt path; actual acceptance is a separate gate.
    monkeypatch.setattr(r, "accepted_receipt", lambda: {})
    monkeypatch.setattr(r, "ACCEPTANCE", r.PROTOCOL)
    out = tmp_path / "prepared"
    r.prepare(out)
    frozen = r.verify(out)
    assert frozen["training"] is False and len(frozen["jobs"]["probes"]) == 18
    assert frozen["jobs"]["generations"][0]["history_ids"]
    path = out / "freeze.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    r.prepare(out)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not (out / "generations").exists()


def test_receipt_binds_implementation_and_rejects_changed_dependency(tmp_path, monkeypatch):
    binding = tmp_path / "implementation.py"
    binding.write_text("fixture", encoding="utf-8")
    receipt = dict(status="accepted_for_matched_opening_scope_cloze_only", fifth_inference=False,
                   sha256={"implementation.py":r.old.digest(binding)})
    document = tmp_path / "acceptance.md"
    document.write_text("```json\n" + json.dumps(receipt) + "\n```", encoding="utf-8")
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "ACCEPTANCE", "acceptance.md")
    assert r.accepted_receipt() == receipt
    binding.write_text("changed fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency changed"):
        r.accepted_receipt()


def test_semantic_evidence_cannot_be_forced_stem_alone():
    job = dict(id="test", control_of=None, prefix_text="Thank you. To delete")
    result = {"test": {"response": "Thank you. To delete your order."}}
    row = dict(id="test", first_object="whole", first_object_quote="To delete your order",
               full_task="conflicting", full_quote="your order", reason="Wrong scope",
               help="Wrong", risk="Not assessed", entities="None", repetition="None", fluency="Readable")
    r.validate_reviews([row], [job], result)
    row["first_object_quote"] = "To delete"
    with pytest.raises(ValueError, match="forced history/stem"):
        r.validate_reviews([row], [job], result)


@pytest.mark.parametrize("values,expected", [([.1,.2],"positive"), ([-.1,-.2],"negative"),
                                            ([0,1e-7],"flat"), ([.1,-.1],"wording_dependent"),
                                            ([None,.1],"unavailable")])
def test_two_wording_rule(values, expected):
    assert r.classify(values) == expected


@pytest.mark.parametrize("kind", ["scores", "probes"])
def test_numerical_stage_requires_frozen_reviews(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(r, "verify", lambda x: {})
    monkeypatch.setattr(r.old, "load_model", lambda x: pytest.fail("Must not load weights before review"))
    with pytest.raises(FileNotFoundError):
        r.run_stage(kind, tmp_path)


class EOSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
    def forward(self, ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        values = torch.zeros((*ids.shape, 4000)) + self.marker
        values[..., :4] += 100  # Must be blocked by the original greedy function.
        values[..., 4] += 5
        return {"logits": values}


def test_synthetic_full_pipeline_resume_read_only_and_corruption(setup, tmp_path, monkeypatch):
    p, cfg, tok, jobs = setup
    out = tmp_path / "outputs/test"
    frozen = dict(config=cfg, jobs=jobs, generation=p["generation"], dependencies_sha256={})
    r.old.save_once(out / "freeze.json", frozen)
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "verify", lambda x: r.old.read(out / "freeze.json"))
    monkeypatch.setattr(r, "load_frozen", lambda x: (tok, {}))
    model = EOSModel()
    state = {k:v.clone() for k,v in model.state_dict().items()}
    monkeypatch.setattr(r.old, "load_model", lambda x: (model, tok, state))
    original = r.prefix_reply
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("outage")
        return original(*args, **kwargs)
    monkeypatch.setattr(r, "prefix_reply", interrupted)
    with pytest.raises(RuntimeError, match="outage"):
        r.run_stage("generations", out)
    existing = {f:(f.read_bytes(),f.stat().st_mtime_ns) for f in (out/"generations").glob("*.json")}
    assert len(existing) == 2
    monkeypatch.setattr(r, "prefix_reply", original)
    r.run_stage("generations", out)
    assert all((f.read_bytes(), f.stat().st_mtime_ns) == old for f,old in existing.items())
    rows = [dict(id=j["id"], first_object="unavailable", first_object_quote="", full_task="unavailable",
                 full_quote="", reason="EOS without action", help="None", risk="None", entities="None",
                 repetition="None", fluency="No continuation") for j in jobs["generations"]]
    r.old.save_once(out/"semantic-reviews.json", rows)
    r.freeze_reviews(out)
    r.run_stage("scores", out)
    r.run_stage("probes", out)
    r.summarize(out)
    assert len(r.old.read(out/"summary.json")["comparisons"]) == 4
    before = {f:(f.read_bytes(),f.stat().st_mtime_ns) for f in out.rglob("*.json")}
    monkeypatch.setattr(r.old, "load_model", lambda x: pytest.fail("Completed stage must not load weights"))
    for kind in ("generations", "scores", "probes"):
        r.run_stage(kind, out)
    r.freeze_reviews(out)
    r.summarize(out)
    assert all((f.read_bytes(),f.stat().st_mtime_ns)==value for f,value in before.items())
    target = next((out/"probes").glob("*.json"))
    target.write_text("{}",encoding="utf-8")
    with pytest.raises(ValueError,match="dependency changed"):
        r.run_stage("probes",out)
