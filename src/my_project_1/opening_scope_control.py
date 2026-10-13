"""Opening-scope cloze intervention; explicit history/stem provenance, no training.

Real inference is a separate CLI action after method acceptance and freezing.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from . import condition_measurement_v2 as old
from .reply_prefix import branch_nll, checked_prefix, prefix_reply
from .tokenization import encode_parts, load_frozen

ROOT = old.ROOT
PROTOCOL = "docs/opening-scope-control-v1-protocol.md"
ACCEPTANCE = "docs/opening-scope-control-v1-acceptance.md"
OUT = ROOT / "outputs/opening-scope-control-v1-20261006"


def read_block(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if old.digest(ROOT / name) != expected:
            raise ValueError("Frozen dependency changed: " + name)


def load_protocol():
    plan = read_block(ROOT / PROTOCOL)
    verify_hashes(plan["evidence_sha256"])
    verify_hashes({plan["generation"]["checkpoint"]: plan["generation"]["checkpoint_sha256"],
                   plan["generation"]["tokenizer"]: plan["generation"]["tokenizer_revision"]})
    return plan


def accepted_receipt():
    receipt = read_block(ROOT / ACCEPTANCE)
    if receipt["status"] != "accepted_for_matched_opening_scope_cloze_only" or receipt["fifth_inference"]:
        raise ValueError("No valid method acceptance")
    verify_hashes(receipt["sha256"])
    return receipt


def exact_text(tokenizer, text, cfg):
    ids = checked_prefix(tokenizer, text, cfg)
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise ValueError("Text decode mismatch")
    if text and (text.endswith(" ") or tokenizer.decode([ids[-1]], skip_special_tokens=False).isspace()):
        raise ValueError("Standalone trailing whitespace")
    return ids


def expand_jobs(plan, tok, cfg):
    generation = plan["generation"]
    required = dict(device="cpu", cpu_threads=2, seed=42, context=512, total_reply_budget=320,
                    no_repeat_ngram_size=0, decoding="greedy", eos_allowed=True)
    if any(generation[k] != v for k, v in required.items()) or cfg["max_length"] != 512:
        raise ValueError("Changed frozen generation contract")
    if generation["blocked_tokens"] != ["<PAD>", "<BOS>", "<USER>", "<ASSISTANT>"]:
        raise ValueError("Changed role masking")
    if cfg["tokenizer_revision"] != generation["tokenizer_revision"]:
        raise ValueError("Tokenizer revision mismatch")
    history = exact_text(tok, plan["history"], cfg)
    if not plan["history"].endswith(".") or len(history) != 12:
        raise ValueError("Invalid complete history")
    jobs = dict(generations=[], scores=[], probes=[])
    question_ids = {}
    for q in plan["questions"]:
        qids, _, full = encode_parts(tok, q["instruction"], "")
        question_ids[q["id"]] = qids
        if len(qids) != 8:
            raise ValueError("Unmatched question length")
        for opening in plan["openings"]:
            stem = exact_text(tok, opening["stem"], cfg)
            if len(stem) != (0 if opening["id"] == "free" else 2):
                raise ValueError("Unmatched stem length")
            prefix_text = plan["history"] + opening["stem"]
            if exact_text(tok, prefix_text, cfg) != history + stem:
                raise ValueError("History/stem concatenation mismatch")
            base = dict(question_id=q["id"], question_verb=q["verb"], scope=q["scope"],
                        instruction=q["instruction"], opening=opening["id"],
                        history_text=plan["history"], history_ids=history,
                        stem_text=opening["stem"], stem_ids=stem, prefix_text=prefix_text,
                        prefix_ids=history + stem, prompt_ids=full[:-1], control_of=None)
            jobs["generations"].append(dict(base, id=f"{q['id']}-{opening['id']}"))
            if opening["id"] == "free":
                continue
            for group in plan["branches"]:
                lengths = []
                for scope in ("partial", "whole"):
                    target = exact_text(tok, group[scope], cfg)
                    lengths.append(len(target))
                    if exact_text(tok, prefix_text + group[scope], cfg) != history + stem + target:
                        raise ValueError("Candidate concatenation mismatch")
                    if len(full[:-1] + history + stem + target) > cfg["max_length"]:
                        raise ValueError("Candidate overflow; no truncation")
                    jobs["scores"].append(dict(base, id=f"{q['id']}-{opening['id']}-{group['id']}-{scope}",
                                               wording=group["id"], candidate_scope=scope,
                                               candidate=group[scope], target_ids=target))
                if lengths != group["token_lengths"] or lengths[0] != lengths[1]:
                    raise ValueError("Unmatched candidate length")
            for probe in plan["decision_probes"]:
                suffix = exact_text(tok, probe["suffix"], cfg)
                text = prefix_text + probe["suffix"]
                if exact_text(tok, text, cfg) != history + stem + suffix:
                    raise ValueError("Probe boundary mismatch")
                targets = [tok.token_to_id(piece) for piece in probe["tokens"]]
                if any(t is None or t <= cfg["eos_token_id"] for t in targets) or len(set(targets)) != 2:
                    raise ValueError("Bad probe targets")
                context = full[:-1] + history + stem + suffix
                if len(context) >= cfg["max_length"]:
                    raise ValueError("Probe overflow")
                jobs["probes"].append(dict(base, id=f"{q['id']}-{opening['id']}-{probe['id']}",
                                           probe=probe["id"], suffix_text=probe["suffix"], suffix_ids=suffix,
                                           context_ids=context, target_ids=targets))
    # Only the range tokens change inside each question-verb comparison.
    for verb in ("delete", "remove"):
        a, b = question_ids[verb + "-partial"], question_ids[verb + "-whole"]
        if [i for i, (x, y) in enumerate(zip(a, b)) if x != y] != [5, 6]:
            raise ValueError("Uncontrolled question difference")
    for kind in jobs:
        if kind == "generations":
            selected = [j for j in jobs[kind] if any(
                j["question_id"] == c["question"] and j["opening"] == c["opening"]
                for c in plan["controls"]["generation"])]
        else:
            control = plan["controls"]["score" if kind == "scores" else "probes"]
            selected = [j for j in jobs[kind] if j["question_id"] == control["question"]
                        and j["opening"] == control["opening"]]
        jobs[kind].extend(dict(j, id=j["id"] + "-repeat", control_of=j["id"]) for j in selected)
    if {k: len(v) for k, v in jobs.items()} != {"generations": 14, "scores": 36, "probes": 18}:
        raise ValueError("Wrong preregistered counts")
    return jobs


@torch.no_grad()
def probe_distribution(model, job, cfg):
    """Exactly one causal forward on the fixed fork; never infer missing top5 values."""
    context, targets = job["context_ids"], job["target_ids"]
    blocked = [cfg[f"{n}_token_id"] for n in ("pad", "bos", "user", "assistant")]
    if not context or len(context) >= cfg["max_length"]:
        raise ValueError("Probe overflow; refuse truncation")
    if len(targets) != 2 or len(set(targets)) != 2 or any(
            t in blocked or t == cfg["eos_token_id"] or not 0 <= t < cfg["vocab_size"] for t in targets):
        raise ValueError("Invalid probe targets")
    inputs = torch.tensor([context], dtype=torch.long, device=next(model.parameters()).device)
    was_training = model.training
    model.eval()
    try:
        logits = model(inputs, attention_mask=torch.ones_like(inputs))["logits"][0, -1].clone()
        if not logits.isfinite().all():
            raise ValueError("Nonfinite probe logits")
        raw_target = logits[targets].tolist()
        raw_eos = float(logits[cfg["eos_token_id"]])
        logits[blocked] = float("-inf")
        logprobs = logits.log_softmax(-1)
        values = [dict(token_id=t, probability=float(logprobs[t].exp()), log_probability=float(logprobs[t]),
                       raw_logit=float(v)) for t, v in zip(targets, raw_target)]
        top = logits.topk(5).indices.tolist()
        return dict(target_ids=targets, input_length=len(context), blocked_ids=blocked,
                    targets=values, log_odds=values[0]["log_probability"] - values[1]["log_probability"],
                    normalization_logsumexp=float(logits.logsumexp(-1)), raw_eos_logit=raw_eos,
                    eos_probability=float(logprobs[cfg["eos_token_id"]].exp()),
                    top5=[dict(token_id=t, probability=float(logprobs[t].exp())) for t in top])
    finally:
        model.train(was_training)


def validate_probe(job, result, cfg):
    if result["target_ids"] != job["target_ids"] or result["input_length"] != len(job["context_ids"]):
        raise ValueError("Wrong probe provenance")
    blocked = [cfg[f"{n}_token_id"] for n in ("pad", "bos", "user", "assistant")]
    if result["blocked_ids"] != blocked or len(result["targets"]) != 2:
        raise ValueError("Wrong probe mask/targets")
    norm = result["normalization_logsumexp"]
    for value, token in zip(result["targets"], job["target_ids"]):
        if value["token_id"] != token or not all(math.isfinite(value[k]) for k in ("probability", "log_probability", "raw_logit")):
            raise ValueError("Invalid probe values")
        if not 0 <= value["probability"] <= 1 or value["log_probability"] > 1e-6:
            raise ValueError("Invalid probability range")
        if not math.isclose(value["log_probability"], value["raw_logit"] - norm, abs_tol=2e-5, rel_tol=1e-6):
            raise ValueError("Invalid masked normalization")
        if not math.isclose(value["probability"], math.exp(value["log_probability"]), abs_tol=1e-6, rel_tol=1e-5):
            raise ValueError("Probability/log probability mismatch")
    delta = result["targets"][0]["log_probability"] - result["targets"][1]["log_probability"]
    if not math.isfinite(result["log_odds"]) or not math.isclose(delta, result["log_odds"], abs_tol=1e-6):
        raise ValueError("Wrong probe log odds")
    if not math.isfinite(norm) or not math.isfinite(result["raw_eos_logit"]):
        raise ValueError("Invalid probe normalizer")
    expected_eos = math.exp(result["raw_eos_logit"] - norm)
    if not 0 <= result["eos_probability"] <= 1 or not math.isclose(result["eos_probability"], expected_eos, abs_tol=1e-6):
        raise ValueError("EOS masked or probability wrong")
    top = result["top5"]
    if len(top) != 5 or len({x["token_id"] for x in top}) != 5 or any(
            x["token_id"] in blocked or not 0 <= x["token_id"] < cfg["vocab_size"]
            or not math.isfinite(x["probability"]) or not 0 <= x["probability"] <= 1 for x in top):
        raise ValueError("Invalid probe top5")


def check_controls(jobs, values, kind):
    if kind != "probes":
        old.check_controls(jobs, values, kind)
        return
    for job in jobs:
        if job["control_of"]:
            a, b = values[job["id"]], values[job["control_of"]]
            if any(a[k] != b[k] for k in ("input_length", "target_ids", "blocked_ids")):
                raise ValueError("Probe control provenance failed")
            if [v["token_id"] for v in a["top5"]] != [v["token_id"] for v in b["top5"]]:
                raise ValueError("Probe control top5 changed")
            pairs = [(a["eos_probability"], b["eos_probability"]), (a["log_odds"], b["log_odds"])]
            pairs += [(x[k], y[k]) for x, y in zip(a["targets"], b["targets"])
                      for k in ("probability", "log_probability")]
            pairs += [(x["probability"], y["probability"]) for x, y in zip(a["top5"], b["top5"])]
            if any(not math.isfinite(x) or not math.isfinite(y) or abs(x - y) > 1e-6 for x, y in pairs):
                raise ValueError("Probe repeat control failed")


def validate_reviews(rows, jobs, results):
    index = {j["id"]: j for j in jobs}
    labels = {"matching", "conflicting", "mixed", "unavailable"}
    if len(rows) != len(jobs) or {r["id"] for r in rows} != set(index):
        raise ValueError("Incomplete/duplicate reviews")
    for row in rows:
        if row["first_object"] not in {"partial", "whole", "mixed", "unavailable"} or row["full_task"] not in labels:
            raise ValueError("Invalid semantic labels")
        for key in ("reason", "help", "risk", "entities", "repetition", "fluency"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("Missing semantic explanation")
        response = results[row["id"]]["response"]
        prefix_length = len(index[row["id"]]["prefix_text"])
        for key, label in (("first_object_quote", "first_object"), ("full_quote", "full_task")):
            quote = row[key]
            if not isinstance(quote, str):
                raise ValueError("Evidence must be text")
            at = response.find(quote) if isinstance(quote, str) and quote else -1
            if (not quote and row[label] != "unavailable") or (quote and (at < 0 or at + len(quote) <= prefix_length)):
                raise ValueError("Evidence must include generated text, not forced history/stem alone")
    by_id = {r["id"]: r for r in rows}
    for job in jobs:
        if job["control_of"]:
            a, b = by_id[job["id"]], by_id[job["control_of"]]
            if any(a.get(k) != b.get(k) for k in set(a) | set(b) if k != "id"):
                raise ValueError("Inconsistent semantic control")


def check_method():
    plan = load_protocol()
    cfg = old.read(ROOT / plan["generation"]["model_config"])
    tok, _ = load_frozen(ROOT / cfg["tokenizer_artifact"])
    jobs = expand_jobs(plan, tok, cfg)
    return plan, cfg, tok, jobs


def prepare(out=OUT):
    accepted_receipt()
    plan, cfg, _, jobs = check_method()
    out = Path(out)
    if (out / "freeze.json").exists():
        verify(out)
        print("Existing freeze verified; no inference.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Unregistered output is not empty")
    paths = [ROOT / PROTOCOL, ROOT / ACCEPTANCE, Path(__file__),
             ROOT / "tests/test_opening_scope_control.py", ROOT / plan["generation"]["model_config"]]
    paths += list((ROOT / "src/my_project_1").glob("*.py"))
    paths += list((ROOT / cfg["tokenizer_artifact"]).glob("*.json"))
    paths += list((ROOT / "outputs").rglob("*.xlsx"))
    deps = {p.relative_to(ROOT).as_posix(): old.digest(p) for p in paths}
    deps.update(plan["evidence_sha256"])
    deps[plan["generation"]["checkpoint"]] = plan["generation"]["checkpoint_sha256"]
    old.save_once(out / "freeze.json", dict(environment=old.environment(), config=cfg,
        generation=plan["generation"], jobs=jobs, dependencies_sha256=deps, training=False,
        registered_at_utc=datetime.now(timezone.utc).isoformat(), ordering="generate -> review -> scores/probes"))
    print("Frozen inputs/implementation/environment. No model loaded.")


def verify(out):
    out = Path(out)
    if (out / "invalidity.json").exists():
        raise ValueError("Invalid batch")
    frozen = old.read(out / "freeze.json")
    verify_hashes(frozen["dependencies_sha256"])
    if frozen["environment"] != old.environment():
        raise ValueError("Runtime changed")
    _, cfg, _, jobs = check_method()
    if frozen["jobs"] != jobs or frozen["config"] != cfg:
        raise ValueError("Changed jobs/configuration")
    return frozen


def validator(kind, tok, cfg):
    if kind == "generations":
        return lambda j, v: old.validate_generation(j, v, tok, cfg)
    if kind == "scores":
        return old.validate_score
    return lambda j, v: validate_probe(j, v, cfg)


def freeze_reviews(out=OUT):
    out = Path(out)
    frozen = verify(out)
    verify_hashes(old.read(out / "generations-freeze.json")["sha256"])
    path = out / "review-freeze.json"
    if path.exists():
        verify_hashes(old.read(path)["sha256"])
        return
    if any((out / k).exists() and any((out / k).glob("*.json")) for k in ("scores", "probes")):
        raise ValueError("Cannot freeze semantic reviews after numerical scoring")
    jobs = frozen["jobs"]["generations"]
    results = {j["id"]: old.read(out / "generations" / (j["id"] + ".json"))["payload"]["result"] for j in jobs}
    validate_reviews(old.read(out / "semantic-reviews.json"), jobs, results)
    paths = [out / "semantic-reviews.json", out / "generations-freeze.json"]
    old.save_once(path, dict(stage="semantics_before_scores_and_probes", sha256={
        p.relative_to(ROOT).as_posix(): old.digest(p) for p in paths}))


def run_stage(kind, out=OUT):
    if kind not in ("generations", "scores", "probes"):
        raise ValueError("Unknown stage")
    out = Path(out)
    frozen = verify(out)
    if kind != "generations":
        verify_hashes(old.read(out / "generations-freeze.json")["sha256"])
        verify_hashes(old.read(out / "review-freeze.json")["sha256"])
    done = out / (kind + "-freeze.json")
    if done.exists():
        verify_hashes(old.read(done)["sha256"])
        print("Completed stage verified; no new inference.")
        return
    jobs = frozen["jobs"][kind]
    allowed = {j["id"] + ".json" for j in jobs}
    if (out / kind).exists() and any(p.name not in allowed and not p.name.startswith(".partial-")
                                    for p in (out / kind).glob("*.json")):
        raise ValueError("Unexpected record")
    cfg = frozen["config"]
    tok, _ = load_frozen(ROOT / cfg["tokenizer_artifact"])
    validate = validator(kind, tok, cfg)
    freeze_hash = old.digest(out / "freeze.json")
    for job in jobs:
        if (out / kind / (job["id"] + ".json")).exists():
            old.record(out, kind, job, freeze_hash, None, validate)
    model, tok, initial = old.load_model(frozen)
    results = {}
    for i, job in enumerate(jobs, 1):
        if kind == "generations":
            compute = lambda: prefix_reply(model, tok, cfg, job["instruction"], job["prefix_text"], total_reply_budget=320)
        elif kind == "scores":
            compute = lambda: branch_nll(model, tok, cfg, job["instruction"], job["prefix_ids"], job["candidate"])
        else:
            compute = lambda: probe_distribution(model, job, cfg)
        results[job["id"]] = old.record(out, kind, job, freeze_hash, compute, validate)
        print(f"{kind} {i}/{len(jobs)} {job['id']}", flush=True)
    try:
        check_controls(jobs, results, kind)
        old.assert_unchanged(model, initial)
        verify_hashes(frozen["dependencies_sha256"])
    except ValueError as exc:
        old.save_once(out / "invalidity.json", dict(stage=kind, reason=str(exc)))
        raise
    status = dict(records=len(jobs), controls_pass=True, model_state_unchanged=True,
                  gradients_none=True, optimizer_updates=0, stage=kind)
    if kind == "generations":
        status["stops"] = dict(Counter(v["stop_reason"] for v in results.values()))
        status["semantic_review"] = "pending"
    old.save_once(out / (kind + "-status.json"), status)
    paths = [out / kind / (j["id"] + ".json") for j in jobs] + [out / "freeze.json", out / (kind + "-status.json")]
    if kind != "generations":
        paths += [out / "semantic-reviews.json", out / "review-freeze.json", out / "generations-freeze.json"]
    old.save_once(done, dict(sha256={p.relative_to(ROOT).as_posix(): old.digest(p) for p in paths}))


def classify(values):
    if len(values) != 2 or any(v is None or not math.isfinite(v) for v in values):
        return "unavailable"
    if all(abs(v) <= 1e-6 for v in values):
        return "flat"
    if all(v > 1e-6 for v in values):
        return "positive"
    if all(v < -1e-6 for v in values):
        return "negative"
    return "wording_dependent"


def summarize(out=OUT):
    out = Path(out)
    verify(out)
    for kind in ("generations", "scores", "probes"):
        verify_hashes(old.read(out / (kind + "-freeze.json"))["sha256"])
    rows = []
    for verb in ("delete", "remove"):
        for opening in ("delete", "remove"):
            contrasts, probes = [], []
            for wording in ("explicit", "alternative"):
                scores = {(q, c): old.read(out / "scores" / f"{verb}-{q}-{opening}-{wording}-{c}.json")["payload"]["result"]
                          for q in ("partial", "whole") for c in ("partial", "whole")}
                margins = [scores[(q, "whole")]["mean_nll"] - scores[(q, "partial")]["mean_nll"] for q in ("partial", "whole")]
                total = [scores[(q, "whole")]["total_nll"] - scores[(q, "partial")]["total_nll"] for q in ("partial", "whole")]
                counts = {v["token_count"] for v in scores.values()}
                if len(counts) != 1:
                    raise ValueError("Candidate comparison lost equal-length contract")
                n = counts.pop()
                if any(not math.isclose(t, m * n, abs_tol=5e-5, rel_tol=1e-6) for m, t in zip(margins, total)):
                    raise ValueError("Mean/total margin arithmetic mismatch")
                contrasts.append(dict(wording=wording, M_mean=margins, M_total=total,
                                      D_mean=margins[0] - margins[1], D_total=total[0] - total[1]))
            for name in ("after-the", "after-verb"):
                values = [old.read(out / "probes" / f"{verb}-{q}-{opening}-{name}.json")["payload"]["result"]["log_odds"]
                          for q in ("partial", "whole")]
                probes.append(dict(probe=name, R=values, delta_R=values[0] - values[1]))
            rows.append(dict(question_verb=verb, opening=opening, contrasts=contrasts, probes=probes,
                             classification=classify([c["D_mean"] for c in contrasts]),
                             each_prefers_own=all(c["M_mean"][0] > 1e-6 and c["M_mean"][1] < -1e-6 for c in contrasts)))
    old.save_once(out / "summary.json", dict(scope="matched_cloze_only_not_understanding_or_forgetting", comparisons=rows,
        semantic_reviews="semantic-reviews.json", primary_generations=12, training=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "generate", "freeze-review", "score", "probe", "summarize"))
    stage = parser.parse_args().stage
    if stage == "check":
        _, _, _, jobs = check_method()
        print("Tokenizer/encoding preflight only; no model load:", {k: len(v) for k, v in jobs.items()})
    elif stage == "prepare":
        prepare()
    elif stage == "freeze-review":
        freeze_reviews()
    elif stage == "summarize":
        summarize()
    else:
        run_stage({"generate": "generations", "score": "scores", "probe": "probes"}[stage])


if __name__ == "__main__":
    main()
