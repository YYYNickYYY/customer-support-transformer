"""Frozen v2 execution: generate -> semantic review freeze -> candidate scoring.

Append-only, hash-bound records. No optimizer or training entry point.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

import torch

from .model import TinyDecoderLM
from .reply_prefix import branch_nll, checked_prefix, prefix_reply
from .tokenization import encode_parts, load_frozen

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/condition-use-measurement-v2-protocol.md"
ACCEPTANCE = "docs/condition-use-measurement-v2-acceptance.md"
OUT = ROOT / "outputs/condition-use-measurement-v2-20261006"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_once(path, value):
    """Atomic exclusive publication; equal existing data is read-only, conflicts fail."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError("Refuse overwrite: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".partial-", suffix=".json", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)  # No replacement if another writer published first.
        except FileExistsError:
            if read(path) != value:
                raise ValueError("Concurrent conflicting publication")
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def environment():
    import tokenizers
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=str(torch.__version__), tokenizers=tokenizers.__version__, device="cpu",
                cpu_threads=2, seed=42, deterministic=torch.are_deterministic_algorithms_enabled())


def verify_hashes(mapping):
    for relative, expected in mapping.items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen dependency changed: " + relative)


def load_protocol():
    text = (ROOT / PROTOCOL).read_text(encoding="utf-8")
    plan = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
    verify_hashes(plan["dependencies"])
    return plan


def expand_jobs(plan, tokenizer, cfg):
    scores, generations = [], []
    for case in plan["cases"]:
        family = plan["families"][case["family"]]
        for hi, prefix in enumerate(plan["prefixes"]):
            history = checked_prefix(tokenizer, prefix, cfg)
            if prefix and (not prefix.endswith(".") or prefix != prefix.strip()):
                raise ValueError("Noncanonical history")
            if tokenizer.decode(history, skip_special_tokens=False) != prefix:
                raise ValueError("History decode mismatch")
            for qi, question in enumerate(case["questions"]):
                prompt = encode_parts(tokenizer, question, "")[2][:-1]
                base = dict(pair_id=case["id"], family=case["family"], question_index=qi,
                            task=family["a" if qi == 0 else "b"], source_row=case["source_rows"][qi],
                            instruction=question, history_index=hi, prefix_text=prefix,
                            prefix_ids=history, prompt_ids=prompt, control_of=None)
                generations.append(dict(base, id=f"{case['id']}-h{hi}-q{qi}"))
                for wi, candidates in enumerate(family["branches"]):
                    lengths = [len(checked_prefix(tokenizer, c, cfg)) for c in candidates]
                    if abs(lengths[0] - lengths[1]) > plan["candidate_length_difference_max"]:
                        raise ValueError("Candidate length mismatch")
                    for ci, candidate in enumerate(candidates):
                        target = checked_prefix(tokenizer, candidate, cfg)
                        if not target or not candidate.startswith(" ") or candidate.endswith(" "):
                            raise ValueError("Bad action boundary")
                        if tokenizer.encode(prefix + candidate, add_special_tokens=False).ids != history + target:
                            raise ValueError("Response concatenation mismatch")
                        if tokenizer.decode(history + target, skip_special_tokens=False) != prefix + candidate:
                            raise ValueError("Response decode mismatch")
                        if len(prompt + history + target) > cfg["max_length"]:
                            raise ValueError("Candidate overflow, no truncation")
                        scores.append(dict(base, id=f"{case['id']}-h{hi}-w{wi}-q{qi}-c{ci}",
                                           wording=wi, candidate_index=ci, candidate=candidate, target_ids=target))
    for jobs in (generations, scores):
        chosen = [j for j in jobs if j["pair_id"] == plan["cases"][0]["id"]
                  and j["history_index"] == 1 and ("wording" not in j or j["wording"] == 0)]
        jobs.extend(dict(j, id=j["id"] + "-repeat", control_of=j["id"]) for j in chosen)
    if len(generations) != 26 or len(scores) != 148:
        raise ValueError("Wrong v2 counts")
    return dict(generations=generations, scores=scores)


def prepare(out=OUT):
    out = Path(out)
    plan = load_protocol()
    cfg = read(ROOT / plan["config"])
    tok, tc = load_frozen(ROOT / cfg["tokenizer_artifact"])
    if tc["tokenizer_sha256"] != cfg["tokenizer_revision"]:
        raise ValueError("Tokenizer/config mismatch")
    jobs = expand_jobs(plan, tok, cfg)
    if (out / "freeze.json").exists():
        frozen = verify(out)
        if frozen["jobs"] != jobs:
            raise ValueError("Changed frozen jobs")
        print("Existing execution freeze verified; no new inference.", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Unregistered nonempty output directory")
    deps = dict(plan["dependencies"])
    paths = [ROOT / PROTOCOL, ROOT / ACCEPTANCE, Path(__file__),
             ROOT / "tests/test_condition_measurement_v2_runner.py",
             ROOT / "tests/test_condition_use_measurement_v2.py"]
    paths += list((ROOT / "src/my_project_1").glob("*.py"))
    paths += list((ROOT / cfg["tokenizer_artifact"]).glob("*.json"))
    paths += list((ROOT / "outputs").rglob("*.xlsx"))
    deps.update({p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    frozen = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=digest(ROOT / PROTOCOL), dependencies_sha256=deps,
                  environment=environment(), config=cfg, generation=plan["future_generation"],
                  jobs=jobs, training=False, ordering="generate -> review-freeze -> score")
    save_once(out / "freeze.json", frozen)
    print("Frozen 26 continuations and 148 branch forwards before inference.", flush=True)


def verify(out):
    out = Path(out)
    if (out / "invalidity.json").exists():
        raise ValueError("Batch marked invalid; cannot continue")
    frozen = read(out / "freeze.json")
    verify_hashes(frozen["dependencies_sha256"])
    if frozen["environment"] != environment():
        raise ValueError("Runtime changed")
    return frozen


def record(out, kind, job, freeze_hash, compute, validate):
    path = Path(out) / kind / (job["id"] + ".json")
    if path.exists():
        wrapped = read(path)
        value = wrapped["payload"]
        if wrapped["sha256"] != payload_hash(value):
            raise ValueError("Corrupt record: " + job["id"])
        if value["freeze_sha256"] != freeze_hash or value["job"] != job:
            raise ValueError("Conflicting record/freeze: " + job["id"])
        validate(job, value["result"])
        return value["result"]
    result = compute()
    validate(job, result)
    value = dict(job=job, result=result, freeze_sha256=freeze_hash)
    save_once(path, dict(payload=value, sha256=payload_hash(value)))
    return result


def validate_generation(job, result, tok, cfg):
    prefix, new = result["prefix_token_ids"], result["continuation_token_ids"]
    if prefix != job["prefix_ids"] or result["prefix_text"] != job["prefix_text"]:
        raise ValueError("Wrong generation history")
    if not new or any(type(t) is not int or not 0 <= t < cfg["vocab_size"] for t in new):
        raise ValueError("Invalid generated tokens")
    if result["generated_token_ids"] != prefix + new or result["generated_tokens"] != len(prefix + new):
        raise ValueError("Generated count mismatch")
    if result["prefix_tokens"] != len(prefix) or result["continuation_tokens"] != len(new):
        raise ValueError("Continuation count mismatch")
    if result["prompt_tokens"] != len(job["prompt_ids"]):
        raise ValueError("Prompt count mismatch")
    if result["response"] != tok.decode(prefix + new, skip_special_tokens=True):
        raise ValueError("Response decode mismatch")
    if result["continuation"] != tok.decode(new, skip_special_tokens=True):
        raise ValueError("Continuation decode mismatch")
    blocked = {cfg[f"{n}_token_id"] for n in ("pad", "bos", "user", "assistant")}
    if any(t in blocked for t in new) or cfg["eos_token_id"] in new[:-1]:
        raise ValueError("Generated control tokens")
    if len(prefix + new) > 320 or len(job["prompt_ids"] + prefix + new) > cfg["max_length"]:
        raise ValueError("Generation exceeded budget")
    expected_stop = ("eos" if new[-1] == cfg["eos_token_id"] else
                     "context_limit" if len(job["prompt_ids"] + prefix + new) == cfg["max_length"]
                     else "max_new_tokens" if len(prefix + new) == 320 else None)
    if result["stop_reason"] != expected_stop:
        raise ValueError("Wrong stop reason")
    if len(result["steps"]) != len(new):
        raise ValueError("Missing generation steps")
    for i, step in enumerate(result["steps"]):
        if (step["step"], step["token_id"], step["input_length"]) != (
                i + 1, new[i], len(job["prompt_ids"]) + len(prefix) + i):
            raise ValueError("Wrong trace alignment")
        if not math.isfinite(step["eos_probability"]) or not 0 <= step["eos_probability"] <= 1:
            raise ValueError("Invalid trace probabilities")
        if len(step["top5"]) != 5 or any(
                not math.isfinite(v["probability"]) or not 0 <= v["probability"] <= 1
                or v["token_id"] in blocked for v in step["top5"]):
            raise ValueError("Invalid top5 trace")


def validate_score(job, result):
    losses = result["per_token_nll"]
    if result["target_ids"] != job["target_ids"] or result["token_count"] != len(job["target_ids"]):
        raise ValueError("Wrong score targets")
    if len(losses) != len(job["target_ids"]) or any(not math.isfinite(v) or v < 0 for v in losses):
        raise ValueError("Invalid NLL")
    if result["input_length"] != len(job["prompt_ids"] + job["prefix_ids"] + job["target_ids"]) - 1:
        raise ValueError("Wrong score input length")
    if not math.isclose(sum(losses), result["total_nll"], abs_tol=2e-5, rel_tol=1e-6):
        raise ValueError("Wrong NLL total")
    if not math.isclose(sum(losses) / len(losses), result["mean_nll"], abs_tol=2e-6, rel_tol=1e-6):
        raise ValueError("Wrong NLL mean")


def assert_unchanged(model, initial):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Unexpected gradients")
    state = model.state_dict()
    if set(state) != set(initial) or any(not torch.equal(state[k].cpu(), v) for k, v in initial.items()):
        raise ValueError("Model state changed")


def load_model(frozen):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cp = torch.load(ROOT / frozen["generation"]["checkpoint"], map_location="cpu", weights_only=True)
    if cp["model_config"] != frozen["config"]:
        raise ValueError("Checkpoint model config mismatch")
    model = TinyDecoderLM.from_config(cp["model_config"])
    model.load_state_dict(cp["model_state_dict"], strict=True)
    tok, _ = load_frozen(ROOT / cp["model_config"]["tokenizer_artifact"])
    return model, tok, cp["model_state_dict"]


def check_controls(jobs, results, kind):
    for job in jobs:
        if not job["control_of"]:
            continue
        a, b = results[job["id"]], results[job["control_of"]]
        if kind == "generations":
            keys = ("response", "continuation", "generated_token_ids", "stop_reason")
            okay = all(a[k] == b[k] for k in keys)
        else:
            okay = all(a[k] == b[k] for k in ("target_ids", "token_count", "input_length"))
            okay = okay and len(a["per_token_nll"]) == len(b["per_token_nll"])
            okay = okay and all(abs(x - y) <= 1e-6 for x, y in zip(a["per_token_nll"], b["per_token_nll"]))
        if not okay:
            raise ValueError("Repeat control failed: " + job["id"])


def validate_reviews(rows, jobs, results):
    if len(rows) != len(jobs) or len({r["id"] for r in rows}) != len(jobs):
        raise ValueError("Incomplete/duplicate semantic review")
    by_id = {j["id"]: j for j in jobs}
    labels = {"matching", "conflicting", "mixed", "unavailable"}
    for row in rows:
        if row["id"] not in by_id or row["first_action"] not in labels or row["full_task"] not in labels:
            raise ValueError("Wrong review ID/label")
        for key in ("reason", "help", "risk", "entities", "repetition", "fluency"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError("Missing semantic explanation")
        text = results[row["id"]]["continuation"]
        for key, label in (("first_action_quote", "first_action"), ("full_quote", "full_task")):
            quote = row[key]
            if not isinstance(quote, str) or (quote and quote not in text) or (not quote and row[label] != "unavailable"):
                raise ValueError("Review evidence not in continuation")
    index = {r["id"]: r for r in rows}
    for job in jobs:
        if job["control_of"]:
            a, b = index[job["id"]], index[job["control_of"]]
            if any(a[k] != b[k] for k in a if k != "id"):
                raise ValueError("Inconsistent control semantic review")


def freeze_reviews(out=OUT):
    out = Path(out)
    frozen = verify(out)
    verify_hashes(read(out / "generation-freeze.json")["sha256"])
    jobs = frozen["jobs"]["generations"]
    results = {j["id"]: read(out / "generations" / (j["id"] + ".json"))["payload"]["result"] for j in jobs}
    validate_reviews(read(out / "semantic-reviews.json"), jobs, results)
    save_once(out / "review-freeze.json", dict(
        stage="semantic_judgments_before_candidate_scoring", reviewer="AI; not independent blind human review",
        sha256={(out / "semantic-reviews.json").relative_to(ROOT).as_posix(): digest(out / "semantic-reviews.json"),
                (out / "generation-freeze.json").relative_to(ROOT).as_posix(): digest(out / "generation-freeze.json")}))
    print("26 semantic judgments frozen before candidate scoring.", flush=True)


def run_stage(kind, out=OUT):
    out = Path(out)
    frozen = verify(out)
    done = out / ("generation-freeze.json" if kind == "generations" else "score-freeze.json")
    if kind == "scores":
        verify_hashes(read(out / "generation-freeze.json")["sha256"])
        verify_hashes(read(out / "review-freeze.json")["sha256"])
    if done.exists():
        verify_hashes(read(done)["sha256"])
        print("Completed stage verified; no new inference.", flush=True)
        return
    jobs = frozen["jobs"][kind]
    expected_paths = {j["id"] + ".json" for j in jobs}
    if (out / kind).exists() and any(p.name not in expected_paths and not p.name.startswith(".partial-")
                                  for p in (out / kind).glob("*.json")):
        raise ValueError("Unexpected record in stage")
    cfg = frozen["config"]
    tok, _ = load_frozen(ROOT / cfg["tokenizer_artifact"])
    freeze_hash = digest(out / "freeze.json")
    results = {}
    validate = ((lambda j, v: validate_generation(j, v, tok, cfg)) if kind == "generations" else validate_score)
    # Validate every existing record before loading weights or filling any missing record.
    for job in jobs:
        if (out / kind / (job["id"] + ".json")).exists():
            record(out, kind, job, freeze_hash, None, validate)
    model, tok, initial = load_model(frozen)
    for i, job in enumerate(jobs, 1):
        if kind == "generations":
            compute = lambda: prefix_reply(model, tok, cfg, job["instruction"], job["prefix_text"], total_reply_budget=320)
        else:
            compute = lambda: branch_nll(model, tok, cfg, job["instruction"], job["prefix_ids"], job["candidate"])
        results[job["id"]] = record(out, kind, job, freeze_hash, compute, validate)
        print(f"{kind} {i}/{len(jobs)} {job['id']}", flush=True)
    try:
        check_controls(jobs, results, kind)
        assert_unchanged(model, initial)
        verify_hashes(frozen["dependencies_sha256"])
    except ValueError as exc:
        save_once(out / "invalidity.json", dict(stage=kind, reason=str(exc)))
        raise
    paths = [out / kind / (j["id"] + ".json") for j in jobs] + [out / "freeze.json"]
    status = dict(stage=kind, records=len(jobs), repeated_controls=2 if kind == "generations" else 4,
                  controls_pass=True, model_state_unchanged=True, gradients_none=True, optimizer_updates=0)
    if kind == "generations":
        status["stops"] = dict(Counter(v["stop_reason"] for v in results.values()))
        status["semantic_review"] = "pending"
    save_once(out / (kind + "-status.json"), status)
    paths.append(out / (kind + "-status.json"))
    if kind == "scores":
        paths += [out / "review-freeze.json", out / "semantic-reviews.json", out / "generation-freeze.json"]
    save_once(done, dict(sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in paths}))


def classify(mean, total):
    values = mean + total
    if len(mean) != 3 or len(total) != 3 or any(v is None or not math.isfinite(v) for v in values):
        return "unavailable"
    if all(abs(v) <= 1e-6 for v in values):
        return "flat"
    if all(v > 1e-6 for v in values):
        return "positive"
    if all(v < -1e-6 for v in values):
        return "negative"
    return "discordant"


def summarize(out=OUT):
    out = Path(out)
    frozen = verify(out)
    verify_hashes(read(out / "score-freeze.json")["sha256"])
    rows = []
    for case in load_protocol()["cases"]:
        for hi in range(3):
            contrasts = []
            for wi in range(3):
                values = {(qi, ci): read(out / "scores" / f"{case['id']}-h{hi}-w{wi}-q{qi}-c{ci}.json")["payload"]["result"]
                          for qi in range(2) for ci in range(2)}
                means = [values[(qi, 1)]["mean_nll"] - values[(qi, 0)]["mean_nll"] for qi in range(2)]
                totals = [values[(qi, 1)]["total_nll"] - values[(qi, 0)]["total_nll"] for qi in range(2)]
                contrasts.append(dict(wording=wi, margins_mean=means, margins_total=totals,
                                      D_mean=means[0] - means[1], D_total=totals[0] - totals[1]))
            rows.append(dict(pair_id=case["id"], history_index=hi, contrasts=contrasts,
                             classification=classify([c["D_mean"] for c in contrasts], [c["D_total"] for c in contrasts])))
    reviews = read(out / "semantic-reviews.json")
    valid_reviews = [r for r in reviews if not r["id"].endswith("-repeat")]
    result = dict(scope="candidate_sentence_sensitivity_not_understanding_or_forgetting", pairs=rows,
                  semantic_primary_n=24, semantic_first_action=dict(Counter(r["first_action"] for r in valid_reviews)),
                  semantic_full_task=dict(Counter(r["full_task"] for r in valid_reviews)), training=False,
                  checkpoint=frozen["generation"]["checkpoint"])
    save_once(out / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "generate", "freeze-review", "score", "summarize"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "freeze-review":
        freeze_reviews()
    elif args.stage == "summarize":
        summarize()
    else:
        run_stage("generations" if args.stage == "generate" else "scores")


if __name__ == "__main__":
    main()
