"""Frozen 135-pair decoding comparison; baseline parity gates the candidate arm."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import sys
import time

import torch

from my_project_1.decoding_comparison import banned_fourgram_tokens, constrained_reply
from my_project_1.model import TinyDecoderLM
from my_project_1.tokenization import encode_parts, load_frozen

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "outputs/decoding-repetition-preregister-20261006-v1"
OUT = ROOT / "outputs/decoding-repetition-comparison-20261006-v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(payload):
    return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


def save(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise ValueError("Refuse conflicting overwrite: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def save_jsonl(path, records):
    text = "".join(canonical(r) + "\n" for r in records)
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("Refuse conflicting JSONL: " + str(path))
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)


def verify_preregistration():
    frozen = read(PRE / "freeze.json")
    verify_hashes(frozen["source_sha256"])
    plan = read(PRE / "plan.json")
    if plan["status"] != "preregistered_not_run" or (plan["control"], plan["candidate"]) != (0, 4):
        raise ValueError("Wrong preregistration")
    print("Frozen preregistration verified; no inference.", flush=True)
    return plan


def environment():
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "torch": str(torch.__version__), "tokenizers": importlib.metadata.version("tokenizers"),
            "numpy": importlib.metadata.version("numpy"), "device": "cpu", "cpu_threads": 2,
            "seed": 42, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def prepare():
    plan = verify_preregistration()
    prior = read(PRE / "freeze.json")
    dependencies = dict(prior["source_sha256"])
    paths = [PRE / "freeze.json", ROOT / "scripts/run_decoding_comparison.py",
             ROOT / "src/my_project_1/decoding_comparison.py",
             ROOT / "tests/test_decoding_comparison.py", ROOT / "tests/test_decoding_comparison_runner.py",
             ROOT / "tests/test_decoding_preregistration.py"]
    paths += list((ROOT / "configs").glob("*.json")) + list((ROOT / ".vscode").glob("*.json"))
    dependencies.update({p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    save(OUT / "execution-freeze.json", {"experiment_id": plan["experiment_id"],
         "status": "prepared_generation_not_complete", "environment": environment(),
         "dependencies_sha256": dependencies, "plan_sha256": digest(PRE / "plan.json"),
         "expected_ids": prior["ordered_ids"], "training_started": False})
    print("Execution code/runtime frozen before inference.", flush=True)


def matches_history(result, previous):
    return (result["response"] == previous["generated_response"] and
            all(result[k] == previous[k] for k in ("generated_token_ids", "generated_tokens", "stop_reason")))


def validate_trace(record, sample, arm, freeze_hash, tokenizer, cfg):
    if record["payload_sha256"] != payload_hash(record["payload"]):
        raise ValueError("Corrupt resumed record")
    item = record["payload"]
    if any(item[k] != sample[k] for k in ("id", "intent", "cohort", "instruction")):
        raise ValueError("Wrong resumed case")
    if item["arm"] != arm or item["execution_freeze_sha256"] != freeze_hash:
        raise ValueError("Wrong resumed arm/freeze")
    size = 0 if arm == "control" else 4
    prompt_tokens = len(encode_parts(tokenizer, sample["instruction"], "")[2]) - 1
    ids, steps = item["generated_token_ids"], item["steps"]
    if (not ids or len(ids) != item["generated_tokens"] or len(steps) != len(ids) or
            item["prompt_tokens"] != prompt_tokens or item["no_repeat_ngram_size"] != size or
            item["response"] != tokenizer.decode(ids, skip_special_tokens=True)):
        raise ValueError("Incomplete resumed trace")
    roles = [cfg[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant")]
    eos = cfg["eos_token_id"]
    for i, (token, step) in enumerate(zip(ids, steps)):
        banned = banned_fourgram_tokens(ids[:i], eos) if size else []
        if (step["token_id"] != token or step["step"] != i + 1 or
                step["input_length"] != prompt_tokens + i or
                step["ngram_blocked_token_ids"] != banned or step["role_blocked_token_ids"] != roles or
                token in roles or token in banned or (token == eos and i != len(ids) - 1) or
                step["constraint_changed_choice"] != (token != step["baseline_greedy_token_id"])):
            raise ValueError("Invalid resumed decoding step")
    total = prompt_tokens + len(ids)
    reason = "eos" if ids[-1] == eos else "context_limit" if total == cfg["max_length"] else "max_new_tokens"
    if (total > cfg["max_length"] or len(ids) > 320 or item["stop_reason"] != reason or
            (reason == "max_new_tokens" and len(ids) != 320)):
        raise ValueError("Invalid resumed stop")
    return item


def check_unchanged_model(model, checkpoint_state):
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Unexpected model gradient")
    if any(not torch.equal(value, checkpoint_state[key]) for key, value in model.state_dict().items()):
        raise ValueError("Model state changed during decoding")


def run_arm(arm, samples, old, model, tokenizer, config, freeze_hash):
    results = []
    cfg = config["model_config"]
    for index, sample in enumerate(samples, 1):
        path = OUT / "traces" / arm / (sample["id"].rsplit(":", 1)[-1] + ".json")
        if path.exists():
            item = validate_trace(read(path), sample, arm, freeze_hash, tokenizer, cfg)
        else:
            start = time.perf_counter()
            result = constrained_reply(model, tokenizer, cfg, sample["instruction"],
                                       no_repeat_ngram_size=0 if arm == "control" else 4,
                                       max_new_tokens=config["max_new_tokens"])
            item = {**sample, **result, "arm": arm, "execution_freeze_sha256": freeze_hash,
                    "inference_seconds": time.perf_counter() - start}
            envelope = {"payload": item, "payload_sha256": payload_hash(item)}
            validate_trace(envelope, sample, arm, freeze_hash, tokenizer, cfg)
            if arm == "control" and not matches_history(item, old[sample["id"]]):
                save(OUT / "failures" / path.name, {"case": sample, "result": item,
                                                     "historical": old[sample["id"]]})
                raise ValueError("Baseline replay differs; candidate arm is blocked")
            save(path, envelope)
        if arm == "control" and not matches_history(item, old[sample["id"]]):
            raise ValueError("Resumed baseline differs; candidate arm is blocked")
        results.append(item)
        print(f"{arm} {index}/135: {sample['id'].rsplit(':', 1)[-1]} "
              f"{item['stop_reason']} {item['generated_tokens']} tokens; "
              f"choices changed={sum(s['constraint_changed_choice'] for s in item['steps'])}", flush=True)
    records = [{k: v for k, v in r.items() if k != "steps"} for r in results]
    save_jsonl(OUT / f"{arm}-replies.jsonl", records)
    return results


def placeholder_occurrences(text):
    return [{"text": m.group(), "start": m.start(), "end": m.end()}
            for m in re.finditer(r"\{\{[^{}]+\}\}", text)]


def finish(control, candidate, frozen):
    pairs = []
    for a, b in zip(control, candidate):
        prefix = 0
        for x, y in zip(a["generated_token_ids"], b["generated_token_ids"]):
            if x != y:
                break
            prefix += 1
        pairs.append({"id": a["id"], "intent": a["intent"], "cohort": a["cohort"],
                      "same_tokens": a["generated_token_ids"] == b["generated_token_ids"],
                      "same_text": a["response"] == b["response"],
                      "common_prefix_tokens": prefix,
                      "constraint_changed_steps": sum(s["constraint_changed_choice"] for s in b["steps"]),
                      "control_stop": a["stop_reason"], "candidate_stop": b["stop_reason"],
                      "control_tokens": a["generated_tokens"], "candidate_tokens": b["generated_tokens"],
                      "control_placeholders": placeholder_occurrences(a["response"]),
                      "candidate_placeholders": placeholder_occurrences(b["response"]),
                      "semantic_ratings": None, "entity_audit": None})
    save(OUT / "mechanism-pairs.json", pairs)
    shuffled = [{"id": r["id"], "arm": r["arm"], "record": r} for r in control + candidate]
    random.Random(42).shuffle(shuffled)
    review, key = [], []
    for index, item in enumerate(shuffled, 1):
        blind_id = f"D{index:04d}"
        r = item["record"]
        review.append({"blind_id": blind_id, "instruction": r["instruction"], "response": r["response"],
                       "customer_request": None, "intent_score": None, "helpfulness_score": None,
                       "fluency_score": None, "repetition": None, "unsupported_claim": None,
                       "entity_audit": None, "evidence": None, "reviewer": None})
        key.append({"blind_id": blind_id, "id": r["id"], "arm": item["arm"],
                    "intent": r["intent"], "cohort": r["cohort"], "stop_reason": r["stop_reason"]})
    save(OUT / "review-blinded.json", review)
    save(OUT / "review-key.json", key)
    summary = {"status": "generation_complete_evaluation_pending", "pairs": 135,
               "baseline_exact_replays": 135, "different_token_sequences": sum(not p["same_tokens"] for p in pairs),
               "different_texts": sum(not p["same_text"] for p in pairs),
               "candidate_constraint_changed_steps": sum(p["constraint_changed_steps"] for p in pairs),
               "stop_counts": {"control": dict(Counter(r["stop_reason"] for r in control)),
                               "candidate": dict(Counter(r["stop_reason"] for r in candidate))},
               "generation_seconds": {"control": sum(r["inference_seconds"] for r in control),
                                      "candidate": sum(r["inference_seconds"] for r in candidate)},
               "optimizer_updates": 0, "model_state_unchanged": True, "model_gradients_none": True,
               "semantic_scores": None, "entity_audit": None, "adoption_approved": False,
               "scope": "fixed dev diagnostic only; not sealed_test",
               "execution_freeze_sha256": digest(OUT / "execution-freeze.json")}
    save(OUT / "summary.json", summary)
    files = list((OUT / "traces").rglob("*.json")) + [OUT / name for name in
        ("control-replies.jsonl", "candidate-replies.jsonl", "mechanism-pairs.json",
         "review-blinded.json", "review-key.json", "summary.json")]
    save(OUT / "generation-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
         "dependencies_sha256": frozen["dependencies_sha256"], "status": summary["status"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def run():
    plan = verify_preregistration()
    frozen = read(OUT / "execution-freeze.json")
    verify_hashes(frozen["dependencies_sha256"])
    if environment() != frozen["environment"]:
        raise ValueError("Runtime changed after execution freeze")
    if (OUT / "generation-freeze.json").exists():
        verify_hashes(read(OUT / "generation-freeze.json")["sha256"])
        print("Completed generation artifacts verified; no new inference.", flush=True)
        return
    config = read(ROOT / plan["baseline_config"])
    selection = read(ROOT / plan["selection"])["samples"]
    previous = [json.loads(line) for line in (ROOT / plan["baseline_replies"]).read_text(encoding="utf-8").splitlines()]
    if [r["id"] for r in previous] != frozen["expected_ids"] or [s["id"] for s in selection] != frozen["expected_ids"]:
        raise ValueError("Wrong order or IDs")
    samples = [{**s, "instruction": r["instruction"]} for s, r in zip(selection, previous)]
    old = {r["id"]: r for r in previous}
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    checkpoint = torch.load(ROOT / config["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["model_config"] != config["model_config"]:
        raise ValueError("Wrong model config")
    cfg = config["model_config"]
    tokenizer, tc = load_frozen(ROOT / cfg["tokenizer_artifact"])
    if tc["tokenizer_sha256"] != cfg["tokenizer_revision"] or tc["vocab_size"] != cfg["vocab_size"]:
        raise ValueError("Wrong tokenizer")
    model = TinyDecoderLM.from_config(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    freeze_hash = digest(OUT / "execution-freeze.json")
    control = run_arm("control", samples, old, model, tokenizer, config, freeze_hash)
    check_unchanged_model(model, checkpoint["model_state_dict"])
    verify_hashes(frozen["dependencies_sha256"])
    save(OUT / "control-parity.json", {"exact_replays": 135, "candidate_gate_passed": True,
         "checked_fields": ["generated_token_ids", "response", "generated_tokens", "stop_reason"],
         "control_replies_sha256": digest(OUT / "control-replies.jsonl"), "execution_freeze_sha256": freeze_hash})
    print("GATE PASSED: all 135 baseline cases exactly match. Starting candidate.", flush=True)
    candidate = run_arm("candidate", samples, old, model, tokenizer, config, freeze_hash)
    check_unchanged_model(model, checkpoint["model_state_dict"])
    verify_hashes(frozen["dependencies_sha256"])
    finish(control, candidate, frozen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "prepare", "run"))
    args = parser.parse_args()
    {"verify": verify_preregistration, "prepare": prepare, "run": run}[args.stage]()
