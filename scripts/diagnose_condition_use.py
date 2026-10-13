"""Frozen supported-task prompts, original greedy inference and paired target NLL."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import run_decoding_comparison as r
from my_project_1.condition_use import build_plan, condition_nll
from my_project_1.generation_diagnostic import trace_reply
from my_project_1.model import TinyDecoderLM
from my_project_1.tokenization import load_frozen

ROOT = r.ROOT
OUT = ROOT / "outputs/condition-use-20261006-v1"
AUDIT = ROOT / "outputs/train-input-target-audit-20261006-v1"
SENSITIVITY = ROOT / "outputs/input-sensitivity-20261006-v1"


def prepare():
    if OUT.exists():
        raise FileExistsError("Already registered; verify or run, never overwrite plan")
    audit = r.read(AUDIT / "preregistration-freeze.json")
    r.verify_hashes(audit["files_sha256"])
    for freeze in ("scan-freeze.json", "review-freeze.json"):
        r.verify_hashes({(AUDIT / name).relative_to(ROOT).as_posix(): expected
                         for name, expected in r.read(AUDIT / freeze)["files_sha256"].items()})
    prior = r.read(SENSITIVITY / "freeze.json")
    r.verify_hashes(prior["sha256"])
    cfg = prior["generation"]
    assert "20261005T132346-full-ebf624ab40734b858b46553b4da37537" in cfg["checkpoint"]
    plan = {"protocol": "docs/condition-use-protocol-v1.md",
            "generation": cfg, "version": "condition-use-v1", "training": False,
            "no_repeat_ngram_size": 0, "max_new_tokens": 320,
            **build_plan(r.read(AUDIT / "samples.json"), r.read(AUDIT / "reviews.json"))}
    OUT.mkdir(parents=True)
    r.save(OUT / "plan.json", plan)
    paths = [OUT / "plan.json", ROOT / plan["protocol"], Path(__file__),
             ROOT / "scripts/run_decoding_comparison.py", ROOT / cfg["checkpoint"],
             AUDIT / "plan.json", AUDIT / "samples.json", AUDIT / "reviews.json",
             AUDIT / "preregistration-freeze.json", AUDIT / "scan-freeze.json", AUDIT / "review-freeze.json",
             SENSITIVITY / "freeze.json", SENSITIVITY / "assessment-freeze.json"]
    paths += list((ROOT / "src/my_project_1").glob("*.py"))
    paths += list((ROOT / cfg["model_config"]["tokenizer_artifact"]).glob("*.json"))
    paths += list((ROOT / "configs").glob("*.json"))
    paths += [ROOT / "tests/test_condition_use.py"]
    paths += list((ROOT / "outputs").rglob("*.xlsx"))
    r.save(OUT / "freeze.json", {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "before_inference", "environment": r.environment(),
        "sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})
    print("Registered 8 sources / 17 generations / 16 fixed-target evaluations before inference.")
    for row in plan["source_rows"]:
        print(row["family"], row["id"], row["instruction"])


def run():
    frozen = r.read(OUT / "freeze.json")
    r.verify_hashes(frozen["sha256"])
    if frozen["environment"] != r.environment():
        raise ValueError("Changed runtime")
    if (OUT / "generation-freeze.json").exists():
        r.verify_hashes(r.read(OUT / "generation-freeze.json")["sha256"])
        print("Existing complete generation verified; no new inference.")
        return
    plan = r.read(OUT / "plan.json")
    cfg = plan["generation"]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cp = torch.load(ROOT / cfg["checkpoint"], map_location="cpu", weights_only=True)
    assert cp["model_config"] == cfg["model_config"]
    model = TinyDecoderLM.from_config(cp["model_config"])
    model.load_state_dict(cp["model_state_dict"], strict=True)
    tokenizer, tc = load_frozen(ROOT / cp["model_config"]["tokenizer_artifact"])
    assert tc["tokenizer_sha256"] == cp["model_config"]["tokenizer_revision"]
    freeze_hash = r.digest(OUT / "freeze.json")
    generated = []
    for i, case in enumerate(plan["cases"], 1):
        path = OUT / "traces" / (case["id"] + ".json")
        if path.exists():
            saved = r.read(path)
            row = saved["payload"]
            if saved["sha256"] != r.payload_hash(row) or row["freeze_sha256"] != freeze_hash:
                raise ValueError("Corrupt or conflicting partial trace")
            if any(row[k] != v for k, v in case.items()):
                raise ValueError("Wrong resumed case")
        else:
            result = trace_reply(model, tokenizer, cp["model_config"], case["instruction"], max_new_tokens=320)
            row = {**case, **result, "freeze_sha256": freeze_hash}
            r.save(path, {"payload": row, "sha256": r.payload_hash(row)})
        assert len(row["steps"]) == len(row["generated_token_ids"]) == row["generated_tokens"]
        assert tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) == row["response"]
        generated.append(row)
        print(f"{i}/17 {case['id']} {row['stop_reason']} {row['generated_tokens']} tokens", flush=True)
    exact = all(generated[0][k] == generated[-1][k] for k in ("response", "generated_token_ids", "stop_reason"))
    if not exact:
        r.save(OUT / "invalidity.json", {"identical_input_control": False})
        raise ValueError("Control failed; batch invalid")
    cases = {c["id"]: c for c in plan["cases"]}
    sources = {s["id"]: s for s in plan["source_rows"]}
    losses = []
    for pair in plan["pairs"]:
        for a, b in (pair, list(reversed(pair))):
            own, other = cases[a], cases[b]
            source = sources[own["source_id"]]
            result = condition_nll(model, tokenizer, own["instruction"], other["instruction"],
                                   source["response"], cp["model_config"]["max_length"])
            losses.append({"case_id": a, "swapped_case_id": b, "source_id": source["id"], **result})
            print(f"NLL {a}: matched={result['matched_nll']:.6f} swapped={result['swapped_nll']:.6f}", flush=True)
    r.check_unchanged_model(model, cp["model_state_dict"])
    r.verify_hashes(frozen["sha256"])
    r.save_jsonl(OUT / "replies.jsonl", [{k: v for k, v in row.items() if k != "steps"} for row in generated])
    r.save(OUT / "target-nll.json", losses)
    r.save(OUT / "generation-summary.json", {"status": "generated_assessment_pending",
        "cases": 17, "source_cases": 8, "target_comparisons": 8, "target_forwards": 16,
        "identical_input_control": exact, "stops": dict(Counter(v["stop_reason"] for v in generated)),
        "model_state_unchanged": True, "model_gradients_none": True, "optimizer_updates": 0,
        "nll_interpretation": "same-target common-prefix NLL, not autoregressive task accuracy"})
    paths = list((OUT / "traces").glob("*.json")) + [OUT / name for name in
        ("replies.jsonl", "target-nll.json", "generation-summary.json", "freeze.json")]
    r.save(OUT / "generation-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage]()
