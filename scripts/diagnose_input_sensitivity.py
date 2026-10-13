"""Freeze stimuli before one-pass inference with the original fifth greedy model."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from my_project_1.generation_diagnostic import trace_reply
from my_project_1.model import TinyDecoderLM
from my_project_1.tokenization import load_frozen

spec = importlib.util.spec_from_file_location("sensitivity_runner", Path(__file__).with_name("run_decoding_comparison.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
ROOT = r.ROOT
OUT = ROOT / "outputs/input-sensitivity-20261006-v1"


def stimuli(plan):
    rows = [{**case, "group": group["id"], "factor": group["factor"]}
            for group in plan["groups"] for case in group["cases"]]
    if len(rows) != 22 or len({v["id"] for v in rows}) != 22 or len(plan["groups"]) != 9:
        raise ValueError("Wrong frozen diagnostic scope")
    return rows


def prepare():
    plan = r.read(OUT / "plan.json")
    rows = stimuli(plan)
    cfg = r.read(ROOT / plan["generation_config"])
    closure = ROOT / "outputs/decoding-review-closure-20261006-v1/calibrated/freeze.json"
    frozen_review = r.read(closure)
    r.verify_hashes(frozen_review["sources_sha256"])
    r.verify_hashes(frozen_review["files_sha256"])
    paths = [OUT / "plan.json", ROOT / plan["protocol"], Path(__file__).resolve(),
             ROOT / "tests/test_input_sensitivity.py", ROOT / plan["generation_config"],
             ROOT / cfg["checkpoint"], closure,
             ROOT / "scripts/run_decoding_comparison.py"]
    paths += list((ROOT / "src/my_project_1").glob("*.py"))
    paths += list((ROOT / cfg["model_config"]["tokenizer_artifact"]).glob("*.json"))
    r.save(OUT / "freeze.json", {"plan": plan, "cases": rows, "generation": cfg,
        "environment": r.environment(), "training": False, "no_repeat_ngram_size": 0,
        "sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})
    print("22 stimuli in 9 groups frozen before inference; no training.")


def run():
    frozen = r.read(OUT / "freeze.json")
    r.verify_hashes(frozen["sha256"])
    if frozen["environment"] != r.environment():
        raise ValueError("Changed frozen runtime")
    if (OUT / "generation-freeze.json").exists():
        r.verify_hashes(r.read(OUT / "generation-freeze.json")["sha256"])
        print("Complete diagnostic verified; no new inference.")
        return
    cfg = frozen["generation"]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cp = torch.load(ROOT / cfg["checkpoint"], map_location="cpu", weights_only=True)
    if cp["model_config"] != cfg["model_config"]:
        raise ValueError("Wrong model config")
    tokenizer, tc = load_frozen(ROOT / cp["model_config"]["tokenizer_artifact"])
    if tc["tokenizer_sha256"] != cp["model_config"]["tokenizer_revision"]:
        raise ValueError("Wrong tokenizer")
    model = TinyDecoderLM.from_config(cp["model_config"])
    model.load_state_dict(cp["model_state_dict"], strict=True)
    rows = []
    freeze_hash = r.digest(OUT / "freeze.json")
    for i, case in enumerate(frozen["cases"], 1):
        path = OUT / "traces" / (case["id"] + ".json")
        if path.exists():
            item = r.read(path)
            if item["sha256"] != r.payload_hash(item["payload"]):
                raise ValueError("Corrupt partial trace")
            row = item["payload"]
            if any(row[k] != v for k, v in case.items()) or row["freeze_sha256"] != freeze_hash:
                raise ValueError("Conflicting resumed case")
        else:
            result = trace_reply(model, tokenizer, cp["model_config"], case["instruction"], max_new_tokens=320)
            row = {**case, **result, "freeze_sha256": freeze_hash}
            r.save(path, {"payload": row, "sha256": r.payload_hash(row)})
        if len(row["steps"]) != row["generated_tokens"] or len(row["generated_token_ids"]) != row["generated_tokens"]:
            raise ValueError("Incomplete trace")
        if tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) != row["response"]:
            raise ValueError("Trace text differs from IDs")
        rows.append(row)
        print(f"{i}/22 {case['id']} {row['stop_reason']} {row['generated_tokens']} tokens", flush=True)
    control = [v for v in rows if v["group"] == "G09"]
    exact_control = all(control[0][k] == control[1][k] for k in ("response", "generated_token_ids", "stop_reason"))
    if not exact_control:
        r.save(OUT / "invalidity.json", {"identical_input_control": False})
        raise ValueError("Identical-input control failed; diagnostic invalid")
    r.check_unchanged_model(model, cp["model_state_dict"])
    r.verify_hashes(frozen["sha256"])
    r.save_jsonl(OUT / "replies.jsonl", [{k: v for k, v in row.items() if k != "steps"} for row in rows])
    r.save(OUT / "generation-summary.json", {"status": "generated_assessment_pending", "cases": 22,
        "groups": 9, "identical_input_control": exact_control, "optimizer_updates": 0,
        "model_state_unchanged": True, "model_gradients_none": True, "semantic_assessment": None})
    files = list((OUT / "traces").glob("*.json")) + [OUT / "replies.jsonl", OUT / "generation-summary.json", OUT / "freeze.json"]
    r.save(OUT / "generation-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in files}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage]()
