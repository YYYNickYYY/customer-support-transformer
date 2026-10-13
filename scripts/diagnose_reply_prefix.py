"""Preregister, parity-gate, probe all trajectories and generate controlled prefixes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import torch
import run_decoding_comparison as r
from my_project_1.model import TinyDecoderLM
from my_project_1.tokenization import load_frozen
from my_project_1.reply_prefix import (NEUTRAL, TASK_PREFIXES, BRANCHES, STAGES,
                                      branch_nll, prefix_reply, trajectory_prefixes)

ROOT = r.ROOT
PRIOR = ROOT / "outputs/condition-use-20261006-v1"
OUT = ROOT / "outputs/reply-prefix-20261006-v1"


def prepare():
    if OUT.exists():
        raise FileExistsError("Already registered; no conflicting overwrite")
    prior = r.read(PRIOR / "plan.json")
    for name in ("freeze.json", "generation-freeze.json", "assessment-freeze.json"):
        r.verify_hashes(r.read(PRIOR / name)["sha256"])
    originals = [v for v in prior["cases"] if v["variant"] == "original"]
    cases = [dict(v, id=v["id"] + "-empty", original_id=v["id"], prefix_condition="empty", prefix_text="") for v in originals]
    for original in originals:
        for condition, prefix in (("neutral", NEUTRAL), ("task", TASK_PREFIXES[original["family"]])):
            cases.append(dict(original, id=original["id"] + "-" + condition,
                              original_id=original["id"], prefix_condition=condition, prefix_text=prefix))
    cases.append(dict(cases[8], id="identical-neutral-control"))
    plan = {"version": "reply-prefix-v1", "protocol": "docs/reply-prefix-protocol-v1.md",
            "generation": prior["generation"], "total_reply_budget": 320, "stages": list(STAGES),
            "branches": BRANCHES, "task_prefixes": TASK_PREFIXES, "neutral": NEUTRAL,
            "cases": cases, "pairs": prior["pairs"], "originals": originals,
            "training": False, "no_repeat_ngram_size": 0}
    OUT.mkdir(parents=True)
    r.save(OUT / "plan.json", plan)
    paths = [OUT / "plan.json", ROOT / plan["protocol"], Path(__file__), ROOT / "scripts/run_decoding_comparison.py",
             ROOT / "tests/test_reply_prefix.py", ROOT / plan["generation"]["checkpoint"]]
    paths += [PRIOR / name for name in ("plan.json", "replies.jsonl", "freeze.json", "generation-freeze.json", "assessment-freeze.json")]
    paths += list((ROOT / "src/my_project_1").glob("*.py"))
    paths += list((ROOT / plan["generation"]["model_config"]["tokenizer_artifact"]).glob("*.json"))
    paths += list((ROOT / "configs").glob("*.json")) + list((ROOT / "outputs").rglob("*.xlsx"))
    r.save(OUT / "freeze.json", {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "before_inference", "environment": r.environment(),
        "sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})
    print("Registered 8 sources; 25 generations; 4 stages per trajectory, no training.")


def cached(path, identity, calculate):
    if path.exists():
        saved = r.read(path)
        row = saved["payload"]
        if saved["sha256"] != r.payload_hash(row) or any(row[k] != v for k, v in identity.items()):
            raise ValueError("Corrupt/conflicting partial result: " + str(path))
        return row
    row = {**identity, **calculate()}
    r.save(path, {"payload": row, "sha256": r.payload_hash(row)})
    return row


def run():
    frozen = r.read(OUT / "freeze.json")
    r.verify_hashes(frozen["sha256"])
    if frozen["environment"] != r.environment():
        raise ValueError("Changed runtime")
    if (OUT / "execution-freeze.json").exists():
        r.verify_hashes(r.read(OUT / "execution-freeze.json")["sha256"])
        print("Completed experiment verified; no new inference.")
        return
    plan = r.read(OUT / "plan.json")
    cfg = plan["generation"]["model_config"]
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cp = torch.load(ROOT / plan["generation"]["checkpoint"], map_location="cpu", weights_only=True)
    assert cp["model_config"] == cfg
    tokenizer, tc = load_frozen(ROOT / cfg["tokenizer_artifact"])
    assert tc["tokenizer_sha256"] == cfg["tokenizer_revision"]
    model = TinyDecoderLM.from_config(cfg)
    model.load_state_dict(cp["model_state_dict"], strict=True)
    history = {v["id"]: v for v in [json.loads(line) for line in (PRIOR / "replies.jsonl").read_text(encoding="utf-8").splitlines()]}
    freeze_hash = r.digest(OUT / "freeze.json")
    rows = []
    # Gate all eight baselines before any probe or intervention.
    for case in plan["cases"][:8]:
        row = cached(OUT / "traces" / (case["id"] + ".json"), {**case, "freeze_sha256": freeze_hash},
                     lambda case=case: prefix_reply(model, tokenizer, cfg, case["instruction"], "", 320))
        previous = history[case["original_id"]]
        if any(row[k] != previous[k] for k in ("response", "generated_token_ids", "stop_reason")):
            r.save(OUT / "invalidity.json", {"baseline_parity": False, "case": case["id"]})
            raise ValueError("Baseline parity failure")
        rows.append(row)
        print("Parity", case["id"], "passed", flush=True)
    r.save(OUT / "parity.json", {"baseline_parity": True, "matched": 8, "freeze_sha256": freeze_hash})
    by_id = {v["id"]: v for v in plan["originals"]}
    probes = []
    for pair in plan["pairs"]:
        a, b = [by_id[s] for s in pair]
        for source in (a, b):
            stages = trajectory_prefixes(tokenizer, history[source["id"]]["generated_token_ids"], cfg["eos_token_id"])
            for stage in stages:
                identity = {"source_original_id": source["id"], "family": source["family"],
                            "pair": pair, **stage, "freeze_sha256": freeze_hash}
                path = OUT / "probes" / f"{source['id']}-{stage['stage']}.json"
                def measure():
                    if not stage["available"]:
                        return {"question_scores": None, "condition_separation": None, "own_correct_margin": None}
                    scores = {}
                    for question in (a, b):
                        branches = {candidate["family"]: branch_nll(model, tokenizer, cfg,
                            question["instruction"], stage["prefix_ids"], BRANCHES[candidate["family"]]) for candidate in (a, b)}
                        scores[question["id"]] = {"branches": branches,
                            "margin_A_over_B": branches[b["family"]]["mean_nll"] - branches[a["family"]]["mean_nll"]}
                    own = scores[source["id"]]["margin_A_over_B"] * (1 if source["id"] == a["id"] else -1)
                    separation = scores[a["id"]]["margin_A_over_B"] - scores[b["id"]]["margin_A_over_B"]
                    return {"question_scores": scores, "condition_separation": separation, "own_correct_margin": own}
                probe = cached(path, identity, measure)
                probes.append(probe)
            print("Probed", source["id"], "4 registered stages", flush=True)
    for i, case in enumerate(plan["cases"][8:], 9):
        row = cached(OUT / "traces" / (case["id"] + ".json"), {**case, "freeze_sha256": freeze_hash},
                     lambda case=case: prefix_reply(model, tokenizer, cfg, case["instruction"], case["prefix_text"], 320))
        rows.append(row)
        print(f"{i}/25 {case['id']} {row['stop_reason']} {row['generated_tokens']} total tokens", flush=True)
    control = all(rows[8][k] == rows[-1][k] for k in ("response", "generated_token_ids", "stop_reason"))
    if not control:
        r.save(OUT / "invalidity.json", {"identical_neutral_control": False})
        raise ValueError("Repeated neutral control failed")
    r.check_unchanged_model(model, cp["model_state_dict"])
    r.verify_hashes(frozen["sha256"])
    r.save_jsonl(OUT / "replies.jsonl", [{k: v for k, v in row.items() if k != "steps"} for row in rows])
    r.save(OUT / "trajectory-probes.json", probes)
    r.save(OUT / "execution-summary.json", {"status": "generated_assessment_pending", "generations": 25,
        "baseline_parity": 8, "identical_neutral_control": control, "probes": len(probes),
        "available_probes": sum(v["available"] for v in probes), "optimizer_updates": 0,
        "model_state_unchanged": True, "model_gradients_none": True})
    paths = list((OUT / "traces").glob("*.json")) + list((OUT / "probes").glob("*.json"))
    paths += [OUT / name for name in ("replies.jsonl", "trajectory-probes.json", "execution-summary.json", "parity.json", "freeze.json")]
    r.save(OUT / "execution-freeze.json", {"sha256": {p.relative_to(ROOT).as_posix(): r.digest(p) for p in paths}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage]()
