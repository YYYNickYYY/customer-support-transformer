"""Frozen intent-stratified dev generation and blank human rating records."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import torch

from utils.device import get_device
from .dataset import CustomerSupportDataset
from .generate import generate_reply
from .model import TinyDecoderLM
from .overfit import validate_tokenizer
from .tokenization import file_hash
from .train import _save_json


RATING_FIELDS = ["intent_score_0_2", "helpfulness_score_0_2", "fluency_score_0_2",
                 "repetition_0_1", "unsupported_claim_0_1", "error_tags", "notes", "reviewer"]


def choose_rows(rows, seed=42):
    """One row per actual dev intent, chosen by SHA256(seed:id), not outputs."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    groups = {}
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("Duplicate dev sample ID")
        seen.add(row["id"])
        groups.setdefault(row["intent"], []).append(row)
    if not groups:
        raise ValueError("Empty dev selection")
    return [min(groups[intent], key=lambda row: (
        hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]))
        for intent in sorted(groups)]


def frozen_selection(dataset, directory, seed, *, create=False):
    directory = Path(directory)
    selection_path = directory / "selection.json"
    payload = {"version": "dev-check-v1", "split": "dev", "seed": seed,
        "dev_sha256": dataset.config["split_sha256"]["dev.jsonl"],
        "method": "one per sorted intent; minimum SHA256(seed:id); no response/model filtering",
        "samples": [{key: row[key] for key in ("id", "intent", "category", "template_id")}
                    for row in choose_rows(dataset.rows, seed)]}
    if create and not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        _save_json(selection_path, payload)
        _save_json(directory / "freeze.json", {"file_sha256": {
            "selection.json": file_hash(selection_path)}})
    manifest = json.loads((directory / "freeze.json").read_text(encoding="utf-8"))
    if file_hash(selection_path) != manifest["file_sha256"]["selection.json"]:
        raise ValueError("Frozen dev selection hash mismatch")
    saved = json.loads(selection_path.read_text(encoding="utf-8"))
    if saved != payload:
        raise ValueError("Frozen dev selection or seed does not match current dev")
    by_id = {row["id"]: row for row in dataset.rows}
    return [by_id[item["id"]] for item in saved["samples"]]


def rating_record(row, result):
    return {"id": row["id"], "category": row["category"], "intent": row["intent"],
        "instruction": row["instruction"], "reference_response": row["response"],
        "generated_response": result["response"], "stop_reason": result["stop_reason"],
        "generated_tokens": result["generated_tokens"], **{key: "" for key in RATING_FIELDS}}


def csv_safe(value):
    # CSV opened in Excel must not execute question/response text as a formula.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def run_dev_check(config, *, freeze=False):
    if type(config["cpu_threads"]) is not int or config["cpu_threads"] <= 0:
        raise ValueError("cpu_threads must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config["cpu_threads"])
    try:
        return _run_dev_check(config, freeze)
    finally:
        torch.set_num_threads(previous_threads)


def _run_dev_check(config, freeze):
    checkpoint_path = Path(config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    dataset = CustomerSupportDataset("dev", tokenizer_dir=model_config["tokenizer_artifact"])
    validate_tokenizer(model_config, dataset)
    rows = frozen_selection(dataset, config["selection_dir"], config["seed"], create=freeze)
    device = get_device(config["device"])
    model = TinyDecoderLM.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
    directory = checkpoint_path.parent / "dev-checks" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    _save_json(directory / "config.json", {**config, "model_config": model_config,
        "checkpoint_sha256": file_hash(checkpoint_path),
        "selection_sha256": file_hash(Path(config["selection_dir"]) / "selection.json"),
        "decoding": "greedy; block PAD/BOS/USER/ASSISTANT", "resolved_device": device,
        "scope": "intent-balanced dev diagnostic; NOT sealed_test or population estimate"})
    fields = list(rating_record(rows[0], {"response": "", "stop_reason": "", "generated_tokens": 0}))
    stops = Counter()
    exact = 0
    print(f"Fixed dev check: {len(rows)} examples; output: {directory}", flush=True)
    with (directory / "replies.jsonl").open("x", encoding="utf-8") as results, \
         (directory / "human-ratings.csv").open("x", encoding="utf-8-sig", newline="") as ratings, \
         (directory / "review.md").open("x", encoding="utf-8") as review:
        writer = csv.DictWriter(ratings, fieldnames=fields)
        writer.writeheader()
        review.write("# 固定 dev 生成检查\n\n人工评分另填 human-ratings.csv。参考回复不是唯一正确答案；业务事实无法核实时标记需核实。\n\n")
        for index, row in enumerate(rows, 1):
            result = generate_reply(model, dataset.tokenizer, model_config, row["instruction"],
                                    max_new_tokens=config["max_new_tokens"])
            record = rating_record(row, result)
            results.write(json.dumps({**record, "generated_token_ids": result["generated_token_ids"]},
                                     ensure_ascii=False) + "\n")
            results.flush()
            writer.writerow({key: csv_safe(value) for key, value in record.items()})
            ratings.flush()
            stops[result["stop_reason"]] += 1
            exact += result["response"] == row["response"]
            review.write(f"## {index}. {row['intent']}\n\nID: {row['id']}\n\n问题：{row['instruction']}\n\n"
                         f"生成回复：\n\n{result['response']}\n\n参考回复：\n\n{row['response']}\n\n"
                         f"停止：{result['stop_reason']}；生成 token（含 EOS）：{result['generated_tokens']}\n\n")
            review.flush()
            print(f"{index}/{len(rows)} {row['intent']}: {result['stop_reason']}, "
                  f"{result['generated_tokens']} tokens", flush=True)
    _save_json(directory / "summary.json", {"examples": len(rows),
        "intents": len({row["intent"] for row in rows}),
        "categories": len({row["category"] for row in rows}), "stop_counts": dict(stops),
        "exact_matches": exact, "human_ratings_completed": 0,
        "warning": "Stop/exact match counts are not quality scores; human fields left blank"})
    print(f"Saved replies and blank human rating table: {directory}", flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dev-check.json"))
    parser.add_argument("--freeze", action="store_true", help="Create v1 selection once; never overwrite")
    args = parser.parse_args()
    run_dev_check(json.loads(args.config.read_text(encoding="utf-8")), freeze=args.freeze)


if __name__ == "__main__":
    main()
