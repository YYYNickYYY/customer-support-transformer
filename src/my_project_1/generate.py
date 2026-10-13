"""Greedy autoregressive replies: prompt only, then feed back our own tokens."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import torch

from utils.device import get_device
from .dataset import CustomerSupportDataset
from .model import TinyDecoderLM
from .overfit import validate_tokenizer
from .tokenization import encode_parts, file_hash
from .train import _save_json


@torch.no_grad()
def generate_reply(model, tokenizer, model_config, instruction, *, max_new_tokens=320):
    """One unpadded prompt; greedy decoding without a KV cache.

    EOS counts toward max_new_tokens and the frozen full-sequence budget.
    Never truncate the question silently or insert a fabricated EOS.
    """
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    _, _, sequence = encode_parts(tokenizer, instruction, "")
    prompt = sequence[:-1]  # BOS USER instruction ASSISTANT; no reply, no EOS.
    budget = model_config["max_length"]
    if len(prompt) >= budget:
        raise ValueError("Prompt leaves no room for generation within max_length")
    device = next(model.parameters()).device
    eos = model_config["eos_token_id"]
    blocked = [model_config[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant")]
    generated = []
    reason = "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            ids = torch.tensor([prompt + generated], dtype=torch.long, device=device)
            logits = model(ids, attention_mask=torch.ones_like(ids))["logits"][0, -1].clone()
            if not logits.isfinite().all():
                raise ValueError("Non-finite generation logits")
            # These controls belong only in prompts, not a reply.
            logits[blocked] = float("-inf")
            token = int(logits.argmax())
            generated.append(token)
            if token == eos:
                reason = "eos"
                break
            if len(prompt) + len(generated) >= budget:
                reason = "context_limit"
                break
    finally:
        model.train(was_training)
    return {"response": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_token_ids": generated, "prompt_tokens": len(prompt),
            "generated_tokens": len(generated), "stop_reason": reason}


def run_generation(config, instruction=None):
    threads = config["cpu_threads"]
    if type(threads) is not int or threads <= 0:
        raise ValueError("cpu_threads must be a positive integer")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return _run_generation(config, instruction)
    finally:
        torch.set_num_threads(previous_threads)


def _run_generation(config, instruction):
    checkpoint_path = Path(config["checkpoint"])
    device = get_device(config["device"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = checkpoint["model_config"]
    dataset = CustomerSupportDataset("train", tokenizer_dir=model_config["tokenizer_artifact"])
    validate_tokenizer(model_config, dataset)
    model = TinyDecoderLM.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if instruction is None:
        metadata = json.loads((checkpoint_path.parent / "experiment.json").read_text(encoding="utf-8"))
        if metadata["model_config"] != model_config:
            raise ValueError("Checkpoint and experiment model configurations differ")
        if metadata["split_sha256"] != dataset.config["split_sha256"]["train.jsonl"]:
            raise ValueError("Experiment train split hash mismatch")
        rows_by_id = {row["id"]: row for row in dataset.rows}
        rows = [rows_by_id[sample_id] for sample_id in metadata["sample_ids"]]
    else:
        rows = [{"id": "custom", "instruction": instruction}]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex
    output_dir = checkpoint_path.parent / "generations" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    _save_json(output_dir / "config.json", {
        **config, "resolved_device": device, "decoding": "greedy",
        "checkpoint_sha256": file_hash(checkpoint_path), "model_config": model_config,
        "suppressed_tokens": ["<PAD>", "<BOS>", "<USER>", "<ASSISTANT>"],
        "instruction": instruction, "scope": "memorized train examples or custom prompt; no dev/test",
    })
    exact_matches = 0
    with (output_dir / "replies.jsonl").open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            result = generate_reply(model, dataset.tokenizer, model_config, row["instruction"],
                                    max_new_tokens=config["max_new_tokens"])
            record = {"id": row["id"], "instruction": row["instruction"], **result}
            if "response" in row:
                record["reference_response"] = row["response"]
                record["exact_match"] = result["response"] == row["response"]
                exact_matches += int(record["exact_match"])
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"\nExample {index}: {row['instruction']}", flush=True)
            print(f"Generated: {result['response']}", flush=True)
            if "response" in row:
                print(f"Reference: {row['response']}", flush=True)
            print(f"Stop: {result['stop_reason']}; generated tokens: {result['generated_tokens']}", flush=True)
    _save_json(output_dir / "summary.json", {"examples": len(rows),
        "exact_matches": exact_matches if instruction is None else None,
        "scope": "train memorization check, NOT generalization"})
    print(f"\nSaved replies: {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/generation.json"))
    parser.add_argument("--instruction", help="Optional new question instead of the memorized examples")
    args = parser.parse_args()
    run_generation(json.loads(args.config.read_text(encoding="utf-8")), args.instruction)


if __name__ == "__main__":
    main()
