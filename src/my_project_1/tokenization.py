"""Train-only byte-level BPE and immutable, versioned sequence statistics."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<USER>", "<ASSISTANT>", "<EOS>"]
TEMPLATE = "<BOS><USER>{instruction}<ASSISTANT>{response}<EOS>"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            for field in ("instruction", "response"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Empty or non-string {field}")
                if any(special in value for special in SPECIAL_TOKENS):
                    raise ValueError(f"Reserved special token in {field}")
            yield row


def encode_parts(tokenizer, instruction, response):
    """Encode fields separately, then insert control IDs without BPE boundary merging."""
    for text in (instruction, response):
        if any(special in text for special in SPECIAL_TOKENS):
            raise ValueError("Reserved special token in sample text")
    instruction_ids = tokenizer.encode(instruction, add_special_tokens=False).ids
    response_ids = tokenizer.encode(response, add_special_tokens=False).ids
    bos, user, assistant, eos = [tokenizer.token_to_id(s) for s in SPECIAL_TOKENS[1:]]
    return instruction_ids, response_ids, [bos, user, *instruction_ids, assistant, *response_ids, eos]


def prepare_example(tokenizer, instruction, response, max_length):
    """Right-truncate full sequence, then shift; never add a false EOS after truncation.

    max_length counts the FULL sequence, so shifted model inputs have at most
    max_length-1 tokens. Only response tokens and a retained EOS contribute loss.
    """
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    inst, resp, full = encode_parts(tokenizer, instruction, response)
    response_start = len(inst) + 3
    kept = full[:max_length]
    labels = [token if position >= response_start else -100
              for position, token in enumerate(kept)][1:]
    return {"input_ids": kept[:-1], "labels": labels,
            "truncated": len(full) > max_length,
            "supervised_tokens": sum(token != -100 for token in labels)}


def summary(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Empty split")
    def percentile(p):
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
    return {"min": ordered[0], "mean": round(mean(values), 3), "max": ordered[-1],
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)}


def profile(tokenizer, rows, candidates):
    lengths = {name: [] for name in ("instruction", "response", "full_sequence")}
    starts = []
    for row in rows:
        inst, resp, full = encode_parts(tokenizer, row["instruction"], row["response"])
        for name, value in zip(lengths, (len(inst), len(resp), len(full))):
            lengths[name].append(value)
        starts.append(len(inst) + 3)
    n = len(starts)
    distributions = {name: summary(values) for name, values in lengths.items()}
    truncation = {}
    for limit in candidates:
        counts = {name: sum(length > limit for length in values)
                  for name, values in lengths.items()}
        truncation[str(limit)] = {
            "over_limit": {name: {"count": count, "rate": count / n}
                           for name, count in counts.items()},
            "lost_sequence_tokens": sum(max(0, length-limit) for length in lengths["full_sequence"]),
            "lost_response_tokens": sum(min(length, max(0, start + length - limit))
                                        for start, length in zip(starts, lengths["response"])),
            "zero_supervised_samples": sum(start >= limit for start in starts),
        }
    return {"rows": n, "lengths": distributions, "truncation": truncation}


def build_artifact(split_dir, output, vocab_size=4000):
    split_dir, output = Path(split_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Frozen output already exists: {output}")
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))
    # Read only train/dev; sealed_test is never opened by this workflow.
    for name in ("train", "dev"):
        if file_hash(split_dir / f"{name}.jsonl") != manifest["file_sha256"][f"{name}.jsonl"]:
            raise ValueError(f"{name} hash mismatch")
    train = list(read_rows(split_dir / "train.jsonl"))
    dev = list(read_rows(split_dir / "dev.jsonl"))
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tokenizer.train_from_iterator((row[field] for row in train
                                   for field in ("instruction", "response")), trainer=trainer)
    candidates = [128, 256, 512, 1024]
    statistics = {"train": profile(tokenizer, train, candidates),
                  "dev": profile(tokenizer, dev, candidates)}
    # Choose the shortest candidate with <=1% truncation in BOTH splits.
    acceptable = [limit for limit in candidates if all(
        stats["truncation"][str(limit)]["over_limit"]["full_sequence"]["rate"] <= .01
        and stats["truncation"][str(limit)]["zero_supervised_samples"] == 0
        for stats in statistics.values())]
    if not acceptable:
        raise ValueError("No candidate meets the <=1% truncation criterion")
    max_length = min(acceptable)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save(str(output / "tokenizer.json"))
    config = {
        "artifact_version": "bitext-bpe-v1", "algorithm": "byte-level BPE",
        "tokenizers_version": tokenizers.__version__, "normalization": "none",
        "training_split": "train", "training_fields": ["instruction", "response"],
        "requested_vocab_size": vocab_size, "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": 2, "special_token_ids": {s: tokenizer.token_to_id(s) for s in SPECIAL_TOKENS},
        "template": TEMPLATE, "assembly": "encode fields separately, insert special IDs",
        "max_length": max_length, "length_scope": "full sequence before next-token shift",
        "model_context_length": max_length,
        "truncation_side": "right", "loss_scope": "response and retained EOS; prompt/padding labels=-100",
        "selection_rule": "shortest of 128/256/512/1024 with <=1% full sequence truncation in train and dev and zero unsupervised samples",
        "source_revision": manifest["revision"],
        "split_sha256": {name: manifest["file_sha256"][name] for name in ("train.jsonl", "dev.jsonl")},
        "tokenizer_sha256": file_hash(output / "tokenizer.json"),
    }
    for name, value in (("config.json", config), ("length-statistics.json", statistics)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    freeze = {"revision": config["tokenizer_sha256"],
              "file_sha256": {name: file_hash(output/name) for name in
                              ("tokenizer.json", "config.json", "length-statistics.json")}}
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    return config, statistics


def load_frozen(output):
    """Verify artifact bytes before loading the tokenizer and its configuration."""
    output = Path(output)
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for name in ("tokenizer.json", "config.json", "length-statistics.json"):
        if file_hash(output / name) != freeze["file_sha256"][name]:
            raise ValueError(f"Frozen artifact hash mismatch: {name}")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if config["tokenizer_sha256"] != freeze["revision"]:
        raise ValueError("Tokenizer revision mismatch")
    return Tokenizer.from_file(str(output / "tokenizer.json")), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("data/splits/bitext-v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer/bitext-bpe-v1"))
    args = parser.parse_args()
    config, _ = build_artifact(args.splits, args.output)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
