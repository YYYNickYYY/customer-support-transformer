"""Deterministic supported-case selection and equal-target teacher-forcing controls."""
import hashlib

import torch
from .tokenization import encode_parts

POOLS = {
    "partial": ["1256", "1298", "1469", "1568", "1584", "1607", "1698", "1709"],
    "whole": ["378", "435", "443", "460", "612", "824", "969"],
    "request": ["16597", "16831", "16867"],
    "policy": ["6365", "6461", "6666", "6688"],
}
PARAPHRASES = {
    "1256": "Please help me remove some items from order {{Order Number}}.",
    "1298": "I would like to remove some items from purchase {{Order Number}}.",
    "1469": "What steps should I follow to remove some products from purchase {{Order Number}}?",
    "1568": "Could I remove some products from order {{Order Number}}?",
    "1584": "Could I delete some products from purchase {{Order Number}}?",
    "1607": "Please help me delete one item from order {{Order Number}}.",
    "1698": "Please delete several items from purchase {{Order Number}}.",
    "1709": "I am trying to delete some items from order {{Order Number}}.",
    "378": "Please assist me in cancelling order {{Order Number}}.",
    "435": "Please help me cancel purchase {{Order Number}}.",
    "443": "I need assistance cancelling purchase {{Order Number}}.",
    "460": "Could you help me cancel purchase {{Order Number}}?",
    "612": "I am unsure how to cancel order {{Order Number}}. Please explain.",
    "824": "Where can I find the option to cancel purchase {{Order Number}}?",
    "969": "I need help cancelling purchase {{Order Number}}.",
    "16597": "I am unsure how to obtain a refund. Please explain the process.",
    "16831": "I am unsure how to request my money back. Please explain how to apply.",
    "16867": "Please assist me with applying for a refund.",
    "6365": "Please explain the circumstances in which I can request reimbursement.",
    "6461": "Please help me find out when I am eligible to request a refund.",
    "6666": "I am trying to find out which situations allow me to request a refund.",
    "6688": "Please help me check the conditions under which I can request a refund.",
}


def build_plan(samples, reviews):
    by_suffix = {r["id"].rsplit(":", 1)[-1]: r for r in samples}
    judgments = {r["id"]: r["task_alignment"] for r in reviews}
    selected, cases, pairs = {}, [], []
    for family, suffixes in POOLS.items():
        pool = [by_suffix[s] for s in suffixes]
        if any(judgments[r["id"]] != "aligned" for r in pool):
            raise ValueError("Source not confirmed aligned")
        ranked = sorted(pool, key=lambda r: (
            hashlib.sha256(f'condition-use-v1:42:{family}:{r["id"]}'.encode()).hexdigest(), r["id"]))
        selected[family] = ranked[:2]
        for index, row in enumerate(ranked[:2], 1):
            source = row["id"].rsplit(":", 1)[-1]
            for variant, text in (("original", row["instruction"]), ("paraphrase", PARAPHRASES[source])):
                cases.append({"id": f"{family}-{index}-{variant}", "source_id": row["id"],
                              "family": family, "variant": variant, "instruction": text})
    for a, b in (("partial", "whole"), ("request", "policy")):
        for index in (1, 2):
            pairs.append([f"{a}-{index}-original", f"{b}-{index}-original"])
    cases.append(dict(cases[0], id="identical-control", variant="control"))
    return {"cases": cases, "pairs": pairs,
            "source_rows": [dict(row, family=family) for family, rows in selected.items() for row in rows]}


def matched_target_batches(tokenizer, instruction, swapped_instruction, response, max_length):
    prompts = [encode_parts(tokenizer, text, "")[2][:-1] for text in (instruction, swapped_instruction)]
    target = encode_parts(tokenizer, instruction, response)[1]
    eos = tokenizer.token_to_id("<EOS>")
    common = min(len(target) + 1, max_length - max(map(len, prompts)))
    if common <= 0:
        raise ValueError("No shared target budget")
    kept = (target + [eos])[:common]
    batches = []
    for prompt in prompts:
        full = prompt + kept
        batches.append({"input_ids": full[:-1], "labels": ([-100] * len(prompt) + kept)[1:]})
    return batches, {"prompt_lengths": list(map(len, prompts)), "response_tokens": len(target),
                     "scored_tokens": common, "retained_response_tokens": min(common, len(target)),
                     "lost_response_tokens": max(0, len(target) - common),
                     "eos_retained": common == len(target) + 1,
                     "original_training_retained_response_tokens": min(len(target), max(0, max_length - len(prompts[0])))}


@torch.no_grad()
def condition_nll(model, tokenizer, instruction, swapped_instruction, response, max_length):
    batches, metadata = matched_target_batches(tokenizer, instruction, swapped_instruction, response, max_length)
    was_training = model.training
    model.eval()
    try:
        losses = []
        for batch in batches:
            inputs = {key: torch.tensor([value], dtype=torch.long, device=next(model.parameters()).device)
                      for key, value in batch.items()}
            result = model(**inputs, attention_mask=torch.ones_like(inputs["input_ids"]))
            loss = float(result["loss"])
            if not torch.isfinite(result["loss"]):
                raise ValueError("Non-finite loss")
            losses.append(loss)
    finally:
        model.train(was_training)
    return {**metadata, "matched_nll": losses[0], "swapped_nll": losses[1],
            "swapped_minus_matched": losses[1] - losses[0]}
