"""Observer-only task probes and explicit prefix continuations, not production decoding."""
import torch
from torch.nn import functional as F
from .tokenization import encode_parts, SPECIAL_TOKENS

NEUTRAL = "I'm here to help. "
TASK_PREFIXES = {
    "partial": "I can help you remove specific items from your order. ",
    "whole": "I can help you cancel your entire order. ",
    "request": "I can help you submit a refund request. ",
    "policy": "I can help you understand refund eligibility. ",
}
BRANCHES = {
    "partial": "\n\nPlease tell me which items you want to remove.",
    "whole": "\n\nPlease confirm that you want to cancel the entire order.",
    "request": "\n\nPlease provide your purchase details to start a refund request.",
    "policy": "\n\nThe refund policy explains which situations are eligible for a refund.",
}
STAGES = (0, 32, 64, 96)


def trajectory_prefixes(tokenizer, token_ids, eos):
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    stops = []
    for index in range(1, len(token_ids) + 1):
        if token_ids[index - 1] == eos:
            break
        prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
        if prefix and text.startswith(prefix) and prefix.rstrip().endswith((".", "!", "?")):
            stops.append(index)
    rows = [{"stage": 0, "prefix_tokens": 0, "prefix_ids": [], "prefix_text": "", "available": True}]
    for minimum in STAGES[1:]:
        index = next((value for value in stops if value >= minimum), None)
        rows.append({"stage": minimum, "available": index is not None,
                     "prefix_tokens": index, "prefix_ids": token_ids[:index] if index is not None else None,
                     "prefix_text": tokenizer.decode(token_ids[:index], skip_special_tokens=True) if index is not None else None})
    return rows


def checked_prefix(tokenizer, text, config):
    if not isinstance(text, str) or any(token in text for token in SPECIAL_TOKENS):
        raise ValueError("Invalid prefix text")
    ids = tokenizer.encode(text, add_special_tokens=False).ids if text else []
    forbidden = {config[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant", "eos")}
    if any(token in forbidden for token in ids):
        raise ValueError("Control/EOS in prefix")
    return ids


@torch.no_grad()
def branch_nll(model, tokenizer, config, instruction, prefix_ids, branch):
    prompt = encode_parts(tokenizer, instruction, "")[2][:-1]
    checked_prefix(tokenizer, branch, config)
    target = tokenizer.encode(branch, add_special_tokens=False).ids
    if not target:
        raise ValueError("Empty branch")
    if any(v in {config[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant", "eos")} for v in prefix_ids):
        raise ValueError("Invalid trajectory prefix")
    full = prompt + prefix_ids + target
    if len(full) > config["max_length"]:
        raise ValueError("Full branch does not fit, refuse truncation")
    base = len(prompt) + len(prefix_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor([([-100] * base + target)[1:]], dtype=torch.long, device=device)
    previous = model.training
    model.eval()
    try:
        logits = model(input_ids, attention_mask=torch.ones_like(input_ids))["logits"]
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="none")
        selected = losses[labels.reshape(-1) != -100]
        if not selected.isfinite().all():
            raise ValueError("Nonfinite branch NLL")
        return {"token_count": len(target), "total_nll": float(selected.sum()),
                "mean_nll": float(selected.mean()), "per_token_nll": selected.tolist(),
                "target_ids": target, "input_length": len(full) - 1}
    finally:
        model.train(previous)


@torch.no_grad()
def prefix_reply(model, tokenizer, config, instruction, prefix_text, total_reply_budget=320):
    if type(total_reply_budget) is not int or total_reply_budget <= 0:
        raise ValueError("Invalid total reply budget")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Instruction must be nonempty")
    prompt = encode_parts(tokenizer, instruction, "")[2][:-1]
    prefix = checked_prefix(tokenizer, prefix_text, config)
    if len(prefix) >= total_reply_budget or len(prompt) + len(prefix) >= config["max_length"]:
        raise ValueError("No continuation budget")
    blocked = [config[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant")]
    eos = config["eos_token_id"]
    device = next(model.parameters()).device
    generated, steps = [], []
    reason = "max_new_tokens"
    previous = model.training
    model.eval()
    try:
        for index in range(total_reply_budget - len(prefix)):
            ids = torch.tensor([prompt + prefix + generated], dtype=torch.long, device=device)
            logits = model(ids, attention_mask=torch.ones_like(ids))["logits"][0, -1].clone()
            if not logits.isfinite().all():
                raise ValueError("Nonfinite logits")
            logits[blocked] = float("-inf")
            token = int(logits.argmax())
            probabilities = logits.softmax(-1)
            values, candidates = logits.topk(min(5, logits.numel() - len(blocked)))
            generated.append(token)
            steps.append({"step": index + 1, "token_id": token, "input_length": ids.shape[1],
                          "eos_probability": float(probabilities[eos]),
                          "top5": [{"token_id": int(t), "probability": float(probabilities[t])} for t in candidates]})
            if token == eos:
                reason = "eos"
                break
            if len(prompt) + len(prefix) + len(generated) >= config["max_length"]:
                reason = "context_limit"
                break
    finally:
        model.train(previous)
    all_ids = prefix + generated
    return {"prefix_text": prefix_text, "prefix_token_ids": prefix, "prefix_tokens": len(prefix),
            "continuation_token_ids": generated, "continuation_tokens": len(generated),
            "response": tokenizer.decode(all_ids, skip_special_tokens=True),
            "continuation": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_token_ids": all_ids, "generated_tokens": len(all_ids),
            "prompt_tokens": len(prompt), "stop_reason": reason, "steps": steps}
