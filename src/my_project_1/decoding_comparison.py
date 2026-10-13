"""Isolated greedy 0-vs-4 experiment; does not change the active generator."""

import torch

from .tokenization import encode_parts


def banned_fourgram_tokens(generated, eos_token_id):
    """Overlapping reply-only 4-grams; EOS remains eligible."""
    if len(generated) < 3:
        return []
    prefix = generated[-3:]
    return sorted({generated[i + 3] for i in range(len(generated) - 3)
                   if generated[i:i + 3] == prefix and generated[i + 3] != eos_token_id})


def choose_token(raw_logits, generated, roles, eos, no_repeat_ngram_size):
    if no_repeat_ngram_size not in (0, 4) or type(no_repeat_ngram_size) is not int:
        raise ValueError("Only preregistered sizes 0 and 4 are permitted")
    if raw_logits.ndim != 1 or not raw_logits.isfinite().all():
        raise ValueError("Non-finite or invalid generation logits")
    if eos in roles:
        raise ValueError("EOS must not be role-blocked")
    logits = raw_logits.clone()
    logits[roles] = float("-inf")
    baseline_token = int(logits.argmax())
    eos_rank_before = 1 + int((logits > logits[eos]).sum())
    banned = banned_fourgram_tokens(generated, eos) if no_repeat_ngram_size else []
    logits[banned] = float("-inf")
    if not logits.isfinite().any():
        raise ValueError("No finite decoding candidate; do not relax constraints")
    token = int(logits.argmax())
    return token, {"baseline_greedy_token_id": baseline_token,
                   "ngram_blocked_token_ids": banned, "role_blocked_token_ids": list(roles),
                   "constraint_changed_choice": token != baseline_token,
                   "eos_rank_before_constraint": eos_rank_before,
                   "eos_rank_after_constraint": 1 + int((logits > logits[eos]).sum())}


@torch.no_grad()
def constrained_reply(model, tokenizer, model_config, instruction, *,
                      no_repeat_ngram_size=0, max_new_tokens=320):
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(no_repeat_ngram_size) is not int or no_repeat_ngram_size not in (0, 4):
        raise ValueError("Only preregistered sizes 0 and 4 are permitted")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    _, _, full = encode_parts(tokenizer, instruction, "")
    prompt = full[:-1]
    budget = model_config["max_length"]
    if len(prompt) >= budget:
        raise ValueError("Prompt leaves no room for generation")
    device = next(model.parameters()).device
    eos = model_config["eos_token_id"]
    roles = [model_config[f"{name}_token_id"] for name in ("pad", "bos", "user", "assistant")]
    generated, steps = [], []
    reason = "max_new_tokens"
    was_training = model.training
    model.eval()
    try:
        for index in range(max_new_tokens):
            ids = torch.tensor([prompt + generated], dtype=torch.long, device=device)
            raw = model(ids, attention_mask=torch.ones_like(ids))["logits"][0, -1]
            token, event = choose_token(raw, generated, roles, eos, no_repeat_ngram_size)
            steps.append({"step": index + 1, "input_length": ids.shape[1],
                          "token_id": token, **event})
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
            "generated_tokens": len(generated), "stop_reason": reason, "steps": steps,
            "no_repeat_ngram_size": no_repeat_ngram_size}
