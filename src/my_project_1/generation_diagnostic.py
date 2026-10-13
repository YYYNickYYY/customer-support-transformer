"""Observer-only greedy traces; must reproduce the existing generator exactly."""
import torch

from .tokenization import encode_parts


@torch.no_grad()
def trace_reply(model, tokenizer, model_config, instruction, *, max_new_tokens=320):
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be a positive integer')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be nonempty')
    _, _, full = encode_parts(tokenizer, instruction, '')
    prompt = full[:-1]
    budget = model_config['max_length']
    if len(prompt) >= budget:
        raise ValueError('Prompt leaves no room for generation')
    device = next(model.parameters()).device
    eos = model_config['eos_token_id']
    blocked = [model_config[f'{name}_token_id'] for name in ('pad', 'bos', 'user', 'assistant')]
    generated, steps = [], []
    reason = 'max_new_tokens'
    was_training = model.training
    model.eval()
    try:
        for index in range(max_new_tokens):
            ids = torch.tensor([prompt + generated], dtype=torch.long, device=device)
            logits = model(ids, attention_mask=torch.ones_like(ids))['logits'][0, -1].clone()
            if not logits.isfinite().all():
                raise ValueError('Non-finite generation logits')
            logits[blocked] = float('-inf')
            token = int(logits.argmax())
            probabilities = logits.softmax(-1)
            values, candidates = logits.topk(min(5, logits.numel() - len(blocked)))
            steps.append({'step': index + 1, 'input_length': ids.shape[1], 'token_id': token,
                          'token_piece': tokenizer.id_to_token(token),
                          'selected_probability': float(probabilities[token]),
                          'eos_probability': float(probabilities[eos]),
                          'eos_rank_strict': 1 + int((logits > logits[eos]).sum()),
                          'selected_minus_eos_logit': float(logits[token] - logits[eos]),
                          'top1_minus_top2_logit': float(values[0] - values[1]),
                          'top5': [{'token_id': int(t), 'token_piece': tokenizer.id_to_token(int(t)),
                                    'probability': float(probabilities[t])} for t in candidates]})
            generated.append(token)
            if token == eos:
                reason = 'eos'
                break
            if len(prompt) + len(generated) >= budget:
                reason = 'context_limit'
                break
    finally:
        model.train(was_training)
    return {'response': tokenizer.decode(generated, skip_special_tokens=True),
            'generated_token_ids': generated, 'prompt_tokens': len(prompt),
            'generated_tokens': len(generated), 'stop_reason': reason, 'steps': steps,
            'context_remaining': budget - len(prompt) - len(generated)}


def locate_quote(tokenizer, token_ids, quote, *, occurrence=0):
    """Map exact text evidence to 1-based generated token indices, safely for byte BPE."""
    if not isinstance(quote, str) or not quote or type(occurrence) is not int or occurrence < 0:
        raise ValueError('Invalid quote occurrence')
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    start, offset = -1, 0
    for _ in range(occurrence + 1):
        start = text.find(quote, offset)
        if start < 0:
            raise ValueError('Quote missing from generated text')
        offset = start + len(quote)
    end = start + len(quote)
    prefixes = [(i, tokenizer.decode(token_ids[:i], skip_special_tokens=True))
                for i in range(1, len(token_ids) + 1)]
    # Skip incomplete UTF-8 byte prefixes whose replacement characters aren't final text.
    valid = [(i, len(prefix)) for i, prefix in prefixes if text.startswith(prefix)]
    first = max([0, *(i for i, length in valid if length <= start)]) + 1
    last = next((i for i, length in valid if length >= end), None)
    if first is None or last is None:
        raise ValueError('Could not map evidence token span')
    return {'quote': quote, 'char_start': start, 'char_end_exclusive': end,
            'first_token': first, 'last_token': last}
