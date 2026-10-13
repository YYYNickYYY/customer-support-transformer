import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from my_project_1.generate import generate_reply
from my_project_1.generation_diagnostic import trace_reply, locate_quote


class Tok:
    specials = {'<PAD>': 0, '<BOS>': 1, '<USER>': 2, '<ASSISTANT>': 3, '<EOS>': 4}
    def token_to_id(self, name):
        return self.specials[name]
    def id_to_token(self, index):
        return str(index)
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[5] if text else [])
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(i) for i in ids if i > 4)


class Model(torch.nn.Module):
    def __init__(self, answers, invalid=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.answers, self.inputs, self.invalid = answers, [], invalid
    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert attention_mask.eq(1).all()
        logits = torch.zeros(1, input_ids.shape[1], 10)
        logits[0, -1, :4] = 100
        logits[0, -1, self.answers[len(self.inputs)]] = 10
        self.inputs.append(input_ids.tolist()[0])
        if self.invalid:
            logits[0, -1, 7] = float('nan')
        return {'logits': logits}


def config(length=12):
    return dict(max_length=length, pad_token_id=0, bos_token_id=1, user_token_id=2,
                assistant_token_id=3, eos_token_id=4)


@pytest.mark.parametrize('answers,length,limit', [([6, 7, 4], 12, 10), ([6, 7], 12, 2),
                                                ([6, 7], 6, 10), ([6, 4], 6, 10)])
def test_observer_has_exact_generator_parity(answers, length, limit):
    original = Model(answers)
    observer = Model(answers)
    expected = generate_reply(original, Tok(), config(length), 'question', max_new_tokens=limit)
    result = trace_reply(observer, Tok(), config(length), 'question', max_new_tokens=limit)
    assert {k: result[k] for k in expected} == expected
    assert observer.inputs == original.inputs
    assert observer.training and observer.anchor.grad is None
    assert len(result['steps']) == result['generated_tokens']
    for step, token in zip(result['steps'], answers):
        assert step['token_id'] == token
        assert not {t['token_id'] for t in step['top5']} & {0, 1, 2, 3}
        assert 0 <= step['eos_probability'] <= 1
        assert step['eos_rank_strict'] == (1 if token == 4 else 2)
        assert step['selected_minus_eos_logit'] == (0 if token == 4 else 10)


@pytest.mark.parametrize('limit', [True, 0, -1, 1.2])
def test_reject_bad_limit(limit):
    with pytest.raises(ValueError):
        trace_reply(Model([4]), Tok(), config(), 'question', max_new_tokens=limit)


def test_invalid_logits_restores_mode_no_gradients():
    model = Model([4], invalid=True)
    with pytest.raises(ValueError, match='Non-finite'):
        trace_reply(model, Tok(), config(), 'question')
    assert model.training and model.anchor.grad is None


def test_quote_occurrences_and_no_invented_evidence():
    assert locate_quote(Tok(), [6, 7, 6, 4], '6', occurrence=1) == {
        'quote': '6', 'char_start': 4, 'char_end_exclusive': 5, 'first_token': 3, 'last_token': 3}
    with pytest.raises(ValueError, match='missing'):
        locate_quote(Tok(), [6, 7], '8')


def test_byte_bpe_quote_mapping_includes_incomplete_byte_prefix():
    class ByteTok:
        def decode(self, ids, skip_special_tokens=True):
            return bytes(ids).decode('utf-8', errors='replace')
    ids = list('a£b'.encode())
    span = locate_quote(ByteTok(), ids, '£')
    assert span['first_token'] == 2 and span['last_token'] == 3


def test_diagnostic_selection_is_deterministic_and_keeps_all_limit_cases():
    path = Path(__file__).resolve().parents[1] / 'scripts/diagnose_fifth_generation.py'
    spec = importlib.util.spec_from_file_location('fifth_trace_script', path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    ids = ['x:16355', 'x:26320', 'x:26438', 'x:N1', 'x:N2', 'x:P1', 'x:P2',
           'x:C1', 'x:C2', 'x:C3', 'x:T1', 'x:T2']
    replies = [{'id': k, 'instruction': k, 'intent': 'test',
                'stop_reason': 'max_new_tokens' if ':T' in k else 'eos'} for k in ids]
    ratings = {'ratings': [{'model': 'fifth', 'id': k, 'repetition': 0 if ':C' in k else 1,
                            'unsupported_claim': 0 if ':C' in k else 1} for k in ids],
               'pairs': [{'id': k, 'fourth': 0 if ':N' in k else 1, 'fifth': 1}
                         for k in ids if ':N' in k or ':P' in k]}
    selected = script.choose_cases(replies, ratings)
    assert selected == script.choose_cases(list(reversed(replies)), ratings)
    assert len(selected) == len({r['id'] for r in selected}) == 12
    assert {r['id'] for r in selected if r['stratum'] == 'non-eos'} == {'x:T1', 'x:T2'}
