from types import SimpleNamespace
import math

import pytest
import torch

from my_project_1.beam_search import beam_search_reply, rank_score
from my_project_1.generate import generate_reply


class Tokenizer:
    def token_to_id(self, token):
        return {'<PAD>': 0, '<BOS>': 1, '<USER>': 2, '<ASSISTANT>': 3, '<EOS>': 4}[token]

    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[5] if text else [])

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(str(x) for x in ids if x > 4)


CONFIG = dict(max_length=12, pad_token_id=0, bos_token_id=1, user_token_id=2,
              assistant_token_id=3, eos_token_id=4)


class TreeModel(torch.nn.Module):
    def __init__(self, invalid=False, looping=False, ties=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.invalid, self.looping, self.ties = invalid, looping, ties

    def forward(self, input_ids, attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        result = torch.full((len(input_ids), input_ids.shape[1], 10), -30.0)
        for i, row in enumerate(input_ids.tolist()):
            self.calls.append(row)
            if self.looping:
                probabilities = {6: .9, 7: .1}
            elif self.ties:
                probabilities = {6: .5, 7: .5} if len(row) == 4 else {4: 1.0}
            elif len(row) == 4:
                probabilities = {6: .55, 7: .45}
            elif row[4] == 7:
                probabilities = {4: .99, 8: .01}
            else:
                probabilities = {4: .4, 8: .3, 9: .3}
            for token, probability in probabilities.items():
                result[i, -1, token] = math.log(probability)
            if self.invalid:
                result[i, -1, 5] = float('nan')
        return {'logits': result}


def test_lower_first_choice_can_win_later_and_trace_is_reconstructible():
    model = TreeModel()
    result = beam_search_reply(model, Tokenizer(), CONFIG, 'q', beam_width=2,
                               num_return_sequences=2, length_penalty=0, max_new_tokens=4)
    assert [c['generated_token_ids'] for c in result['candidates']] == [[7, 4], [6, 4]]
    assert result['candidates'][0]['log_probability'] == pytest.approx(math.log(.45 * .99), abs=1e-6)
    assert result['trace'][0]['kept'][0]['token'] == 6
    previous = [[]]
    for step in result['trace']:
        previous = [previous[b['parent_rank'] - 1] + ([] if b['carried'] else [b['token']]) for b in step['kept']]
    assert previous == [c['generated_token_ids'] for c in result['candidates']]
    assert model.training and model.anchor.grad is None


@pytest.mark.parametrize('alpha', [0, .7, 1.0])
def test_width_one_equals_greedy(alpha):
    greedy = generate_reply(TreeModel(), Tokenizer(), CONFIG, 'q', max_new_tokens=4)
    beam = beam_search_reply(TreeModel(), Tokenizer(), CONFIG, 'q', beam_width=1,
                            num_return_sequences=1, length_penalty=alpha, max_new_tokens=4)
    assert beam['candidates'][0]['generated_token_ids'] == greedy['generated_token_ids']
    assert beam['candidates'][0]['stop_reason'] == greedy['stop_reason']


def test_eos_never_extended_and_forbidden_controls_never_generated():
    model = TreeModel()
    result = beam_search_reply(model, Tokenizer(), CONFIG, 'q', beam_width=4, num_return_sequences=4,
                               max_new_tokens=5)
    assert all(4 not in row[4:] for row in model.calls)
    for c in result['candidates']:
        assert not set(c['generated_token_ids']).intersection({0, 1, 2, 3})
        if c['stop_reason'] == 'eos':
            assert c['generated_token_ids'][-1] == 4


def test_token_limit_and_context_limit_no_fabricated_eos():
    result = beam_search_reply(TreeModel(looping=True), Tokenizer(), CONFIG, 'q',
                               max_new_tokens=2, beam_width=2, num_return_sequences=2)
    assert all(c['stop_reason'] == 'max_new_tokens' and len(c['generated_token_ids']) == 2 for c in result['candidates'])
    result = beam_search_reply(TreeModel(looping=True), Tokenizer(), {**CONFIG, 'max_length': 6}, 'q',
                               beam_width=2, num_return_sequences=2)
    assert all(c['stop_reason'] == 'context_limit' for c in result['candidates'])
    assert all(4 not in c['generated_token_ids'] for c in result['candidates'])


def test_ties_are_stable_and_length_score_explicit():
    result = beam_search_reply(TreeModel(ties=True), Tokenizer(), CONFIG, 'q',
                               beam_width=2, num_return_sequences=2, max_new_tokens=2, length_penalty=.7)
    assert result['candidates'][0]['generated_token_ids'] == [6, 4]
    assert result['candidates'][1]['generated_token_ids'] == [7, 4]
    for c in result['candidates']:
        assert c['score'] == rank_score(c['log_probability'], c['generated_tokens'], .7)


def test_nan_restores_mode_and_no_grad():
    model = TreeModel(invalid=True)
    with pytest.raises(ValueError, match='Non-finite'):
        beam_search_reply(model, Tokenizer(), CONFIG, 'q')
    assert model.training and model.anchor.grad is None


@pytest.mark.parametrize('kwargs', [dict(beam_width=0), dict(beam_width=True), dict(beam_width=17),
    dict(num_return_sequences=5), dict(max_new_tokens=0), dict(max_new_tokens=True),
    dict(length_penalty=-1), dict(length_penalty=float('nan')), dict(length_penalty=True)])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        beam_search_reply(TreeModel(), Tokenizer(), CONFIG, 'q', **kwargs)


@pytest.mark.parametrize('question', ['', ' ', None, 'q <EOS>'])
def test_invalid_prompts(question):
    with pytest.raises(ValueError):
        beam_search_reply(TreeModel(), Tokenizer(), CONFIG, question)


def test_long_prompt_rejected():
    with pytest.raises(ValueError, match='no room'):
        beam_search_reply(TreeModel(), Tokenizer(), {**CONFIG, 'max_length': 4}, 'q')


def test_saved_real_demo_trace_and_separate_stages():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs/beam-search-fifth-demo/20261007T021508-39fb54baab4a4a1e9231dbe8fbe0f59b'
    if not out.exists():
        pytest.skip('Local ignored demo artifacts not present')
    rows = [json.loads(x) for x in (out / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2
    for row in rows:
        previous = [([], 0.0)]
        for step in row['beam']['trace']:
            current = []
            for b in step['kept']:
                ids, score = previous[b['parent_rank'] - 1]
                tokens = ids + ([] if b['carried'] else [b['token']])
                total = score + (0 if b['carried'] else b['token_log_probability'])
                assert total == pytest.approx(b['log_probability'])
                assert len(tokens) == b['length']
                assert rank_score(total, len(tokens), .7) == pytest.approx(b['score'])
                current.append((tokens, total))
            previous = current
        assert [x[0] for x in previous] == [c['generated_token_ids'] for c in row['beam']['candidates']]
        assert row['checker_selection']['status'] == 'not_implemented'
        assert row['checker_selection']['gain_vs_beam_top1'] is None
    notes = json.loads((out / 'demo-observations.json').read_text(encoding='utf-8'))
    for r, n in zip(rows, notes['candidate_generation']):
        assert r['id'] == n['id'] and n['greedy_evidence'] in r['greedy']['response']
        assert all(n['beam_evidence_each'] in c['response'] for c in r['beam']['candidates'])
    assert not notes['candidate_generation'][1]['acceptable_scope_candidate_present']


def test_historical_training_freeze_and_workspace_entry():
    import json
    from pathlib import Path
    from my_project_1.training_sixth import load_plan
    root = Path(__file__).resolve().parents[1]
    if not (root / 'runs').exists():
        pytest.skip('Local frozen training artifacts not present')
    load_plan()  # Includes unchanged historical .vscode/tasks.json and fifth weights.
    workspace = json.loads((root / '项目工作台.code-workspace').read_text(encoding='utf-8'))
    task = workspace['tasks']['tasks'][0]
    assert 'my_project_1.beam_search' in task['args']
    assert '--train' not in task['args']
    assert task['options']['cwd'] == '${workspaceFolder}'
