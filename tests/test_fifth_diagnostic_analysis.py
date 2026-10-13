import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fifth_trace_analysis', ROOT / 'scripts/analyze_fifth_diagnostic.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_literal_repeat_nonoverlap_and_different_words_not_semantic_score():
    assert analysis.repeated_ngrams([1, 2, 3, 4, 9, 1, 2, 3, 4]) == [
        {'first_token': 1, 'repeat_token': 6, 'ids': [1, 2, 3, 4]}]
    assert analysis.repeated_ngrams([1, 2, 3, 4, 9, 1, 2, 5, 4]) == []
    assert analysis.repeated_ngrams([1, 1, 1, 1, 1]) == []


def test_old_reply_prompt_compatibility_is_in_memory_only(monkeypatch):
    monkeypatch.chdir(ROOT)
    path = ROOT / 'scripts/diagnose_fifth_generation_v11.py'
    spec = importlib.util.spec_from_file_location('fifth_trace_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.original_lines = lambda p: [{'id': 'x', 'instruction': 'hello', 'generated_token_ids': [4]}]
    records = module.compatible_lines(module.module.SOURCE / 'replies.jsonl')
    assert records[0]['prompt_tokens'] > 3
    assert records[0]['generated_token_ids'] == [4]
    assert 'prompt_tokens' not in module.original_lines(None)[0]


@pytest.mark.skipif(not (ROOT / analysis.OUT / 'summary.json').exists(),
                    reason='Frozen local diagnostic traces are not shipped with source')
def test_real_frozen_traces_match_original_and_have_no_optimizer_updates():
    from my_project_1.tokenization import file_hash
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    directory = ROOT / analysis.OUT
    summary = read(directory / 'summary.json')
    original = {r['id']: r for r in [json.loads(line) for line in
                 (ROOT / 'outputs/dev-expanded-fifth-20261005/replies.jsonl').read_text(encoding='utf-8').splitlines()]}
    assert summary['exact_replays'] == 14
    assert summary['optimizer_updates'] == 0 and summary['model_gradients_none']
    assert summary['source_stop_counts_all135'] == {'eos': 131, 'max_new_tokens': 4}
    assert summary['stop_counts_cases'] == {'eos': 10, 'max_new_tokens': 4}
    frozen = read(directory / 'freeze.json')
    for mapping in (frozen['dependencies_sha256'], summary['trace_sha256'],
                    read(directory / 'analysis-freeze.json')['sha256']):
        for p, h in mapping.items():
            assert file_hash(ROOT / p) == h
    for item in summary['cases']:
        trace = read(ROOT / item['path'])
        old = original[trace['id']]
        assert trace['generated_token_ids'] == old['generated_token_ids']
        assert trace['response'] == old['generated_response'] and trace['stop_reason'] == old['stop_reason']
        assert len(trace['steps']) == old['generated_tokens']
        for step, token in zip(trace['steps'], trace['generated_token_ids']):
            assert step['token_id'] == token
            assert 0 <= step['selected_probability'] <= 1
            assert 0 <= step['eos_probability'] <= 1
            assert not {r['token_id'] for r in step['top5']} & {0, 1, 2, 3}
        if trace['stop_reason'] == 'max_new_tokens':
            assert trace['generated_tokens'] == 320
            assert 176 <= trace['context_remaining'] <= 184
            assert trace['steps'][-1]['eos_rank_strict'] > 1
