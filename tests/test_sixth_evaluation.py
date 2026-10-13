"""Read-only checks of the completed sixth evaluation; no inference/training."""
import json
from pathlib import Path

from my_project_1.tokenization import file_hash
from my_project_1.training_sixth import load_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dev-expanded-sixth-20261007'
KEYS = ('intent_score', 'helpfulness_score', 'fluency_score', 'repetition', 'unsupported_claim')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def replies(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def test_frozen_sources_and_weights():
    load_plan()
    for filename in ('freeze.json', 'rating-freeze.json'):
        payload = read(OUT / filename)
        hashes = payload.get('dependencies_sha256', payload.get('sources_sha256'))
        assert hashes
        for path, expected in hashes.items():
            assert file_hash(ROOT / path) == expected
    assert file_hash(OUT / 'ai-expert-ratings.json') == read(OUT / 'rating-freeze.json')['ratings_sha256']


def test_same_135_and_only_checkpoint_changed():
    plan = load_plan()
    current = read(OUT / 'config.json')
    old = plan['generation_baseline']
    assert {k: v for k, v in current.items() if k not in ('checkpoint', 'checkpoint_sha256')} == {
        k: v for k, v in old.items() if k not in ('checkpoint', 'checkpoint_sha256')}
    rows = replies(OUT / 'replies.jsonl')
    baseline = replies(ROOT / plan['baseline_replies'])
    assert len(rows) == len(baseline) == 135
    for new, prior in zip(rows, baseline):
        assert all(new[k] == prior[k] for k in ('id', 'instruction', 'intent', 'cohort', 'reference_response'))


def test_tokens_and_stops_complete():
    config = read(OUT / 'config.json')['model_config']
    rows = replies(OUT / 'replies.jsonl')
    inputs = read(OUT / 'freeze.json')['inputs']
    blocked = {config[k + '_token_id'] for k in ('pad', 'bos', 'user', 'assistant')}
    for row, prompt in zip(rows, inputs):
        tokens = row['generated_token_ids']
        assert row['id'] == prompt['id'] and row['prompt_tokens'] == len(prompt['input_ids'])
        assert len(tokens) == row['generated_tokens'] and len(tokens) <= 320
        assert row['prompt_tokens'] + len(tokens) <= config['max_length']
        assert not blocked.intersection(tokens)
        if row['stop_reason'] == 'eos':
            assert tokens[-1] == config['eos_token_id']
            assert config['eos_token_id'] not in tokens[:-1]
        else:
            assert row['stop_reason'] == 'max_new_tokens' and len(tokens) == 320
    summary = read(OUT / 'summary.json')
    assert summary['weights_unchanged'] and summary['optimizer_updates'] == 0
    assert summary['replies_sha256'] == file_hash(OUT / 'replies.jsonl')


def test_full_text_ratings_and_paired_arithmetic():
    scored = read(OUT / 'ai-expert-ratings.json')
    raw = replies(OUT / 'replies.jsonl')
    assert len(scored['ratings']) == len(scored['pairs']) == 135
    for rating, row in zip(scored['ratings'], raw):
        assert rating['id'] == row['id']
        assert rating['notes'] and rating['evidence']
        assert all(q in row['generated_response'] for q in rating['evidence'])
        assert rating['risk_rubric'] == 'customer-support-risk-review-v1.1'
        assert rating['other_rubric'] == 'customer-support-review-v1.0'
        assert all(type(rating[k]) is int and 0 <= rating[k] <= (2 if i < 3 else 1) for i, k in enumerate(KEYS))
    for i, k in enumerate(KEYS):
        total = sum(r[k] for r in scored['ratings'])
        assert scored['metrics']['sixth'][k] == (total / 135 if i < 3 else total)
        assert sum(scored['paired_counts'][k].values()) == 135
    assert scored['metrics']['fifth']['repetition'] == 63
    assert scored['metrics']['fifth']['unsupported_claim'] == 43


def test_predefined_scope_and_primary_outcome():
    scored = read(OUT / 'ai-expert-ratings.json')
    assert len(scored['per_intent']['sixth']) == 27
    assert all(r['n'] == 5 for r in scored['per_intent']['sixth'].values())
    assert scored['per_cohort']['sixth']['original27']['n'] == 27
    assert scored['per_cohort']['sixth']['new108']['n'] == 108
    assert scored['best_dev_loss']['sixth'] > scored['best_dev_loss']['fifth']
    for k in ('intent_score', 'helpfulness_score'):
        assert scored['metrics']['sixth'][k] < scored['metrics']['fifth'][k]
