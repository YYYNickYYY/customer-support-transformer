"""Reproducible read-only train selection and lexical diagnostics, not risk scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

INTENTS = ('cancel_order', 'change_order', 'edit_account', 'get_refund',
           'place_order', 'registration_problems', 'track_order')
OUT = Path('outputs/train-quality-audit-20261004-v1')
TRAIN = Path('data/splits/bitext-v1/train.jsonl')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def split_sentences(text):
    # Do not split at decimal points; enough for a lexical diagnostic, not NLP parsing.
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def signals(text):
    sentences = split_sentences(text.lower())
    ensure = lambda s: bool(re.search(r'\b(?:ensure|ensures|ensuring)\b', s))
    smooth = lambda s: bool(re.search(r'\b(?:smooth|seamless|satisfaction|satisfactory)\b', s))
    return {
        'guarantee_or_promise': bool(re.search(r'\b(?:guarantee(?:d|s|ing)?|promis(?:e|es|ed|ing))\b', text.lower())),
        'ensure_smooth_or_satisfaction': any(ensure(s) and smooth(s) for s in sentences),
        'refund_action': any('refund' in s and re.search(r'\b(?:issu(?:e|es|ed|ing)|process(?:es|ed|ing)?|provid(?:e|es|ed|ing)|receiv(?:e|es|ed|ing)|get|gets|getting)\b', s) for s in sentences),
        'uncertainty_or_goal': any(re.search(r'\b(?:goal|aim|strive|try|committed|may|might|if|eligible)\b|do our best|work towards', s) for s in sentences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--finalize', action='store_true', help='Join explicit manual evidence to frozen train samples')
    args = parser.parse_args()
    content = TRAIN.read_bytes()
    source_hash = digest(content)
    assert source_hash == load(Path('artifacts/tokenizer/bitext-bpe-v1/config.json'))['split_sha256']['train.jsonl']
    rows = [json.loads(line) for line in content.decode('utf-8').splitlines()]
    groups = {intent: [r for r in rows if r['intent'] == intent] for intent in INTENTS}
    selected = []
    for intent in INTENTS:
        selected.extend(sorted(groups[intent], key=lambda r: (
            digest(('train-quality-audit-v1:42:' + r['id']).encode()), r['id']))[:5])
    payload = {'protocol': 'train-quality-audit-v1', 'seed': 42, 'samples_per_intent': 5,
               'train_sha256': source_hash,
               'selection_rule': 'minimum SHA256(train-quality-audit-v1:42:ID) within intent; no content filtering',
               'intents': list(INTENTS), 'ids': [r['id'] for r in selected]}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'selection.json'
    if args.freeze:
        if target.exists():
            assert load(target) == payload
        else:
            with target.open('x', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with (OUT / 'freeze.json').open('x', encoding='utf-8') as f:
                json.dump({'selection_sha256': digest(target.read_bytes()),
                           'protocol_sha256': digest(Path('docs/train-quality-audit-protocol-v1.md').read_bytes())}, f, indent=2)
        print('Frozen 35 IDs before displaying or scanning response content.')
        return
    assert load(target) == payload
    frozen = load(OUT / 'freeze.json')
    assert digest(target.read_bytes()) == frozen['selection_sha256']
    assert digest(Path('docs/train-quality-audit-protocol-v1.md').read_bytes()) == frozen['protocol_sha256']
    if args.finalize:
        decisions = load(OUT / 'manual-decisions.json')
        assert len(decisions) == len(selected) == 35
        reviewed = []
        for row, decision in zip(selected, decisions):
            suffix, categories, evidence, notes = decision
            assert row['id'].rsplit(':', 1)[-1] == suffix
            assert evidence in row['response'], (row['id'], evidence)
            reviewed.append({'id': row['id'], 'intent': row['intent'],
                'categories': categories, 'evidence': evidence, 'notes': notes,
                'reviewer': 'AI客服专家审查', 'rubric_id': 'customer-support-review-v1.0'})
        kinds = Counter(k for r in reviewed for k in r['categories'])
        risk_review = load(OUT / 'risk-review.json')
        assert {r['intent'] for r in risk_review['reviews']} == set(INTENTS)
        for stage, source_dir in [('old', Path('runs/20261004T015453-full-22c1eff31be445aa83c00458c4582e6a/dev-checks/20261004T091445-fdf38127e5b2435597a623e0b2ff61c5')),
                                  ('new', Path('runs/20261004T095337-full-666ad47f70d1422485fede4e65c16b1f/dev-checks/20261004T115457-6dd058d531f146e29eea8302929c588f'))]:
            raw = {r['intent']: r for r in map(json.loads, (source_dir / 'replies.jsonl').read_text(encoding='utf-8').splitlines())}
            for review in risk_review['reviews']:
                assert review[stage + '_evidence'] in raw[review['intent']]['generated_response']
        with (OUT / 'samples-reviewed.json').open('x', encoding='utf-8') as f:
            json.dump({'train_sha256': source_hash, 'selection_sha256': frozen['selection_sha256'],
                       'sample_count': len(reviewed), 'category_counts': dict(kinds),
                       'note': 'Categories overlap; counts are NOT risk rates; no historical score changes',
                       'reviews': reviewed}, f, ensure_ascii=False, indent=2)
        print('Verified all 35 manual evidence excerpts and 14 paired excerpts; categories:', dict(kinds))
        return
    stats = {}
    phrases = ('ensure a seamless experience', 'ensure a smooth', 'ensure your satisfaction', 'issue a refund')
    for intent, items in groups.items():
        counts = Counter()
        for row in items:
            counts.update(k for k, v in signals(row['response']).items() if v)
        stats[intent] = {'rows': len(items), 'candidate_counts': dict(counts),
            'candidate_union': sum(any(v for k, v in signals(r['response']).items() if k != 'uncertainty_or_goal') for r in items),
            'phrase_counts': {p: sum(p in r['response'].lower() for r in items) for p in phrases}}
    with (OUT / 'lexical-statistics.json').open('x', encoding='utf-8') as f:
        json.dump({'scope': 'seven target intents in frozen train; lexical candidates NOT v1.0 risk scores',
                   'train_sha256': source_hash, 'statistics': stats}, f, ensure_ascii=False, indent=2)
    with (OUT / 'samples.jsonl').open('x', encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps({**row, 'lexical_signals': signals(row['response'])}, ensure_ascii=False) + '\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
