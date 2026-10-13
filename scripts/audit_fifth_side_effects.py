"""Freeze complete regression cases, response-independent samples and lexical evidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT = Path('outputs/fifth-side-effect-audit-20261006')
EVAL = Path('outputs/dev-expanded-fifth-20261005')
BASE = Path('data/derived/result-privacy-v2/train.jsonl')
FINAL = Path('data/derived/refund-execution-v3.1')
PHRASES = ('ensure a smooth resolution', 'ensure a smooth and satisfactory resolution')

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]

def rank(row):
    return hashlib.sha256(('fifth-side-effects-v1:42:'+row['id']).encode()).hexdigest(), row['id']

def norm(text):
    return ' '.join(text.casefold().split())

def choose(rows_, intents, changed):
    selected = {}
    for intent in sorted(intents):
        layers = ['all'] + (['changed', 'unchanged'] if intent in ('get_refund', 'track_refund') else [])
        for layer in layers:
            pool = [r for r in rows_ if r['intent'] == intent and (layer == 'all' or (r['id'] in changed) == (layer == 'changed'))]
            for r in sorted(pool, key=rank)[:3]:
                item = selected.setdefault(r['id'], {'id': r['id'], 'intent': intent, 'layers': []})
                item['layers'].append(layer)
    return list(selected.values())

def main():
    manifest = read(FINAL/'manifest.json')
    for p, h in manifest['dependencies_sha256'].items():
        assert file_hash(p) == h, p
    for p, h in manifest['files_sha256'].items():
        assert file_hash(FINAL/p) == h, p
    frozen = read(EVAL/'rating-freeze.json')
    assert file_hash(EVAL/'ai-expert-ratings.json') == frozen['ratings_sha256']
    for p, h in frozen['sources_sha256'].items():
        assert file_hash(p) == h, p
    ratings = read(EVAL/'ai-expert-ratings.json')
    repeats = [r for r in ratings['pairs'] if r['from'] == 'fourth' and r['repetition'] == 1]
    risks = [r for r in ratings['focused_refund_audit'] if r['model'] == 'fifth' and r['refund_result_tag'] == 1]
    assert len(repeats) == 24 and len(risks) == 3
    keys = {r['id'] for r in repeats + risks}
    raw = {m: {r['id']: r for r in rows((EVAL if m == 'fifth' else Path('outputs/dev-expanded-20261005-v1/fourth'))/'replies.jsonl')} for m in ('fourth', 'fifth')}
    by_score = {(r['model'], r['id']): r for r in ratings['ratings']}
    train = rows(FINAL/'train.jsonl'); base = {r['id']: r for r in rows(BASE)}
    changes = {r['id']: r for r in rows(FINAL/'changes.jsonl')}
    assert len(train) == len(base) == 21528
    for r in train:
        b = base[r['id']]
        assert {k:v for k,v in r.items() if k != 'response'} == {k:v for k,v in b.items() if k != 'response'}
        assert (r['response'] != b['response']) == (r['id'] in changes)
        if r['id'] in changes:
            assert changes[r['id']]['before_response'] == b['response'] and changes[r['id']]['after_response'] == r['response']
    cases = [{'id': key, 'intent': raw['fifth'][key]['intent'], 'cohort': raw['fifth'][key]['cohort'],
        'layers': (['new_repetition'] if key in {r['id'] for r in repeats} else []) + (['residual_refund_result'] if key in {r['id'] for r in risks} else []),
        'instruction': raw['fifth'][key]['instruction'],
        'fourth': {'response': raw['fourth'][key]['generated_response'], 'scores': by_score['fourth',key], 'stop': raw['fourth'][key]['stop_reason']},
        'fifth': {'response': raw['fifth'][key]['generated_response'], 'scores': by_score['fifth',key], 'stop': raw['fifth'][key]['stop_reason']}}
        for key in sorted(keys)]
    selected = choose(train, {r['intent'] for r in cases}, set(changes))
    deps = [BASE, FINAL/'manifest.json', FINAL/'train.jsonl', FINAL/'changes.jsonl', EVAL/'ai-expert-ratings.json',
        EVAL/'rating-freeze.json', EVAL/'replies.jsonl', Path('outputs/dev-expanded-20261005-v1/fourth/replies.jsonl'),
        Path('docs/fifth-side-effect-audit-protocol-v1.md'), Path(__file__)]
    save(OUT/'selection.json', {'protocol': 'fifth-side-effects-v1', 'cases': [{'id':r['id'],'intent':r['intent'],'cohort':r['cohort'],'layers':r['layers']} for r in cases],
        'train_samples': selected, 'dependencies_sha256': {str(p):file_hash(p) for p in deps}})
    save(OUT/'cases.json', cases)
    by_train = {r['id']: r for r in train}
    save(OUT/'targets.json', [{**s, 'instruction':by_train[s['id']]['instruction'], 'v2':base[s['id']]['response'],
        'v31':by_train[s['id']]['response'], 'changed':s['id'] in changes} for s in selected])
    quotes = [{'case_id':r['id'],'quote':q} for r in cases for q in r['fifth']['scores']['evidence'] if len(q.split()) >= 5]
    quotes += [{'case_id':'residual-phrase-probe','quote':q} for q in PHRASES]
    diagnostics = []
    for probe in quotes:
        found = [r for r in train if norm(probe['quote']) in norm(r['response'])]
        diagnostics.append({**probe, 'row_count':len(found), 'intents':dict(Counter(r['intent'] for r in found)),
            'changed_row_count':sum(r['id'] in changes for r in found), 'ids':[r['id'] for r in found]})
    save(OUT/'exact-span-diagnostics.json', {'method':'Case/whitespace normalized exact substring, not semantic classifier or causal attribution', 'probes':diagnostics})
    groups = {}
    for r in train:
        if r['intent'] not in ('get_refund', 'track_refund'): continue
        sentences = re.split(r'(?<=[.!?])\s+', r['response'])
        for i, sentence in enumerate(sentences):
            if not any(norm(q) in norm(sentence) for q in PHRASES): continue
            context = ' '.join(sentences[max(0,i-1):i+2])
            key = (r['intent'], context)
            g = groups.setdefault(key, {'intent':r['intent'],'sentence':sentence,'context':context,'ids':[]})
            g['ids'].append(r['id'])
    material = []
    for i, g in enumerate(sorted(groups.values(), key=lambda g:(g['intent'], g['context'])),1):
        sample = min(g['ids'], key=lambda k:rank(by_train[k]))
        material.append({'review_id':f'R{i:03}', **g, 'sample_id':sample,'instruction':by_train[sample]['instruction'],
            'v2':base[sample]['response'],'v31':by_train[sample]['response'], 'changed_ids':[k for k in g['ids'] if k in changes]})
    save(OUT/'residual-contexts.json', material)
    save(OUT/'material-freeze.json', {n:file_hash(OUT/n) for n in ('selection.json','cases.json','targets.json','exact-span-diagnostics.json','residual-contexts.json')})
    print(json.dumps({'cases':len(cases),'train_samples':len(selected),'residual_context_groups':len(material),'residual_rows':len({k for g in material for k in g['ids']}),
        'phrase_probes':diagnostics[-2:]},ensure_ascii=False,indent=2))

if __name__ == '__main__':main()
