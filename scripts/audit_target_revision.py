"""Read-only target intervention diagnostics; outputs never alter frozen sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from my_project_1.target_revision import derive, sentence_bounds

ROOT = Path('data/derived/result-privacy-v1')
OUT = Path('outputs/target-revision-audit-20261005-v1')
PROTOCOL = Path('docs/target-revision-audit-protocol-v1.md')
INTENTS = ('create_account', 'newsletter_subscription')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def save(name, value):
    target = OUT / name
    content = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    if target.exists():
        if target.read_text(encoding='utf-8') != content:
            raise ValueError('Refuse overwrite differing audit artifact: ' + name)
    else:
        target.write_text(content, encoding='utf-8')


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def context_at(text, position):
    left, right = sentence_bounds(text, position)
    prev = sentences(text[:left])
    after = sentences(text[right:])
    return {'previous': prev[-1] if prev else '', 'sentence': text[left:right], 'next': after[0] if after else ''}


def signals(sentence):
    s = sentence.lower()
    privacy = bool(re.search(r'\b(?:information|data|privacy|details)\b', s) and re.search(r'\b(?:care|protect\w*|secure\w*|safe\w*|confidential\w*|handl\w*)\b', s))
    result = bool(re.search(r'\b(?:ensur\w*|guarantee\w*|promis\w*)\b', s) and re.search(r'\b(?:resolution|resolv\w*|cancel\w*|unsubscrib\w*|result|outcome)\b', s))
    return [k for k, yes in [('privacy', privacy), ('result', result)] if yes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    manifest = load(ROOT / 'manifest.json')
    paths = {str(ROOT / p): h for p, h in manifest['files_sha256'].items()}
    paths.update(manifest['dependencies_sha256'])
    paths[str(Path(manifest['source_train_path']))] = manifest['source_train_sha256']
    for p, h in paths.items():
        if digest(Path(p).read_bytes()) != h:
            raise ValueError('Frozen hash mismatch ' + p)
    source, derived, changes = rows(manifest['source_train_path']), rows(ROOT / 'train.jsonl'), rows(ROOT / 'changes.jsonl')
    changed_ids = {c['id'] for c in changes}
    rank = lambda r: (digest(('target-revision-audit-v1:42:' + r['id']).encode()), r['id'])
    selection = []
    for intent in INTENTS:
        pool = [r for r in derived if r['intent'] == intent]
        for layer in ('all', 'changed', 'unchanged'):
            items = [r for r in pool if layer == 'all' or (r['id'] in changed_ids) == (layer == 'changed')]
            selection.extend({'id': r['id'], 'intent': intent, 'layer': layer} for r in sorted(items, key=rank)[:5])
    frozen = {'protocol_sha256': digest(PROTOCOL.read_bytes()), 'script_sha256': digest(Path(__file__).read_bytes()),
              'source_hashes': paths, 'selection_rule': 'lowest SHA256(target-revision-audit-v1:42:ID), tie ID, up to5 per intent/layer', 'selection': selection}
    OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        save('selection.json', frozen)
        print('Frozen selection before displaying sample replies:', len({s['id'] for s in selection}), 'unique IDs')
        return
    if load(OUT / 'selection.json') != frozen:
        raise ValueError('Audit freeze mismatch')
    replay, replay_changes, skipped = derive(source)
    if replay != derived or replay_changes != changes:
        raise ValueError('Exact replay mismatch')
    by_id = {r['id']: r for r in derived}
    full = []
    groups = {}
    for c in changes:
        offset = 0
        for e in c['edits']:
            pos = e['start'] + offset
            old, new = context_at(c['before_response'], e['start']), context_at(c['after_response'], pos)
            item = {'id': c['id'], 'intent': c['intent'], 'rule_id': e['rule_id'], 'before': old, 'after': new}
            key = json.dumps([old, new], sort_keys=True)
            group = groups.setdefault(key, {'index': len(groups) + 1, 'before': old, 'after': new, 'members': []})
            group['members'].append({'id': c['id'], 'intent': c['intent'], 'rule_id': e['rule_id']})
            full.append(item)
            offset += len(e['after']) - len(e['before'])
    candidates = []
    counts = Counter()
    for r in derived:
        for sentence in sentences(r['response']):
            kinds = signals(sentence)
            if kinds:
                counts.update((r['intent'], k) for k in kinds)
                if r['intent'] in INTENTS or r['id'] in changed_ids:
                    candidates.append({'id': r['id'], 'intent': r['intent'], 'changed': r['id'] in changed_ids, 'kinds': kinds,
                                       'context': context_at(r['response'], r['response'].index(sentence))})
    cgroups = {}
    for c in candidates:
        key = json.dumps(c['context'], sort_keys=True)
        g = cgroups.setdefault(key, {'index': len(cgroups) + 1, 'context': c['context'], 'members': []})
        g['members'].append({k: v for k, v in c.items() if k != 'context'})
    selected = [{**by_id[id_], 'layers': [s['layer'] for s in selection if s['id'] == id_]} for id_ in dict.fromkeys(s['id'] for s in selection)]
    save('changed-contexts.json', {'rows': full, 'groups': list(groups.values())})
    save('candidate-contexts.json', {'groups': list(cgroups.values())})
    save('selected-samples.json', selected)
    save('complete-changed-responses.json', changes)
    save('checks.json', {'train_rows': len(derived), 'changed_rows': len(changes), 'edits': len(full), 'exact_replay': True,
                         'all_frozen_hashes_match': True, 'context_groups': len(groups), 'candidate_context_groups': len(cgroups),
                         'selected_unique_ids': len(selected), 'intent_counts': {intent: {'all': sum(r['intent'] == intent for r in derived),
                         'changed': sum(c['intent'] == intent for c in changes)} for intent in INTENTS},
                         'candidate_sentence_counts_NOT_risk_rates': {intent: {k: counts[intent, k] for k in ('privacy', 'result')} for intent in sorted({r['intent'] for r in derived})},
                         'skipped_matches': len(skipped), 'note': 'Replay/lexical checks are not independent risk or grammar scores; AI context review follows separately.'})
    print(json.dumps(load(OUT / 'checks.json'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
