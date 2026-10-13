"""Immutable, approved whole-sentence target edits on top of revision v1."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .target_revision import json_bytes, digest, load_revision as load_v1, rows_bytes
from .tokenization import file_hash, read_rows, load_frozen, encode_parts, prepare_example

VERSION = 'result-privacy-target-v2'
BASE = Path('data/derived/result-privacy-v1/manifest.json')
SOURCE = Path('data/splits/bitext-v1/train.jsonl')
OUT = Path('data/derived/result-privacy-v2')
PROPOSALS = Path('outputs/target-revision-v2-review-20261005')
DEPENDENCIES = ['docs/train-target-revision-v2.md', 'src/my_project_1/target_revision_v2.py',
    'src/my_project_1/revisions.py', 'src/my_project_1/dataset.py', 'configs/training-fourth.json',
    'configs/model.json', 'configs/generation.json', 'configs/dev-check.json',
    'data/splits/bitext-v1/dev.jsonl', 'artifacts/tokenizer/bitext-bpe-v1/config.json',
    'artifacts/tokenizer/bitext-bpe-v1/tokenizer.json']
QUALIFIER = re.compile(r'\b(?:goals?|aims?|striv\w*|try\w*|may|might)\b|\bdo(?:ing)?\b[^.!?\n]{0,45}\b(?:best|everything[^.!?\n]{0,15}can)\b|\bwork\w*\s+(?:towards|closely|diligently|together|with)\b|\b(?:help|want)\s+(?:to\s+)?ensure\b', re.I)
PRIVACY = re.compile(r'\b(?:information|data|details|privacy|email(?: address)?)\b', re.I)
PROTECT = re.compile(r'\b(?:handl\w*|treat\w*|kept|keep|stor\w*|protect\w*|safeguard\w*|secur\w*)\b', re.I)
CARE = re.compile(r'\b(?:care|confidential\w*|safe\w*|secur\w*|protect\w*|privacy)\b', re.I)
DEFINITE = re.compile(r"\bwill\b|\bwe'll\b|\b(?:guarantee\w*|promis\w*)\b", re.I)
RESULT = re.compile(r'\b(?:ensur\w*|guarantee\w*|promis\w*)\b[^.!?\n]{0,100}\b(?:resolution|resolved|successful outcome)\b', re.I)
CANCEL = re.compile(r'\b(?:ensur\w*|guarantee\w*|promis\w*)\b[^.!?\n]{0,110}\b(?:cancel(?:l)?ed|unsubscribed|cancellation|unsubscribe|removed|taken off|executed|processed)\b', re.I)
SAFE = 'Please share only the necessary personal information through an official registration or support channel.'
ASSIST = 'We can help you review the issue and consider the next steps.'
CANCEL_ASSIST = 'We can help you review the unsubscribe steps and submit a cancellation request through an official support channel.'
SEAMS = {
 'To work towards a resolution process, could you please provide me with the details of your claim?': 'To help us review your claim, could you please provide me with its details?',
 'This will expedite the process and work towards a resolution.': 'This will help us review your request and consider the next steps.',
 "We appreciate your patience and cooperation, and we're committed to working closely with you to work towards a resolution.": "We appreciate your patience and cooperation, and we're here to work with you to review the issue and consider the next steps.",
 'To work towards a resolution process, I recommend following these steps:': 'To help you report the payment issue, I recommend following these steps:',
 'We appreciate your patience and understanding while we work to work towards a resolution for you.': 'We appreciate your patience and understanding while we review your request and consider the next steps.',
}


def sentence_spans(text):
    """Preserve numbered-line markers and all whitespace outside sentence edits."""
    start = 0
    for m in re.finditer(r'(?<=[.!?])\s+|\n+', text):
        # A standalone enumerator is not a sentence boundary.
        if re.fullmatch(r'\s*\d+\.', text[start:m.start()]):
            continue
        if text[start:m.start()].strip():
            yield start, m.start(), text[start:m.start()]
        start = m.end()
    if text[start:].strip():
        yield start, len(text), text[start:]


def propose_sentence(sentence, intent):
    s = sentence.strip()
    if s in SEAMS:
        return SEAMS[s], 'v1-seam', None
    # Disallow whole-sentence replacement of mixed steps/questions/business actions.
    if QUALIFIER.search(s):
        return None, None, 'qualified-service-intent'
    if PRIVACY.search(s) and DEFINITE.search(s) and PROTECT.search(s) and CARE.search(s):
        if re.search(r'\b(?:password|click|provide me|provide us|please provide|could you|once I have|this will help|to get started)\b', s, re.I):
            return None, None, 'mixed-privacy-actions-needs-review'
        return SAFE, 'privacy-whole-sentence', None
    if re.search(r"\bensure\w*\b[^.!?]{0,60}\baccount\b[^.!?]{0,35}\bset up\b[^.!?]{0,35}\bsecurely\b", s, re.I):
        return 'I can help you review the account setup steps.', 'privacy-account-setup', None
    if RESULT.search(s):
        if '?' in s or ':' in s or re.search(r'\b(?:please provide|could you|share|details|information|documentation|evidence|steps|order|receipt)\b', s, re.I):
            return None, None, 'mixed-resolution-actions-needs-review'
        return ASSIST, 'resolution-whole-sentence', None
    if intent == 'newsletter_subscription' and CANCEL.search(s) and re.search(r'\b(?:subscription|newsletter|mailing list|unsubscribe|unsubscribed)\b', s, re.I):
        if '?' in s or re.search(r'\b(?:please provide|could you|click|follow|this information|this will|details)\b', s, re.I):
            return None, None, 'mixed-cancel-actions-needs-review'
        return CANCEL_ASSIST, 'cancel-whole-sentence', None
    return None, None, None


def propose(rows):
    rules, excluded = {}, []
    for row in rows:
        for start, end, sentence in sentence_spans(row['response']):
            after, kind, reason = propose_sentence(sentence, row['intent'])
            if after is not None:
                key = (row['intent'] if kind.startswith('cancel') else '*', sentence)
                entry = rules.setdefault(key, {'id': 'sentence-' + digest(json_bytes(key))[:16], 'intent': key[0], 'before': sentence, 'after': after, 'kind': kind, 'members': []})
                entry['members'].append({'id': row['id'], 'intent': row['intent'], 'instruction': row['instruction'], 'response': row['response'], 'start': start, 'end': end})
            elif reason:
                excluded.append({'id': row['id'], 'intent': row['intent'], 'sentence': sentence, 'reason': reason})
    return sorted(rules.values(), key=lambda r: r['id']), excluded


def derive(rows, rules):
    index = {}
    for r in rules:
        key = (r['intent'], r['before'])
        if key in index or not r['before'] or not r['after']:
            raise ValueError('Duplicate/empty approved rule')
        index[key] = r
    output, changes, matched = [], [], Counter()
    for row in rows:
        edits = []
        for start, end, sentence in sentence_spans(row['response']):
            rule = index.get((row['intent'], sentence), index.get(('*', sentence)))
            if rule:
                edits.append({'start': start, 'end': end, 'before': sentence, 'after': rule['after'], 'rule_id': rule['id'], 'kind': rule['kind']})
                matched[rule['id']] += 1
        revised = row['response']
        for e in reversed(edits):
            revised = revised[:e['start']] + e['after'] + revised[e['end']:]
        output.append({**row, 'response': revised})
        if edits:
            changes.append({'id': row['id'], 'intent': row['intent'], 'before_response': row['response'], 'after_response': revised, 'edits': edits})
    if set(matched) != {r['id'] for r in rules}:
        raise ValueError('Approved rule with no match')
    return output, changes


def baseline(source):
    return load_v1(BASE, source)[0]


def approved_rules(proposal, approval):
    ids = approval['approved_rule_ids']
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate approval')
    rules = [{k: r[k] for k in ('id', 'intent', 'before', 'after', 'kind')} for r in proposal['rules'] if r['id'] in ids]
    if len(rules) != len(ids):
        raise ValueError('Unknown approval ID')
    for r in rules:
        if r['id'] in approval.get('after_overrides', {}):
            r['after'] = approval['after_overrides'][r['id']]
    return rules + approval.get('extra_rules', [])


def freeze_revision(directory=OUT):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('V2 exists; refuse overwrite')
    source = list(read_rows(SOURCE))
    base = baseline(source)
    proposal = json.loads((PROPOSALS / 'proposals.json').read_text(encoding='utf-8'))
    approval = json.loads((PROPOSALS / 'approval.json').read_text(encoding='utf-8'))
    if approval['proposals_sha256'] != file_hash(PROPOSALS / 'proposals.json'):
        raise ValueError('Review proposal hash mismatch')
    if proposal['protocol_sha256'] != file_hash(DEPENDENCIES[0]):
        raise ValueError('Protocol changed since proposal')
    rules = approved_rules(proposal, approval)
    if not rules:
        raise ValueError('Incomplete approvals')
    output, changes = derive(base, rules)
    tokenizer, config = load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    lengths = []
    changed = {c['id'] for c in changes}
    for old, new in zip(base, output):
        if {k: v for k, v in old.items() if k != 'response'} != {k: v for k, v in new.items() if k != 'response'}:
            raise ValueError('Nonresponse mutation')
        if old['id'] in changed:
            lengths.append([len(encode_parts(tokenizer, r['instruction'], r['response'])[2]) for r in (old, new)])
            if prepare_example(tokenizer, new['instruction'], new['response'], config['max_length'])['supervised_tokens'] < 1:
                raise ValueError('Zero supervision')
    rank = lambda r: digest(('target-controls-v2:42:' + r['id']).encode())
    controls = {'selection': 'lowest SHA256(target-controls-v2:42:ID), up to5 per kind; excluded up to20',
        'positive_ids': {kind: [r['id'] for r in sorted([c for c in changes if any(e['kind'] == kind for e in c['edits'])], key=rank)[:5]] for kind in sorted({r['kind'] for r in rules})},
        'excluded': sorted(proposal['excluded'], key=rank)[:20]}
    payloads = {'train.jsonl': rows_bytes(output), 'changes.jsonl': rows_bytes(changes), 'rules.json': json_bytes(rules), 'controls.json': json_bytes(controls)}
    dependencies = DEPENDENCIES + [str(PROPOSALS / 'proposals.json'), str(PROPOSALS / 'approval.json')]
    manifest = {'version': VERSION, 'frozen': True, 'source_train_sha256': file_hash(SOURCE), 'baseline_manifest_sha256': file_hash(BASE),
        'baseline_train_sha256': file_hash(BASE.parent / 'train.jsonl'), 'files_sha256': {n: digest(b) for n, b in payloads.items()},
        'dependencies_sha256': {p: file_hash(p) for p in dependencies}, 'train_samples': len(output), 'changed_samples': len(changes),
        'edits': sum(len(c['edits']) for c in changes), 'rule_counts': dict(Counter(e['kind'] for c in changes for e in c['edits'])),
        'changed_length_checks': {'old_max_full_tokens': max(n[0] for n in lengths), 'new_max_full_tokens': max(n[1] for n in lengths),
        'old_truncated': sum(n[0] > config['max_length'] for n in lengths), 'new_truncated': sum(n[1] > config['max_length'] for n in lengths), 'zero_supervision': 0},
        'review_method': 'AI-reviewed approved exact whole sentences; not independent human or all-training risk scoring', 'created_date': '2026-10-05'}
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    (directory / 'manifest.json').write_bytes(json_bytes(manifest))
    load_revision(directory / 'manifest.json', source)
    return manifest


def load_revision(manifest_path, source_rows):
    path = Path(manifest_path)
    m = json.loads(path.read_text(encoding='utf-8'))
    expected_deps = set(DEPENDENCIES + [str(PROPOSALS / 'proposals.json'), str(PROPOSALS / 'approval.json')])
    if m.get('version') != VERSION or m.get('frozen') is not True or m.get('source_train_sha256') != file_hash(SOURCE) or m.get('baseline_manifest_sha256') != file_hash(BASE) or m.get('baseline_train_sha256') != file_hash(BASE.parent / 'train.jsonl'):
        raise ValueError('V2 source/baseline/version mismatch')
    if set(m.get('files_sha256', {})) != {'train.jsonl', 'changes.jsonl', 'rules.json', 'controls.json'} or set(m.get('dependencies_sha256', {})) != expected_deps:
        raise ValueError('Incomplete V2 freeze')
    for name, expected in m['files_sha256'].items():
        if file_hash(path.parent / name) != expected:
            raise ValueError('V2 artifact mismatch: ' + name)
    for name, expected in m['dependencies_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('V2 dependency mismatch: ' + name)
    approved = json.loads((PROPOSALS / 'approval.json').read_text(encoding='utf-8'))
    proposals = json.loads((PROPOSALS / 'proposals.json').read_text(encoding='utf-8'))
    rules = json.loads((path.parent / 'rules.json').read_text(encoding='utf-8'))
    expected_rules = approved_rules(proposals, approved)
    if rules != expected_rules or approved['proposals_sha256'] != file_hash(PROPOSALS / 'proposals.json') or proposals['protocol_sha256'] != file_hash(DEPENDENCIES[0]):
        raise ValueError('Rules differ from approvals')
    expected, changes = derive(baseline(source_rows), rules)
    ledger = [json.loads(line) for line in (path.parent / 'changes.jsonl').read_text(encoding='utf-8').splitlines()]
    if list(read_rows(path.parent / 'train.jsonl')) != expected or ledger != changes:
        raise ValueError('V2 exact replay mismatch')
    if len(expected) != m['train_samples'] or len(changes) != m['changed_samples'] or sum(len(c['edits']) for c in changes) != m['edits']:
        raise ValueError('V2 count mismatch')
    return expected, {'manifest_path': str(path), 'manifest_sha256': file_hash(path), 'source_train_sha256': m['source_train_sha256'],
        'baseline_manifest_sha256': m['baseline_manifest_sha256'], 'actual_train_sha256': file_hash(path.parent / 'train.jsonl'), 'version': VERSION, 'changed_samples': m['changed_samples'], 'edits': m['edits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--propose', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if sum([args.propose, args.freeze, args.verify]) != 1:
        parser.error('Choose exactly one of --propose/--freeze/--verify')
    if args.propose:
        if PROPOSALS.exists():
            raise ValueError('Proposal directory exists')
        rules, excluded = propose(baseline(list(read_rows(SOURCE))))
        PROPOSALS.mkdir(parents=True)
        (PROPOSALS / 'proposals.json').write_bytes(json_bytes({'rules': rules, 'excluded': excluded, 'protocol_sha256': file_hash(DEPENDENCIES[0])}))
        print(json.dumps({'rules': len(rules), 'matches': sum(len(r['members']) for r in rules), 'excluded': len(excluded)}))
    elif args.freeze:
        print(json.dumps(freeze_revision(), indent=2))
    else:
        _, provenance = load_revision(OUT / 'manifest.json', list(read_rows(SOURCE)))
        print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
