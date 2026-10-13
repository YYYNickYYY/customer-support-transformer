"""Conservative exact-span target intervention and immutable train provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .tokenization import encode_parts, file_hash, load_frozen, prepare_example, read_rows

VERSION = 'result-privacy-target-v1'
PROTOCOL = Path('docs/train-target-revision-v1.md')
OUT = Path('data/derived/result-privacy-v1')
SOURCE = Path('data/splits/bitext-v1/train.jsonl')
DEPENDENCIES = ['docs/train-target-revision-v1.md', 'src/my_project_1/target_revision.py',
    'data/splits/bitext-v1/dev.jsonl', 'artifacts/tokenizer/bitext-bpe-v1/config.json',
    'artifacts/tokenizer/bitext-bpe-v1/tokenizer.json', 'configs/model.json',
    'configs/generation.json', 'configs/dev-check.json']
EXCLUDE = r'\b(?:goals?|aims?|striv\w*|try\w*|may|might)\b|\bdo\b[^.!?\n]{0,40}\bbest\b|\bmake\b[^.!?\n]{0,40}\beffort\b|\bwork\w*\s+towards\b|\bwork\w*\s+(?:diligently|together|with you)\b|\bdo everything (?:we|I) can\b|\bdo (?:my|our|your) utmost\b|\bhelp\s+(?:to\s+)?ensure\b|\bwant to ensure\b|\byour (?:cooperation|insights?|inputs?|feedback|assistance)\b'
RULES = []
for verb, replacement in [('ensure', 'work'), ('ensuring', 'working')]:
    for adjective in ['smooth and satisfactory', 'satisfactory', 'smooth', 'fair', 'swift']:
        RULES.append({'id': f'resolution-{verb}-{adjective.replace(" ", "-")}',
                      'before': f'{verb} a {adjective} resolution',
                      'after': f'{replacement} towards a resolution',
                      'reason': 'Direct assurance of resolution changed to work towards resolution',
                      'kind': 'result'})
for subject in ['we will', "we'll", 'and will']:
    for personal in ['personal ', '']:
        for suffix in [' and protection', '']:
            RULES.append({'id': f'privacy-{len(RULES)}',
                'before': f'{subject} handle your {personal}information with the utmost care{suffix}',
                'after': ('and' if subject == 'and will' else 'we') + ' recommend sharing only the necessary information through an official support channel',
                'reason': 'Replace unsupported privacy handling assurance with a channel recommendation',
                'kind': 'privacy'})
for before in ['ensure that your subscription is promptly canceled, and you will no longer receive our newsletter',
               "ensure that your subscription is canceled promptly and that you won't receive any further newsletters",
               'ensure that your subscription is promptly canceled',
               'ensure that your subscription is canceled promptly',
               'ensure your subscription is canceled promptly']:
    RULES.append({'id': f'cancel-{len(RULES)}', 'before': before,
                  'after': 'help you submit your subscription cancellation request for review',
                  'reason': 'Assist submission for review without promising cancellation', 'kind': 'result'})


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rule_hash():
    return digest(json_bytes({'version': VERSION, 'rules': RULES, 'exclude': EXCLUDE}))


def sentence_bounds(text, start):
    # Only whitespace after sentence punctuation/newline separates the context.
    separators = list(re.finditer(r'(?<=[.!?])\s+|\n+', text))
    left = max((m.end() for m in separators if m.end() <= start), default=0)
    right = min((m.start() for m in separators if m.start() > start), default=len(text))
    return left, right


def revise_response(text):
    """Return exact response and nonoverlapping edits; never touch other fields."""
    edits, skipped = [], []
    for rule in RULES:
        ending = r'(?=\s*[,.;!?]|\s*$)' if rule['kind'] == 'privacy' else r'(?!\w)'
        pattern = r'(?<!\w)' + re.escape(rule['before']) + ending
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if any(match.start() < e['end'] and match.end() > e['start'] for e in edits):
                continue
            left, right = sentence_bounds(text, match.start())
            context = text[left:right]
            if re.search(EXCLUDE, context, re.IGNORECASE):
                skipped.append({'rule_id': rule['id'], 'context': context, 'reason': 'service-intent qualifier'})
                continue
            replacement = rule['after']
            if match.group()[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            edits.append({'rule_id': rule['id'], 'start': match.start(), 'end': match.end(),
                          'before': match.group(), 'after': replacement, 'context': context,
                          'reason': rule['reason']})
    edits.sort(key=lambda e: e['start'])
    revised = text
    for edit in reversed(edits):
        revised = revised[:edit['start']] + edit['after'] + revised[edit['end']:]
    return revised, edits, skipped


def derive(rows):
    output, changes, skipped = [], [], []
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    for row in rows:
        text, edits, exclusions = revise_response(row['response'])
        output.append({**row, 'response': text})
        if edits:
            changes.append({'id': row['id'], 'intent': row['intent'],
                'before_response': row['response'], 'after_response': text, 'edits': edits})
        skipped.extend({'id': row['id'], 'intent': row['intent'], **e} for e in exclusions)
    return output, changes, skipped


def rows_bytes(rows):
    return ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')


def freeze_revision(directory=OUT):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Revision directory already exists; refuse overwrite')
    config = json.loads(Path('artifacts/tokenizer/bitext-bpe-v1/config.json').read_text(encoding='utf-8'))
    if file_hash(SOURCE) != config['split_sha256']['train.jsonl']:
        raise ValueError('Frozen source train mismatch')
    rows = list(read_rows(SOURCE))
    output, changes, skipped = derive(rows)
    if not changes:
        raise ValueError('No confirmed edits')
    tokenizer, token_config = load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    length_checks = []
    changed_ids = {c['id'] for c in changes}
    for old, new in zip(rows, output):
        if {k: v for k, v in old.items() if k != 'response'} != {k: v for k, v in new.items() if k != 'response'}:
            raise ValueError('Nonresponse field changed')
        if old['id'] in changed_ids:
            lengths = [len(encode_parts(tokenizer, r['instruction'], r['response'])[2]) for r in (old, new)]
            example = prepare_example(tokenizer, new['instruction'], new['response'], token_config['max_length'])
            if example['supervised_tokens'] < 1:
                raise ValueError('No supervised token after intervention')
            length_checks.append((lengths[0], lengths[1]))
    # Fixed controls selection is independent of content and model outcomes.
    rank = lambda r: digest(('target-controls-v1:42:' + r['id']).encode())
    by_rule = {rule['id']: [] for rule in RULES}
    for change in changes:
        for rule in {e['rule_id'] for e in change['edits']}:
            by_rule[rule].append(change)
    controls = {'selection': 'lowest SHA256(target-controls-v1:42:ID), up to5 per active rule',
        'positive_ids': {key: [r['id'] for r in sorted(value, key=rank)[:5]] for key, value in by_rule.items() if value},
        'excluded_service_intent': sorted(skipped, key=rank)[:20],
        'unchanged_ids': [r['id'] for r in sorted([r for r, o in zip(rows, output) if r == o], key=rank)[:20]]}
    payloads = {'train.jsonl': rows_bytes(output), 'changes.jsonl': rows_bytes(changes),
                'controls.json': json_bytes(controls), 'rules.json': json_bytes({'rules': RULES, 'exclude': EXCLUDE})}
    manifest = {'version': VERSION, 'frozen': True, 'source_train_path': str(SOURCE),
        'source_train_sha256': file_hash(SOURCE), 'rules_sha256': rule_hash(),
        'files_sha256': {p: digest(b) for p, b in payloads.items()},
        'dependencies_sha256': {p: file_hash(p) for p in DEPENDENCIES},
        'train_samples': len(rows), 'changed_samples': len(changes),
        'changed_fraction': len(changes)/len(rows), 'edits': sum(len(c['edits']) for c in changes),
        'rule_counts': dict(Counter(e['rule_id'] for c in changes for e in c['edits'])),
        'intent_counts': dict(Counter(c['intent'] for c in changes)),
        'excluded_qualified_candidates': len(skipped),
        'changed_length_checks': {'max_length': token_config['max_length'],
            'old_max_full_tokens': max(x[0] for x in length_checks),
            'new_max_full_tokens': max(x[1] for x in length_checks),
            'old_truncated': sum(x[0] > token_config['max_length'] for x in length_checks),
            'new_truncated': sum(x[1] > token_config['max_length'] for x in length_checks),
            'zero_supervision': 0, 'scope': 'changed rows only'},
        'review_method': 'Conservative confirmed exact-span rules with sentence-context exclusion; not independent manual scoring of every row',
        'created_date': '2026-10-05'}
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        (directory / name).write_bytes(content)
    # Write last, so an interrupted construction cannot look frozen.
    (directory / 'manifest.json').write_bytes(json_bytes(manifest))
    load_revision(directory / 'manifest.json', rows)
    return manifest


def load_revision(manifest_path, source_rows):
    """Fail closed: validate source, dependencies, exact replay and edit ledger."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION or manifest.get('frozen') is not True:
        raise ValueError('Train revision is not frozen or has wrong version')
    if manifest.get('source_train_sha256') != file_hash(SOURCE) or manifest.get('rules_sha256') != rule_hash():
        raise ValueError('Train revision source/rules mismatch')
    if set(manifest.get('files_sha256', {})) != {'train.jsonl', 'changes.jsonl', 'controls.json', 'rules.json'}:
        raise ValueError('Incomplete train revision file hashes')
    if set(manifest.get('dependencies_sha256', {})) != set(DEPENDENCIES):
        raise ValueError('Incomplete train revision dependency hashes')
    for name, expected in manifest['files_sha256'].items():
        if Path(name).name != name or file_hash(manifest_path.parent / name) != expected:
            raise ValueError('Train revision file hash mismatch: ' + name)
    for name, expected in manifest['dependencies_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError('Train revision dependency mismatch: ' + name)
    expected_rows, expected_changes, _ = derive(source_rows)
    derived_rows = list(read_rows(manifest_path.parent / 'train.jsonl'))
    ledger = [json.loads(s) for s in (manifest_path.parent / 'changes.jsonl').read_text(encoding='utf-8').splitlines()]
    if derived_rows != expected_rows or ledger != expected_changes:
        raise ValueError('Train revision does not match exact approved replay')
    if len(source_rows) != manifest['train_samples'] or len(ledger) != manifest['changed_samples']:
        raise ValueError('Train revision count mismatch')
    if manifest['edits'] != sum(len(c['edits']) for c in ledger):
        raise ValueError('Train revision edit count mismatch')
    return derived_rows, {'manifest_path': str(manifest_path), 'manifest_sha256': file_hash(manifest_path),
        'source_train_sha256': manifest['source_train_sha256'],
        'actual_train_sha256': file_hash(manifest_path.parent / 'train.jsonl'),
        'version': VERSION, 'changed_samples': manifest['changed_samples'], 'edits': manifest['edits']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    args = parser.parse_args()
    if not args.freeze:
        parser.error('Use --freeze to create a new immutable revision')
    manifest = freeze_revision(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
