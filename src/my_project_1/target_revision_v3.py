"""Frozen refund execution-only target revision layered on immutable v2."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from .target_revision import json_bytes, rows_bytes, digest
from .target_revision_v2 import derive, load_revision as load_v2
from .tokenization import read_rows, file_hash, load_frozen, prepare_example

VERSION='refund-execution-target-v3'
SOURCE=Path('data/splits/bitext-v1/train.jsonl')
BASE=Path('data/derived/result-privacy-v2/manifest.json')
REVIEW=Path('outputs/target-revision-v3-review-20261005')
OUT=Path('data/derived/refund-execution-v3')
DEPS=['docs/train-target-revision-v3.md','src/my_project_1/target_revision_v3.py',
    'src/my_project_1/training_fifth.py','src/my_project_1/training.py',
    'scripts/propose_target_v3.py','scripts/review_target_v3.py',
    'scripts/approve_target_v3.py','scripts/finalize_target_v3_review.py',
    'configs/training-fifth.json','configs/model.json','configs/generation.json','configs/dev-check.json',
    'data/splits/bitext-v1/dev.jsonl','artifacts/tokenizer/bitext-bpe-v1/config.json',
    'artifacts/tokenizer/bitext-bpe-v1/tokenizer.json']+[str(REVIEW/n) for n in ['proposals.json','triage.json','approval.json','approval-final.json','approval-frozen.json','approval-freeze-ready.json']]

def check_settings():
    old=json.loads(Path('configs/training-fourth.json').read_text(encoding='utf-8'))
    new=json.loads(Path('configs/training-fifth.json').read_text(encoding='utf-8'))
    if new['train_revision']!=str(OUT/'manifest.json').replace('\\','/'):
        raise ValueError('Unexpected fifth revision path')
    if {k:v for k,v in old.items() if k!='train_revision'}!={k:v for k,v in new.items() if k!='train_revision'}:
        raise ValueError('Non-factor training settings changed')

def check_rows(base,output):
    if len(base)!=len(output):raise ValueError('Row count changed')
    entity=lambda s:Counter(re.findall(r'\{\{[^{}]+\}\}',s))
    for a,b in zip(base,output):
        if {k:v for k,v in a.items() if k!='response'}!={k:v for k,v in b.items() if k!='response'}:
            raise ValueError('Nonresponse mutation')
        if entity(a['response'])!=entity(b['response']):raise ValueError('Entity mutation '+a['id'])

def review():
    p=json.loads((REVIEW/'proposals.json').read_text(encoding='utf-8'))
    a=json.loads((REVIEW/'approval-freeze-ready.json').read_text(encoding='utf-8'))
    if a['proposals_sha256']!=file_hash(REVIEW/'proposals.json') or a['initial_approval_sha256']!=file_hash(REVIEW/'approval.json'):
        raise ValueError('Approval provenance mismatch')
    if p['baseline_sha256']!=file_hash(BASE):raise ValueError('Proposal baseline changed')
    rules=a['rules']; decisions={r['id']:r for r in a['reviews']}
    if len({r['id'] for r in rules})!=len(rules):raise ValueError('Duplicate approval')
    for r in rules:
        d=decisions[r['id']]
        if not d['approved'] or d['before']!=r['before'] or d['after']!=r['after'] or r['kind']!='refund-execution-whole-sentence':
            raise ValueError('Unapproved rule')
        if not r['after'] or r['after']==r['before']:raise ValueError('Invalid replacement')
    return p,a,rules

def freeze():
    if OUT.exists():raise ValueError('V3 exists; refuse overwrite')
    check_settings();source=list(read_rows(SOURCE));base,_=load_v2(BASE,source)
    p,a,rules=review();output,changes=derive(base,rules);check_rows(base,output)
    tokenizer,config=load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    changed_ids={c['id'] for c in changes}
    for r in output:
        if r['id'] in changed_ids and prepare_example(tokenizer,r['instruction'],r['response'],config['max_length'])['supervised_tokens']<1:
            raise ValueError('Zero supervision')
    payload={'train.jsonl':rows_bytes(output),'changes.jsonl':rows_bytes(changes),'rules.json':json_bytes(rules)}
    m={'version':VERSION,'frozen':True,'source_train_sha256':file_hash(SOURCE),
        'baseline_manifest_sha256':file_hash(BASE),'baseline_train_sha256':file_hash(BASE.parent/'train.jsonl'),
        'files_sha256':{k:digest(v) for k,v in payload.items()},'dependencies_sha256':{p:file_hash(p) for p in DEPS},
        'train_samples':len(output),'changed_samples':len(changes),'edits':sum(len(c['edits']) for c in changes),
        'approved_rules':len(rules),'changed_intents':dict(Counter(c['intent'] for c in changes)),
        'review_method':a['method'],'initial_nomination_protocol_sha256':p['protocol_sha256'],
        'final_protocol_sha256':file_hash(DEPS[0]),'protocol_amendment':'Before freeze clarify sentence review and lexical screening separately; no claim every excluded response reviewed.',
        'zero_supervision':0,'entity_checks':True,'training_settings_equal_except_revision':True}
    OUT.mkdir(parents=True,exist_ok=False)
    for n,b in payload.items():(OUT/n).write_bytes(b)
    (OUT/'manifest.json').write_bytes(json_bytes(m))
    load_revision(OUT/'manifest.json',source)
    return m

def load_revision(manifest_path,source_rows):
    path=Path(manifest_path);m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('version')!=VERSION or m.get('frozen')is not True:raise ValueError('Invalid V3 version')
    if m['source_train_sha256']!=file_hash(SOURCE) or m['baseline_manifest_sha256']!=file_hash(BASE) or m['baseline_train_sha256']!=file_hash(BASE.parent/'train.jsonl'):
        raise ValueError('V3 source/baseline mismatch')
    if set(m['dependencies_sha256'])!=set(DEPS) or set(m['files_sha256'])!={'train.jsonl','changes.jsonl','rules.json'}:
        raise ValueError('Incomplete V3 freeze')
    for p,h in m['dependencies_sha256'].items():
        if file_hash(p)!=h:raise ValueError('V3 dependency mismatch '+p)
    for n,h in m['files_sha256'].items():
        if file_hash(path.parent/n)!=h:raise ValueError('V3 artifact mismatch '+n)
    check_settings();base,_=load_v2(BASE,source_rows);_,a,rules=review()
    if json.loads((path.parent/'rules.json').read_text(encoding='utf-8'))!=rules:raise ValueError('Rules mismatch')
    output,changes=derive(base,rules);check_rows(base,output)
    actual=list(read_rows(path.parent/'train.jsonl'))
    ledger=[json.loads(l) for l in (path.parent/'changes.jsonl').read_text(encoding='utf-8').splitlines()]
    if actual!=output or ledger!=changes:raise ValueError('V3 exact replay mismatch')
    if len(output)!=m['train_samples'] or len(changes)!=m['changed_samples'] or sum(len(c['edits']) for c in changes)!=m['edits']:
        raise ValueError('V3 counts mismatch')
    return output,{'manifest_path':str(path),'manifest_sha256':file_hash(path),'source_train_sha256':m['source_train_sha256'],
        'baseline_manifest_sha256':m['baseline_manifest_sha256'],'actual_train_sha256':file_hash(path.parent/'train.jsonl'),
        'version':VERSION,'changed_samples':m['changed_samples'],'edits':m['edits']}

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--verify',action='store_true');a=p.parse_args()
    result=freeze() if a.freeze else load_revision(OUT/'manifest.json',list(read_rows(SOURCE)))[1]
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
