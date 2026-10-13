"""Pre-launch refinement of v3 replacement completeness; v3 draft preserved."""
import argparse
from collections import Counter
import json
from pathlib import Path
from . import target_revision_v3 as draft
from .target_revision_v2 import derive, load_revision as load_v2
from .target_revision import json_bytes, rows_bytes, digest
from .tokenization import read_rows, file_hash, load_frozen, prepare_example

OUT=Path('data/derived/refund-execution-v3.1')
VERSION='refund-execution-target-v3.1'
DEPS=draft.DEPS+['data/derived/refund-execution-v3/manifest.json',
    'src/my_project_1/target_revision_v31.py','src/my_project_1/training_fifth_final.py',
    'configs/training-fifth-final.json']
# Only replacement wording of previously approved refund-execution fragments changes.
OVERRIDES={
 '31ce382a4ab10ec1':'If you suspect unauthorized charges, we can help you review the transaction and any refund request through official support.',
 '6eee66b0a5352e29':'We take unauthorized charges seriously and can help you review the transaction and any refund request through official support.',
 '199dc10c591ed1be':'We take unauthorized charges seriously and can help you review the transaction and any refund request through official support.',
 'a3f99fecdbfa6abe':'We take unauthorized charges seriously and can help you review the transaction and any refund request through official support.',
 'ab28cd729d5c0982':'We can help you review the duplicate charge and any refund request through official support.',
 '9f7ab77738b5558d':'We can help you review the duplicate charge and any refund request through official support.',
 'b8b3dd8d08de718e':'We can help you review the duplicate charge and any refund request through official support.',
 'a9b6aed92fc9b67b':'We can help you review the duplicate charge and any refund request through official support.',
 '4b683cc037716cc0':'We can help you review the duplicate charge and any refund request through official support.',
 '3f683d8fc51ddb2d':'We can help you review the duplicate charge and any refund request through official support.',
}

def check_settings():
    draft.check_settings()
    old=json.loads(Path('configs/training-fourth.json').read_text(encoding='utf-8'))
    new=json.loads(Path('configs/training-fifth-final.json').read_text(encoding='utf-8'))
    if new['train_revision']!=OUT.as_posix()+'/manifest.json' or {k:v for k,v in old.items() if k!='train_revision'}!={k:v for k,v in new.items() if k!='train_revision'}:
        raise ValueError('Final fifth settings changed')

def rules_and_base(source):
    draft.load_revision(draft.OUT/'manifest.json',source)
    base,_=load_v2(draft.BASE,source);_,_,rules=draft.review()
    if not set(OVERRIDES)<={r['id'] for r in rules}:raise ValueError('Unknown wording override')
    return base,[{**r,'after':OVERRIDES.get(r['id'],r['after'])} for r in rules]

def validate(base,output,changes):
    draft.check_rows(base,output)
    for c in changes:
        after=[e['after'] for e in c['edits']]
        if len(after)!=len(set(after)):raise ValueError('Duplicate rewritten sentence '+c['id'])
        if c['after_response'].count('**')!=c['before_response'].count('**'):raise ValueError('Heading mutation')

def freeze():
    if OUT.exists():raise ValueError('V3.1 exists; refuse overwrite')
    check_settings();source=list(read_rows(draft.SOURCE));base,rules=rules_and_base(source)
    output,changes=derive(base,rules);validate(base,output,changes)
    tokenizer,config=load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    ids={r['id'] for r in changes}
    for r in output:
        if r['id'] in ids and prepare_example(tokenizer,r['instruction'],r['response'],config['max_length'])['supervised_tokens']<1:raise ValueError('Zero supervision')
    payload={'train.jsonl':rows_bytes(output),'changes.jsonl':rows_bytes(changes),'rules.json':json_bytes(rules)}
    m={'version':VERSION,'frozen':True,'baseline_manifest_sha256':file_hash(draft.BASE),
        'baseline_train_sha256':file_hash(draft.BASE.parent/'train.jsonl'),'source_train_sha256':file_hash(draft.SOURCE),
        'files_sha256':{n:digest(b) for n,b in payload.items()},'dependencies_sha256':{p:file_hash(p) for p in DEPS},
        'train_samples':len(output),'changed_samples':len(changes),'edits':sum(len(c['edits']) for c in changes),
        'approved_rules':len(rules),'changed_intents':dict(Counter(c['intent'] for c in changes)),
        'replacement_refinement':OVERRIDES,'reason':'six responses had duplicate generic guidance in pre-launch v3; ten exact approved sentence replacements now preserve unauthorized/duplicate-charge context. Same factor and same approved fragments.',
        'entity_checks':True,'no_duplicate_rewritten_sentences':True,'zero_supervision':0,'same_training_settings_except_revision':True}
    OUT.mkdir(parents=True,exist_ok=False)
    for n,b in payload.items():(OUT/n).write_bytes(b)
    (OUT/'manifest.json').write_bytes(json_bytes(m))
    load_revision(OUT/'manifest.json',source)
    return m

def load_revision(path,source):
    path=Path(path);m=json.loads(path.read_text(encoding='utf-8'))
    if m.get('version')!=VERSION or m.get('frozen')is not True or set(m['dependencies_sha256'])!=set(DEPS) or set(m['files_sha256'])!={'train.jsonl','changes.jsonl','rules.json'}:raise ValueError('Incomplete V3.1 freeze')
    for p,h in m['dependencies_sha256'].items():
        if file_hash(p)!=h:raise ValueError('V3.1 dependency mismatch '+p)
    for n,h in m['files_sha256'].items():
        if file_hash(path.parent/n)!=h:raise ValueError('V3.1 artifact mismatch '+n)
    check_settings();base,rules=rules_and_base(source);output,changes=derive(base,rules);validate(base,output,changes)
    if json.loads((path.parent/'rules.json').read_text(encoding='utf-8'))!=rules or list(read_rows(path.parent/'train.jsonl'))!=output or [json.loads(l) for l in (path.parent/'changes.jsonl').read_text(encoding='utf-8').splitlines()]!=changes:raise ValueError('V3.1 exact replay mismatch')
    if m['train_samples']!=len(output) or m['changed_samples']!=len(changes) or m['edits']!=sum(len(c['edits']) for c in changes):raise ValueError('V3.1 counts mismatch')
    return output,{'manifest_path':str(path),'manifest_sha256':file_hash(path),'source_train_sha256':m['source_train_sha256'],'baseline_manifest_sha256':m['baseline_manifest_sha256'],'actual_train_sha256':file_hash(path.parent/'train.jsonl'),'version':VERSION,'changed_samples':len(changes),'edits':m['edits']}

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--verify',action='store_true');a=p.parse_args()
    m=freeze() if a.freeze else load_revision(OUT/'manifest.json',list(read_rows(draft.SOURCE)))[1]
    print(json.dumps(m,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
