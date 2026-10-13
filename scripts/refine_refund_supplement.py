"""Refine only approved replacement wording; never activate a training revision."""
import importlib.util
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import read_rows, file_hash, load_frozen, prepare_example

ROOT=Path('outputs/refund-supplement-20261006-v1')
OUT=Path('outputs/refund-supplement-20261006-v1.1')
def main():
    manifest=read(ROOT/'manifest.json')
    for p,h in manifest['sha256'].items():
        if file_hash(p)!=h:raise ValueError('Parent changed '+p)
    spec=importlib.util.spec_from_file_location('supplement',Path('scripts/finalize_refund_supplement.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.verify_material()
    refinement=read(ROOT/'wording-refinement.json');reviews=read(ROOT/'reviews-final.json')
    approved={c['review_id'] for c in reviews if c['approved']}
    if not set(refinement['overrides'])<=approved:raise ValueError('Unknown approved rule')
    for c in reviews:
        if c['review_id'] in refinement['overrides']:
            c['draft_after']=c['after'];c['after']=refinement['overrides'][c['review_id']]
            c['wording_refinement_reason']=refinement['reason']
    base=list(read_rows(helper.BASE));output,changes=helper.replay(base,reviews)
    tok,cfg=load_frozen('artifacts/tokenizer/bitext-bpe-v1');checks=[]
    ids={c['id'] for c in changes}
    for b,a in zip(base,output):
        if a['id'] not in ids:continue
        old=prepare_example(tok,b['instruction'],b['response'],cfg['max_length'])
        new=prepare_example(tok,a['instruction'],a['response'],cfg['max_length'])
        if new['supervised_tokens']<1:raise ValueError('No reply supervision')
        checks.append({'id':a['id'],'before_supervised_tokens':old['supervised_tokens'],
            'after_supervised_tokens':new['supervised_tokens'],'before_truncated':old['truncated'],
            'after_truncated':new['truncated']})
    save(OUT/'reviews-final.json',reviews)
    save(OUT/'approved-rules.json',[c for c in reviews if c['approved']])
    save(OUT/'changes-preview.json',changes);save(OUT/'token-check.json',checks)
    paths=[OUT/n for n in ('reviews-final.json','approved-rules.json','changes-preview.json','token-check.json')]
    paths += [ROOT/'manifest.json',ROOT/'wording-refinement.json',Path(__file__)]
    frozen={**{k:v for k,v in manifest.items() if k!='sha256'},
        'version':'refund-supplement-list-v1.1','parent_manifest_sha256':file_hash(ROOT/'manifest.json'),
        'refined_replacement_rules':len(refinement['overrides']),
        'sha256':{str(p):file_hash(p) for p in paths},
        'dependency_sha256':manifest['sha256']}
    helper.verify_material();save(OUT/'manifest.json',frozen)
    print('Frozen v1.1:',len(changes),'records;',len(refinement['overrides']),'replacement wordings refined; no training')
if __name__=='__main__':main()
