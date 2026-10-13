"""Freeze full response material, mapping and focused refund audit before scoring."""
import json
from pathlib import Path
from my_project_1.expanded_dev import read, save, digest
from my_project_1.tokenization import file_hash

OUT=Path('outputs/dev-expanded-fifth-20261005')
def main():
    rows=[json.loads(l) for l in (OUT/'replies.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==135 and len({r['id'] for r in rows})==135
    material=[];mapping=[]
    for i,r in enumerate(sorted(rows,key=lambda r:digest('fifth-review-v1:42:'+r['id'])),1):
        key=f'F{i:03}'
        material.append({'review_id':key,'instruction':r['instruction'],'generated_response':r['generated_response'],'stop_reason':r['stop_reason']})
        mapping.append({'review_id':key,'id':r['id'],'intent':r['intent'],'cohort':r['cohort']})
    save(OUT/'review-material.json',material);save(OUT/'review-mapping.json',mapping)
    save(OUT/'review-freeze.json',{'sources_sha256':{str(OUT/n):file_hash(OUT/n) for n in ['replies.jsonl','config.json','freeze.json','review-material.json','review-mapping.json']},
        'method':'Single known fifth model, shuffled full-text AI review without reference/old scores in material. Not blind or independent human. Risk v1.1, other v1.0; five separate metrics, no overall score.',
        'focused_refund_audit':{'intents':['check_refund_policy','get_refund','track_refund'],'n':15,
            'definition':'Supplementary factual tag, not sixth score: definite unauthorized refund execution/payment completion/rectification of billing result. Include issue/process/provide refund and ensure receive, even if necessary/applicable; exclude effort-qualified assistance/request guidance, may/typically policy descriptions, plain query capability or page-feature claims. Read full outputs of each of four models; record quotes and binary mark.',
            'not_population_rate':True}})

if __name__=='__main__':main()
