"""Freeze shuffled fourth/fifth complete text for repetition-only review."""
import hashlib
import json
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT=Path('outputs/repetition-review-20261006-v1')
def main():
    paths={m:Path(p) for m,p in [('fourth','outputs/dev-expanded-20261005-v1/fourth/replies.jsonl'),('fifth','outputs/dev-expanded-fifth-20261005/replies.jsonl')]}
    rows=[{'model':m,**json.loads(line)} for m,p in paths.items() for line in p.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==270 and len({(r['model'],r['id']) for r in rows})==270
    material=[];mapping=[]
    for i,r in enumerate(sorted(rows,key=lambda r:hashlib.sha256(('repeat-review-v1:42:'+r['model']+':'+r['id']).encode()).hexdigest()),1):
        key=f'P{i:03}'
        material.append({'review_id':key,'instruction':r['instruction'],'response':r['generated_response']})
        mapping.append({'review_id':key,'model':r['model'],'id':r['id'],'intent':r['intent'],'cohort':r['cohort'],'stop_reason':r['stop_reason']})
    save(OUT/'material.json',material);save(OUT/'mapping.json',mapping)
    deps=[*paths.values(),Path('outputs/dev-expanded-fifth-20261005/ai-expert-ratings.json'),Path('docs/evaluation-rubric-v1.md'),Path(__file__)]
    save(OUT/'freeze.json',{'version':'repetition-review-v1','scope':'Only repetition, all fourth/fifth same135; hidden labels mixed full text; AI known context not strict blind; no other scores altered',
        'dependencies_sha256':{str(p):file_hash(p) for p in deps},'material_sha256':file_hash(OUT/'material.json'),'mapping_sha256':file_hash(OUT/'mapping.json')})
    print('Frozen270 full texts, repetition v1.0 only')
if __name__=='__main__':main()
