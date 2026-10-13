"""Verify paired generations and freeze model-hidden new108 review material."""
import json
from pathlib import Path
from my_project_1.expanded_dev import OUT, ROOT, RUNS, OPTIONS, read, save, digest
from my_project_1.tokenization import file_hash

HIST={
 'second':('outputs/dev-risk-review-20261004-v1-1/experiment-2-ratings.json','runs/20261004T095337-full-666ad47f70d1422485fede4e65c16b1f/dev-checks/20261004T115457-6dd058d531f146e29eea8302929c588f/replies.jsonl'),
 'third':('outputs/dev-check-20261005-third-098e916c/ai-expert-ratings.json','runs/20261005T014129-full-c492479de8f8407c8c437a435e7e6bd9/dev-checks/20261005T072340-098e916c77894c8794783bacce6cf37e/replies.jsonl'),
 'fourth':('outputs/dev-check-20261005-fourth-9bc759a8/ai-expert-ratings.json','runs/20261005T093449-full-6181db843d8543ba96a9d83e60a81857/dev-checks/20261005T114804-9bc759a8f78e4b5985ae90b48389b7f1/replies.jsonl')}

def rows(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines()]

def main():
    materials=[]; inherited=[]; hashes={}; previous=None; previous_config=None
    for model in RUNS:
        records=rows(OUT/model/'replies.jsonl');config=read(OUT/model/'config.json');summary=read(OUT/model/'summary.json')
        assert len(records)==135 and len({r['id'] for r in records})==135
        assert summary['replies_sha256']==file_hash(OUT/model/'replies.jsonl')
        metadata=[[r[k] for k in ('id','intent','instruction','reference_response','cohort')] for r in records]
        if previous is not None:assert metadata==previous
        previous=metadata
        settings={k:config[k] for k in [*OPTIONS,'model_config','selection_sha256']}
        if previous_config is not None:assert settings==previous_config
        previous_config=settings
        assert config['checkpoint_sha256']==read(ROOT/'freeze.json')['models'][model]['sha256']
        historic=read(HIST[model][0])['ratings'];old={r['id']:r for r in rows(HIST[model][1])}
        for r in records:
            if r['cohort']=='original27':
                assert r['generated_response']==old[r['id']]['generated_response']
                assert r['generated_token_ids']==old[r['id']]['generated_token_ids']
                rating=next(p for p in historic if p['id']==r['id'])
                inherited.append({'model':model,'id':r['id'],'rating':rating,'source':HIST[model][0]})
            else:
                materials.append({'model':model,'id':r['id'],'instruction':r['instruction'],
                      'generated_response':r['generated_response'],'stop_reason':r['stop_reason']})
        hashes[str(OUT/model/'replies.jsonl')]=file_hash(OUT/model/'replies.jsonl')
        hashes[HIST[model][0]]=file_hash(HIST[model][0])
    materials.sort(key=lambda r:(digest('expanded-blind-v1:42:'+r['model']+':'+r['id']),r['id']))
    mapping=[{'blind_id':f'B{i:04d}','model':r['model'],'id':r['id']} for i,r in enumerate(materials,1)]
    hidden=[{'blind_id':f'B{i:04d}',**{k:r[k] for k in ('instruction','generated_response','stop_reason')}} for i,r in enumerate(materials,1)]
    save(OUT/'blind-material.json',hidden);save(OUT/'blind-mapping.json',mapping);save(OUT/'inherited-original27.json',inherited)
    save(OUT/'review-freeze.json',{'dependencies_sha256':hashes,'blind_material_sha256':file_hash(OUT/'blind-material.json'),
         'blind_mapping_sha256':file_hash(OUT/'blind-mapping.json'),'selection_sha256':file_hash(ROOT/'selection.json')})
    print('Frozen:',len(hidden),'new model-hidden replies; inherited original:',len(inherited))

if __name__=='__main__':main()
