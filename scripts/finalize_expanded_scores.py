"""Assemble explicit AI reviews; never infers scores from keywords."""
import json
from pathlib import Path
from my_project_1.expanded_dev import OUT, ROOT, RUNS, read, save
from my_project_1.tokenization import file_hash

KEYS=['intent_score','helpfulness_score','fluency_score','repetition','unsupported_claim']

def metric(rows):
    return {'n':len(rows),**{k:sum(r[k] for r in rows)/len(rows) for k in KEYS[:3]},
       **{k:sum(r[k] for r in rows) for k in KEYS[3:]}}

def main():
    material={r['blind_id']:r for r in read(OUT/'blind-material.json')}
    mapping={r['blind_id']:r for r in read(OUT/'blind-mapping.json')}
    frozen=read(OUT/'review-freeze.json')
    for p,h in frozen['dependencies_sha256'].items():assert file_hash(p)==h
    assert file_hash(OUT/'blind-material.json')==frozen['blind_material_sha256']
    assert file_hash(OUT/'blind-mapping.json')==frozen['blind_mapping_sha256']
    decisions=[]
    for p in sorted(Path('.tools/dev-sheet-builder').glob('expanded-decisions-*.json')):decisions.extend(read(p))
    assert len(decisions)==324 and {r[0] for r in decisions}==set(material)
    ratings=[]
    for blind,scores,quotes,notes in decisions:
        assert len(scores)==5 and all(type(v)is int and 0<=v<=(2 if i<3 else 1) for i,v in enumerate(scores))
        for q in quotes:
            if q not in material[blind]['generated_response']:raise ValueError(blind+' bad quote: '+q)
        ratings.append({**mapping[blind],'cohort':'new108',**dict(zip(KEYS,scores)),
            'evidence':quotes,'notes':notes,'reviewer':'AI model-hidden review, risk v1.1 / other v1.0'})
    for inherited in read(OUT/'inherited-original27.json'):
        r=inherited['rating']
        ratings.append({'model':inherited['model'],'id':inherited['id'],'cohort':'original27',
            **{k:r[k] for k in KEYS},'evidence':r.get('evidence',[]),'notes':r['notes'],
            'reviewer':r['reviewer'],'inherited_from':inherited['source']})
    summaries={m:{c:metric([r for r in ratings if r['model']==m and (c=='all135' or r['cohort']==c)])
           for c in ('original27','new108','all135')} for m in RUNS}
    selection=read(ROOT/'selection.json')['samples'];intents={r['id']:r['intent'] for r in selection}
    by={(r['model'],r['id']):r for r in ratings}
    pairs=[]
    for a,b in [('second','third'),('third','fourth')]:
        for item in selection:
            old,new=by[a,item['id']],by[b,item['id']]
            pairs.append({'from':a,'to':b,'id':item['id'],'intent':item['intent'],'cohort':item['cohort'],
                 'delta':{k:new[k]-old[k] for k in KEYS}})
    counts={a+'→'+b:{c:{k:{'improved':sum((p['delta'][k]>0 if i<3 else p['delta'][k]<0) for p in subset),
             'unchanged':sum(p['delta'][k]==0 for p in subset),
             'worsened':sum((p['delta'][k]<0 if i<3 else p['delta'][k]>0) for p in subset)} for i,k in enumerate(KEYS)}
          for c in ('original27','new108','all135')
          for subset in [[p for p in pairs if p['from']==a and p['to']==b and (c=='all135' or p['cohort']==c)]]}
         for a,b in [('second','third'),('third','fourth')]}
    per_intent={m:{intent:metric([r for r in ratings if r['model']==m and intents[r['id']]==intent])
         for intent in sorted(set(intents.values()))} for m in RUNS}
    payload={'review_date':'2026-10-05','rubric_risk':'customer-support-risk-review-v1.1',
       'rubric_other':'customer-support-review-v1.0','method':'AI explicit full text new324 reviews, model labels hidden and mixed; style can reveal model, not strict blind or independent human. Original81 scores inherited only after exact text/token agreement. No merchant evidence. Dev diagnostic not final test.',
       'selection_sha256':file_hash(ROOT/'selection.json'),'review_freeze_sha256':file_hash(OUT/'review-freeze.json'),
       'decision_sha256':{str(p):file_hash(p) for p in sorted(Path('.tools/dev-sheet-builder').glob('expanded-decisions-*.json'))},
       'ratings':ratings,'metrics':summaries,'paired_counts':counts,'pairs':pairs,'per_intent':per_intent}
    save(OUT/'ai-expert-ratings.json',payload)
    print(json.dumps({'metrics':summaries,'third_fourth_new108':counts['third→fourth']['new108']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
