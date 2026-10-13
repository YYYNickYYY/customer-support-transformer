"""Validate explicit full-text judgments and paired arithmetic, never auto-score."""
from collections import Counter
import json
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash
from my_project_1.training_sixth import load_plan

OUT = Path('outputs/dev-expanded-seventh-20261007')
KEYS = ('intent_score','helpfulness_score','fluency_score','repetition','unsupported_claim')


def metrics(rows):
    return {'n':len(rows), **{k:sum(r[k] for r in rows)/len(rows) if i<3 else sum(r[k] for r in rows)
                              for i,k in enumerate(KEYS)}}


def counts(pairs, keys=KEYS):
    return {k:{'improved':sum(p[k]>0 if j<3 else p[k]<0 for p in pairs),
               'unchanged':sum(p[k]==0 for p in pairs),
               'worsened':sum(p[k]<0 if j<3 else p[k]>0 for p in pairs)} for j,k in enumerate(keys)}


def main():
    plan = load_plan()
    for p,h in read(OUT/'freeze.json')['dependencies_sha256'].items():
        assert file_hash(p)==h,p
    raw=[json.loads(x) for x in (OUT/'replies.jsonl').read_text(encoding='utf-8').splitlines()]
    old=[json.loads(x) for x in Path(plan['baseline_replies']).read_text(encoding='utf-8').splitlines()]
    baseline={r['id']:r for r in read(plan['baseline_ratings'])['ratings'] if r['model']=='fifth'}
    decisions=read(OUT/'decisions.json')
    entity=read(OUT/'entity-decisions.json')
    assert entity['reviewed_cases']==list(range(1,136))
    oldentity={i:(qs,note) for i,qs,note in entity['fifth_positive_decisions']}
    assert len(oldentity)==len(entity['fifth_positive_decisions'])
    assert len(raw)==len(old)==len(baseline)==len(decisions)==135
    assert [d[0] for d in decisions]==list(range(1,136))
    ratings,pairs,entity_pairs=[],[],[]
    for i,(r,prior,d) in enumerate(zip(raw,old,decisions),1):
        assert all(r[k]==prior[k] for k in ('id','instruction','intent','cohort','reference_response'))
        _,scores,quotes,note,newentity=d
        assert len(scores)==5 and all(type(v) is int and 0<=v<=(2 if j<3 else 1) for j,v in enumerate(scores))
        assert note and quotes and all(q and q in r['generated_response'] for q in quotes),i
        assert type(newentity) is int and newentity in (0,1)
        if i in oldentity:
            assert all(q and q in prior['generated_response'] for q in oldentity[i][0]),('entity',i)
        rating={k:r[k] for k in ('id','intent','cohort','stop_reason')}
        rating.update(dict(zip(KEYS,scores)),evidence=quotes,notes=note,entity_issue=newentity,
                      model='seventh',reviewer='AI full-text nonblind; known history/loss, no keyword scoring',
                      risk_rubric='customer-support-risk-review-v1.1',other_rubric='customer-support-review-v1.0')
        ratings.append(rating)
        pairs.append({'id':r['id'],'intent':r['intent'],'cohort':r['cohort'],
                      **{k:rating[k]-baseline[r['id']][k] for k in KEYS}})
        entity_pairs.append({'id':r['id'],'case':i,'fifth':int(i in oldentity),'seventh':newentity,
                             'fifth_evidence':oldentity.get(i,([], 'Full text reviewed; no entity defect under stated scope'))[0],
                             'fifth_notes':oldentity.get(i,([], 'Full text reviewed; no entity defect under stated scope'))[1]})
    allrows={'fifth':list(baseline.values()),'seventh':ratings}
    sources=[str(OUT/n) for n in ('replies.jsonl','decisions.json','entity-decisions.json','freeze.json','summary.json','config.json')]
    sources += [plan['baseline_ratings'],plan['baseline_replies'],str(Path(__file__)),
                'docs/seventh-depth-protocol-v1.md','docs/evaluation-rubric-v1.md','docs/evaluation-risk-rubric-v1.1.md']
    payload={'method':'All fixed 135; AI nonblind full-text. Fifth frozen five scores unchanged; both entity annotations newly reviewed. No total score or generalization claim.',
             'ratings':ratings,'metrics':{m:metrics(rs) for m,rs in allrows.items()},'pairs':pairs,'paired_counts':counts(pairs),
             'entity_pairs':entity_pairs,'entity_issues':{m:sum(p[m] for p in entity_pairs) for m in ('fifth','seventh')},
             'entity_counts':{'improved':sum(p['seventh']<p['fifth'] for p in entity_pairs),
                              'unchanged':sum(p['seventh']==p['fifth'] for p in entity_pairs),
                              'worsened':sum(p['seventh']>p['fifth'] for p in entity_pairs)},
             'per_cohort':{m:{c:metrics([r for r in rs if r['cohort']==c]) for c in ('original27','new108')} for m,rs in allrows.items()},
             'per_intent':{m:{intent:metrics([r for r in rs if r['intent']==intent]) for intent in sorted({r['intent'] for r in rs})} for m,rs in allrows.items()},
             'stop_counts':{m:dict(Counter(r['stop_reason'] for r in rs)) for m,rs in [('fifth',old),('seventh',raw)]},
             'best_dev_loss':{'fifth':plan['baseline_best_dev_loss'],'seventh':read(OUT/'summary.json')['training_summary']['best_dev_loss']},
             'sources_sha256':{p:file_hash(p) for p in sources}}
    save(OUT/'ai-expert-ratings.json',payload)
    save(OUT/'rating-freeze.json',{'complete':True,'ratings_sha256':file_hash(OUT/'ai-expert-ratings.json'),'sources_sha256':payload['sources_sha256']})
    print(json.dumps({k:payload[k] for k in ('metrics','paired_counts','entity_issues','entity_counts','per_cohort','stop_counts')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
