"""Apply explicit paired boundary calibration to preserved draft; never infer scores."""
import copy
import json
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash
from assemble_seventh_review import OUT, KEYS, metrics, counts


def main():
    draft=read(OUT/'ai-expert-ratings.json')
    for p,h in draft['sources_sha256'].items():
        assert file_hash(p)==h,p
    assert read(OUT/'rating-freeze.json')['ratings_sha256']==file_hash(OUT/'ai-expert-ratings.json')
    result=copy.deepcopy(draft)
    calibration=read(OUT/'calibration.json')
    raw=[json.loads(x) for x in (OUT/'replies.jsonl').read_text(encoding='utf-8').splitlines()]
    changed=set()
    for c in calibration['changes']:
        key=(c['case'],c['field'])
        assert key not in changed and c['field'] in KEYS
        changed.add(key)
        rating=result['ratings'][c['case']-1]
        assert rating[c['field']]==c['from']
        assert all(q and q in raw[c['case']-1]['generated_response'] for q in c['evidence'])
        rating[c['field']]=c['to']
        rating.setdefault('calibration',[]).append(c)
        rating['evidence'] += [q for q in c['evidence'] if q not in rating['evidence']]
        rating['notes'] += '；最终校准：'+c['reason']
        # Old notes remain in immutable draft; explicit final reasons override them.
        result['pairs'][c['case']-1][c['field']] += c['to']-c['from']
    result['metrics']['seventh']=metrics(result['ratings'])
    result['paired_counts']=counts(result['pairs'])
    for c in result['per_cohort']['seventh']:
        result['per_cohort']['seventh'][c]=metrics([r for r in result['ratings'] if r['cohort']==c])
    for intent in result['per_intent']['seventh']:
        result['per_intent']['seventh'][intent]=metrics([r for r in result['ratings'] if r['intent']==intent])
    result['authority']='Final calibrated v1.0/v1.1 review; draft retained for audit, not current comparison'
    result['decision']='recommend_seventh_for_learning_baseline_with_explicit_limitations_not_deployment'
    result['decision_reason']='Lower dev loss, positive paired intention/help and fewer repetition/entity defects; modest average gains including small new108 intention/help gains, new108 risk +1; new refund/subscribe failures persist. No generalization claim. Original defaults unchanged.'
    result['final_sources_sha256']={str(p):file_hash(p) for p in [OUT/'ai-expert-ratings.json',OUT/'rating-freeze.json',OUT/'calibration.json',Path(__file__)]}
    save(OUT/'final-review.json',result)
    print(json.dumps({k:result[k] for k in ('metrics','paired_counts','entity_issues','per_cohort','decision')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
