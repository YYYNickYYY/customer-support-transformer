"""Validate explicit AI review decisions; preserve historical scores and sources."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT = Path('outputs/repetition-review-20261006-v1')

def assemble(material, mapping, decisions, old):
    texts = {r['review_id']: r for r in material}
    keys = {r['review_id']: r for r in mapping}
    assert len(texts) == len(keys) == len(material) == len(mapping) == 270
    assert len(decisions) == 270 and len({d[0] for d in decisions}) == 270
    assert {d[0] for d in decisions} == set(texts) == set(keys)
    historical = {(r['model'], r['id']): r for r in old['ratings']}
    result = deepcopy(old)
    updated = {(r['model'], r['id']): r for r in result['ratings']}
    reviews = []
    for key, score, quotes, reason in decisions:
        assert type(score) is int and score in (0, 1)
        assert quotes and reason and all(q and q in texts[key]['response'] for q in quotes), key
        m = keys[key]; identity = (m['model'], m['id'])
        original = historical[identity]
        updated[identity]['repetition'] = score
        reviews.append({**m, 'old_repetition': original['repetition'],
                        'repetition': score, 'evidence': quotes, 'reason': reason,
                        'changed': score != original['repetition']})
    # All fields except repetition remain byte-equivalent as JSON values.
    for k, r in updated.items():
        assert {f:v for f,v in r.items() if f!='repetition'} == {
            f:v for f,v in historical[k].items() if f!='repetition'}
    counts = {}
    for model in ('fourth', 'fifth'):
        counts[model] = {}
        for cohort in ('original27', 'new108', 'all135'):
            subset = [r for r in reviews if r['model']==model and
                      (cohort=='all135' or r['cohort']==cohort)]
            counts[model][cohort] = {'n':len(subset),
                'old_repeat':sum(r['old_repetition'] for r in subset),
                'repeat':sum(r['repetition'] for r in subset),
                'old0_new1':sum(r['old_repetition']==0 and r['repetition']==1 for r in subset),
                'old1_new0':sum(r['old_repetition']==1 and r['repetition']==0 for r in subset)}
    lookup = {(r['model'],r['id']):r for r in reviews}
    ids = {r['id'] for r in reviews if r['model']=='fourth'}
    assert len(ids)==135 and ids=={r['id'] for r in reviews if r['model']=='fifth'}
    pairs = [{'id':k,'intent':lookup['fourth',k]['intent'],
              'cohort':lookup['fourth',k]['cohort'],
              'fourth':lookup['fourth',k]['repetition'],
              'fifth':lookup['fifth',k]['repetition']} for k in sorted(ids)]
    transitions = Counter(f"{r['fourth']}->{r['fifth']}" for r in pairs)
    # Old aggregate metrics are stale after repetition changes; do not label them current.
    result = {'method':'AI full-text repetition-only v1.0 re-review; labels hidden, not independent blind review',
              'ratings':result['ratings'], 'repetition_metrics':counts,
              'paired_repetition':dict(sorted(transitions.items())), 'pairs':pairs,
              'best_dev_loss_unchanged':old['best_dev_loss'],
              'other_dimensions_unchanged':True,
              'historical_nonrepetition_evidence':True,
              'historical_source':'outputs/dev-expanded-fifth-20261005/ai-expert-ratings.json'}
    return result, reviews

def main():
    freeze=read(OUT/'freeze.json')
    for p,h in freeze['dependencies_sha256'].items():
        assert file_hash(p)==h,p
    for name in ('material','mapping'):
        assert file_hash(OUT/f'{name}.json')==freeze[f'{name}_sha256']
    paths=sorted(OUT.glob('decisions-*.json'))
    decisions=[d for p in paths for d in read(p)]
    result,reviews=assemble(read(OUT/'material.json'), read(OUT/'mapping.json'), decisions,
        read('outputs/dev-expanded-fifth-20261005/ai-expert-ratings.json'))
    save(OUT/'reviews.json',reviews)
    save(OUT/'ai-expert-ratings-repetition-revised.json',result)
    save(OUT/'review-freeze.json',{
        'frozen':True,'complete_reviews':270,'repetition_version':'v1.0',
        'source_material_freeze_sha256':file_hash(OUT/'freeze.json'),
        'sha256':{str(p):file_hash(p) for p in [*paths,OUT/'reviews.json',
            OUT/'ai-expert-ratings-repetition-revised.json',Path(__file__)]}})
    print(result['repetition_metrics']); print(result['paired_repetition'])

if __name__=='__main__':main()
