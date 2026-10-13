"""Validate explicit full-text judgments; no automatic semantic scoring."""
import json
from collections import Counter
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash

OUT = Path('outputs/dev-expanded-eighth-20261013')
OLD = Path('outputs/dev-expanded-seventh-20261007')
KEYS = ('intent_score', 'helpfulness_score', 'fluency_score', 'repetition', 'unsupported_claim', 'entity_issue')


def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]


def metrics(rs):
    return {'n': len(rs), **{k: sum(r[k] for r in rs) / len(rs) if j < 3 else sum(r[k] for r in rs)
                            for j, k in enumerate(KEYS)}}


def build():
    for p, h in read(OLD/'completion-freeze.json')['sha256'].items():
        assert file_hash(p) == h, p
    for p, h in read(OUT/'freeze.json')['dependencies_sha256'].items():
        assert file_hash(p) == h, p
    raw, oldraw = rows(OUT/'replies.jsonl'), rows(OLD/'replies.jsonl')
    baseline = read(OLD/'final-review.json')['ratings']
    ds = read(OUT/'decisions.json')
    assert len(raw) == len(oldraw) == len(baseline) == len(ds) == 135
    assert [d[0] for d in ds] == list(range(1, 136))
    ratings, pairs = [], []
    for i, (r, old, b, d) in enumerate(zip(raw, oldraw, baseline, ds), 1):
        assert all(r[k] == old[k] for k in ('id','instruction','intent','cohort','reference_response'))
        assert r['id'] == b['id']
        _, scores, quotes, note, entity = d
        scores = [*scores, entity]
        assert len(scores) == 6
        assert all(type(v) is int and 0 <= v <= (2 if j < 3 else 1) for j, v in enumerate(scores)), i
        assert note and quotes and all(q and q in r['generated_response'] for q in quotes), i
        rating = {k:r[k] for k in ('id','intent','cohort','stop_reason')}
        rating.update(zip(KEYS, scores))
        rating.update(case=i, model='eighth', evidence=quotes, notes=note,
                      reviewer='AI nonblind full-text review; known training/loss and seventh calibration',
                      risk_rubric='customer-support-risk-review-v1.1', other_rubric='customer-support-review-v1.0')
        ratings.append(rating)
        pairs.append({'case':i,'id':r['id'],'intent':r['intent'], **{k:rating[k]-b[k] for k in KEYS}})
    models = {'seventh':baseline, 'eighth':ratings}
    sources = [str(OUT/n) for n in ('decisions.json','replies.jsonl','summary.json','config.json','freeze.json')]
    sources += [str(OLD/n) for n in ('final-review.json','calibration.json','replies.jsonl','completion-freeze.json')]
    sources += [__file__, 'docs/evaluation-rubric-v1.md','docs/evaluation-risk-rubric-v1.1.md']
    return {
        'authority':'Final eighth paired review; seventh frozen final review unchanged',
        'method':'135 full replies read; explicit per-case AI nonblind decisions, not keyword scoring. Risk v1.1, other v1.0 with seventh calibration. All entities reviewed. No composite score or deployment/generalization claim.',
        'ratings':ratings, 'pairs':pairs,
        'metrics':{m:metrics(rs) for m,rs in models.items()},
        'paired_counts':{k:{'improved':sum(p[k]>0 if j<3 else p[k]<0 for p in pairs),
                            'unchanged':sum(p[k]==0 for p in pairs),
                            'worsened':sum(p[k]<0 if j<3 else p[k]>0 for p in pairs)} for j,k in enumerate(KEYS)},
        'per_cohort':{m:{c:metrics([r for r in rs if r['cohort']==c]) for c in ('original27','new108')} for m,rs in models.items()},
        'per_intent':{m:{t:metrics([r for r in rs if r['intent']==t]) for t in sorted({r['intent'] for r in rs})} for m,rs in models.items()},
        'stop_counts':{m:dict(Counter(r['stop_reason'] for r in rs)) for m,rs in [('seventh',oldraw),('eighth',raw)]},
        'best_dev_loss':{'seventh':read(OLD/'summary.json')['training_summary']['best_dev_loss'],
                         'eighth':read(OUT/'summary.json')['training_summary']['best_dev_loss']},
        'sources_sha256':{p:file_hash(p) for p in sources}}


def main():
    result = build()
    save(OUT/'final-review.json', result)
    print(json.dumps({k:result[k] for k in ('metrics','paired_counts','per_cohort','stop_counts')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
