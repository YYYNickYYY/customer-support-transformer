"""Read-only acceptance: generation contracts, calibrated scores and evidence."""
import importlib.util
import json
from pathlib import Path

import pytest

from my_project_1.tokenization import file_hash
from my_project_1.training_sixth import load_plan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dev-expanded-seventh-20261007'
KEYS=('intent_score','helpfulness_score','fluency_score','repetition','unsupported_claim')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def completed():
    if not (OUT/'final-review.json').exists():
        pytest.skip('Frozen local model evaluation artifacts are not distributed')
    return read(OUT/'final-review.json')


def test_sources_and_immutable_draft(completed):
    for key in ('sources_sha256','final_sources_sha256'):
        for p,h in completed[key].items():
            assert file_hash(ROOT/p)==h,p
    assert read(OUT/'rating-freeze.json')['ratings_sha256']==file_hash(OUT/'ai-expert-ratings.json')
    for p,h in read(OUT/'freeze.json')['dependencies_sha256'].items():
        assert file_hash(ROOT/p)==h,p


def test_same_inputs_and_depth_only(completed):
    plan=load_plan()
    config=read(OUT/'config.json')
    expected={**plan['generation_baseline'],'checkpoint':config['checkpoint'],
              'checkpoint_sha256':config['checkpoint_sha256'],
              'model_config':{**plan['generation_baseline']['model_config'],'n_layers':6}}
    assert config==expected
    new,old=rows(OUT/'replies.jsonl'),rows(ROOT/plan['baseline_replies'])
    assert len(new)==len(old)==135
    for a,b in zip(new,old):
        assert all(a[k]==b[k] for k in ('id','instruction','intent','cohort','reference_response'))


def test_token_stops_and_weights(completed):
    raw=rows(OUT/'replies.jsonl')
    inputs=read(OUT/'freeze.json')['inputs']
    mc=read(OUT/'config.json')['model_config']
    blocked={mc[k+'_token_id'] for k in ('pad','bos','user','assistant')}
    for r,prompt in zip(raw,inputs):
        tokens=r['generated_token_ids']
        assert r['id']==prompt['id'] and r['prompt_tokens']==len(prompt['input_ids'])
        assert len(tokens)==r['generated_tokens']<=320
        assert r['prompt_tokens']+len(tokens)<=512 and not blocked.intersection(tokens)
        if r['stop_reason']=='eos':
            assert tokens[-1]==mc['eos_token_id'] and mc['eos_token_id'] not in tokens[:-1]
        else:
            assert r['stop_reason']=='max_new_tokens' and len(tokens)==320
    summary=read(OUT/'summary.json')
    assert summary['weights_unchanged'] and summary['optimizer_updates']==0
    assert summary['replies_sha256']==file_hash(OUT/'replies.jsonl')


def test_complete_judgments_and_arithmetic(completed):
    raw=rows(OUT/'replies.jsonl')
    assert len(completed['ratings'])==len(completed['pairs'])==135
    for r,s in zip(completed['ratings'],raw):
        assert r['id']==s['id'] and r['notes'] and all(q and q in s['generated_response'] for q in r['evidence'])
        assert r['risk_rubric']=='customer-support-risk-review-v1.1'
        assert r['other_rubric']=='customer-support-review-v1.0'
        assert all(type(r[k]) is int and 0<=r[k]<=(2 if j<3 else 1) for j,k in enumerate(KEYS))
    for j,k in enumerate(KEYS):
        total=sum(r[k] for r in completed['ratings'])
        assert completed['metrics']['seventh'][k]==(total/135 if j<3 else total)
        assert sum(completed['paired_counts'][k].values())==135
    assert len(completed['per_intent']['seventh'])==27
    assert all(s['n']==5 for s in completed['per_intent']['seventh'].values())


def test_explicit_calibration_only(completed):
    draft=read(OUT/'ai-expert-ratings.json')
    changes={(c['case'],c['field']):c for c in read(OUT/'calibration.json')['changes']}
    for i,(old,new) in enumerate(zip(draft['ratings'],completed['ratings']),1):
        for k in KEYS:
            if (i,k) in changes:
                c=changes[i,k]
                assert old[k]==c['from'] and new[k]==c['to']
            else:
                assert old[k]==new[k]
    assert draft['metrics']['fifth']==completed['metrics']['fifth']
    assert draft['entity_pairs']==completed['entity_pairs']


def test_entity_review_full_coverage(completed):
    plan=load_plan()
    old=rows(ROOT/plan['baseline_replies'])
    assert read(OUT/'entity-decisions.json')['reviewed_cases']==list(range(1,136))
    assert len(completed['entity_pairs'])==135
    for pair,r in zip(completed['entity_pairs'],old):
        assert pair['id']==r['id']
        assert all(q in r['generated_response'] for q in pair['fifth_evidence'])
    assert completed['entity_issues']=={'fifth':21,'seventh':18}
    assert completed['entity_counts']=={'improved':3,'unchanged':132,'worsened':0}


def test_completed_generation_rerun_no_inference(completed,monkeypatch):
    path=ROOT/'scripts/evaluate_seventh_expanded.py'
    spec=importlib.util.spec_from_file_location('seventh_eval_check',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args,**kwargs):
        pytest.fail('No new model load/inference/training in completed replay')
    monkeypatch.setattr(module.torch,'load',forbidden)
    monkeypatch.setattr(module,'generate_reply',forbidden)
    module.main()
