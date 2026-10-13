"""Read-only acceptance for the completed eighth greedy evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest
from my_project_1.tokenization import file_hash, encode_parts
from my_project_1.dataset import CustomerSupportDataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/dev-expanded-eighth-20261013'
OLD = ROOT/'outputs/dev-expanded-seventh-20261007'
KEYS = ('intent_score','helpfulness_score','fluency_score','repetition','unsupported_claim','entity_issue')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def review():
    if not (OUT/'final-review.json').exists():
        pytest.skip('Local evaluation artifacts not distributed')
    return read(OUT/'final-review.json')


def test_sources_and_history(review):
    for p,h in review['sources_sha256'].items():
        assert file_hash(ROOT/p)==h,p
    for p,h in read(OUT/'freeze.json')['dependencies_sha256'].items():
        assert file_hash(ROOT/p)==h,p
    for p,h in read(OLD/'completion-freeze.json')['sha256'].items():
        assert file_hash(ROOT/p)==h,p


def test_only_checkpoint_changed_and_all_inputs_match(review):
    new,old=read(OUT/'config.json'),read(OLD/'config.json')
    assert new=={**old,'checkpoint':new['checkpoint'],'checkpoint_sha256':new['checkpoint_sha256']}
    assert new['checkpoint_sha256']=='8a85e9886b8f866f580eca453550210879511aad29a958afed88e3d690527405'
    assert read(OUT/'freeze.json')['inputs']==read(OLD/'freeze.json')['inputs']
    data=CustomerSupportDataset('dev',tokenizer_dir=new['model_config']['tokenizer_artifact'])
    a,b=rows(OUT/'replies.jsonl'),rows(OLD/'replies.jsonl')
    assert len(a)==len(b)==135 and len({r['id'] for r in a})==135
    for r,prior,prompt in zip(a,b,read(OUT/'freeze.json')['inputs']):
        assert all(r[k]==prior[k] for k in ('id','instruction','intent','cohort','reference_response'))
        assert prompt['input_ids']==encode_parts(data.tokenizer,r['instruction'],'')[2][:-1]
        assert data.tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True)==r['generated_response']


def test_training_completed_and_first_five_epochs_match(review):
    new=ROOT/read(OUT/'config.json')['checkpoint']
    old=ROOT/read(OLD/'config.json')['checkpoint']
    a,b=read(new.parent/'config.json'),read(old.parent/'config.json')
    assert {k:v for k,v in a.items() if k!='run_id'}=={**{k:v for k,v in b.items() if k!='run_id'},'epochs':8}
    a,b=rows(new.parent/'metrics.jsonl'),rows(old.parent/'metrics.jsonl')
    assert len(a)==9 and a[:6]==b
    assert min(r['dev']['loss'] for r in a[1:])==review['best_dev_loss']['eighth']
    status=read(new.parent/'status.json')
    assert (status['state'],status['completed_epochs'],status['completed_steps'])==('completed',8,21528)


def test_token_stops_and_no_weight_updates(review):
    mc=read(OUT/'config.json')['model_config']
    blocked={mc[k+'_token_id'] for k in ('pad','bos','user','assistant')}
    for r,prompt in zip(rows(OUT/'replies.jsonl'),read(OUT/'freeze.json')['inputs']):
        ts=r['generated_token_ids']
        assert r['id']==prompt['id'] and r['prompt_tokens']==len(prompt['input_ids'])
        assert len(ts)==r['generated_tokens']<=320
        assert len(ts)+r['prompt_tokens']<=512 and not blocked.intersection(ts)
        if r['stop_reason']=='eos':
            assert ts[-1]==mc['eos_token_id'] and mc['eos_token_id'] not in ts[:-1]
        else:
            assert r['stop_reason']=='max_new_tokens' and len(ts)==320
    s=read(OUT/'summary.json')
    assert s['weights_unchanged'] and s['optimizer_updates']==0
    assert s['replies_sha256']==file_hash(OUT/'replies.jsonl')


def test_full_evidence_and_paired_arithmetic(review):
    rs=review['ratings']
    old=read(OLD/'final-review.json')['ratings']
    assert len(rs)==len(review['pairs'])==135
    for i,(r,raw,b,p) in enumerate(zip(rs,rows(OUT/'replies.jsonl'),old,review['pairs']),1):
        assert r['case']==i and r['id']==raw['id']==b['id']==p['id']
        assert r['notes'] and r['evidence'] and all(q and q in raw['generated_response'] for q in r['evidence'])
        assert r['risk_rubric']=='customer-support-risk-review-v1.1'
        assert r['other_rubric']=='customer-support-review-v1.0'
        for j,k in enumerate(KEYS):
            assert type(r[k]) is int and 0<=r[k]<=(2 if j<3 else 1)
            assert p[k]==r[k]-b[k]
    for j,k in enumerate(KEYS):
        for model,ratings in [('seventh',old),('eighth',rs)]:
            total=sum(r[k] for r in ratings)
            assert review['metrics'][model][k]==(total/135 if j<3 else total)
        pairs=review['pairs']
        assert review['paired_counts'][k]=={
            'improved':sum(p[k]>0 if j<3 else p[k]<0 for p in pairs),
            'unchanged':sum(p[k]==0 for p in pairs),
            'worsened':sum(p[k]<0 if j<3 else p[k]>0 for p in pairs)}
    assert len(review['per_intent']['eighth'])==27
    assert all(r['n']==5 for r in review['per_intent']['eighth'].values())


def test_finished_rerun_does_not_load_or_generate(review,monkeypatch):
    spec=importlib.util.spec_from_file_location('eighth_eval',ROOT/'scripts/evaluate_eighth_expanded.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    def forbidden(*a,**kw):
        pytest.fail('Complete replay must not load model or generate again')
    monkeypatch.setattr(mod.torch,'load',forbidden)
    monkeypatch.setattr(mod,'generate_reply',forbidden)
    mod.main()
