"""Synthetic-only method acceptance; checkpoints forbidden for every test."""
from copy import deepcopy
import json
import math
import pytest
import torch

from my_project_1 import workflow_prefix_control as r
from my_project_1.reply_prefix import prefix_reply, branch_nll


@pytest.fixture(autouse=True)
def forbid_checkpoint(monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('No real checkpoint during method acceptance'))
    before=torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def setup():
    return r.check_method()


def test_full_matrix_boundaries_and_no_answer_leakage(setup):
    p,c,t,jobs=setup
    assert len(jobs)==14 and sum(j['control_of'] is not None for j in jobs)==2
    assert len({len(j['prefix_ids']) for j in jobs})==1
    for j in jobs:
        assert j['prefix_ids']==j['common_ids']+j['route_ids']
        assert t.encode(j['prefix_text']).ids==j['prefix_ids']
        assert len(j['prompt_ids'])==11 and len(j['route_ids'])==13
        assert j['prefix_text'].endswith('.') and not j['prefix_text'].endswith(' ')
        assert not any(w in j['prefix_text'].lower() for w in ['selected','entire','cancel','remove','delete'])
        # A new continuation begins at a genuine boundary; no independent trailing space.
        for suffix in (' Please tell me which items to remove.',' Let us return to the order request.','\n3. Please provide the order number.'):
            assert t.encode(j['prefix_text']+suffix).ids==j['prefix_ids']+t.encode(suffix).ids


@pytest.mark.parametrize('kind',['question','route','mask','budget','control','matrix','incomplete'])
def test_contract_changes_refused(setup,kind):
    p,c,t,_=setup; p=deepcopy(p)
    if kind=='question': p['questions'][0]['instruction']='Please remove some selected items immediately.'
    elif kind=='route': p['prefixes'][0]['route']+=' '
    elif kind=='mask': p['generation']['blocked_tokens'].append('<EOS>')
    elif kind=='budget': p['generation']['total_reply_budget']=321
    elif kind=='control': p['controls']=[]
    elif kind=='matrix': p['prefixes'][0]['domain']='billing'
    else: p['prefixes'][0]['route']=p['prefixes'][0]['route'][:-1]
    with pytest.raises(ValueError): r.expand_jobs(p,t,c)


class EOSModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.marker=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64))
    def forward(self,ids,attention_mask):
        assert not self.training and not torch.is_grad_enabled()
        assert attention_mask.eq(1).all()
        v=torch.zeros((*ids.shape,4000),dtype=torch.float64)+self.marker
        v[...,:4]=100; v[...,4]=5
        return {'logits':v}


def test_synthetic_eos_mask_positions_and_state(setup):
    _,c,t,jobs=setup; model=EOSModel(); initial={k:v.clone() for k,v in model.state_dict().items()}
    for j in jobs:
        v=prefix_reply(model,t,c,j['instruction'],j['prefix_text'])
        r.old.validate_generation(j,v,t,c)
        assert v['continuation_token_ids']==[4] and v['stop_reason']=='eos'
        assert v['steps'][0]['input_length']==len(j['prompt_ids']+j['prefix_ids'])
        assert v['steps'][0]['eos_probability']==pytest.approx(math.exp(5)/(math.exp(5)+3995))
    r.old.assert_unchanged(model,initial); assert model.training


@pytest.mark.parametrize('mode',['budget','nonfinite','exception'])
def test_error_restoration_and_budget(setup,mode):
    _,c,t,jobs=setup; j=jobs[0]
    class Broken(EOSModel):
        def forward(self,*a,**k):
            if mode=='exception': raise RuntimeError('synthetic failure')
            v=super().forward(*a,**k); v['logits'][...,20]=float('nan'); return v
    m=Broken()
    with pytest.raises((RuntimeError,ValueError)):
        prefix_reply(m,t,c,j['instruction'],j['prefix_text'],total_reply_budget=len(j['prefix_ids']) if mode=='budget' else 320)
    assert m.training and m.marker.grad is None


def test_shifted_supervision_excludes_prefix_independent_oracle(setup):
    # Method does not score candidates. This checks the retained helper boundary,
    # not a fifth-model metric: prefix/prompt excluded, no EOS target invented.
    _,c,t,jobs=setup; j=jobs[0]; m=EOSModel(); target=' Please provide the specific items.'
    v=branch_nll(m,t,c,j['instruction'],j['prefix_ids'],target)
    expected=math.log(4*math.exp(100)+math.exp(5)+3995)
    assert v['target_ids']==t.encode(target).ids
    assert v['per_token_nll']==pytest.approx([expected]*len(v['target_ids']))
    assert v['input_length']==len(j['prompt_ids']+j['prefix_ids']+v['target_ids'])-1
    assert m.marker.grad is None and m.training


def review(j):
    return dict(id=j['id'],next_object='unavailable',next_object_quote='',next_action='unavailable',next_action_quote='',
                full_task='unavailable',full_task_quote='',recovery='unavailable' if j['domain']=='billing' else 'not_applicable',recovery_quote='',
                reason='No action at EOS',help='None',risk='None',entities='None',repetition='None',fluency='No continuation')


def test_semantics_not_forced_history_and_recovery_consistency():
    j=dict(id='x',domain='billing',control_of=None); results={'x':{'continuation':' Please specify the items to remove.'}}
    row=review(j); row.update(next_object='partial',next_object_quote='specify the items',next_action='matching',next_action_quote='specify the items',full_task='matching',full_task_quote='specify the items',recovery='implicit_return',recovery_quote='specify the items')
    r.validate_reviews([row],[j],results)
    row['next_object_quote']='Review the billing details'
    with pytest.raises(ValueError,match='generated continuation'): r.validate_reviews([row],[j],results)
    row['next_object_quote']='specify the items'; row['full_task']='conflicting'
    with pytest.raises(ValueError,match='Implicit recovery'): r.validate_reviews([row],[j],results)


@pytest.mark.parametrize('bad',['other_label','missing','duplicate','control'])
def test_review_rejections(setup,bad):
    _,_,_,jobs=setup; rows=[review(j) for j in jobs]; values={j['id']:{'continuation':''} for j in jobs}
    if bad=='other_label': rows[0]['next_object']='account'
    elif bad=='missing': rows[0]['risk']=''
    elif bad=='duplicate': rows[-1]=rows[0]
    else: rows[-1]['help']='changed judgment'
    with pytest.raises(ValueError): r.validate_reviews(rows,jobs,values)


def test_no_acceptance_no_output(tmp_path,monkeypatch):
    def rejected(): raise ValueError('No valid acceptance')
    monkeypatch.setattr(r,'accepted_receipt',rejected)
    with pytest.raises(ValueError,match='acceptance'): r.prepare(tmp_path/'absent')
    assert not (tmp_path/'absent').exists()


def test_prepare_freeze_repeat_without_inference(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'accepted_receipt',lambda:{})
    monkeypatch.setattr(r,'ACCEPTANCE',r.PROTOCOL)
    out=tmp_path/'prepared'; r.prepare(out); assert len(r.verify(out)['jobs'])==14
    before=(out/'freeze.json').read_bytes(),(out/'freeze.json').stat().st_mtime_ns
    r.prepare(out)
    assert ((out/'freeze.json').read_bytes(),(out/'freeze.json').stat().st_mtime_ns)==before
    assert not (out/'generations').exists()


def test_corrupt_partial_rejected_before_load(setup,tmp_path,monkeypatch):
    p,c,t,jobs=setup; out=tmp_path/'outputs/run'; f=dict(config=c,generation=p['generation'],jobs=jobs,dependencies_sha256={})
    r.old.save_once(out/'freeze.json',f); r.old.save_once(out/'generations'/(jobs[0]['id']+'.json'),{'payload':{},'sha256':'bad'})
    monkeypatch.setattr(r,'verify',lambda _:f); monkeypatch.setattr(r,'load_frozen',lambda _:(t,{}))
    monkeypatch.setattr(r.old,'load_model',lambda _:pytest.fail('Must precheck before load'))
    with pytest.raises(ValueError,match='Corrupt'): r.generate(out)


def test_synthetic_pipeline_outage_recovery_completed_readonly(setup,tmp_path,monkeypatch):
    p,c,t,jobs=setup; out=tmp_path/'outputs/run'; f=dict(config=c,generation=p['generation'],jobs=jobs,dependencies_sha256={})
    r.old.save_once(out/'freeze.json',f)
    monkeypatch.setattr(r,'ROOT',tmp_path); monkeypatch.setattr(r,'verify',lambda _:f)
    monkeypatch.setattr(r,'load_frozen',lambda _:(t,{}))
    m=EOSModel(); state={k:v.clone() for k,v in m.state_dict().items()}
    monkeypatch.setattr(r.old,'load_model',lambda _:(m,t,state))
    original=r.prefix_reply; calls=[]
    def outage(*a,**k):
        calls.append(1)
        if len(calls)==3: raise RuntimeError('outage')
        return original(*a,**k)
    monkeypatch.setattr(r,'prefix_reply',outage)
    with pytest.raises(RuntimeError,match='outage'): r.generate(out)
    existing={x:(x.read_bytes(),x.stat().st_mtime_ns) for x in (out/'generations').glob('*.json')}
    assert len(existing)==2
    monkeypatch.setattr(r,'prefix_reply',original); r.generate(out)
    assert all((x.read_bytes(),x.stat().st_mtime_ns)==v for x,v in existing.items())
    with pytest.raises(FileNotFoundError): r.summarize(out)
    r.old.save_once(out/'semantic-reviews.json',[review(j) for j in jobs]); r.freeze_reviews(out); r.summarize(out)
    assert all(x['status']=='unavailable' for x in r.old.read(out/'summary.json')['pairs'])
    before={x:(x.read_bytes(),x.stat().st_mtime_ns) for x in out.rglob('*.json')}
    monkeypatch.setattr(r.old,'load_model',lambda _:pytest.fail('Complete rerun must not load'))
    r.generate(out); r.freeze_reviews(out); r.summarize(out)
    assert all((x.read_bytes(),x.stat().st_mtime_ns)==v for x,v in before.items())
    target=next((out/'generations').glob('*.json')); target.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='dependency changed'): r.generate(out)


def test_changed_top5_control_rejected(setup):
    _,c,t,jobs=setup; m=EOSModel(); values={j['id']:prefix_reply(m,t,c,j['instruction'],j['prefix_text']) for j in jobs}
    r.check_controls(jobs,values); repeat=next(j for j in jobs if j['control_of'])
    values[repeat['id']]['steps'][0]['top5'][0]['probability']+=.01
    with pytest.raises(ValueError,match='control probability'): r.check_controls(jobs,values)


def test_real_inputs_have_scope_and_domain_at_matched_positions(setup):
    _,c,t,jobs=setup
    class Capture(EOSModel):
        def __init__(self): super().__init__(); self.inputs=[]
        def forward(self,ids,attention_mask):
            self.inputs.append(ids[0].tolist())
            return super().forward(ids,attention_mask)
    m=Capture()
    for j in jobs[:12]: prefix_reply(m,t,c,j['instruction'],j['prefix_text'])
    assert m.inputs==[j['prompt_ids']+j['prefix_ids'] for j in jobs[:12]]
    assert len({len(x) for x in m.inputs})==1
    assert [i for i,(a,b) in enumerate(zip(m.inputs[0],m.inputs[6])) if a!=b]==[7,8]
    assert [i for i,(a,b) in enumerate(zip(m.inputs[0],m.inputs[1])) if a!=b]==[len(jobs[0]['prompt_ids']+jobs[0]['common_ids'])+5]


def test_receipt_hash_binding(tmp_path,monkeypatch):
    binding=tmp_path/'runner.py'; binding.write_text('fixture',encoding='utf-8')
    receipt=dict(status='accepted_for_complete_workflow_prefix_only',fifth_inference=False,sha256={'runner.py':r.old.digest(binding)})
    doc=tmp_path/'acceptance.md'; doc.write_text('```json\n'+json.dumps(receipt)+'\n```',encoding='utf-8')
    monkeypatch.setattr(r,'ROOT',tmp_path); monkeypatch.setattr(r,'ACCEPTANCE','acceptance.md')
    assert r.accepted_receipt()==receipt
    binding.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='dependency changed'): r.accepted_receipt()


@pytest.mark.parametrize('failure',['control','state','dependency'])
def test_stop_and_mark_invalid_on_end_checks(setup,tmp_path,monkeypatch,failure):
    p,c,t,jobs=setup; out=tmp_path/'outputs/run'; f=dict(config=c,generation=p['generation'],jobs=jobs,dependencies_sha256={})
    r.old.save_once(out/'freeze.json',f); monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'verify',lambda _:f); monkeypatch.setattr(r,'load_frozen',lambda _:(t,{}))
    m=EOSModel(); state={k:v.clone() for k,v in m.state_dict().items()}
    monkeypatch.setattr(r.old,'load_model',lambda _:(m,t,state))
    def fail(*a,**k): raise ValueError('synthetic '+failure+' failure')
    target={'control':(r,'check_controls'),'state':(r.old,'assert_unchanged'),'dependency':(r,'verify_hashes')}[failure]
    monkeypatch.setattr(*target,fail)
    with pytest.raises(ValueError,match='synthetic'): r.generate(out)
    assert (out/'invalidity.json').exists() and not (out/'generations-freeze.json').exists()


@pytest.mark.parametrize('stage',['freeze','summary'])
def test_no_semantic_freeze_no_summary_or_review_before_generation(tmp_path,monkeypatch,stage):
    monkeypatch.setattr(r,'verify',lambda _: {})
    with pytest.raises(FileNotFoundError):
        (r.freeze_reviews if stage=='freeze' else r.summarize)(tmp_path)
