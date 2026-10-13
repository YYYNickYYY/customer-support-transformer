"""Matched complete workflow prefixes; method check never loads real weights."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import condition_measurement_v2 as old
from . import opening_scope_control as scope
from .reply_prefix import prefix_reply
from .tokenization import encode_parts, load_frozen

ROOT = old.ROOT
PROTOCOL = 'docs/workflow-prefix-control-v1-protocol.md'
ACCEPTANCE = 'docs/workflow-prefix-control-v1-acceptance.md'
OUT = ROOT / 'outputs/workflow-prefix-control-v1-20261006'


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if old.digest(ROOT / name) != expected:
            raise ValueError('Frozen dependency changed: ' + name)


def accepted_receipt():
    receipt = scope.read_block(ROOT / ACCEPTANCE)
    if receipt['status'] != 'accepted_for_complete_workflow_prefix_only' or receipt['fifth_inference']:
        raise ValueError('No valid acceptance')
    verify_hashes(receipt['sha256'])
    return receipt


def expand_jobs(plan, tok, cfg):
    required = dict(device='cpu', cpu_threads=2, seed=42, context=512,
                    total_reply_budget=320, no_repeat_ngram_size=0, decoding='greedy', eos_allowed=True)
    g = plan['generation']
    if any(g[k] != v for k, v in required.items()) or cfg['max_length'] != 512:
        raise ValueError('Changed generation contract')
    if g['blocked_tokens'] != ['<PAD>', '<BOS>', '<USER>', '<ASSISTANT>'] or g['tokenizer_revision'] != cfg['tokenizer_revision']:
        raise ValueError('Changed tokenizer/mask')
    if [q['id'] for q in plan['questions']] != ['partial', 'whole']:
        raise ValueError('Changed question IDs')
    if [(p['id'], p['domain'], p['wording']) for p in plan['prefixes']] != [
        (d+'-'+w, d, w) for w in ('a', 'b') for d in ('neutral', 'order', 'billing')]:
        raise ValueError('Changed prefix matrix')
    common = scope.exact_text(tok, plan['common'].rstrip('\n'), cfg)
    # Encode newline with route, avoiding a forced standalone trailing whitespace token.
    jobs, qids, routes = [], [], {}
    for q in plan['questions']:
        inst, _, full = encode_parts(tok, q['instruction'], '')
        qids.append(inst)
        if len(inst) != 8:
            raise ValueError('Unmatched questions')
        for p in plan['prefixes']:
            text = plan['common'] + p['route']
            if not p['route'].endswith('.') or not text.endswith('.'):
                raise ValueError('Incomplete route sentence')
            route = scope.exact_text(tok, '\n'+p['route'], cfg)
            prefix = scope.exact_text(tok, text, cfg)
            if prefix != common + route or len(route) != 13:
                raise ValueError('Prefix boundary/length mismatch')
            if len(prefix) >= 320 or len(full[:-1]+prefix) >= cfg['max_length']:
                raise ValueError('No continuation budget')
            routes[p['id']] = route
            jobs.append(dict(id=q['id']+'-'+p['id'], scope=q['id'], instruction=q['instruction'],
                             domain=p['domain'], wording=p['wording'], common_text=plan['common'].rstrip('\n'),
                             common_ids=common, route_text='\n'+p['route'], route_ids=route,
                             prefix_text=text, prefix_ids=prefix, prompt_ids=full[:-1], control_of=None))
    if [i for i, (a,b) in enumerate(zip(*qids)) if a != b] != [5,6]:
        raise ValueError('Question scope positions changed')
    for w in ('a','b'):
        base = routes['neutral-'+w]
        for d in ('order','billing'):
            if [i for i,(a,b) in enumerate(zip(base,routes[d+'-'+w])) if a!=b] != [5]:
                raise ValueError('Unmatched domain position')
    if len({len(j['prefix_ids']) for j in jobs}) != 1 or plan['controls'] != ['partial-neutral-a','partial-billing-a']:
        raise ValueError('Unmatched prefixes/controls')
    jobs += [dict(j,id=j['id']+'-repeat',control_of=j['id']) for j in jobs if j['id'] in plan['controls']]
    if len(jobs)!=14 or len({j['id'] for j in jobs})!=14:
        raise ValueError('Wrong counts')
    return jobs


def check_method():
    p = scope.read_block(ROOT / PROTOCOL)
    verify_hashes(p['evidence_sha256'])
    g = p['generation']
    verify_hashes({g['checkpoint']:g['checkpoint_sha256'],g['tokenizer']:g['tokenizer_revision']})
    cfg = old.read(ROOT / g['model_config'])
    tok, _ = load_frozen(ROOT / cfg['tokenizer_artifact'])
    return p, cfg, tok, expand_jobs(p,tok,cfg)


def prepare(out=OUT):
    accepted_receipt()
    p,cfg,_,jobs = check_method()
    out=Path(out)
    if (out/'freeze.json').exists():
        verify(out)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('Unregistered output not empty')
    paths=[ROOT/PROTOCOL,ROOT/ACCEPTANCE,Path(__file__),ROOT/'tests/test_workflow_prefix_control.py',ROOT/p['generation']['model_config']]
    paths+=list((ROOT/'src/my_project_1').glob('*.py'))+list((ROOT/cfg['tokenizer_artifact']).glob('*.json'))
    paths+=list((ROOT/'outputs').rglob('*.xlsx'))
    deps={x.relative_to(ROOT).as_posix():old.digest(x) for x in paths}
    deps.update(p['evidence_sha256'])
    deps[p['generation']['checkpoint']]=p['generation']['checkpoint_sha256']
    old.save_once(out/'freeze.json',dict(config=cfg,generation=p['generation'],jobs=jobs,
        environment=old.environment(),dependencies_sha256=deps,training=False,
        registered_at_utc=datetime.now(timezone.utc).isoformat(),ordering='generate -> review -> summarize'))


def verify(out):
    out=Path(out)
    if (out/'invalidity.json').exists():
        raise ValueError('Invalid batch')
    f=old.read(out/'freeze.json')
    verify_hashes(f['dependencies_sha256'])
    p,cfg,_,jobs=check_method()
    if f['environment']!=old.environment() or f['config']!=cfg or f['jobs']!=jobs or f['generation']!=p['generation']:
        raise ValueError('Changed execution condition')
    return f


def check_controls(jobs,values):
    old.check_controls(jobs,values,'generations')
    for j in jobs:
        if not j['control_of']:
            continue
        a,b=values[j['id']]['steps'],values[j['control_of']]['steps']
        if len(a)!=len(b):
            raise ValueError('Changed control trace')
        for x,y in zip(a,b):
            if [v['token_id'] for v in x['top5']]!=[v['token_id'] for v in y['top5']] or abs(x['eos_probability']-y['eos_probability'])>1e-6 or any(abs(v['probability']-w['probability'])>1e-6 for v,w in zip(x['top5'],y['top5'])):
                raise ValueError('Changed control probability')


def generate(out=OUT):
    out=Path(out); f=verify(out); done=out/'generations-freeze.json'
    if done.exists():
        verify_hashes(old.read(done)['sha256']); return
    jobs=f['jobs']; cfg=f['config']; tok,_=load_frozen(ROOT/cfg['tokenizer_artifact'])
    allowed={j['id']+'.json' for j in jobs}
    if any(x.name not in allowed for x in (out/'generations').glob('*.json')):
        raise ValueError('Unexpected generation record')
    validate=lambda j,v:old.validate_generation(j,v,tok,cfg)
    fh=old.digest(out/'freeze.json')
    for j in jobs:
        if (out/'generations'/(j['id']+'.json')).exists():
            old.record(out,'generations',j,fh,None,validate)
    model,tok,initial=old.load_model(f)
    values={}
    try:
        for i,j in enumerate(jobs,1):
            compute=lambda:prefix_reply(model,tok,cfg,j['instruction'],j['prefix_text'],total_reply_budget=320)
            values[j['id']]=old.record(out,'generations',j,fh,compute,validate)
            print(f"generations {i}/14 {j['id']}",flush=True)
        check_controls(jobs,values); old.assert_unchanged(model,initial)
        verify_hashes(f['dependencies_sha256'])
    except ValueError as exc:
        old.save_once(out/'invalidity.json',dict(reason=str(exc))); raise
    old.save_once(out/'generations-status.json',dict(records=14,controls_pass=True,model_state_unchanged=True,
        gradients_none=True,optimizer_updates=0,stops=dict(Counter(v['stop_reason'] for v in values.values()))))
    paths=[out/'freeze.json',out/'generations-status.json']+[out/'generations'/(j['id']+'.json') for j in jobs]
    old.save_once(done,dict(sha256={x.relative_to(ROOT).as_posix():old.digest(x) for x in paths}))


def validate_reviews(rows,jobs,results):
    if len(rows)!=len(jobs) or len({r['id'] for r in rows})!=len(jobs) or {r['id'] for r in rows}!={j['id'] for j in jobs}:
        raise ValueError('Incomplete/duplicate reviews')
    byid={r['id']:r for r in rows}
    action={'matching','conflicting','mixed','unavailable'}
    for j in jobs:
        r=byid[j['id']]
        allowed={'explicit_return','implicit_return','continues_wrong_domain','mixed','unavailable'} if j['domain']=='billing' else {'not_applicable'}
        if r['next_object'] not in {'partial','whole','other','mixed','unavailable'} or r['next_action'] not in action or r['full_task'] not in action or r['recovery'] not in allowed:
            raise ValueError('Invalid semantic label')
        if r['recovery']=='implicit_return' and (r['next_action']!='matching' or r['full_task']!='matching'):
            raise ValueError('Implicit recovery requires matching actions/full task')
        if r['recovery']=='explicit_return' and r['full_task']!='matching':
            raise ValueError('Explicit recovery requires matching full task')
        for k in ('reason','help','risk','entities','repetition','fluency'):
            if not isinstance(r[k],str) or not r[k].strip():
                raise ValueError('Missing explanation')
        text=results[j['id']]['continuation']
        for label in ('next_object','next_action','full_task','recovery'):
            quote=r[label+'_quote']
            if not isinstance(quote,str) or (quote and quote not in text) or (not quote and r[label] not in {'unavailable','not_applicable'}):
                raise ValueError('Evidence must be generated continuation')
        if j['control_of'] and any(r[k]!=byid[j['control_of']][k] for k in r if k!='id'):
            raise ValueError('Inconsistent control review')


def freeze_reviews(out=OUT):
    out=Path(out); f=verify(out)
    verify_hashes(old.read(out/'generations-freeze.json')['sha256'])
    path=out/'review-freeze.json'
    if path.exists():
        verify_hashes(old.read(path)['sha256']); return
    results={j['id']:old.read(out/'generations'/(j['id']+'.json'))['payload']['result'] for j in f['jobs']}
    validate_reviews(old.read(out/'semantic-reviews.json'),f['jobs'],results)
    paths=[out/'semantic-reviews.json',out/'generations-freeze.json']
    old.save_once(path,dict(sha256={x.relative_to(ROOT).as_posix():old.digest(x) for x in paths}))


def summarize(out=OUT):
    out=Path(out); verify(out)
    for name in ('generations-freeze.json','review-freeze.json'):
        verify_hashes(old.read(out/name)['sha256'])
    rows={r['id']:r for r in old.read(out/'semantic-reviews.json')}
    pairs=[]
    for d in ('neutral','order','billing'):
        for w in ('a','b'):
            pair=[rows[q+'-'+d+'-'+w] for q in ('partial','whole')]
            status='unavailable' if any('unavailable' in (r['next_action'],r['full_task']) for r in pair) else 'both_match' if all(r['next_action']==r['full_task']=='matching' for r in pair) else 'not_both_match'
            pairs.append(dict(domain=d,wording=w,status=status,ids=[r['id'] for r in pair],
                              next_objects=[r['next_object'] for r in pair],next_actions=[r['next_action'] for r in pair],
                              full_tasks=[r['full_task'] for r in pair],recovery=[r['recovery'] for r in pair]))
    agreement={d:len({p['status'] for p in pairs if p['domain']==d})==1 for d in ('neutral','order','billing')}
    old.save_once(out/'summary.json',dict(scope='fixed_domain_cued_complete_prefixes_only',pairs=pairs,
        pair_status_agreement=agreement,primary_generations=12,training=False,not_general_accuracy=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('check','prepare','generate','freeze-review','summarize'))
    stage=parser.parse_args().stage
    if stage=='check':
        _,_,_,jobs=check_method()
        print('Token-only preflight; no model load:',len(jobs),'jobs; prefix',len(jobs[0]['prefix_ids']))
    else:
        {'prepare':prepare,'generate':generate,'freeze-review':freeze_reviews,'summarize':summarize}[stage]()


if __name__=='__main__':
    main()
