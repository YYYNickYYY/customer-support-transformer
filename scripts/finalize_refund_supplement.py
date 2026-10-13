"""Freeze reviewed supplement and exact replay preview, without activating training."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
from my_project_1.expanded_dev import read, save
from my_project_1.target_revision_v2 import sentence_spans
from my_project_1.tokenization import read_rows, file_hash, load_frozen, prepare_example

OUT=Path('outputs/refund-supplement-20261006-v1')
BASE=Path('data/derived/refund-execution-v3.1/train.jsonl')
REASONS={
 'conditional_or_status_description':'条件、状态选项或历史材料描述，不断言退款已执行或必到账。',
 'nonexecution_policy_or_navigation':'一般政策说明/查询路径，不属于本补充因素的确定执行、到账或补救结果；其他风险未顺带修改。',
 'customer_need_confirmation':'确认客户想知道或期待的结果，不是客服承诺能实现该结果。',
 'explicit_goal_desire_or_effort':'goal/want/aim/strive/work towards/diligently/do best等在该句限定整个结果目标为意愿或努力，不视为独立确定保证。',
 'support_or_process_purpose_not_result':'承诺的是支持/沟通用途或流程目的，没有独立断言退款已执行、到账或成功解决。',
 'customer_action_or_eligibility_check':'客户准备材料或检查条件的指示，不是客服执行结果保证。',
 'query_accuracy_or_ordinary_service_purpose':'指导/核查用途、信息或普通服务目的，不在本次退款执行/补救结果改写范围。',
 'broad_experience_satisfaction_attitude_or_effort':'宽泛满意/体验/态度或明确努力限定，按v1.1正常服务意愿保留。'
}

def decisions(candidates, draft, addendum):
    approved={**draft['approved'],**addendum['initial_wording_overrides'],**addendum['extra_approved']}
    kept={}
    for prefix,groups in [('S',draft['retained_groups']),('E',addendum['extra_retained_groups'])]:
        for category,numbers in groups.items():
            for n in numbers:
                key=f'{prefix}{n:03}'
                if key in kept:raise ValueError('Duplicate decision '+key)
                kept[key]=REASONS[category]
    ids={c['review_id'] for c in candidates}
    if len(ids)!=len(candidates) or set(approved)&set(kept) or set(approved)|set(kept)!=ids:
        raise ValueError('Decisions must cover every candidate exactly once')
    result=[]
    for c in candidates:
        key=c['review_id'];yes=key in approved
        if yes:
            after=approved[key]
            if not after or after==c['before']:raise ValueError('Empty or unchanged approval')
            reason='混合服务句仍含独立退款执行、到账或补救结果保证；整句保留协助/澄清/条件功能，改为核查请求与可用下一步。'
            if 'actively' in c['before'].lower():reason='无执行记录却声称正在处理/加速退款；改为可协助核查处理状态。'
        else:after=c['before'];reason=kept[key]
        reason=addendum.get('borderline_notes',{}).get(key,reason)
        result.append({**c,'approved':yes,'after':after,'reason':reason,
                       'reviewer':addendum['review_identity']})
    return result

def entities(text):
    return Counter(re.findall(r'\{\{[^{}]+\}\}',text))

def replay(rows, reviews):
    rules={c['before']:c for c in reviews if c['approved']}
    if len(rules)!=sum(c['approved'] for c in reviews):raise ValueError('Duplicate approved sentence')
    expected=Counter((c['review_id'],m['id'],m['start'],m['end'])
                     for c in rules.values() for m in c['members'])
    seen=Counter();output=deepcopy(rows);changes=[]
    for base,row in zip(rows,output):
        edits=[]
        for start,end,s in sentence_spans(base['response']):
            if s not in rules:continue
            c=rules[s];key=(c['review_id'],base['id'],start,end);seen[key]+=1
            edits.append({'review_id':c['review_id'],'start':start,'end':end,
                          'before':s,'after':c['after'],'reason':c['reason']})
        if not edits:continue
        text=base['response']
        for e in reversed(edits):
            assert text[e['start']:e['end']]==e['before']
            text=text[:e['start']]+e['after']+text[e['end']:]
        row['response']=text
        if entities(text)!=entities(base['response']):raise ValueError('Entity mutation '+row['id'])
        if text.count('**')!=base['response'].count('**'):raise ValueError('Heading mutation')
        if re.findall(r'(?m)^\s*\d+\.',text)!=re.findall(r'(?m)^\s*\d+\.',base['response']):
            raise ValueError('Step marker mutation')
        before_counts=Counter(s for _,_,s in sentence_spans(base['response']))
        after_counts=Counter(s for _,_,s in sentence_spans(text))
        for e in edits:
            if after_counts[e['after']]>max(1,before_counts[e['after']]):
                raise ValueError('New exact duplicate sentence '+row['id'])
        if {k:v for k,v in row.items() if k!='response'}!={k:v for k,v in base.items() if k!='response'}:
            raise ValueError('Nonresponse mutation')
        changes.append({'id':row['id'],'intent':row['intent'],'instruction':row['instruction'],
                        'before_response':base['response'],'after_response':text,'edits':edits})
    if seen!=expected:raise ValueError('Unreviewed/missing approved occurrence')
    if len(rows)!=len(output) or [r['id'] for r in rows]!=[r['id'] for r in output]:
        raise ValueError('ID/order mutation')
    return output,changes

def verify_material():
    for name in ('material-freeze.json','extra-material-freeze.json'):
        frozen=read(OUT/name)
        for p,h in frozen.get('dependencies_sha256',frozen.get('sha256',{})).items():
            if file_hash(p)!=h:raise ValueError('Frozen dependency changed '+p)
        if 'candidates_sha256' in frozen and file_hash(OUT/'candidates.json')!=frozen['candidates_sha256']:
            raise ValueError('Initial material changed')
    baseline=read(BASE.parent/'manifest.json')
    for p,h in baseline['dependencies_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Baseline dependency changed '+p)
    for name,h in baseline['files_sha256'].items():
        if file_hash(BASE.parent/name)!=h:raise ValueError('Baseline artifact changed '+name)

def main():
    verify_material()
    candidates=read(OUT/'candidates.json')+read(OUT/'extra-candidates.json')
    reviews=decisions(candidates,read(OUT/'decisions-draft.json'),read(OUT/'decisions-final-addendum.json'))
    rows=list(read_rows(BASE));output,changes=replay(rows,reviews)
    tokenizer,config=load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    profiles=[]
    changed_ids={r['id'] for r in changes}
    for base,after in zip(rows,output):
        if after['id'] not in changed_ids:continue
        b=prepare_example(tokenizer,base['instruction'],base['response'],config['max_length'])
        a=prepare_example(tokenizer,after['instruction'],after['response'],config['max_length'])
        if a['supervised_tokens']<1:raise ValueError('Zero supervision')
        profiles.append({'id':after['id'],'before_supervised_tokens':b['supervised_tokens'],
            'after_supervised_tokens':a['supervised_tokens'],'before_truncated':b['truncated'],
            'after_truncated':a['truncated']})
    # Immutable review snapshots only. No derived train or active configuration is written.
    save(OUT/'reviews-final.json',reviews)
    save(OUT/'approved-rules.json',[c for c in reviews if c['approved']])
    save(OUT/'changes-preview.json',changes)
    save(OUT/'token-check.json',profiles)
    verify_material()
    paths=[OUT/n for n in ('material-freeze.json','extra-material-freeze.json','candidates.json',
        'extra-candidates.json','decisions-draft.json','decisions-final-addendum.json',
        'reviews-final.json','approved-rules.json','changes-preview.json','token-check.json')]
    paths+=[Path(__file__),Path('scripts/finalize_repeat_review.py'),
        Path('configs/training-fifth-final.json'),Path('configs/model.json'),
        Path('configs/generation.json'),Path('configs/dev-check.json'),
        Path('data/splits/bitext-v1/dev.jsonl'),
        Path('artifacts/tokenizer/bitext-bpe-v1/config.json'),
        Path('artifacts/tokenizer/bitext-bpe-v1/tokenizer.json'),BASE,BASE.parent/'manifest.json']
    manifest={'version':'refund-supplement-list-v1','frozen':True,'activated':False,
        'training_started':False,'training_revision_created':False,
        'reviewed_candidate_rules':len(reviews),'candidate_occurrences':sum(len(c['members']) for c in reviews),
        'approved_rules':sum(c['approved'] for c in reviews),'retained_rules':sum(not c['approved'] for c in reviews),
        'changed_samples_preview':len(changes),'approved_occurrences':sum(len(c['edits']) for c in changes),
        'changed_intents':dict(Counter(c['intent'] for c in changes)),
        'zero_supervision':0,'truncated_before':sum(p['before_truncated'] for p in profiles),
        'truncated_after':sum(p['after_truncated'] for p in profiles),
        'replay_entities_ids_order_nonresponse_headings_passed':True,
        'new_exact_duplicate_rewritten_sentence':0,
        'sha256':{str(p):file_hash(p) for p in paths},
        'scope_limits':read(OUT/'decisions-final-addendum.json')['scope_limits']}
    save(OUT/'manifest.json',manifest)
    print({k:v for k,v in manifest.items() if k!='sha256'})
if __name__=='__main__':main()
