"""Additional explicit review of79 execution/remedy exclusions before freeze."""
import re
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import file_hash
from approve_target_v3 import replacement
ROOT=Path('outputs/target-revision-v3-review-20261005')
EXTRA={1,54,105,197,204,337,343,507,604,711,744,755,869,909,966,1068,1105,1128,1150,1170,1174,1176,1385,1643,1715,1719,1725,1814,1852,1858,1932,2045,2289,2507,2635,2863}

def main():
    p=read(ROOT/'approval.json');reviews=[]
    for r in p['screened_excluded']:
        if not re.search(r'\b(?:rectif\w*|issue a refund|process a refund|provide a refund)\b',r['before'],re.I):continue
        yes=r['number'] in EXTRA
        reviews.append({**r,'approved':yes,'after':replacement(r['before']) if yes else None,
            'reason':'AI逐句复核：独立执行/补救承诺（含缩写或混合句），条件及标题保留' if yes else 'AI逐句复核：努力/可能/一般政策/引导申请或非确定结果，保守保留'})
    assert len(reviews)==79 and EXTRA<={r['number'] for r in reviews}
    rules=p['rules']+[{'id':r['id'],'intent':'*','before':r['before'],'after':r['after'],'kind':'refund-execution-whole-sentence'} for r in reviews if r['approved']]
    overrides={
        26:'If the delivery issue cannot be resolved, we can help you review any refund request through an official support channel.',
        78:'With this information, we can locate your transaction and help you review a refund request through an official support channel.',
        221:'6. Confirmation and Refund: After canceling the order, you will receive a confirmation email. Please contact an official support channel to review any refund request according to the applicable refund policy.',
        2230:'5. If your request is approved, you will receive a confirmation. Please contact an official support channel to review the next steps according to the applicable refund policy.',
        2891:'6. Confirmation and Refund: After successfully cancelling your order, you will receive a confirmation email. If applicable, please contact an official support channel to review your refund request according to the refund policy.'}
    numbers={r['id']:r['number'] for r in p['reviews']+reviews}
    for r in rules:
        r['after']=replacement(r['before'])
        if r['before']=='In order to process the rebate, I will need some additional information from you.':
            r['after']='To help review your rebate request, I will need some additional information from you.'
        if numbers[r['id']] in overrides:r['after']=overrides[numbers[r['id']]]
    # Freeze the repaired heading/condition-preserving replacements; retain earlier draft for audit.
    by={r['id']:r['after'] for r in rules}
    decisions=p['reviews']+reviews
    for d in decisions:
        if d['approved']:d['after']=by[d['id']]
    save(ROOT/'approval-freeze-ready.json',{'initial_approval_sha256':file_hash(ROOT/'approval.json'),
        'proposals_sha256':p['proposals_sha256'],'reviews':decisions,
        'screened_only':[r for r in p['screened_excluded'] if r['id'] not in {x['id'] for x in reviews}],
        'rules':rules,'method':'AI full-sentence review of603 nominations plus79 execution/remedy exclusions; necessary ambiguous contexts checked, not all member full responses. Screened-only exclusions not claimed manually reviewed. No merchant facts verified.'})
    print('Final rules',len(rules),'explicit decisions',len(p['reviews'])+len(reviews))

if __name__=='__main__':main()
