"""Freeze context-aware lexical nominations; never auto-approve or change train."""
import re
from pathlib import Path
from my_project_1.expanded_dev import save
from my_project_1.target_revision_v2 import sentence_spans
from my_project_1.tokenization import read_rows, file_hash

OUT=Path('outputs/refund-supplement-20261006-v1')
BASE=Path('data/derived/refund-execution-v3.1/train.jsonl')
MONEY=re.compile(r'\b(refund\w*|reimburse\w*|rebate\w*|compensat\w*|restitution)\b',re.I)
PREDICATE=re.compile(
    r'\b(?:ensur\w*|guarantee\w*|promis\w*)\b.{0,130}\b(?:resolution|resolved|compensation|in order|processed|completed|credited|issued)\b'
    r'|\b(?:ensur\w*|guarantee\w*|promis\w*)\b.{0,60}\breceiv\w*\b.{0,60}\b(?:refund\w*|reimburse\w*|rebate\w*|compensat\w*|restitution)\b'
    r'|\b(?:refund|reimbursement|rebate|compensation)\b.{0,50}\bwill be (?:processed|issued|completed|credited)\b'
    r'|\b(?:you will|you.ll) (?:receive|get)\b.{0,60}\b(?:refund|reimbursement|rebate|compensation)\b'
    r'|\b(?:actively|already)\b.{0,80}\b(?:process\w*|issu\w*)\b',re.I)

def nominate(rows):
    rules={};scoped=0
    for r in rows:
        if r['intent'] not in ('get_refund','track_refund','check_refund_policy') and not MONEY.search(r['response']):
            continue
        scoped+=1
        spans=list(sentence_spans(r['response']))
        for i,(start,end,sentence) in enumerate(spans):
            if not PREDICATE.search(sentence):continue
            rule=rules.setdefault(sentence,{'before':sentence,'members':[]})
            rule['members'].append({'id':r['id'],'intent':r['intent'],
                'instruction':r['instruction'],'response':r['response'],
                'start':start,'end':end,
                'context':r['response'][spans[max(0,i-1)][0]:spans[min(len(spans)-1,i+1)][1]]})
    result=[]
    for i,s in enumerate(sorted(rules),1):
        result.append({'review_id':f'S{i:03}',**rules[s]})
    return result,scoped

def main():
    rules,scoped=nominate(list(read_rows(BASE)))
    save(OUT/'candidates.json',rules)
    deps=[BASE,BASE.parent/'manifest.json',Path('docs/refund-supplement-protocol-20261006.md'),
          Path('docs/evaluation-risk-rubric-v1.1.md'),Path(__file__),
          Path('outputs/repetition-review-20261006-v1/review-freeze.json')]
    save(OUT/'material-freeze.json',{'candidate_rules':len(rules),
         'occurrences':sum(len(r['members']) for r in rules),'scope_rows':scoped,
         'candidates_sha256':file_hash(OUT/'candidates.json'),
         'dependencies_sha256':{str(p):file_hash(p) for p in deps},
         'not_automatic_approval':True,'not_semantic_exhaustiveness':True})
    print(len(rules),sum(len(r['members']) for r in rules),scoped)
if __name__=='__main__':main()
