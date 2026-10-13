"""Freeze wider sentence inventory within the explicitly approved records only."""
import re
from pathlib import Path
from my_project_1.expanded_dev import read, save
from my_project_1.target_revision_v2 import sentence_spans
from my_project_1.tokenization import read_rows, file_hash

OUT=Path('outputs/refund-supplement-20261006-v1')
def main():
    candidates=read(OUT/'candidates.json'); draft=read(OUT/'decisions-draft.json')
    ids={m['id'] for c in candidates if c['review_id'] in draft['approved'] for m in c['members']}
    existing={c['before'] for c in candidates}
    rules={}
    for row in read_rows('data/derived/refund-execution-v3.1/train.jsonl'):
        if row['id'] not in ids:continue
        for start,end,s in sentence_spans(row['response']):
            if s in existing or not re.search(r'\b(?:ensur\w*|guarantee\w*|promis\w*|secur\w*)\b|\bmake sure\b|\bdeliver a satisfactory resolution\b',s,re.I):continue
            rule=rules.setdefault(s,{'before':s,'members':[]})
            rule['members'].append({'id':row['id'],'intent':row['intent'],
                'instruction':row['instruction'],'response':row['response'],'start':start,'end':end})
    result=[{'review_id':f'E{i:03}',**rules[s]} for i,s in enumerate(sorted(rules),1)]
    save(OUT/'extra-candidates.json',result)
    save(OUT/'extra-material-freeze.json',{'scope':'Every broader-marker sentence in initial approved records, not all train',
        'candidates':len(result),'occurrences':sum(len(c['members']) for c in result),
        'sha256':{str(p):file_hash(p) for p in [OUT/'candidates.json',OUT/'decisions-draft.json',
            OUT/'extra-candidates.json',Path(__file__)]}})
    print(len(result),sum(len(c['members']) for c in result))
if __name__=='__main__':main()
