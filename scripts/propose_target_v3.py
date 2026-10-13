"""Nominate whole sentences for explicit review, not automatic approval."""
import re
from pathlib import Path
from my_project_1.target_revision_v2 import sentence_spans, load_revision
from my_project_1.target_revision import json_bytes, digest
from my_project_1.tokenization import read_rows, file_hash

OUT=Path('outputs/target-revision-v3-review-20261005')

def main():
    rows,_=load_revision('data/derived/result-privacy-v2/manifest.json',list(read_rows('data/splits/bitext-v1/train.jsonl')))
    rules={}
    for r in rows:
        refund=bool(re.search(r'\b(refund\w*|reimburse\w*|rebate\w*)\b',r['response'],re.I))
        if not refund:continue
        for start,end,s in sentence_spans(r['response']):
            direct=re.search(r'\b(?:process\w*|issu\w*|receiv\w*|complet\w*|credit\w*|return\w*|approv\w*|initiat\w*)\b[^.!?\n]{0,85}\b(?:refund\w*|reimburse\w*|rebate\w*)\b|\b(?:refund\w*|reimburse\w*|rebate\w*)\b[^.!?\n]{0,85}\b(?:process\w*|issu\w*|receiv\w*|complet\w*|credit\w*|return\w*|approv\w*|initiat\w*)\b',s,re.I)
            remedy=re.search(r'\b(?:rectif\w*|revers\w*|correct\w*)\b',s,re.I)
            if direct or remedy:
                entry=rules.setdefault(s,{'id':digest(s.encode())[:16],'before':s,'members':[]})
                entry['members'].append({'id':r['id'],'intent':r['intent'],'response':r['response'],'start':start,'end':end})
    out={'protocol_sha256':file_hash('docs/train-target-revision-v3.md'),'baseline_sha256':file_hash('data/derived/result-privacy-v2/manifest.json'),'rules':sorted(rules.values(),key=lambda x:x['id'])}
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/'proposals.json'
    if p.exists():raise ValueError('Refuse overwrite')
    p.write_bytes(json_bytes(out))
    print(len(out['rules']),'unique candidates',sum(len(r['members']) for r in out['rules']),'occurrences')

if __name__=='__main__':main()
