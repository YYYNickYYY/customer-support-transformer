"""Preselected guarantee diagnostics, read-only original/v1/v2 targets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from my_project_1.expanded_dev import save, read, digest
from my_project_1.tokenization import file_hash
from my_project_1.dataset import CustomerSupportDataset

OUT=Path('outputs/dev-expanded-20261005-v1/target-audit')
INTENTS=['newsletter_subscription','get_refund','check_refund_policy','delivery_period','track_order','track_refund']

def candidate(text):
    return bool(re.search(r'\b(?:ensur\w*|guarantee\w*|promis\w*|will|able)\b',text,re.I)
                and re.search(r'\b(?:cancel\w*|unsubscrib\w*|refund\w*|rectif\w*|accurat\w*|resolution|resolv\w*)\b',text,re.I))

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    original=CustomerSupportDataset('train')
    v1=CustomerSupportDataset('train',train_revision='data/derived/result-privacy-v1/manifest.json')
    v2=CustomerSupportDataset('train',train_revision='data/derived/result-privacy-v2/manifest.json')
    maps=[{r['id']:r for r in d.rows} for d in (original,v1,v2)]
    selection=[]; counts=[]
    for intent in INTENTS:
        pool=[r for r in v2.rows if r['intent']==intent]
        ordered=sorted(pool,key=lambda r:(digest('guarantee-audit-v2:42:'+r['id']),r['id']))
        layers={
          'changed':[r for r in ordered if maps[1][r['id']]['response']!=r['response']],
          'unchanged':[r for r in ordered if maps[1][r['id']]['response']==r['response']],
          'candidate':[r for r in ordered if candidate(r['response'])],
        }
        counts.append({'intent':intent,'total':len(pool),**{k:len(v) for k,v in layers.items()}})
        for layer,items in layers.items():
            for r in items[:2 if layer=='candidate' else 3]:
                selection.append({'id':r['id'],'intent':intent,'layer':layer})
    paths=['docs/expanded-dev-protocol-v1.md',str(Path(__file__)),
           'data/splits/bitext-v1/train.jsonl','data/derived/result-privacy-v1/train.jsonl',
           'data/derived/result-privacy-v2/train.jsonl','data/derived/result-privacy-v1/manifest.json',
           'data/derived/result-privacy-v2/manifest.json']
    frozen={'selection':selection,'counts':counts,'dependencies_sha256':{p:file_hash(p) for p in paths},
            'method':'hash before content review; up to3 changed/unchanged, up to2 broad candidates; overlapping layers not population rates'}
    if args.freeze:
        save(OUT/'selection.json',frozen)
        print(json.dumps({'unique_ids':len({r['id'] for r in selection}),'layer_positions':len(selection),'counts':counts}));return
    if read(OUT/'selection.json')!=frozen:raise ValueError('Audit selection hash mismatch')
    unique=list(dict.fromkeys(r['id'] for r in selection))
    records=[]
    for id in unique:
        rows=[m[id] for m in maps]
        records.append({'id':id,'intent':rows[0]['intent'],'instruction':rows[0]['instruction'],
             'layers':[r['layer'] for r in selection if r['id']==id],
             'original':rows[0]['response'],'v1':rows[1]['response'],'v2':rows[2]['response']})
    save(OUT/'targets.json',records)
    print('Saved full target texts:',len(records))

if __name__=='__main__':main()
