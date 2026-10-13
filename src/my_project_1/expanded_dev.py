"""Frozen 5-per-intent dev diagnostics; never loads sealed_test or trains."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from .dataset import CustomerSupportDataset
from .dev_check import rating_record
from .generate import generate_reply
from .model import TinyDecoderLM
from .overfit import validate_tokenizer
from .tokenization import file_hash

ROOT = Path('artifacts/dev-check-expanded-v1')
OUT = Path('outputs/dev-expanded-20261005-v1')
RUNS = {
    'second': 'runs/20261004T095337-full-666ad47f70d1422485fede4e65c16b1f',
    'third': 'runs/20261005T014129-full-c492479de8f8407c8c437a435e7e6bd9',
    'fourth': 'runs/20261005T093449-full-6181db843d8543ba96a9d83e60a81857',
}
OPTIONS = {'seed':42,'device':'cpu','cpu_threads':2,'max_new_tokens':320,
           'decoding':'greedy; block PAD/BOS/USER/ASSISTANT'}

def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, payload):
    """Create once; an identical rerun is idempotent, differing writes fail."""
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)+'\n'
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError('Refuse overwrite: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

def choose_expanded(rows, original, per_intent=5):
    if type(per_intent) is not int or per_intent < 1:
        raise ValueError('Invalid per_intent')
    by_id = {r['id']:r for r in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError('Empty/duplicate rows')
    groups = {}
    for r in rows:
        groups.setdefault(r['intent'], []).append(r)
    old = {}
    for item in original:
        if item['intent'] in old or item['id'] not in by_id:
            raise ValueError('Invalid old selection')
        r=by_id[item['id']]
        if any(item[k] != r[k] for k in ('intent','category','template_id')):
            raise ValueError('Old selection metadata mismatch')
        old[item['intent']] = r
    if set(old) != set(groups):
        raise ValueError('Intent mismatch')
    selected, coverage = [], []
    for intent in sorted(groups):
        pool = sorted(groups[intent],key=lambda r:(digest('expanded-dev-v1:42:'+r['id']),r['id']))
        if len(pool) < per_intent:
            raise ValueError('Too few rows '+intent)
        chosen=[old[intent]]; templates={chosen[0]['template_id']}
        for r in pool:
            if len(chosen)==per_intent:
                break
            if r['template_id'] not in templates:
                chosen.append(r); templates.add(r['template_id'])
        fallback=per_intent-len(chosen)
        ids={r['id'] for r in chosen}
        for r in pool:
            if len(chosen)==per_intent:
                break
            if r['id'] not in ids:
                chosen.append(r); ids.add(r['id'])
        selected.extend({**{k:r[k] for k in ('id','intent','category','template_id')},
                         'cohort':'original27' if i==0 else 'new108'}
                        for i,r in enumerate(chosen))
        coverage.append({'intent':intent,'dev_rows':len(pool),
                         'dev_template_groups':len({r['template_id'] for r in pool}),
                         'selected_template_groups':len(templates),'fallback_rows':fallback})
    return {'version':'dev-check-expanded-v1','split':'dev','per_intent':per_intent,
            'samples':selected,'coverage':coverage}

def freeze(dataset, create=False):
    old_path=Path('artifacts/dev-check-v1/selection.json')
    old_freeze=read(old_path.parent/'freeze.json')
    if file_hash(old_path)!=old_freeze['file_sha256']['selection.json']:
        raise ValueError('Original freeze hash')
    selection=choose_expanded(dataset.rows,read(old_path)['samples'])
    dependencies=['docs/expanded-dev-protocol-v1.md',str(Path(__file__)),
                  str(old_path),'data/splits/bitext-v1/dev.jsonl',
                  'docs/evaluation-risk-rubric-v1.1.md','docs/evaluation-rubric-v1.md']
    models={}
    for name,run in RUNS.items():
        if read(Path(run)/'status.json')['state']!='completed':
            raise ValueError('Incomplete '+name)
        checkpoint=Path(run)/'best.pt'
        models[name]={'checkpoint':str(checkpoint),'sha256':file_hash(checkpoint)}
    manifest={'dependencies_sha256':{p:file_hash(p) for p in dependencies},
              'models':models,'generation':OPTIONS}
    if create:
        save(ROOT/'selection.json',selection)
        manifest['selection_sha256']=file_hash(ROOT/'selection.json')
        save(ROOT/'freeze.json',manifest)
    manifest['selection_sha256']=file_hash(ROOT/'selection.json')
    if read(ROOT/'selection.json')!=selection or read(ROOT/'freeze.json')!=manifest:
        raise ValueError('Expanded freeze mismatch')
    return selection,manifest

def generate(model_name):
    dataset=CustomerSupportDataset('dev')
    selection,manifest=freeze(dataset)
    torch.set_num_threads(2); torch.manual_seed(42)
    checkpoint_path=Path(RUNS[model_name])/'best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    config=checkpoint['model_config'];validate_tokenizer(config,dataset)
    model=TinyDecoderLM.from_config(config);model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    directory=OUT/model_name;directory.mkdir(parents=True,exist_ok=True)
    save(directory/'config.json',{**OPTIONS,'model_config':config,
         'checkpoint':str(checkpoint_path),'checkpoint_sha256':manifest['models'][model_name]['sha256'],
         'selection_sha256':manifest['selection_sha256'],'scope':'fixed dev diagnostic, not sealed_test'})
    rows={r['id']:r for r in dataset.rows}; path=directory/'replies.jsonl'
    prior=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    if len(prior)>len(selection['samples']):
        raise ValueError('Too many prior replies')
    for item,r in zip(selection['samples'],prior):
        source=rows[item['id']]
        if any(r[k]!=source[k] for k in ('id','intent','instruction')) or r['reference_response']!=source['response'] or r['cohort']!=item['cohort']:
            raise ValueError('Invalid prior prefix')
    with path.open('a' if prior else 'x',encoding='utf-8') as stream:
        for index,item in enumerate(selection['samples'][len(prior):],len(prior)+1):
            row=rows[item['id']]
            result=generate_reply(model,dataset.tokenizer,config,row['instruction'],max_new_tokens=320)
            record={**rating_record(row,result),'generated_token_ids':result['generated_token_ids'],'cohort':item['cohort']}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();prior.append(record)
            print(f'{model_name} {index}/135 {row["intent"]}: {result["stop_reason"]}',flush=True)
    save(directory/'summary.json',{'examples':len(prior),'stop_counts':dict(Counter(r['stop_reason'] for r in prior)),
         'replies_sha256':file_hash(path),'cohorts':dict(Counter(r['cohort'] for r in prior))})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--model',choices=list(RUNS))
    args=parser.parse_args()
    if args.freeze:
        selection,manifest=freeze(CustomerSupportDataset('dev'),create=True)
        print(json.dumps({'samples':len(selection['samples']),'sha256':manifest['selection_sha256'],'coverage':selection['coverage']},ensure_ascii=False))
    elif args.model:
        generate(args.model)
    else:
        freeze(CustomerSupportDataset('dev'));print('Frozen expanded dev verified')

if __name__=='__main__':
    main()
