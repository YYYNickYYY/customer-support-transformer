"""Generate fifth responses on the immutable expanded dev set, never train."""
from collections import Counter
import json
from pathlib import Path
import torch
from my_project_1.expanded_dev import freeze, read, save, OPTIONS, ROOT
from my_project_1.dataset import CustomerSupportDataset
from my_project_1.model import TinyDecoderLM
from my_project_1.generate import generate_reply
from my_project_1.dev_check import rating_record
from my_project_1.overfit import validate_tokenizer
from my_project_1.tokenization import file_hash

RUN=Path('runs/20261005T132346-full-ebf624ab40734b858b46553b4da37537')
OUT=Path('outputs/dev-expanded-fifth-20261005')

def main():
    dataset=CustomerSupportDataset('dev');selection,old_freeze=freeze(dataset)
    status=read(RUN/'status.json');summary=read(RUN/'summary.json');config=read(RUN/'config.json')
    assert status['state']=='completed' and status['completed_epochs']==5 and status['completed_steps']==13455
    assert summary['subset_run'] is False and config['train_revision']=='data/derived/refund-execution-v3.1/manifest.json'
    fourth=read('runs/20261005T093449-full-6181db843d8543ba96a9d83e60a81857/config.json')
    assert {k:v for k,v in fourth.items() if k not in ('train_revision','run_id')}=={k:v for k,v in config.items() if k not in ('train_revision','run_id')}
    exp=read(RUN/'experiment.json');manifest=Path(config['train_revision'])
    assert exp['train_revision']['manifest_sha256']==file_hash(manifest)
    assert exp['actual_train_sha256']==file_hash(manifest.parent/'train.jsonl')
    checkpoint=RUN/'best.pt';torch.set_num_threads(2);torch.manual_seed(42)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    validate_tokenizer(data['model_config'],dataset)
    model=TinyDecoderLM.from_config(data['model_config']);model.load_state_dict(data['model_state_dict'],strict=True)
    generation={**OPTIONS,'model_config':data['model_config'],'checkpoint':str(checkpoint),
        'checkpoint_sha256':file_hash(checkpoint),'selection_sha256':old_freeze['selection_sha256'],'scope':'fixed dev diagnostic, not sealed_test'}
    previous=read('outputs/dev-expanded-20261005-v1/fourth/config.json')
    for k in [*OPTIONS,'model_config','selection_sha256']:assert generation[k]==previous[k]
    save(OUT/'config.json',generation)
    deps=[str(checkpoint),str(RUN/'status.json'),str(RUN/'summary.json'),str(RUN/'config.json'),str(RUN/'experiment.json'),
        str(manifest),str(manifest.parent/'train.jsonl'),str(ROOT/'selection.json'),str(ROOT/'freeze.json'),
        'docs/evaluation-risk-rubric-v1.1.md','docs/evaluation-rubric-v1.md','outputs/dev-expanded-20261005-v1/ai-expert-ratings.json',str(Path(__file__))]
    save(OUT/'freeze.json',{'dependencies_sha256':{p:file_hash(p) for p in deps},'generation':generation})
    path=OUT/'replies.jsonl';prior=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    rows={r['id']:r for r in dataset.rows}
    assert len(prior)<=135
    for r,item in zip(prior,selection['samples']):
        source=rows[item['id']]
        assert r['id']==item['id'] and r['cohort']==item['cohort'] and r['instruction']==source['instruction'] and r['reference_response']==source['response']
    with path.open('a' if prior else 'x',encoding='utf-8') as f:
        for i,item in enumerate(selection['samples'][len(prior):],len(prior)+1):
            row=rows[item['id']];result=generate_reply(model,dataset.tokenizer,data['model_config'],row['instruction'],max_new_tokens=320)
            r={**rating_record(row,result),'generated_token_ids':result['generated_token_ids'],'cohort':item['cohort']}
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();prior.append(r)
            print(f'{i}/135 {row["intent"]}: {result["stop_reason"]}',flush=True)
    save(OUT/'summary.json',{'examples':len(prior),'stop_counts':dict(Counter(r['stop_reason'] for r in prior)),
        'replies_sha256':file_hash(path),'same_selection_and_generation':True,'training_summary':summary})

if __name__=='__main__':main()
