"""Frozen eighth best, original greedy and fixed 135 dev; never train."""
from collections import Counter
import json
from pathlib import Path

import torch

from my_project_1.training_sixth import load_plan
from my_project_1.training_seventh import validate_configs
from my_project_1.expanded_dev import read, save
from my_project_1.dataset import CustomerSupportDataset
from my_project_1.model import TinyDecoderLM
from my_project_1.generate import generate_reply
from my_project_1.dev_check import rating_record
from my_project_1.tokenization import file_hash, encode_parts

RUN = Path('runs/20261007T092605-full-2618d87d80534a10b9f1c2b02874d19c')
OUT = Path('outputs/dev-expanded-eighth-20261013')


def verify_run():
    plan = load_plan()
    depth = read('artifacts/experiments/eighth-epochs-v1/freeze.json')
    for p, h in depth['dependencies_sha256'].items():
        assert file_hash(p) == h, p
    fields = read('configs/training-eighth.json')
    mc = read(fields['model_config'])
    assert fields == {**read('configs/training-seventh.json'), 'epochs': 8}
    config = read(RUN / 'config.json')
    assert {k: config[k] for k in fields} == fields
    baseline_config = read(Path(plan['baseline_run']) / 'config.json')
    assert {k:v for k,v in config.items() if k not in ('run_id','model_config','epochs')} == {
        k:v for k,v in baseline_config.items() if k not in ('run_id','model_config','epochs')}
    status = read(RUN / 'status.json')
    assert status['state'] == 'completed' and status['completed_epochs'] == 8 and status['completed_steps'] == 21528
    assert read(RUN / 'experiment.json') == {**read(Path(plan['baseline_run']) / 'experiment.json'), 'model_config': mc}
    summary = read(RUN / 'summary.json')
    assert not summary['subset_run'] and summary['completed_steps'] == 21528
    return plan, fields, mc, summary


def main():
    plan, fields, mc, summary = verify_run()
    dataset = CustomerSupportDataset('dev', tokenizer_dir=mc['tokenizer_artifact'])
    selection = read(plan['selection'])
    rows = {r['id']: r for r in dataset.rows}
    baseline = [json.loads(x) for x in Path(plan['baseline_replies']).read_text(encoding='utf-8').splitlines()]
    assert len(selection['samples']) == len(baseline) == 135
    for s, old in zip(selection['samples'], baseline):
        assert s['id'] == old['id'] and rows[s['id']]['instruction'] == old['instruction']
    checkpoint = RUN / 'best.pt'
    generation = {**plan['generation_baseline'], 'checkpoint': str(checkpoint),
                  'checkpoint_sha256': file_hash(checkpoint), 'model_config': mc}
    save(OUT / 'config.json', generation)
    deps = [str(checkpoint), *(str(RUN/n) for n in ('config.json','summary.json','status.json','experiment.json')),
            'artifacts/experiments/eighth-epochs-v1/freeze.json', 'docs/eighth-epochs-protocol-v1.md',
            str(Path(__file__)), plan['selection'], plan['baseline_replies'], plan['baseline_ratings'],
            'data/splits/bitext-v1/dev.jsonl', 'src/my_project_1/generate.py', 'src/my_project_1/model.py',
            'src/my_project_1/attention.py', 'src/my_project_1/transformer.py', 'src/my_project_1/tokenization.py',
            'docs/evaluation-risk-rubric-v1.1.md', 'docs/evaluation-rubric-v1.md',
            mc['tokenizer_artifact']+'/tokenizer.json', mc['tokenizer_artifact']+'/config.json']
    prompts = [{'id':s['id'], 'input_ids':encode_parts(dataset.tokenizer, rows[s['id']]['instruction'], '')[2][:-1]}
               for s in selection['samples']]
    save(OUT / 'freeze.json', {'dependencies_sha256':{p:file_hash(p) for p in deps},
         'generation':generation, 'inputs':prompts, 'torch_version':str(torch.__version__)})
    path = OUT / 'replies.jsonl'
    prior = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert len(prior) <= 135
    for saved, old in zip(prior, baseline):
        assert all(saved[k] == old[k] for k in ('id','instruction','intent','cohort','reference_response'))
    if len(prior) == 135:
        assert read(OUT/'summary.json')['replies_sha256'] == file_hash(path)
        print('Complete generation verified; no new inference.', flush=True)
        return
    threads = torch.get_num_threads()
    torch.set_num_threads(generation['cpu_threads'])
    try:
        torch.manual_seed(generation['seed'])
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert data['model_config'] == mc and data['training_config'] == fields
        assert data['epoch'] == summary['best_epoch'] and data['step'] == 2691 * data['epoch']
        model = TinyDecoderLM.from_config(mc)
        model.load_state_dict(data['model_state_dict'], strict=True)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with path.open('a' if prior else 'x', encoding='utf-8') as stream:
            for i, s in enumerate(selection['samples'][len(prior):], len(prior)+1):
                row = rows[s['id']]
                result = generate_reply(model, dataset.tokenizer, mc, row['instruction'], max_new_tokens=generation['max_new_tokens'])
                record = {**rating_record(row,result), 'generated_token_ids':result['generated_token_ids'],
                          'prompt_tokens':result['prompt_tokens'], 'cohort':s['cohort']}
                stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                prior.append(record)
                print(f'{i}/135 {row["intent"]}: {result["stop_reason"]}', flush=True)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        for p,h in read(OUT/'freeze.json')['dependencies_sha256'].items():
            assert file_hash(p) == h, p
        save(OUT/'summary.json', {'examples':135,'stop_counts':dict(Counter(r['stop_reason'] for r in prior)),
             'replies_sha256':file_hash(path),'weights_unchanged':True,'optimizer_updates':0,'training_summary':summary})
    finally:
        torch.set_num_threads(threads)


if __name__ == '__main__':
    main()
