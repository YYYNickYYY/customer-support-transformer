"""Run immutable sixth-best generation on the same 135 dev inputs; never train."""
from collections import Counter
import json
from pathlib import Path

import torch

from my_project_1.training_sixth import load_plan, read, validate_configs
from my_project_1.expanded_dev import freeze, save
from my_project_1.dataset import CustomerSupportDataset
from my_project_1.model import TinyDecoderLM
from my_project_1.generate import generate_reply
from my_project_1.dev_check import rating_record
from my_project_1.overfit import validate_tokenizer
from my_project_1.tokenization import file_hash, encode_parts

RUN = Path('runs/20261006T150952-full-917ad4a16bfb43669f2790ea2c903d92')
OUT = Path('outputs/dev-expanded-sixth-20261007')


def main():
    plan = load_plan()
    baseline = read(Path(plan['baseline_run']) / 'config.json')
    config = read(RUN / 'config.json')
    fields = read('configs/training-sixth.json')
    validate_configs(read('configs/training-fifth-final.json'), {k: config[k] for k in fields})
    assert {k: v for k, v in baseline.items() if k not in ('run_id', 'learning_rate')} == {
        k: v for k, v in config.items() if k not in ('run_id', 'learning_rate')}
    summary = read(RUN / 'summary.json')
    status = read(RUN / 'status.json')
    assert status['state'] == 'completed' and status['completed_epochs'] == 5 and status['completed_steps'] == 13455
    assert summary['subset_run'] is False
    assert read(RUN / 'experiment.json') == read(Path(plan['baseline_run']) / 'experiment.json')
    dataset = CustomerSupportDataset('dev')
    selection, old_freeze = freeze(dataset)
    checkpoint = RUN / 'best.pt'
    options = plan['generation_baseline']
    torch.set_num_threads(options['cpu_threads'])
    torch.manual_seed(options['seed'])
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert data['model_config'] == options['model_config']
    assert data['training_config'] == fields
    assert data['epoch'] == summary['best_epoch'] == 5 and data['step'] == 13455
    validate_tokenizer(data['model_config'], dataset)
    generation = {**options, 'checkpoint': str(checkpoint), 'checkpoint_sha256': file_hash(checkpoint)}
    assert generation['selection_sha256'] == old_freeze['selection_sha256']
    save(OUT / 'config.json', generation)
    deps = [str(checkpoint), *(str(RUN / n) for n in ('config.json', 'summary.json', 'status.json', 'experiment.json')),
            'docs/sixth-learning-rate-protocol-v1.md', str(Path(__file__)), plan['selection'], plan['baseline_replies'],
            plan['baseline_ratings'], 'data/splits/bitext-v1/dev.jsonl', 'docs/evaluation-risk-rubric-v1.1.md',
            'docs/evaluation-rubric-v1.md', 'src/my_project_1/generate.py']
    rows = {r['id']: r for r in dataset.rows}
    prompts = [{ 'id': s['id'], 'input_ids': encode_parts(dataset.tokenizer, rows[s['id']]['instruction'], '')[2][:-1]}
               for s in selection['samples']]
    save(OUT / 'freeze.json', {'dependencies_sha256': {p: file_hash(p) for p in deps},
         'generation': generation, 'inputs': prompts, 'scope': 'same 135 dev, sixth best; no training or sealed_test'})
    path = OUT / 'replies.jsonl'
    prior = [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert len(prior) <= 135
    for item, r in zip(selection['samples'], prior):
        source = rows[item['id']]
        assert all(r[k] == source[k] for k in ('id', 'instruction', 'intent'))
        assert r['cohort'] == item['cohort'] and r['reference_response'] == source['response']
    if len(prior) == 135:
        assert read(OUT / 'summary.json')['replies_sha256'] == file_hash(path)
        print('Complete result verified; no new generation.', flush=True)
        return
    model = TinyDecoderLM.from_config(data['model_config'])
    model.load_state_dict(data['model_state_dict'], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with path.open('a' if prior else 'x', encoding='utf-8') as f:
        for i, item in enumerate(selection['samples'][len(prior):], len(prior) + 1):
            row = rows[item['id']]
            result = generate_reply(model, dataset.tokenizer, data['model_config'], row['instruction'],
                                    max_new_tokens=generation['max_new_tokens'])
            record = {**rating_record(row, result), 'generated_token_ids': result['generated_token_ids'],
                      'prompt_tokens': result['prompt_tokens'], 'cohort': item['cohort']}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            prior.append(record)
            print(f'{i}/135 {row["intent"]}: {result["stop_reason"]}', flush=True)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    for p, h in read(OUT / 'freeze.json')['dependencies_sha256'].items():
        assert file_hash(p) == h, p
    save(OUT / 'summary.json', {'examples': len(prior), 'stop_counts': dict(Counter(r['stop_reason'] for r in prior)),
         'replies_sha256': file_hash(path), 'same_selection_and_generation': True, 'training_summary': summary,
         'weights_unchanged': True, 'optimizer_updates': 0})


if __name__ == '__main__':
    main()
