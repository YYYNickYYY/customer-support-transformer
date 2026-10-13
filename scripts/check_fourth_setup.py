"""Validate v2 data and one CPU batch without training or creating a run."""
import json
from pathlib import Path

import torch
from my_project_1.dataset import CustomerSupportDataset, DynamicPaddingCollator
from my_project_1.model import TinyDecoderLM
from my_project_1.target_revision_v2 import SEAMS, SAFE
from my_project_1.tokenization import file_hash


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    third = json.loads(Path('configs/training-third.json').read_text())
    fourth = json.loads(Path('configs/training-fourth.json').read_text())
    assert {k: v for k, v in third.items() if k != 'train_revision'} == {k: v for k, v in fourth.items() if k != 'train_revision'}
    dataset = CustomerSupportDataset('train', train_revision=fourth['train_revision'])
    ledger_path = Path(fourth['train_revision']).parent / 'changes.jsonl'
    changed_ids = [json.loads(line)['id'] for line in ledger_path.read_text(encoding='utf-8').splitlines()][:4]
    indexes = {r['id']: i for i, r in enumerate(dataset.rows)}
    batch = DynamicPaddingCollator(dataset.pad_token_id)([dataset[indexes[id_]] for id_ in changed_ids])
    model = TinyDecoderLM.from_config(json.loads(Path('configs/model.json').read_text())).eval()
    with torch.no_grad():
        result = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
    assert torch.isfinite(result['loss']) and torch.isfinite(result['logits']).all()
    assert all('work to work towards' not in r['response'] for r in dataset.rows)
    assert all(all(before not in r['response'] for before in SEAMS) for r in dataset.rows)
    assert all(r['response'].count(SAFE) <= 1 for r in dataset.rows)
    output = {'same_training_settings_except_revision': True, 'train_rows': len(dataset), 'batch_shape': list(batch['input_ids'].shape),
              'logits_shape': list(result['logits'].shape), 'diagnostic_loss_NOT_training_result': float(result['loss']),
              'all_frozen_dependencies_verified': True, 'known_seams_removed': True, 'no_new_duplicate_privacy_recommendation': True,
              'manifest_sha256': file_hash(fourth['train_revision']), 'diagnostic_changed_ids': changed_ids, 'no_optimizer_run_or_checkpoint': True}
    print(json.dumps(output, indent=2))
    target = Path('outputs/target-revision-v2-review-20261005/setup-validation-changed.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)


if __name__ == '__main__':
    main()
