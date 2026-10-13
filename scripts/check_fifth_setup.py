"""One revised-data batch forward/backward, no optimizer or checkpoint."""
import json
from pathlib import Path
import torch
from my_project_1.target_revision_v31 import OUT, load_revision, check_settings
from my_project_1.tokenization import read_rows, prepare_example, load_frozen, file_hash
from my_project_1.dataset import DynamicPaddingCollator
from my_project_1.model import TinyDecoderLM

def main():
    torch.set_num_threads(2);torch.manual_seed(42);check_settings()
    rows,_=load_revision(OUT/'manifest.json',list(read_rows('data/splits/bitext-v1/train.jsonl')))
    changes=[json.loads(l) for l in (OUT/'changes.jsonl').read_text(encoding='utf-8').splitlines()]
    by={r['id']:r for r in rows}; selected=[by[r['id']] for r in changes[:4]]
    tokenizer,config=load_frozen('artifacts/tokenizer/bitext-bpe-v1')
    examples=[prepare_example(tokenizer,r['instruction'],r['response'],config['max_length']) for r in selected]
    pad=tokenizer.token_to_id('<PAD>')
    batch=DynamicPaddingCollator(pad)(examples)
    model=TinyDecoderLM.from_config(json.loads(Path('configs/model.json').read_text())).train()
    result=model(**batch);result['loss'].backward()
    assert torch.isfinite(result['loss']) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    report={'manifest_sha256':file_hash(OUT/'manifest.json'),'train_rows':len(rows),'shape':list(batch['input_ids'].shape),
        'diagnostic_loss_not_training_result':result['loss'].item(),'finite_loss_and_gradients':True,
        'no_optimizer_update_or_checkpoint':True,'selected_ids':[r['id'] for r in selected]}
    destination=Path('outputs/target-revision-v3-review-20261005/setup-validation-final.json')
    with destination.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
