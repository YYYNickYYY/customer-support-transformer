"""Frozen sixth learning-rate experiment. Explicit --train required; --check is read-only."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from utils.device import get_device
from . import revisions
from . import target_revision_v31 as revision
from . import training
from .condition_measurement_v2 import digest
from .tokenization import load_frozen, read_rows

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/training-sixth.json'
PROTOCOL = ROOT / 'docs/sixth-learning-rate-protocol-v1.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_plan():
    text=PROTOCOL.read_text(encoding='utf-8')
    plan=json.loads(text.split('```json\n',1)[1].split('\n```',1)[0])
    if plan['status']!='frozen_not_started' or not plan['dependencies_sha256']:
        raise ValueError('Sixth experiment not frozen')
    for name,h in plan['dependencies_sha256'].items():
        if digest(ROOT/name)!=h:
            raise ValueError('Sixth frozen dependency changed: '+name)
    return plan


def validate_configs(baseline,candidate):
    # Output location and every training field are identical; only LR may differ.
    if baseline['learning_rate']!=0.0003 or candidate!={**baseline,'learning_rate':0.0001}:
        raise ValueError('Sixth experiment must change only learning_rate 0.0003 -> 0.0001')
    config=training.FullTrainConfig(**candidate)
    if config.epochs!=5 or config.seed!=42 or config.train_limit is not None or config.dev_limit is not None:
        raise ValueError('Only frozen full-data five-epoch experiment allowed')
    return config


def verify_selection(selection,dev,replies,ratings):
    samples=selection['samples']; ids=[s['id'] for s in samples]
    if len(ids)!=135 or len(set(ids))!=135 or set(Counter(s['intent'] for s in samples).values())!={5}:
        raise ValueError('Expected unchanged 27 intents x 5 unique dev IDs')
    byid={r['id']:r for r in dev}
    if len(byid)!=len(dev) or len(replies)!=135 or [r['id'] for r in replies]!=ids:
        raise ValueError('Baseline replies/selection alignment changed')
    for s,r in zip(samples,replies):
        source=byid[s['id']]
        if any(s[k]!=source[k] for k in ('id','intent','category','template_id')) or r['instruction']!=source['instruction'] or r['reference_response']!=source['response']:
            raise ValueError('Fixed dev question/reference changed')
        if r['cohort']!=s['cohort']:
            raise ValueError('Cohort changed')
    fifth=[r for r in ratings['ratings'] if r['model']=='fifth']
    if len(fifth)!=135 or len({r['id'] for r in fifth})!=135 or {r['id'] for r in fifth}!=set(ids):
        raise ValueError('Incomplete fifth baseline ratings')
    for r in fifth:
        if r['intent']!=byid[r['id']]['intent'] or r['risk_rubric']!='customer-support-risk-review-v1.1' or r['other_rubric']!='customer-support-review-v1.0':
            raise ValueError('Baseline rubric/intent changed')
        for k,maximum in [('intent_score',2),('helpfulness_score',2),('fluency_score',2),('repetition',1),('unsupported_claim',1)]:
            if type(r[k]) is not int or not 0<=r[k]<=maximum:
                raise ValueError('Incomplete/invalid baseline scores')
    return dict(n=135,intent_mean=sum(r['intent_score'] for r in fifth)/135,
                help_mean=sum(r['helpfulness_score'] for r in fifth)/135,
                fluency_mean=sum(r['fluency_score'] for r in fifth)/135,
                repetition=sum(r['repetition'] for r in fifth),risk=sum(r['unsupported_claim'] for r in fifth))


def preflight(config_path=CONFIG):
    if Path.cwd().resolve()!=ROOT:
        raise ValueError('Run from the project root; VS Code task sets cwd automatically')
    plan=load_plan()
    baseline=read(ROOT/'configs/training-fifth-final.json')
    config=validate_configs(baseline,read(config_path))
    recorded=read(ROOT/plan['baseline_run']/'config.json')
    if {k:recorded[k] for k in baseline}!=baseline or recorded['subset_run']:
        raise ValueError('Fifth recorded config differs from baseline')
    device=get_device(config.device)
    if device!=recorded['resolved_device'] or str(torch.__version__)!=recorded['torch_version']:
        raise ValueError('Runtime/device changed; do not silently add a hardware factor')
    modelcfg=read(ROOT/config.model_config)
    _,tc=load_frozen(ROOT/modelcfg['tokenizer_artifact'])
    if tc['tokenizer_sha256']!=modelcfg['tokenizer_revision']:
        raise ValueError('Tokenizer revision mismatch')
    # Full immutable revision replay; no model construction, forward or optimizer.
    source=list(read_rows(ROOT/'data/splits/bitext-v1/train.jsonl'))
    train,meta=revision.load_revision(ROOT/config.train_revision,source)
    dev=list(read_rows(ROOT/'data/splits/bitext-v1/dev.jsonl'))
    experiment=read(ROOT/plan['baseline_run']/'experiment.json')
    if len(train)!=21528 or len(dev)!=2688 or meta['actual_train_sha256']!=experiment['actual_train_sha256']:
        raise ValueError('Actual train/dev changed')
    selection=read(ROOT/plan['selection'])
    replies=[json.loads(x) for x in (ROOT/plan['baseline_replies']).read_text(encoding='utf-8').splitlines()]
    metrics=verify_selection(selection,dev,replies,read(ROOT/plan['baseline_ratings']))
    summary=read(ROOT/plan['baseline_run']/'summary.json')
    status=read(ROOT/plan['baseline_run']/'status.json')
    if status['state']!='completed' or status['completed_epochs']!=5 or status['completed_steps']!=13455 or summary['subset_run']:
        raise ValueError('Incomplete fifth baseline run')
    gen=read(ROOT/'outputs/dev-expanded-fifth-20261005/config.json')
    if gen!=plan['generation_baseline'] or gen['selection_sha256']!=digest(ROOT/plan['selection']) or gen['model_config']!=modelcfg:
        raise ValueError('Frozen generation configuration changed')
    if metrics!=plan['baseline_metrics'] or summary['best_dev_loss']!=plan['baseline_best_dev_loss']:
        raise ValueError('Baseline metrics changed')
    return config,dict(verified=True,learning_rate=config.learning_rate,epochs=5,batch_size=8,seed=42,
        resolved_device=device,train_samples=len(train),dev_samples=len(dev),comparison_samples=135,
        train_sha256=meta['actual_train_sha256'],selection_sha256=digest(ROOT/plan['selection']),
        initialization='fresh seeded model; no checkpoint load or resume',baseline_metrics=metrics,
        baseline_best_dev_loss=summary['best_dev_loss'],protocol_sha256=digest(PROTOCOL),
        read_only=True,training_started=False)


def launch(config):
    original=revisions.load_revision
    def routed(path,rows):
        if read(path).get('version')==revision.VERSION:
            return revision.load_revision(path,rows)
        return original(path,rows)
    revisions.load_revision=routed
    try:
        # Same immutable engine as fifth: seeds RNGs, constructs a fresh model,
        # dedicated shuffle generator, AdamW, token-weighted dev, new UUID run dir.
        run_dir=training.run_full_training(config)
    finally:
        revisions.load_revision=original
    return run_dir


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=CONFIG)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true',help='Verify frozen data/config/135 IDs only; never train')
    action.add_argument('--train',action='store_true',help='Explicitly start the five-epoch sixth experiment')
    args=p.parse_args(argv)
    config,report=preflight(args.config)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if args.check:
        return
    print('Starting SIXTH experiment: LR=0.0001, fresh seed42, five epochs; not a resume.',flush=True)
    return launch(config)


if __name__=='__main__':
    main()
