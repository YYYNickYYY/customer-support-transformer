"""Epoch budget 5 -> 8 only, fresh initialization; explicit --train required."""
import argparse
import json
from pathlib import Path

import torch

from . import training_seventh as seventh
from .tokenization import file_hash

ROOT = seventh.ROOT
CONFIG = ROOT / 'configs/training-eighth.json'
PROTOCOL = ROOT / 'docs/eighth-epochs-protocol-v1.md'
FREEZE = ROOT / 'artifacts/experiments/eighth-epochs-v1/freeze.json'


def validate_configs(baseline, candidate):
    if baseline['epochs'] != 5 or candidate != {**baseline, 'epochs': 8}:
        raise ValueError('Only epochs 5 -> 8 may change; no resume or second factor')
    if (candidate['model_config'], candidate['learning_rate'], candidate['batch_size'], candidate['seed']) != (
            'configs/model-six-layer.json', 0.0003, 8, 42):
        raise ValueError('Expected unchanged seventh settings')
    return seventh.baseline_gate.training.FullTrainConfig(**candidate)


def preflight(config_path=CONFIG):
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from project root')
    read = seventh.baseline_gate.read
    plan = read(FREEZE)
    if plan['version'] != 'eighth-epochs-v1' or not plan['dependencies_sha256']:
        raise ValueError('Missing frozen eighth contract')
    for name, expected in plan['dependencies_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Eighth frozen dependency changed: ' + name)
    baseline = read(seventh.CONFIG)
    config = validate_configs(baseline, read(config_path))
    # Full immutable v3.1 replay and tokenizer/runtime/135-question checks.
    # This gate does not construct a model, load weights or run an optimizer.
    _, report = seventh.preflight()
    run = ROOT / plan['baseline_run']
    recorded = read(run / 'config.json')
    if {k:recorded[k] for k in baseline} != baseline or recorded['subset_run']:
        raise ValueError('Recorded seventh baseline changed')
    if recorded['resolved_device'] != report['resolved_device'] or recorded['torch_version'] != str(torch.__version__):
        raise ValueError('Runtime/device changed')
    status, summary = read(run / 'status.json'), read(run / 'summary.json')
    if (status['state'],status['completed_epochs'],status['completed_steps']) != ('completed',5,13455):
        raise ValueError('Incomplete seventh baseline')
    if summary['subset_run'] or summary['best_dev_loss'] != plan['baseline_best_dev_loss']:
        raise ValueError('Seventh baseline summary changed')
    evaluation = read(ROOT / plan['baseline_review'])
    generation = read(ROOT / plan['baseline_generation'])
    if generation != plan['generation_baseline'] or evaluation['metrics']['seventh'] != plan['baseline_metrics']:
        raise ValueError('Seventh final generation/scoring baseline changed')
    if evaluation['best_dev_loss']['seventh'] != summary['best_dev_loss']:
        raise ValueError('Baseline evaluation/run mismatch')
    return config, {**report, 'experiment':'eighth-epochs-v1', 'epochs':8,
        'expected_steps':21528, 'baseline_metrics':plan['baseline_metrics'],
        'baseline_best_dev_loss':summary['best_dev_loss'], 'baseline_model':'seventh 6-layer best',
        'protocol_sha256':file_hash(PROTOCOL), 'freeze_sha256':file_hash(FREEZE),
        'initialization':'fresh seed42, same six-layer architecture; no checkpoint load or resume',
        'comparison':'same 135 dev, original greedy, versus final calibrated seventh scores',
        'read_only':True, 'training_started':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--train', action='store_true')
    args = parser.parse_args(argv)
    config, report = preflight(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        return
    print('Starting EIGHTH: 6 layers, LR0.0003, fresh seed42, 8 epochs; not a resume.', flush=True)
    return seventh.baseline_gate.launch(config)


if __name__ == '__main__':
    main()
