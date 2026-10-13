"""Seventh experiment: depth 4 -> 6 only. No training without explicit --train."""
import argparse
import json
from pathlib import Path

from . import training_sixth as baseline_gate
from .tokenization import file_hash

ROOT = baseline_gate.ROOT
CONFIG = ROOT / 'configs/training-seventh.json'
FREEZE = ROOT / 'artifacts/experiments/seventh-depth-v1/freeze.json'


def validate_configs(old, new, old_model, new_model):
    if new != {**old, 'model_config': 'configs/model-six-layer.json'}:
        raise ValueError('Only the model config path may change; no resume or second training factor')
    if old_model['n_layers'] != 4 or new_model != {**old_model, 'n_layers': 6}:
        raise ValueError('Only n_layers 4 -> 6 may change')
    if (new['learning_rate'], new['epochs'], new['seed'], new['batch_size']) != (0.0003, 5, 42, 8):
        raise ValueError('Expected fifth training settings')
    return baseline_gate.training.FullTrainConfig(**new)


def preflight(config_path=CONFIG):
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from project root')
    plan = baseline_gate.read(FREEZE)
    if plan['version'] != 'seventh-depth-v1' or not plan['dependencies_sha256']:
        raise ValueError('Missing frozen seventh contract')
    for name, expected in plan['dependencies_sha256'].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('Seventh frozen dependency changed: ' + name)
    old = baseline_gate.read(ROOT / 'configs/training-fifth-final.json')
    model = baseline_gate.read(ROOT / 'configs/model-six-layer.json')
    config = validate_configs(old, baseline_gate.read(config_path),
                              baseline_gate.read(ROOT / old['model_config']), model)
    # Reuse immutable baseline gate for full v3.1 replay, fifth run, runtime,
    # tokenizer, all 135 questions and authoritative fifth scores. No model load.
    _, report = baseline_gate.preflight()
    if plan['generation_baseline'] != baseline_gate.load_plan()['generation_baseline']:
        raise ValueError('Original greedy configuration changed')
    return config, {**report, 'experiment': 'seventh-depth-v1',
                    'learning_rate': config.learning_rate, 'n_layers': model['n_layers'],
                    'model_config': model, 'freeze_sha256': file_hash(FREEZE),
                    'initialization': 'fresh seed42; no fifth/sixth checkpoint loaded',
                    'comparison': 'same 135 dev, original greedy; evaluation pending training'}


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
    print('Starting SEVENTH: 6 layers, LR0.0003, fresh seed42, 5 epochs; not a resume.', flush=True)
    # Same fifth engine / revision router. New UUID run directory, never overwrite.
    return baseline_gate.launch(config)


if __name__ == '__main__':
    main()
