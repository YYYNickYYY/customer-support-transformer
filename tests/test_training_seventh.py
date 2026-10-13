"""Contract tests only; no training or checkpoint loading."""
import json

import pytest

from my_project_1 import training_seventh as seventh


def configs():
    read = seventh.baseline_gate.read
    return (read(seventh.ROOT / 'configs/training-fifth-final.json'), read(seventh.CONFIG),
            read(seventh.ROOT / 'configs/model.json'), read(seventh.ROOT / 'configs/model-six-layer.json'))


def test_depth_only():
    old, new, om, nm = configs()
    result = seventh.validate_configs(old, new, om, nm)
    assert result.learning_rate == 0.0003 and result.epochs == 5
    assert [k for k in old if old[k] != new[k]] == ['model_config']
    assert [k for k in om if om[k] != nm[k]] == ['n_layers']


@pytest.mark.parametrize('key,value', [('learning_rate', 0.0001), ('epochs', 6), ('seed', 43),
    ('batch_size', 16), ('resume', 'best.pt'), ('train_limit', 16), ('dev_limit', 16),
    ('train_revision', 'other'), ('device', 'cuda'), ('cpu_threads', 4)])
def test_no_second_training_factor(key, value):
    old, new, om, nm = configs()
    with pytest.raises(ValueError):
        seventh.validate_configs(old, {**new, key: value}, om, nm)


@pytest.mark.parametrize('key,value', [('n_layers', 5), ('d_model', 256), ('n_heads', 8),
                                    ('d_ff', 1024), ('dropout', 0.0), ('max_length', 1024)])
def test_no_second_model_factor(key, value):
    old, new, om, nm = configs()
    with pytest.raises(ValueError):
        seventh.validate_configs(old, new, om, {**nm, key: value})


def test_explicit_mode_and_check_no_launch(monkeypatch):
    with pytest.raises(SystemExit):
        seventh.main([])
    with pytest.raises(SystemExit):
        seventh.main(['--train', '--check'])
    monkeypatch.setattr(seventh, 'preflight', lambda _: (object(), {'training_started': False}))
    monkeypatch.setattr(seventh.baseline_gate, 'launch', lambda _: pytest.fail('No train in check'))
    seventh.main(['--check'])


def test_train_routes_only_validated_configuration(monkeypatch):
    config = object()
    calls = []
    monkeypatch.setattr(seventh, 'preflight', lambda _: (config, {'training_started': False}))
    monkeypatch.setattr(seventh.baseline_gate, 'launch', lambda c: calls.append(c))
    seventh.main(['--train'])
    assert calls == [config]


def test_workspace_entry():
    workspace = json.loads((seventh.ROOT / '第七次容量实验.code-workspace').read_text(encoding='utf-8'))
    check, train = workspace['tasks']['tasks']
    assert check['args'][-1] == '--check' and train['args'][-1] == '--train'
    assert train['dependsOn'] == check['label'] and train['dependsOrder'] == 'sequence'
    assert not any(t.get('group') for t in workspace['tasks']['tasks'])


def test_real_preflight_is_read_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No weights, forward or optimizer in preflight')
    gate = seventh.baseline_gate
    monkeypatch.setattr(gate.torch, 'load', forbidden)
    monkeypatch.setattr(gate.training, 'run_full_training', forbidden)
    monkeypatch.setattr(gate.torch.optim.AdamW, 'step', forbidden)
    config, report = seventh.preflight()
    assert report['verified'] and not report['training_started']
    assert report['n_layers'] == 6 and report['learning_rate'] == config.learning_rate == 0.0003
    assert report['comparison_samples'] == 135 and report['train_samples'] == 21528
