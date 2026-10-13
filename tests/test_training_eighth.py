"""Eighth contract: read-only or mocked launches, no weight load/optimizer."""
import json

import pytest

from my_project_1 import training_eighth as eighth


def configs():
    read=eighth.seventh.baseline_gate.read
    return read(eighth.seventh.CONFIG),read(eighth.CONFIG)


def test_epochs_only():
    old,new=configs()
    assert [k for k in old if old[k]!=new[k]]==['epochs']
    config=eighth.validate_configs(old,new)
    assert (config.epochs,config.learning_rate,config.batch_size,config.seed)==(8,0.0003,8,42)


@pytest.mark.parametrize('key,value', [('epochs',9),('learning_rate',0.0001),('batch_size',16),
    ('seed',43),('device','cuda'),('cpu_threads',4),('model_config','configs/model.json'),
    ('train_revision','other'),('train_limit',16),('dev_limit',16),('resume','best.pt'),
    ('output_dir','old-run'),('weight_decay',0.02),('grad_clip',0.5),('scheduler','cosine')])
def test_reject_second_factor(key,value):
    old,new=configs()
    with pytest.raises(ValueError):
        eighth.validate_configs(old,{**new,key:value})


def test_explicit_mode(monkeypatch):
    with pytest.raises(SystemExit):
        eighth.main([])
    with pytest.raises(SystemExit):
        eighth.main(['--check','--train'])


def test_check_cannot_launch(monkeypatch):
    monkeypatch.setattr(eighth,'preflight',lambda _: (object(),{'training_started':False}))
    monkeypatch.setattr(eighth.seventh.baseline_gate,'launch',lambda _: pytest.fail('No launch in check'))
    eighth.main(['--check'])


def test_train_routes_validated_eight_epochs(monkeypatch):
    _,new=configs()
    old,_=configs()
    config=eighth.validate_configs(old,new)
    monkeypatch.setattr(eighth,'preflight',lambda _: (config,{'training_started':False}))
    calls=[]
    monkeypatch.setattr(eighth.seventh.baseline_gate,'launch',lambda c: calls.append(c))
    eighth.main(['--train'])
    assert calls==[config] and calls[0].epochs==8


def test_workspace_routes_and_no_default():
    workspace=json.loads((eighth.ROOT/'第八次轮数实验.code-workspace').read_text(encoding='utf-8'))
    check,train=workspace['tasks']['tasks']
    assert check['args'][-1]=='--check' and train['args'][-1]=='--train'
    assert train['args'][4]=='my_project_1.training_eighth'
    assert train['dependsOn']==check['label'] and train['dependsOrder']=='sequence'
    assert not any(t.get('group') for t in workspace['tasks']['tasks'])


def test_actual_preflight_no_model_load_training_or_forward(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Read-only gate cannot load/train/forward')
    gate=eighth.seventh.baseline_gate
    monkeypatch.setattr(eighth.torch,'load',forbidden)
    monkeypatch.setattr(gate.training,'run_full_training',forbidden)
    monkeypatch.setattr(eighth.torch.optim.AdamW,'step',forbidden)
    monkeypatch.setattr(gate.training.TinyDecoderLM,'from_config',forbidden)
    config,report=eighth.preflight()
    assert report['verified'] and not report['training_started']
    assert report['epochs']==config.epochs==8 and report['n_layers']==6
    assert report['expected_steps']==21528 and report['comparison_samples']==135
    assert report['baseline_metrics']['helpfulness_score']==1.4
    assert report['baseline_metrics']['repetition']==45
