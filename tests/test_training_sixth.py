"""No training/weight loading: sixth experiment contract and mocked launch tests."""
import ast
import copy
import json

import pytest

from my_project_1 import training_sixth as sixth


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('This test must not train or load weights')
    monkeypatch.setattr(sixth.training, 'run_full_training', forbidden)
    monkeypatch.setattr(sixth.torch, 'load', forbidden)
    monkeypatch.setattr(sixth.torch.optim.AdamW, 'step', forbidden)


def baseline():
    return sixth.read(sixth.ROOT / 'configs/training-fifth-final.json')


def test_only_lr_changed():
    old = baseline()
    new = sixth.read(sixth.CONFIG)
    assert [k for k in old if old[k] != new[k]] == ['learning_rate']
    config = sixth.validate_configs(old, new)
    assert (config.learning_rate, config.seed, config.epochs) == (0.0001, 42, 5)


@pytest.mark.parametrize('key,value', [
    ('epochs', 6), ('seed', 43), ('batch_size', 16), ('learning_rate', 0.0002),
    ('model_config', 'other.json'), ('train_revision', 'other.json'),
    ('output_dir', 'old-run'), ('train_limit', 32), ('dev_limit', 32),
    ('resume', 'best.pt'), ('device', 'cuda'), ('weight_decay', 0.02),
])
def test_reject_second_factor(key, value):
    new = {**baseline(), 'learning_rate': 0.0001, key: value}
    with pytest.raises(ValueError):
        sixth.validate_configs(baseline(), new)


def test_explicit_action_required():
    with pytest.raises(SystemExit) as exc:
        sixth.main([])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        sixth.main(['--check', '--train'])


def test_check_never_launches(monkeypatch, capsys):
    monkeypatch.setattr(sixth, 'preflight', lambda _: (object(), {'training_started': False}))
    assert sixth.main(['--check']) is None
    assert 'false' in capsys.readouterr().out


def test_explicit_train_uses_validated_config(monkeypatch):
    config = object()
    monkeypatch.setattr(sixth, 'preflight', lambda _: (config, {'verified': True}))
    calls = []
    monkeypatch.setattr(sixth, 'launch', lambda c: calls.append(c))
    sixth.main(['--train'])
    assert calls == [config]


@pytest.mark.parametrize('fail', [False, True])
def test_mock_engine_router_restored(monkeypatch, fail):
    original = sixth.revisions.load_revision
    monkeypatch.setattr(sixth, 'read', lambda _: {'version': sixth.revision.VERSION})
    monkeypatch.setattr(sixth.revision, 'load_revision', lambda p, r: (r, 'v3.1'))
    config = object()
    def fake_engine(c):
        assert c is config
        assert sixth.revisions.load_revision('manifest', ['same rows']) == (['same rows'], 'v3.1')
        if fail:
            raise RuntimeError('mock interruption')
        return 'unique-new-run'
    monkeypatch.setattr(sixth.training, 'run_full_training', fake_engine)
    if fail:
        with pytest.raises(RuntimeError):
            sixth.launch(config)
    else:
        assert sixth.launch(config) == 'unique-new-run'
    assert sixth.revisions.load_revision is original


def comparison():
    plan = sixth.load_plan()
    selection = sixth.read(sixth.ROOT / plan['selection'])
    dev = list(sixth.read_rows(sixth.ROOT / 'data/splits/bitext-v1/dev.jsonl'))
    replies = [json.loads(x) for x in (sixth.ROOT / plan['baseline_replies']).read_text(encoding='utf-8').splitlines()]
    ratings = sixth.read(sixth.ROOT / plan['baseline_ratings'])
    return plan, selection, dev, replies, ratings


def test_same_135_and_corrected_baseline():
    plan, selection, dev, replies, ratings = comparison()
    scores = sixth.verify_selection(selection, dev, replies, ratings)
    assert scores == plan['baseline_metrics']
    assert scores['repetition'] == 63
    assert scores['risk'] == 43


@pytest.mark.parametrize('mutation', ['duplicate', 'order', 'question', 'rubric', 'missing'])
def test_reject_comparison_drift(mutation):
    _, selection, dev, replies, ratings = comparison()
    if mutation == 'duplicate':
        selection['samples'][1] = copy.deepcopy(selection['samples'][0])
    elif mutation == 'order':
        replies[0], replies[1] = replies[1], replies[0]
    elif mutation == 'question':
        replies[0]['instruction'] = 'different question'
    elif mutation == 'rubric':
        next(r for r in ratings['ratings'] if r['model'] == 'fifth')['risk_rubric'] = 'v2'
    else:
        ratings['ratings'] = [r for r in ratings['ratings'] if r['model'] != 'fifth'][1:]
    with pytest.raises(ValueError):
        sixth.verify_selection(selection, dev, replies, ratings)


def test_hash_drift_rejected_before_engine(monkeypatch):
    monkeypatch.setattr(sixth, 'digest', lambda _: 'wrong')
    with pytest.raises(ValueError, match='dependency changed'):
        sixth.preflight()


def test_engine_fresh_initialization_contract():
    # AST inspection only: no forward, optimizer creation/update or checkpoint load.
    source = (sixth.ROOT / 'src/my_project_1/training.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_full_training')
    text = ast.get_source_segment(source, function)
    assert 'torch.manual_seed(config.seed)' in text
    assert 'random.seed(config.seed)' in text
    assert 'TinyDecoderLM.from_config' in text
    assert 'torch.load' not in text
    assert 'exist_ok=False' in text


def test_vscode_independent_entry_and_gate():
    tasks = sixth.read(sixth.ROOT / '.vscode/tasks.json')['tasks']
    check = next(t for t in tasks if t['label'] == '核验第六次学习率实验（不训练）')
    train = next(t for t in tasks if t['label'].startswith('第六次训练'))
    assert check['args'][-1] == '--check'
    assert train['args'][-1] == '--train'
    assert train['dependsOn'] == check['label']
    for task in (check, train):
        assert task['options']['cwd'] == '${workspaceFolder}'
        assert task['command'] == '${workspaceFolder}/.venv/Scripts/python.exe'
        assert 'my_project_1.training_sixth' in task['args']
    defaults = [t for t in tasks if isinstance(t.get('group'), dict) and t['group'].get('isDefault')]
    assert len(defaults) == 1 and defaults[0]['label'].startswith('第五次训练')
