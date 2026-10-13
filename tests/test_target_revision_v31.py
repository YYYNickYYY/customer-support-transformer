import json
from pathlib import Path
import pytest
from my_project_1.target_revision_v31 import validate, check_settings, OUT, load_revision
from my_project_1.tokenization import read_rows

def test_final_settings_and_task_match():
    check_settings()
    tasks=json.loads(Path('.vscode/tasks.json').read_text(encoding='utf-8'))['tasks']
    task=next(t for t in tasks if t['label'].startswith('第五次训练'))
    assert 'my_project_1.training_fifth_final' in task['args']
    assert task['args'][-1]=='configs/training-fifth-final.json'

def test_duplicate_guard():
    row={'id':'1','response':'Hello.'}
    with pytest.raises(ValueError,match='Duplicate rewritten'):
        validate([row],[row],[{'id':'1','before_response':'Hello.','after_response':'Hello.', 'edits':[{'after':'Review.'},{'after':'Review.'}]}])

def test_final_replay_preserves_fragments_and_resolves_repetition():
    rows,meta=load_revision(OUT/'manifest.json',list(read_rows('data/splits/bitext-v1/train.jsonl')))
    assert len(rows)==21528 and meta['changed_samples']==495 and meta['edits']==529
    changes=[json.loads(l) for l in (OUT/'changes.jsonl').read_text(encoding='utf-8').splitlines()]
    for c in changes:
        after=[e['after'] for e in c['edits']]
        assert len(after)==len(set(after))
    ids=['6100','6268','6609','6734','6841','6881']
    for r in rows:
        if r['id'].split(':')[-1] in ids:
            assert r['response'].count('We can help you review the issue and any refund request through an official support channel.')<=1

def test_launcher_restores_router_and_uses_same_engine(monkeypatch):
    from my_project_1 import revisions, training_fifth_final as entry
    original=revisions.load_revision
    def fake(config):
        assert config.epochs==5 and config.learning_rate==0.0003
        assert config.train_revision.endswith('refund-execution-v3.1/manifest.json')
        assert revisions.load_revision is not original
        raise RuntimeError('sentinel, no training')
    monkeypatch.setattr(entry,'run_full_training',fake)
    monkeypatch.setattr('sys.argv',['fifth'])
    with pytest.raises(RuntimeError,match='sentinel'):entry.main()
    assert revisions.load_revision is original
