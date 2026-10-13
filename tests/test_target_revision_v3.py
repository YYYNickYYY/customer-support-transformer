import json
from pathlib import Path
import shutil
import pytest
from my_project_1.target_revision_v3 import check_rows, check_settings, load_revision, OUT, freeze, review
from my_project_1.tokenization import read_rows

def test_fifth_only_changes_revision_and_default_task():
    check_settings()
    tasks=json.loads(Path('.vscode/tasks.json').read_text(encoding='utf-8'))['tasks']
    default=[t for t in tasks if isinstance(t.get('group'),dict) and t['group'].get('isDefault')]
    assert len(default)==1 and default[0]['label'].startswith('第五次训练')
    assert default[0]['args'][-1]=='configs/training-fifth-final.json'
    assert default[0]['dependsOn']=='核验第五次冻结清单（不训练）'

def test_field_and_entity_guard():
    a=[{'id':'x','instruction':'hi','intent':'get_refund','response':'Keep {{Refund Amount}}.'}]
    check_rows(a,a)
    with pytest.raises(ValueError,match='Entity mutation'):check_rows(a,[{**a[0],'response':'Gone.'}])
    with pytest.raises(ValueError,match='Nonresponse'):check_rows(a,[{**a[0],'id':'y'}])

def test_approved_replacements_preserve_headings_and_conditions():
    _,a,rules=review()
    by={r['number']:r for r in a['reviews']}
    assert by[4]['after'].startswith('5. **Duplicate Charges:** If you are billed more than once')
    assert 'where the new date is not suitable for you' in by[669]['after']
    assert 'locate your transaction' in by[78]['after']
    assert 'confirmation email' in by[221]['after']
    assert 'some additional information' in by[2773]['after']
    for n in [32,59,121,400,727,914,1207,1894,2428,2725]:assert not by[n]['approved']
    assert all(r['kind']=='refund-execution-whole-sentence' for r in rules)

def test_frozen_v3_replay_tamper_refuse_overwrite(tmp_path):
    source=list(read_rows('data/splits/bitext-v1/train.jsonl'))
    rows,meta=load_revision(OUT/'manifest.json',source)
    assert len(rows)==21528 and meta['changed_samples']>0
    with pytest.raises(ValueError,match='refuse overwrite'):freeze()
    dest=tmp_path/'v3';shutil.copytree(OUT,dest)
    (dest/'rules.json').write_bytes(b'[]')
    with pytest.raises(ValueError,match='artifact mismatch'):load_revision(dest/'manifest.json',source)
