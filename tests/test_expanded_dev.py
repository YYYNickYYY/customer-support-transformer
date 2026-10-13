import pytest
from my_project_1.expanded_dev import choose_expanded

def samples(groups=6):
    return [{'id':f'{i}', 'intent':'a','category':'c','template_id':f't{i%groups}'} for i in range(9)]

def test_deterministic_distinct_groups():
    rows=samples(); old=rows[:1]
    a=choose_expanded(rows,old)
    assert a==choose_expanded(list(reversed(rows)),old)
    assert len(a['samples'])==5
    assert len({r['template_id'] for r in a['samples']})==5
    assert a['samples'][0]['id']=='0'
    assert a['samples'][0]['cohort']=='original27'
    assert a['coverage'][0]['fallback_rows']==0

def test_fallback():
    rows=samples(2);a=choose_expanded(rows,rows[:1])
    assert len({r['id'] for r in a['samples']})==5
    assert a['coverage'][0]['fallback_rows']==3

def test_reject_duplicate_missing_too_few():
    rows=samples()
    with pytest.raises(ValueError):choose_expanded(rows+[rows[0]],rows[:1])
    with pytest.raises(ValueError):choose_expanded(rows,[])
    with pytest.raises(ValueError):choose_expanded(rows[:3],rows[:1])

def test_no_response_filtering():
    rows=samples();a=choose_expanded(rows,rows[:1])
    altered=[{**r,'response':'completely different','instruction':'different'} for r in rows]
    assert a==choose_expanded(altered,rows[:1])
