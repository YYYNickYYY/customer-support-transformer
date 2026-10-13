import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('side_effect_audit',Path(__file__).resolve().parents[1]/'scripts/audit_fifth_side_effects.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def test_sampling_is_response_independent_and_stable():
    rows=[{'id':str(i),'intent':'get_refund','response':'bad' if i%2 else 'good'} for i in range(10)]
    a=audit.choose(rows,{'get_refund'},{'0','1','2','3'})
    assert a==audit.choose([{**r,'response':'changed'} for r in reversed(rows)],{'get_refund'},{'0','1','2','3'})
    for layer in ('all','changed','unchanged'):
        assert sum(layer in r['layers'] for r in a)==3
    assert len(a)==len({r['id'] for r in a})

def test_small_strata_and_normalization():
    rows=[{'id':'a','intent':'track_refund'}]
    assert audit.choose(rows,{'track_refund'},{'a'})==[{'id':'a','intent':'track_refund','layers':['all','changed']}]
    assert audit.norm(' Ensure\n a SMOOTH  resolution ')==audit.norm('ensure a smooth resolution')
