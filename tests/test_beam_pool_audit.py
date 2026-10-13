import copy
from collections import Counter

import pytest

from my_project_1.beam_pool_audit import choose, INTENTS
from my_project_1.expanded_dev import read
from my_project_1.tokenization import read_rows


def test_selection_is_instruction_only_and_deterministic():
    selection = read('artifacts/dev-check-expanded-v1/selection.json')
    rows = list(read_rows('data/splits/bitext-v1/dev.jsonl'))
    chosen = choose(selection, rows)
    assert len(chosen) == 12 and len({s['id'] for s in chosen}) == 12
    assert Counter(s['intent'] for s in chosen) == Counter({i: 2 for i in INTENTS})
    mutated = copy.deepcopy(rows)
    for row in mutated:
        row['response'] = 'arbitrary wrong response'
    assert choose(selection, mutated) == chosen
    selection['samples'].reverse()
    assert choose(selection, rows) == chosen
    assert not any('cannot poay' in s['instruction'] for s in chosen)


def test_not_enough_questions_fails_no_fallback():
    with pytest.raises(ValueError, match='Too few'):
        choose({'samples': []}, [])
