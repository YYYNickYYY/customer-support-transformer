"""Summarize frozen observer traces, not automatic human-quality scoring."""
from collections import Counter
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.generation_diagnostic import locate_quote
from my_project_1.tokenization import file_hash, load_frozen, read_rows

OUT = Path('outputs/fifth-generation-diagnostic-20261006-v1.1')
BASE = Path('data/derived/refund-execution-v3.1/train.jsonl')
PHRASES = [
    'ensure a smooth and satisfactory resolution',
    'ensure a smooth and efficient resolution',
    'Your insights will help us expedite the process and ensure a smooth resolution.',
    'ensure a smooth transition and ensure a smooth transition',
    'ensure a smooth transition',
    'reply to the email with the email to the newsletter email with the newsletter',
    'reply to the email', 'proactive approach',
    'Your proactive approach in notifying us about the issue is greatly appreciated.',
    'take your concerns seriously', 'take customer claims seriously',
]


def repeated_ngrams(ids, n=4):
    """Literal nonoverlapping token repeats; NOT a semantic repetition score."""
    seen, found = {}, []
    for index in range(len(ids) - n + 1):
        gram = tuple(ids[index:index + n])
        if gram in seen and index - seen[gram] >= n:
            found.append({'first_token': seen[gram] + 1, 'repeat_token': index + 1, 'ids': list(gram)})
        elif gram not in seen:
            seen[gram] = index
    return found


def main():
    summary = read(OUT / 'summary.json')
    frozen = read(OUT / 'freeze.json')
    for group in (frozen['dependencies_sha256'], summary['trace_sha256']):
        for p, h in group.items():
            if file_hash(p) != h:
                raise ValueError('Changed trace/source ' + p)
    tokenizer, _ = load_frozen(frozen['generation']['model_config']['tokenizer_artifact'])
    traces = {r['id'].rsplit(':', 1)[-1]: read(r['path']) for r in summary['cases']}
    events = []
    for event in read(OUT / 'interpretation-events.json'):
        trace = traces[event['short_id']]
        span = locate_quote(tokenizer, trace['generated_token_ids'], event['quote'], occurrence=event['occurrence'])
        first, last = span['first_token'], span['last_token']
        after = trace['steps'][last] if last < len(trace['steps']) else None
        events.append({**event, **span, 'stop_reason': trace['stop_reason'],
                       'prefix_before_event': tokenizer.decode(trace['generated_token_ids'][:first - 1],
                                                              skip_special_tokens=True)[-150:],
                       'onset_step': trace['steps'][first - 1],
                       'step_after_event': after,
                       'window': trace['steps'][max(0, first - 3):min(len(trace['steps']), last + 2)]})
    rows = list(read_rows(BASE))
    phrase_counts = [{'phrase': phrase, 'matching_train_records': sum(phrase in r['response'] for r in rows)}
                     for phrase in PHRASES]
    cooccurrence = sum('take your concerns seriously' in r['response'] and
                       'take customer claims seriously' in r['response'] for r in rows)
    literal = [{'id': t['id'], 'stratum': t['stratum'],
                'nonoverlap_4gram_repeats': repeated_ngrams(t['generated_token_ids'])}
               for t in traces.values()]
    result = {'human_interpreted_events': events, 'exact_train_phrase_counts': phrase_counts,
              'both_claim_attitudes_same_train_record': cooccurrence,
              'literal_token_repeats_not_scores': literal,
              'case_stop_counts': dict(Counter(t['stop_reason'] for t in traces.values())),
              'model_untrained_on_v12': True, 'training_started': False,
              'limitations': 'One greedy trajectory; no internal attribution or counterfactual decoding. '
                             'Exact zero matches does not rule out semantic training sources. '
                             'Control means risk0/repeat0, not correct intent or overall quality.'}
    save(OUT / 'analysis.json', result)
    deps = [OUT / 'summary.json', OUT / 'freeze.json', OUT / 'interpretation-events.json',
            OUT / 'analysis.json', BASE, Path(__file__)]
    save(OUT / 'analysis-freeze.json', {'sha256': {str(p): file_hash(p) for p in deps},
                                      'trace_sha256': summary['trace_sha256']})
    print('Interpreted events:', len(events), '; exact substring queries:', len(phrase_counts))


if __name__ == '__main__':
    main()
