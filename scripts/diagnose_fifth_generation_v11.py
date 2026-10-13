"""Versioned adapter for old reply records that omitted prompt_tokens.

Never rewrites the frozen v1 script or historical replies. Prompt length is
derived in memory from the frozen tokenizer/template, not claimed as historical.
"""
import argparse
import importlib.util
from pathlib import Path

from my_project_1.expanded_dev import read, save
from my_project_1.tokenization import encode_parts, load_frozen, file_hash

PARENT = Path('outputs/fifth-generation-diagnostic-20261006-v1')
OUT = Path('outputs/fifth-generation-diagnostic-20261006-v1.1')
spec = importlib.util.spec_from_file_location('fifth_diagnostic_v1', 'scripts/diagnose_fifth_generation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = OUT
original_lines = module.lines
tokenizer, _ = load_frozen(read(module.SOURCE / 'config.json')['model_config']['tokenizer_artifact'])


def compatible_lines(path):
    records = original_lines(path)
    if path == module.SOURCE / 'replies.jsonl':
        for r in records:
            r['prompt_tokens'] = len(encode_parts(tokenizer, r['instruction'], '')[2]) - 1
    return records


def versioned_save(path, payload):
    if path == OUT / 'freeze.json':
        if read(OUT / 'selection.json') != read(PARENT / 'selection.json'):
            raise ValueError('Diagnostic cases changed during schema repair')
        payload = {**payload, 'compatibility_version': 'v1.1',
                   'prompt_tokens_source': 'derived from frozen tokenizer, absent in old records',
                   'dependencies_sha256': {**payload['dependencies_sha256'],
                       str(Path(__file__)): file_hash(__file__),
                       str(PARENT / 'freeze.json'): file_hash(PARENT / 'freeze.json'),
                       str(PARENT / 'selection.json'): file_hash(PARENT / 'selection.json')}}
    save(path, payload)


module.lines = compatible_lines
module.save = versioned_save

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'run'))
    args = parser.parse_args()
    (module.prepare if args.stage == 'prepare' else module.run)()
