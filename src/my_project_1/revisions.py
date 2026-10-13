"""Version routing without modifying immutable v1 rule implementation."""
import json
from pathlib import Path


def load_revision(manifest_path, source_rows):
    version = json.loads(Path(manifest_path).read_text(encoding='utf-8')).get('version')
    if version == 'result-privacy-target-v1':
        from .target_revision import load_revision as load
    elif version == 'result-privacy-target-v2':
        from .target_revision_v2 import load_revision as load
    else:
        raise ValueError('Unknown train revision version')
    return load(manifest_path, source_rows)
