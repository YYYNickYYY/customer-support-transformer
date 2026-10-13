"""V3-aware fifth entry without editing router files frozen by earlier revisions."""
import argparse
import json
from pathlib import Path
from . import revisions
from .target_revision_v3 import VERSION, OUT, check_settings, load_revision
from .training import FullTrainConfig, run_full_training

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/training-fifth.json'));a=p.parse_args()
    check_settings()
    config=FullTrainConfig(**json.loads(a.config.read_text(encoding='utf-8')))
    expected=FullTrainConfig(**json.loads(Path('configs/training-fifth.json').read_text(encoding='utf-8')))
    if config!=expected:raise ValueError('Fifth entry only accepts frozen fifth settings')
    original=revisions.load_revision
    def routed(path,rows):
        if json.loads(Path(path).read_text(encoding='utf-8')).get('version')==VERSION:
            return load_revision(path,rows)
        return original(path,rows)
    # Process-scoped extension: the immutable v1/v2 router and historical hash checks remain intact.
    revisions.load_revision=routed
    try:run_full_training(config)
    finally:revisions.load_revision=original

if __name__=='__main__':main()
