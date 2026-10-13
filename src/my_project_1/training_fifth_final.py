"""Final fifth VS Code training entry, same engine and optimizer as fourth."""
import argparse
import json
from pathlib import Path
from . import revisions
from .target_revision_v31 import VERSION, load_revision, check_settings
from .training import FullTrainConfig, run_full_training

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/training-fifth-final.json'));a=p.parse_args()
    check_settings()
    config=FullTrainConfig(**json.loads(a.config.read_text(encoding='utf-8')))
    expected=FullTrainConfig(**json.loads(Path('configs/training-fifth-final.json').read_text(encoding='utf-8')))
    if config!=expected:raise ValueError('Only frozen final fifth settings accepted')
    original=revisions.load_revision
    def routed(path,rows):
        if json.loads(Path(path).read_text(encoding='utf-8')).get('version')==VERSION:return load_revision(path,rows)
        return original(path,rows)
    revisions.load_revision=routed
    try:run_full_training(config)
    finally:revisions.load_revision=original

if __name__=='__main__':main()
