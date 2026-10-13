"""Verify frozen files on disk or as staged Git bytes before publishing."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def read(path):
        if args.staged:
            return subprocess.check_output(["git", "show", ":" + path.as_posix()], cwd=root)
        return (root / path).read_bytes()

    count = 0
    for manifest in (Path("data/splits/bitext-v1/manifest.json"),
                     Path("artifacts/tokenizer/bitext-bpe-v1/freeze.json"),
                     Path("artifacts/dev-check-v1/freeze.json")):
        entries = json.loads(read(manifest))["file_sha256"]
        for name, expected in entries.items():
            path = manifest.parent / name
            if hashlib.sha256(read(path)).hexdigest() != expected:
                raise ValueError(f"Frozen file hash mismatch: {path}")
            count += 1
    print(f"Verified {count} frozen artifact hashes")


if __name__ == "__main__":
    main()
