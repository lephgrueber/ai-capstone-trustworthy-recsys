"""Immutable run directories and verified, portable artifact manifests."""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from trustworthy_recsys.data.reporting import code_provenance, hashes, write_json


def begin(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / 'INCOMPLETE').write_text('Do not consume this unfinished run.\n')
    return path


def finish(path, metadata):
    write_json(path / 'manifest.json', {
        'format_version': 1, **metadata, 'code': code_provenance(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__},
        'outputs': {p.name: hashes(p)['sha256'] for p in sorted(path.iterdir())
                    if p.is_file() and p.name not in {'INCOMPLETE', 'manifest.json'}},
    })
    (path / 'INCOMPLETE').unlink()


def verify(path):
    path = Path(path)
    if (path / 'INCOMPLETE').exists():
        raise ValueError('Incomplete OPE run')
    manifest = json.loads((path / 'manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid manifest path')
        if hashes(path / name)['sha256'] != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    return manifest
