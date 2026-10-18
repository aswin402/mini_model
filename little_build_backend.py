"""PEP 517 adapter staging immutable LITTLE resources for uv_build wheels.

Source data remains authoritative. Generated package resources exist only in the
build's temporary checkout; runtime databases and knowledge vaults are excluded.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

_RESOURCE_PATTERNS = (
    'schemas/*.json',
    'knowledge/*.json',
    'knowledge/world_corpus/*.md',
    'fixtures/concept_knots/*.json',
)


@contextmanager
def _wheel_source():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix='little-wheel-') as temporary:
        staged = Path(temporary)
        for name in ('pyproject.toml', 'README.md'):
            shutil.copy2(root / name, staged / name)
        shutil.copytree(root / 'src' / 'little', staged / 'src' / 'little', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.db', 'knowledge_vault', '_resources'))
        resources = staged / 'src' / 'little' / '_resources'
        for pattern in _RESOURCE_PATTERNS:
            for source in sorted((root / 'data').glob(pattern)):
                target = resources / source.relative_to(root / 'data')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        previous = Path.cwd()
        try:
            os.chdir(staged)
            yield
        finally:
            os.chdir(previous)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    import uv_build
    output = str(Path(wheel_directory).resolve())
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    with _wheel_source():
        return uv_build.build_wheel(output, config_settings, metadata)


def build_sdist(sdist_directory, config_settings=None):
    import uv_build
    return uv_build.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    import uv_build
    return uv_build.build_editable(wheel_directory, config_settings, metadata_directory)
