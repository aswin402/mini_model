"""Resource discovery from source and clean wheel layouts, without live state."""
import importlib.resources
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from little.core import runtime_paths
from little.core.runtime_paths import RuntimePaths


def test_simulated_installed_resources_are_package_owned(tmp_path, monkeypatch):
    package = tmp_path / 'site-packages' / 'little'
    schema = package / '_resources' / 'schemas'
    schema.mkdir(parents=True)
    source = RuntimePaths.default().schema_directory
    (schema / 'runtime_paths.json').write_text((source / 'runtime_paths.json').read_text())
    monkeypatch.setattr(runtime_paths, '__file__', str(package / 'core' / 'runtime_paths.py'))
    monkeypatch.setattr(importlib.resources, 'files', lambda name: package)
    paths = RuntimePaths.default()
    assert paths.schema_directory == schema
    assert paths.world_corpus_directory == package / '_resources' / 'knowledge' / 'world_corpus'
    assert paths.database == Path('data/little.db')


def test_real_wheel_contains_only_immutable_resources_and_resolves_them(tmp_path):
    root = Path(__file__).resolve().parents[2]
    wheels = tmp_path / 'wheels'
    result = subprocess.run(['uv', 'build', '--wheel', '--out-dir', str(wheels), '--cache-dir', str(tmp_path / 'cache')], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheels.glob('*.whl'))
    installed = tmp_path / 'installed'
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert 'little/_resources/schemas/runtime_paths.json' in names
        assert 'little/_resources/knowledge/commonsense.json' in names
        assert 'little/_resources/knowledge/world_corpus/physics_and_tools.md' in names
        assert 'little/_resources/fixtures/concept_knots/apple.json' in names
        assert not any(name.endswith('.db') or 'knowledge_vault' in name for name in names)
        archive.extractall(installed)
    script = '''import json, sys
sys.path.insert(0, sys.argv[1])
from little.core.runtime_paths import RuntimePaths
paths = RuntimePaths.default()
assert paths.construction_pack.is_file()
assert paths.knowledge_pack.is_file()
assert paths.ontology_pack.is_file()
assert paths.world_corpus_directory.is_dir()
assert (paths.concept_fixture_directory / 'apple.json').is_file()
print(json.dumps({'schema': str(paths.schema_directory), 'database': str(paths.database)}))
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(installed)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert str(installed) in payload['schema']
    assert payload['database'] == 'data/little.db'
