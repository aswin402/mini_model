import json
from pathlib import Path

from little.core.runtime_paths import RuntimePaths


def test_runtime_paths_load_from_versioned_schema(tmp_path: Path):
    (tmp_path / "runtime_paths.json").write_text(
        json.dumps(
            {
                "format": "little.runtime_paths.v1",
                "runtime_paths": {
                    "database": "memory/test.db",
                    "construction_pack": "grammar.json",
                    "knowledge_pack": "knowledge.json",
                    "ontology_pack": "ontology.json",
                    "concept_fixture_directory": "fixtures/concepts",
                },
            }
        ),
        encoding="utf-8",
    )

    paths = RuntimePaths.load(tmp_path)

    assert paths.database == Path("memory/test.db")
    assert paths.schema_directory == tmp_path
    assert paths.construction_pack == tmp_path / "grammar.json"
    assert paths.knowledge_pack == tmp_path / "knowledge.json"
    assert paths.ontology_pack == tmp_path / "ontology.json"
    assert paths.concept_fixture_directory == tmp_path / "fixtures/concepts"
