import json
from pathlib import Path

from little.memory.ontology import seed_commonsense_ontology
from little.memory.store import MemoryStore


def test_ontology_seeder_uses_injected_knowledge_pack(tmp_path: Path):
    pack = tmp_path / "knowledge.json"
    pack.write_text(
        json.dumps(
            {
                "format": "little.knowledge.triples.v1",
                "triples": [
                    ["nebula_x", "is_a", "cosmic_object", 3.0, True],
                    ["nebula_x", "color", "violet", 2.0, True],
                ],
            }
        ),
        encoding="utf-8",
    )

    store = MemoryStore(":memory:")
    seed_commonsense_ontology(store, pack_path=pack)

    assert store.find_relation_by_names("nebula_x", "is_a", "cosmic_object")
    assert store.find_relation_by_names("nebula_x", "color", "violet")
    assert store.get_concept("dog") is None
