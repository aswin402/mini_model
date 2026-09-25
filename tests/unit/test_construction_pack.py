import json
from pathlib import Path

from little.language.construction import ConstructionEngine
from little.memory.store import MemoryStore


def test_memory_store_loads_grammar_from_an_injected_pack(tmp_path: Path):
    pack_path = tmp_path / "constructions.json"
    pack_path.write_text(
        json.dumps(
            {
                "format": "little.constructions.v1",
                "constructions": [
                    {
                        "name": "custom_relation",
                        "pattern_tokens": ["{X}", "glides", "{Y}"],
                        "slot_roles": {"X": "subject", "Y": "object"},
                        "predicate_template": "moves_over",
                        "construction_type": "statement",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with MemoryStore(":memory:", construction_pack=pack_path) as memory:
        parsed = ConstructionEngine.parse_with_constructions(
            "a bird glides clouds", memory
        )

    assert parsed[0].predicate == "moves_over"
    assert parsed[0].subject == "bird"
    assert parsed[0].object_ == "cloud"
