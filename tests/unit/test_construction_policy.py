import json
from pathlib import Path

from little.core.construction_policy import ConstructionPolicy
from little.core.grammar_registry import GrammarRegistry
from little.core.models import Construction
from little.language.construction import ConstructionEngine


def test_construction_policy_loads_default_confidence():
    policy = ConstructionPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.default_confidence == 1.0
    assert policy.learned_confidence == 0.8


def test_construction_engine_has_no_process_global_parser_policy():
    assert "POLICY" not in ConstructionEngine.__dict__


def test_construction_factory_uses_configured_default():
    construction = Construction.create(
        "test", ["{x}", "is", "{y}"], {"x": "subject", "y": "object"}, "is_a"
    )

    assert construction.confidence == ConstructionPolicy.default().default_confidence


def test_grammar_registry_uses_policy_when_confidence_is_omitted(tmp_path: Path):
    pack = {
        "format": "little.constructions.v1",
        "constructions": [
            {
                "name": "test",
                "pattern_tokens": ["{x}", "is", "{y}"],
                "slot_roles": {"x": "subject", "y": "object"},
                "predicate_template": "is_a",
            }
        ],
    }
    path = tmp_path / "constructions.json"
    path.write_text(json.dumps(pack), encoding="utf-8")

    loaded = GrammarRegistry.load(path)

    assert loaded[0].confidence == ConstructionPolicy.default().default_confidence
