import json
from dataclasses import replace
from pathlib import Path

from little.language.parser import LearningEngine, SimpleParser
from little.language.parser_policy import ParserPolicy
from little.language.semantic_predicate_policy import SemanticPredicatePolicy
from little.memory.store import MemoryStore


def test_semantic_predicate_policy_loads_roles_from_data():
    policy = SemanticPredicatePolicy.load(Path("data/schemas"))

    assert policy.taxonomy == "is_a"
    assert policy.part == "part_of"
    assert policy.whole == "has"
    assert policy.location == "located_in"
    assert policy.identity == ("has_name", "name", "is")


def test_parser_policy_can_inject_semantic_predicate_roles(tmp_path: Path, monkeypatch):
    payload = json.loads(
        Path("data/schemas/semantic_predicate_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["semantic_predicate_policy"]["roles"]["taxonomy"] = "subclass_of"
    custom_path = tmp_path / "semantic_predicate_policy.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")
    semantic = SemanticPredicatePolicy.load(tmp_path)
    parser_policy = replace(SimpleParser.POLICY, semantic=semantic)
    monkeypatch.setattr(SimpleParser, "POLICY", parser_policy)

    parsed = SimpleParser.parse_statement("A wolf is an animal")

    assert parsed[0].predicate == "subclass_of"


def test_learning_engine_uses_policy_identity_predicates(tmp_path: Path, monkeypatch):
    payload = json.loads(
        Path("data/schemas/semantic_predicate_policy.json").read_text(
            encoding="utf-8"
        )
    )
    payload["semantic_predicate_policy"]["identity_predicates"] = ["alias"]
    custom_path = tmp_path / "semantic_predicate_policy.json"
    custom_path.write_text(json.dumps(payload), encoding="utf-8")
    semantic = SemanticPredicatePolicy.load(tmp_path)
    monkeypatch.setattr(
        SimpleParser, "POLICY", replace(SimpleParser.POLICY, semantic=semantic)
    )

    memory = MemoryStore(":memory:")
    engine = LearningEngine(memory)
    user = memory.create_concept("user")
    alias = memory.create_concept("aswin")
    memory.add_relation(user.id, "alias", alias.id)

    result = engine.ask("What is my name?")

    assert result.status.value == "SUPPORTED"
    assert "Aswin" in result.answer
