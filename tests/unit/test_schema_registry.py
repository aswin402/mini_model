from pathlib import Path

from little.inference.engine import InferenceEngine
from little.knowledge.registry import SchemaRegistry
from little.memory.store import MemoryStore


def test_registry_loads_relation_and_action_metadata():
    registry = SchemaRegistry.load(Path("data/schemas"))

    part_of = registry.relation("part_of")
    split = registry.action("split")

    assert part_of is not None
    assert part_of.inverse == "has_part"
    assert part_of.axis == "mereology"
    assert split is not None
    assert "mass_conservation" in split.invariants
    assert split.preconditions == ()
    assert registry.action("slice").preconditions == ()
    assert registry.relation("name").attribute_update_policy == "replace"
    assert registry.relation("has_name").attribute_update_policy == "replace"
    assert registry.relation("name").attribute_value_format == "capitalize"
    assert registry.relation("has_name").attribute_value_format == "capitalize"
    assert registry.relation("color").attribute_update_policy == "append"


def test_registry_rejects_invalid_relation_types():
    registry = SchemaRegistry.load(Path("data/schemas"))

    result = registry.validate_relation("is_a", "quantity", "animal")

    assert result.valid is False
    assert "domain" in result.reason.lower() or "range" in result.reason.lower()


def test_registry_can_register_a_persisted_learned_predicate():
    registry = SchemaRegistry.load(Path("data/schemas"))

    schema = registry.register_learned_relation("originated_in")

    assert schema.name == "originated_in"
    assert registry.relation("originated_in") is schema


def test_registry_exposes_reasoning_roles_from_relation_data():
    registry = SchemaRegistry.load(Path("data/schemas"))

    assert "is_a" in registry.predicates(transitive=True)
    assert "located_in" in registry.predicates(transitive=True)
    assert "disjoint_with" in registry.predicates(disjoint=True)
    assert "larger_than" in registry.predicates(reverse_refutation=True)
    assert "can" in registry.predicates(axis="procedural")
    assert registry.predicates(acyclic=True) == {"is_a", "subclass_of", "part_of"}
    assert registry.primary_predicate("taxonomy") == "is_a"

    engine = InferenceEngine(MemoryStore(":memory:"), registry=registry)
    assert engine.transitive_predicates == registry.predicates(transitive=True)
    assert engine.disjoint_predicates == registry.predicates(disjoint=True)
