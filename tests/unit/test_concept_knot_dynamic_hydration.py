import pytest
from little.core.concept_knot import ConceptKnot
from little.memory.store import MemoryStore


def test_concept_knot_dynamic_hydration_from_memory():
    memory = MemoryStore(":memory:")

    # 1. Ingest multi-axis facts for 'cat'
    # Taxonomy
    memory.add_relation("cat", "is_a", "feline")
    memory.add_relation("feline", "is_a", "carnivore")
    memory.add_relation("carnivore", "is_a", "mammal")

    # Mereology
    memory.add_relation("cat", "has_part", "whiskers")
    memory.add_relation("cat", "has_part", "tail")
    memory.add_relation("claw", "part_of", "cat")

    # Invariants (Mutex)
    memory.add_relation("mammal", "disjoint_with", "reptile")
    memory.add_relation("cat", "disjoint_with", "dog")

    # Skills / Procedural actions
    memory.add_relation("cat", "can", "purr")
    memory.add_relation("cat", "can", "climb")

    # Attributes
    cat_concept = memory.get_or_create_concept("cat")
    assert cat_concept is not None
    memory.update_concept_attributes(
        cat_concept.id, {"mass": 4.5, "temperature": 38.5, "freshness": 1.0}
    )

    # 2. Dynamically hydrate 360° Concept Knot directly from memory
    cat_knot = ConceptKnot.from_memory(memory, "cat")
    assert cat_knot is not None
    assert cat_knot.concept_id == "CAT"

    # Verify 90° Taxonomic Axis
    assert "feline" in cat_knot.taxonomy_hypernyms

    # Verify 135° Mereological Axis
    assert "whiskers" in cat_knot.mereology_parts
    assert "tail" in cat_knot.mereology_parts
    assert "claw" in cat_knot.mereology_parts

    # Verify 180° Invariant Axis
    assert "dog" in cat_knot.invariant_disjoints

    # Verify 225° Procedural Skills Axis
    assert "purr" in cat_knot.procedural_skills
    assert "climb" in cat_knot.procedural_skills

    # Verify 45° Continuous Dynamics Axis
    assert cat_knot.invariant_mass == 4.5
    assert cat_knot.dynamics_state.temperature == 38.5
