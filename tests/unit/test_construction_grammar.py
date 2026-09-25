"""Unit tests for Construction Grammar (CxG) and Qualitative Process Theory (QPT).

Verifies:
1. Grammar is stored in memory as persistent SQLite records, not hardcoded Python regexes.
2. Dynamic acquisition of open relational pivots (e.g. 'originated in', 'borders').
3. Specificity pre-emption in Construction Grammar.
4. Qualitative Physics (QPT) generalization across arbitrary physical entities.
"""

from little.core.models import UpdateType
from little.dynamics.transformations import TransformationEngine
from little.language.construction import ConstructionEngine
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_construction_persistence_and_seeding():
    store = MemoryStore(":memory:")
    cxns = store.list_constructions()
    assert len(cxns) >= 20

    is_a_cxn = store.get_construction_by_name("cxn_is_a")
    assert is_a_cxn is not None
    assert is_a_cxn.pattern_tokens == ["{x}", "is", "a", "{y}"]
    assert is_a_cxn.predicate_template == "is_a"

    # Evidence reinforcement
    store.add_construction_evidence(is_a_cxn.id, positive=True)
    updated = store.get_construction(is_a_cxn.id)
    assert updated.evidence_positive == is_a_cxn.evidence_positive + 1
    assert updated.confidence > 0.5


def test_construction_grammar_parsing_and_specificity():
    store = MemoryStore(":memory:")

    # Taxonomic classification
    triples_tax = ConstructionEngine.parse_with_constructions(
        "A tiger is a carnivore.", store
    )
    assert len(triples_tax) == 1
    assert triples_tax[0].subject == "tiger"
    assert triples_tax[0].predicate == "is_a"
    assert triples_tax[0].object_ == "carnivore"

    # Negative / Disjoint statement
    triples_neg = ConstructionEngine.parse_with_constructions(
        "An animal cannot be a machine.", store
    )
    assert len(triples_neg) == 1
    assert triples_neg[0].subject == "animal"
    assert triples_neg[0].predicate == "disjoint_with"
    assert triples_neg[0].object_ == "machine"

    triples_list = ConstructionEngine.parse_with_constructions(
        "A truck has wheels, an engine, and headlights.", store
    )
    assert [triple.object_ for triple in triples_list] == [
        "wheel",
        "engine",
        "headlight",
    ]

    # Question parsing
    q_res = ConstructionEngine.parse_question_with_constructions(
        "Is a tiger a carnivore?", store
    )
    assert q_res == ("tiger", "is_a", "carnivore")

    # Action parsing
    act_res = ConstructionEngine.parse_action_with_constructions(
        "Slice a melon into 6 pieces", store
    )
    assert act_res is not None
    assert act_res[0].upper() == "SLICE"
    assert act_res[1]["object"] == "melon"
    assert act_res[1]["count"] == 6


def test_open_pivot_learning_and_reuse():
    """Novel predicates remain candidates until a relation schema approves them."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    before = store.count_relations()

    first = engine.learn("Apples originated in Central Asia.")
    second = engine.learn("Peaches originated in China.")

    assert first.update_type is UpdateType.NO_OP
    assert second.update_type is UpdateType.NO_OP
    assert store.count_relations() == before
    assert store.get_construction_by_name("cxn_originated_in") is None
    evidence = store.evidence.list_for_claim("apple", "originated", "in central asia")
    evidence += store.evidence.list_for_claim("peach", "originated", "in china")
    assert len(evidence) == 2
    assert all(record.status.value == "candidate" for record in evidence)
    decisions = store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id IN (?, ?)",
        (evidence[0].evidence_id, evidence[1].evidence_id),
    ).fetchall()
    assert len(decisions) == 2
    assert all(row[0] == "unknown" for row in decisions)


def test_qualitative_physics_generalization():
    """Verify that slicing is a universal topological operator, not hardcoded to apples."""
    store = MemoryStore(":memory:")

    # 1. Register a non-food inorganic object: a diamond crystal
    store.create_concept(
        "diamond",
        category="mineral",
        attributes={"interior_color": "clear", "material": "carbon", "decay_tau": 1e12},
    )

    # Slice diamond into 2 pieces
    result = TransformationEngine.slice_object(store, "diamond", num_pieces=2)
    assert result.num_pieces == 2
    assert result.slice_concept == "diamond slice"
    assert len(result.entities_created) == 2

    # Mass conservation: each half is 0.5 mass fraction
    slice_concept = store.get_concept("diamond slice")
    assert slice_concept.attributes["mass_fraction"] == 0.5
    assert slice_concept.attributes["interior_color"] == "clear"
    assert slice_concept.attributes["interior_material"] == "carbon"
    assert slice_concept.attributes["exposed_interior"] is True

    # Inorganic diamond has astronomical tau (does not brown or decay)
    assert result.initial_state.tau_seconds >= 1e10

    # 2. Register an organic root vegetable: carrot with orange interior
    store.create_concept(
        "carrot",
        category="vegetable",
        attributes={"interior_color": "orange", "material": "plant_tissue"},
    )
    TransformationEngine.slice_object(store, "carrot", num_pieces=5)
    carrot_slice = store.get_concept("carrot slice")
    assert carrot_slice.attributes["interior_color"] == "orange"
    assert carrot_slice.attributes["mass_fraction"] == 0.2
