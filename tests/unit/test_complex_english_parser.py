import pytest
from little.language.parser import LearningEngine, SimpleParser
from little.memory.store import MemoryStore
from little.core.models import BeliefStatus


def test_coordinate_compound_clause_learning():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)

    # Compound sentence with pronoun reference
    result = engine.learn("A falcon is a bird and it hunts rodents.")
    assert len(result.relations_created) >= 2

    # Query relations learned from both clauses
    res1 = engine.ask("Is a falcon a bird?")
    assert res1.status == BeliefStatus.SUPPORTED

    res2 = engine.ask("Does a falcon hunt rodents?")
    assert res2.status == BeliefStatus.SUPPORTED or "rodents" in res2.verbalize()


def test_restrictive_relative_clause_learning():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)

    # Restrictive clause defining a concept via its distinctive traits
    result = engine.learn("The animal that has gills and swims in water is a fish.")
    assert len(result.relations_created) >= 1

    res_tax = engine.ask("Is a fish an animal?")
    assert res_tax.status == BeliefStatus.SUPPORTED


def test_indirect_and_polite_queries():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)

    # Paris in Europe is part of seed ontology
    res = engine.ask("Can you tell me if Paris is in Europe?")
    assert res.status == BeliefStatus.SUPPORTED

    res_wh = engine.ask("Do you know where a salmon lives?")
    assert res_wh.status == BeliefStatus.SUPPORTED
    assert "water" in res_wh.verbalize().lower()


def test_multi_turn_pronoun_continuity_in_ask():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)

    # Turn 1: Introduce tiger
    engine.learn("A tiger is a mammal.")

    # Turn 2: Query using pronoun 'it'
    res = engine.ask("Is it an animal?")
    # Transitive deduction: tiger -> mammal -> animal
    assert res.status == BeliefStatus.SUPPORTED
