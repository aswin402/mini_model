import pytest
from little.memory.store import MemoryStore
from little.language.parser import LearningEngine
from little.core.models import BeliefStatus
from little.procedural.math_story import MathStorySolver


def test_sequential_changes_word_problem():
    solver = MathStorySolver()
    problem = (
        "Sarah has 15 apples. She gives 4 apples to Bob, then buys 7 more apples. "
        "How many apples does Sarah have now?"
    )
    res = solver.solve_story(problem)
    assert res is not None
    assert res.status == "SOLVED"
    assert res.result == 18
    assert any("Initial: 15" in s or "15" in s for s in res.steps)
    assert any("- 4" in s or "subtract" in s.lower() or "gives" in s.lower() for s in res.steps)
    assert any("+ 7" in s or "add" in s.lower() or "buys" in s.lower() for s in res.steps)


def test_proportional_rate_word_problem():
    solver = MathStorySolver()
    problem = "A train travels at 60 km/h for 3 hours. How far does the train travel?"
    res = solver.solve_story(problem)
    assert res is not None
    assert res.status == "SOLVED"
    assert res.result == 180


def test_proportional_rate_uses_configured_unit_graph():
    solver = MathStorySolver()
    problem = "A vehicle travels at 10 m/s for 1 hour. How far does it travel?"

    res = solver.solve_story(problem)

    assert res is not None
    assert res.status == "SOLVED"
    assert res.result == 36


def test_fractional_partition_word_problem():
    solver = MathStorySolver()
    problem = (
        "David has 24 candies. He gives half of them to his sister. "
        "How many candies does David have left?"
    )
    res = solver.solve_story(problem)
    assert res is not None
    assert res.status == "SOLVED"
    assert res.result == 12


def test_multi_item_scaling_sum():
    solver = MathStorySolver()
    problem = (
        "Alice bought 3 books for 5 dollars each and 2 pens for 4 dollars each. "
        "How much money did she spend in total?"
    )
    res = solver.solve_story(problem)
    assert res is not None
    assert res.status == "SOLVED"
    assert res.result == 23


def test_math_story_integration_in_learning_engine():
    store = MemoryStore(":memory:", seed_ontology=True)
    engine = LearningEngine(store)

    query = (
        "John had 20 marbles. He lost 5 marbles, and then found 8 marbles. "
        "How many marbles does John have?"
    )
    res = engine.ask(query)
    assert res.status == BeliefStatus.SUPPORTED
    assert res.answer == 23 or "23" in str(res.answer) or "23" in res.verbalize()
    assert any("<math_trace>" in ev for ev in res.evidence) or any("23" in s for s in res.trace)
