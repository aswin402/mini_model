"""Integration tests for math + English synthesis: algebraic word problems and comparative ordering."""

from little.core.models import BeliefStatus
from little.inference.thinking_controller import ThinkingController
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def test_algebraic_age_problem_via_learning_engine():
    memory = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(memory=memory)

    problem = "Alice is twice as old as Bob. In 5 years, the sum of their ages will be 40. How old is Alice?"
    res = engine.ask(problem)

    assert res.status == BeliefStatus.SUPPORTED
    assert "Alice is 20 years old" in str(res.answer)
    assert any("<math_trace>" in ev for ev in res.evidence)
    assert any("Cramer's Rule" in step for step in res.trace)


def test_algebraic_cost_system_via_learning_engine():
    memory = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(memory=memory)

    problem = "A pen costs 3 dollars more than a notebook. 2 pens and 3 notebooks cost 21 dollars. What is the cost of a pen?"
    res = engine.ask(problem)

    assert res.status == BeliefStatus.SUPPORTED
    assert "cost of a pen is $6" in str(res.answer)
    assert any("<math_trace>" in ev for ev in res.evidence)


def test_comparative_poset_reasoning_via_learning_engine():
    memory = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(memory=memory)

    # Compound premises + superlative query
    story = "John is taller than Mary, and Mary is taller than Peter. Who is the tallest?"
    res1 = engine.ask(story)
    assert res1.status == BeliefStatus.SUPPORTED
    assert "John is the tallest" in str(res1.answer)

    # Transitive affirmative question
    res2 = engine.ask("Is John taller than Peter?")
    assert res2.status == BeliefStatus.SUPPORTED
    assert "Yes" in str(res2.answer)
    assert "John is taller than Mary" in str(res2.answer)
    assert "Mary is taller than Peter" in str(res2.answer)

    # Refutation of inverted relation
    res3 = engine.ask("Is Peter taller than John?")
    assert res3.status == BeliefStatus.REFUTED
    assert "No" in str(res3.answer)


def test_thinking_controller_routes_algebraic_and_comparative():
    memory = MemoryStore(":memory:", seed_ontology=False)
    controller = ThinkingController(memory=memory)

    # 1. Algebraic word problem via thinking controller
    problem = "David is 4 years older than Sarah. The sum of their ages is 28. How old is Sarah?"
    res_alg = controller.run(problem)
    assert res_alg.inference.status == BeliefStatus.SUPPORTED
    assert "Sarah is 12 years old" in str(res_alg.inference.answer)

    # 2. Comparative ordering via thinking controller
    story = "Falcon is faster than Hawk, and Hawk is faster than Pigeon. Who is the fastest?"
    res_comp = controller.run(story)
    assert res_comp.inference.status == BeliefStatus.SUPPORTED
    assert "Falcon is the fastest" in str(res_comp.inference.answer)
