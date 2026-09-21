"""Unit tests for Conversational, Identity, and Definition Query handling in LITTLE."""

from little.active.inquisitor import ActiveInquisitor
from little.core.models import BeliefStatus
from little.language.parser import LearningEngine, SimpleParser
from little.memory.store import MemoryStore


def test_conversational_pronoun_protection():
    """Verify that conversational sentences like 'who are you' do not create garbage concepts."""
    triples = SimpleParser.parse_statement("who are you")
    assert len(triples) == 0, f"Expected no triples for conversational input, got: {triples}"

    triples_hello = SimpleParser.parse_statement("hello")
    assert len(triples_hello) == 0

    triples_hii = SimpleParser.parse_statement("hii")
    assert len(triples_hii) == 0


def test_identity_query():
    """Verify that asking 'who are you' or 'what are you' returns LITTLE's identity."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    for q in ["who are you", "who are you?", "what are you", "what can you do"]:
        res = engine.ask(q)
        assert res.status == BeliefStatus.SUPPORTED, f"Failed on: {q}"
        assert "LITTLE" in str(res.answer)
        assert res.confidence == 1.0


def test_concept_definition_query():
    """Verify that 'what is a dog/animal' returns rich definitions from memory."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)

    engine.learn("A dog is an animal.")
    engine.learn("An animal is not a vehicle.")

    res_dog = engine.ask("What is a dog?")
    assert res_dog.status == BeliefStatus.SUPPORTED
    assert "animal" in str(res_dog.answer).lower()

    res_animal = engine.ask("What is an animal?")
    assert res_animal.status == BeliefStatus.SUPPORTED
    assert "vehicle" in str(res_animal.answer).lower()


def test_definition_unknown_curiosity_flow():
    """Verify that asking about an unknown concept prompts curiosity and learns its category."""
    store = MemoryStore(":memory:")
    engine = LearningEngine(store)
    inquisitor = ActiveInquisitor(store, engine)

    # Ask about unknown mango
    res = engine.ask("What is a mango?")
    assert res.status == BeliefStatus.UNKNOWN

    # Inquisitor inspects
    parsed = SimpleParser.parse_question("What is a mango?")
    assert parsed is not None
    s, p, o = parsed
    prompt = inquisitor.inspect_uncertainty(res, s, p, o)
    assert prompt is not None
    assert "mango" in prompt.question_for_user

    # User answers: "fruit"
    learn_res = inquisitor.resolve_response(prompt, "fruit")
    assert "mango" in learn_res.concepts_created or "fruit" in learn_res.concepts_created
    assert any("mango is_a fruit" in r for r in learn_res.relations_created)

    # Re-ask
    res_after = engine.ask("What is a mango?")
    assert res_after.status == BeliefStatus.SUPPORTED
    assert "fruit" in str(res_after.answer).lower()
