from little.core.models import BeliefStatus, InferenceResult
from little.inference.thinking_controller import CognitiveMode, ThinkingController
from little.memory.store import MemoryStore


def _concept_graph(*names: str) -> tuple[MemoryStore, dict[str, str]]:
    memory = MemoryStore(":memory:")
    concepts = {name: memory.create_concept(name) for name in names}
    return memory, {name: concept.id for name, concept in concepts.items()}


def test_direct_verified_relation_routes_to_fast_mode():
    memory, ids = _concept_graph("apple", "fruit")
    memory.add_relation(ids["apple"], "is_a", ids["fruit"])

    result = ThinkingController(memory).run("Is an apple a fruit?")

    assert result.mode is CognitiveMode.FAST
    assert result.inference.status is BeliefStatus.SUPPORTED
    assert result.inference.answer is True
    assert any("FAST" in step for step in result.inference.trace)


def test_multihop_relation_routes_to_think_mode_and_keeps_proof_trace():
    memory, ids = _concept_graph("gala apple", "apple", "pome fruit", "fruit")
    memory.add_relation(ids["gala apple"], "is_a", ids["apple"])
    memory.add_relation(ids["apple"], "is_a", ids["pome fruit"])
    memory.add_relation(ids["pome fruit"], "is_a", ids["fruit"])

    result = ThinkingController(memory).run("Is a gala apple a fruit?")

    assert result.mode is CognitiveMode.THINK
    assert result.inference.status is BeliefStatus.SUPPORTED
    assert any("Transitive path discovered" in step for step in result.inference.trace)
    assert any("THINK" in step for step in result.inference.trace)


def test_multihop_controller_includes_verified_system2_gate_trace():
    memory, ids = _concept_graph("gala apple", "apple", "fruit")
    memory.add_relation(ids["gala apple"], "is_a", ids["apple"])
    memory.add_relation(ids["apple"], "is_a", ids["fruit"])

    result = ThinkingController(memory).run("Is a gala apple a fruit?")
    trace = "\n".join(result.inference.trace)

    assert result.inference.status is BeliefStatus.SUPPORTED
    assert "System 2 proof path" in trace
    assert all(gate in trace for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND"))


def test_unproven_but_well_formed_query_remains_unknown():
    memory, _ = _concept_graph("apple", "spaceship")

    result = ThinkingController(memory).run("Is an apple a spaceship?")

    assert result.mode is CognitiveMode.THINK
    assert result.inference.status is BeliefStatus.UNKNOWN
    assert result.inference.answer is None


def test_unrecognized_input_routes_to_ask_without_guessing():
    result = ThinkingController(MemoryStore(":memory:")).run("qwerty zorp")

    assert result.mode is CognitiveMode.ASK
    assert result.inference.status is BeliefStatus.UNKNOWN
    assert result.inference.answer is None
    assert "not recognized" in result.reason.lower()


def test_explicit_missing_candidate_query_skips_parsing_and_asks(monkeypatch):
    memory, ids = _concept_graph("falcon", "bird")
    memory.add_relation(ids["falcon"], "is_a", ids["bird"])
    controller = ThinkingController(memory)

    def fail_if_parsed(_text):
        raise AssertionError("explicit None must not reparse")

    monkeypatch.setattr(controller, "_parse_question", fail_if_parsed)

    plan = controller.plan("Is a falcon a bird?", parsed_query=None)
    result = controller.run("Is a falcon a bird?", parsed_query=None)

    assert plan.mode is CognitiveMode.ASK
    assert result.mode is CognitiveMode.ASK
    assert result.inference.status is BeliefStatus.UNKNOWN
    assert result.inference.answer is None


def test_statement_routes_to_study_instead_of_answering_as_a_question():
    result = ThinkingController(MemoryStore(":memory:")).plan("A falcon is a bird.")

    assert result.mode is CognitiveMode.STUDY
    assert result.subject == "falcon"
    assert result.predicate == "is_a"
    assert result.target == "bird"


def test_resolver_receives_only_supported_payload_fields():
    memory = MemoryStore(":memory:")
    received = {}

    def resolver(question, parsed_query=None):
        received["question"] = question
        received["parsed_query"] = parsed_query
        return InferenceResult(
            query=question,
            status=BeliefStatus.UNKNOWN,
            answer=None,
            confidence=0.0,
            evidence=[],
            trace=[],
        )

    parsed_query = ("eagle", "can", "fly")
    result = ThinkingController(memory, resolver=resolver).run(
        "Can an eagle fly?",
        parsed_query=parsed_query,
        parsed_queries=[parsed_query],
        question_parts=["Can an eagle fly"],
    )

    assert result.inference.status is BeliefStatus.UNKNOWN
    assert received == {
        "question": "Can an eagle fly?",
        "parsed_query": parsed_query,
    }
