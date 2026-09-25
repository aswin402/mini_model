import pytest

from little.core.kernel import CognitiveKernel
from little.core.models import BeliefStatus, Construction, InferenceResult, UpdateType
from little.inference.thinking_controller import CognitiveMode, ThinkingController
from little.language.parser import SimpleParser
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


def test_default_kernel_keeps_deterministic_perception_identity():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)

    outcome = kernel.process("A falcon is a bird.")

    assert isinstance(kernel.perception, DeterministicPerceptionAdapter)
    assert kernel.learner.perception is kernel.perception
    assert outcome.frame.model_id == "deterministic-parser"


def test_statement_commits_one_candidate_frame_through_learner():
    class RecordingAdapter:
        def __init__(self):
            self.delegate = DeterministicPerceptionAdapter()
            self.calls = 0
            self.context = None
            self.frame = None

        def perceive(self, text, *, memory, context):
            self.calls += 1
            self.context = context
            self.frame = self.delegate.perceive(text, memory=memory, context=context)
            return self.frame

    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = RecordingAdapter()
    kernel = CognitiveKernel(memory, perception=adapter)

    outcome = kernel.process("A falcon is a bird.")

    assert adapter.calls == 1
    assert adapter.context is kernel.learner.dialogue
    assert outcome.frame is adapter.frame
    assert outcome.frame.accepted is False
    assert outcome.mode is CognitiveMode.STUDY
    assert outcome.learning is not None
    assert outcome.learning.update_type is UpdateType.NEW_RELATION
    assert outcome.inference is None
    assert memory.count_relations() == 1
    assert len(memory.evidence.list_by_source("user")) == 1


def test_known_question_uses_controller_and_does_not_learn():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)
    kernel.process("A falcon is a bird.")
    before_relations = memory.count_relations()
    before_evidence = len(memory.evidence.list_by_source("user"))

    outcome = kernel.process("Is a falcon a bird?")

    assert outcome.mode is CognitiveMode.FAST
    assert outcome.frame.intent == "question"
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.SUPPORTED
    assert outcome.inference.answer is True
    assert any("Controller route: FAST" in step for step in outcome.inference.trace)
    assert memory.count_relations() == before_relations
    assert len(memory.evidence.list_by_source("user")) == before_evidence


def test_kernel_multihop_question_exposes_verified_system2_trace():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)
    kernel.process("A falcon is a bird.")
    kernel.process("A bird is an animal.")

    outcome = kernel.process("Is a falcon an animal?")
    trace = "\n".join(outcome.inference.trace)

    assert outcome.mode is CognitiveMode.THINK
    assert outcome.inference.status is BeliefStatus.SUPPORTED
    assert "System 2 proof path" in trace
    assert all(gate in trace for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND"))


def test_kernel_targetless_special_query_is_safe():
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory).process("tell me a fact")

    assert outcome.inference is not None
    assert outcome.inference.status in {BeliefStatus.UNKNOWN, BeliefStatus.SUPPORTED}


def test_kernel_preserves_compound_query_parts_and_answer():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)
    kernel.process("An eagle can fly.")
    kernel.process("An eagle has a wing.")

    outcome = kernel.process("Can an eagle fly and does it have wings?")

    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.SUPPORTED
    assert outcome.inference.answer is True
    assert len(outcome.frame.question_parts) == 2
    assert [query[0:2] for query in outcome.frame.parsed_queries] == [
        ("eagle", "can"),
        ("eagle", "has"),
    ]


def test_kernel_passes_perceived_query_payload_to_resolver():
    memory = MemoryStore(":memory:", seed_ontology=False)
    received = {}

    def resolver(question, **payload):
        received["question"] = question
        received.update(payload)
        return InferenceResult(
            query=question,
            status=BeliefStatus.UNKNOWN,
            answer=None,
            confidence=0.0,
            evidence=[],
            trace=[],
        )

    controller = ThinkingController(memory, resolver=resolver)
    kernel = CognitiveKernel(memory, controller=controller)

    kernel.process("Can an eagle fly and does it have wings?")

    assert received["parsed_queries"] == [
        ("eagle", "can", "fly"),
        ("eagle", "has", "wing"),
    ]
    assert received["question_parts"] == [
        "Can an eagle fly",
        "does it have wings?",
    ]


def test_kernel_routes_to_later_parsed_query_when_first_clause_is_unknown():
    memory = MemoryStore(":memory:", seed_ontology=False)
    received = {}

    def resolver(question, **payload):
        received.update(payload)
        return InferenceResult(
            query=question,
            status=BeliefStatus.SUPPORTED,
            answer=True,
            confidence=1.0,
            evidence=[],
            trace=[],
        )

    controller = ThinkingController(memory, resolver=resolver)
    outcome = CognitiveKernel(memory, controller=controller).process(
        "Why? Is an eagle a bird?"
    )

    assert outcome.mode is CognitiveMode.THINK
    assert outcome.inference is not None
    assert outcome.inference.answer is True
    assert received["parsed_query"] is None
    assert received["parsed_queries"] == [
        None,
        ("eagle", "is_a", "bird"),
    ]


def test_question_forwards_candidate_query_without_controller_reparse(monkeypatch):
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)
    kernel.process("A falcon is a bird.")
    original_parse = SimpleParser.parse_question
    parse_calls = []

    def count_parse(text, *, known_concepts=None):
        parse_calls.append(text)
        return original_parse(text, known_concepts=known_concepts)

    monkeypatch.setattr(SimpleParser, "parse_question", count_parse)

    def fail_if_controller_reparses(_text):
        raise AssertionError("controller reparsed a perceived question")

    monkeypatch.setattr(kernel.controller, "_parse_question", fail_if_controller_reparses)

    outcome = kernel.process("Is a falcon a bird?")

    assert outcome.frame.parsed_query == ("falcon", "is_a", "bird")
    assert outcome.mode is CognitiveMode.FAST
    assert outcome.inference is not None
    assert outcome.inference.answer is True
    assert parse_calls == []


@pytest.mark.parametrize(
    ("target", "expected_mode"),
    [("bird", CognitiveMode.FAST), ("animal", CognitiveMode.THINK)],
)
def test_registered_question_construction_routes_without_controller_reparse(
    monkeypatch, target, expected_mode
):
    memory = MemoryStore(":memory:", seed_ontology=False)
    memory.save_construction(Construction.create(
        name="q_custom_taxonomy",
        pattern_tokens=["does", "{X}", "count", "as", "{Y}"],
        slot_roles={"X": "subject", "Y": "object"},
        predicate_template="is_a",
        construction_type="question",
    ))
    kernel = CognitiveKernel(memory)
    kernel.process("A falcon is a bird.")
    if target == "animal":
        kernel.process("A bird is an animal.")
    before_relations = memory.count_relations()
    before_evidence = len(memory.evidence.list_by_source("user"))

    def fail_if_controller_reparses(_text):
        raise AssertionError("controller reparsed a perceived question")

    monkeypatch.setattr(kernel.controller, "_parse_question", fail_if_controller_reparses)

    outcome = kernel.process(f"Does a falcon count as {target}?")

    assert outcome.frame.intent == "question"
    assert outcome.frame.parsed_query == ("falcon", "is_a", target)
    assert outcome.mode is expected_mode
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.SUPPORTED
    assert outcome.inference.answer is True
    assert memory.count_relations() == before_relations
    assert len(memory.evidence.list_by_source("user")) == before_evidence


def test_unproven_relation_question_remains_unknown_without_learning():
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory).process("Is a falcon a spaceship?")

    assert outcome.mode is CognitiveMode.THINK
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.UNKNOWN
    assert outcome.inference.answer is None
    assert memory.count_relations() == 0
    assert memory.evidence.list_by_source("user") == []


def test_unrecognized_text_returns_explicit_unknown_without_writes():
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory).process("qwerty zorp")

    assert outcome.frame.intent == "unknown"
    assert outcome.mode is CognitiveMode.UNKNOWN
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.query == "qwerty zorp"
    assert outcome.inference.status is BeliefStatus.UNKNOWN
    assert outcome.inference.answer is None
    assert outcome.inference.confidence == 0.0
    assert outcome.inference.evidence == []
    assert outcome.inference.trace
    assert memory.count_relations() == 0
    assert memory.evidence.list_by_source("user") == []
    assert memory._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0


def test_learning_can_be_disabled_without_reperceiving_or_writing():
    memory = MemoryStore(":memory:", seed_ontology=False)

    class RecordingAdapter:
        def __init__(self):
            self.delegate = DeterministicPerceptionAdapter()
            self.calls = 0

        def perceive(self, text, *, memory, context):
            self.calls += 1
            return self.delegate.perceive(text, memory=memory, context=context)

    adapter = RecordingAdapter()
    kernel = CognitiveKernel(memory, perception=adapter)

    outcome = kernel.process("A falcon is a bird.", allow_learning=False)

    assert adapter.calls == 1
    assert outcome.frame.intent == "statement"
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.UNKNOWN
    assert outcome.inference.answer is None
    assert memory.count_relations() == 0
    assert memory.evidence.list_by_source("user") == []
    assert memory._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0


def test_action_uses_study_route_and_commits_through_ledger():
    memory = MemoryStore(":memory:", seed_ontology=False)

    outcome = CognitiveKernel(memory).process("Slice an apple into 4 pieces.")

    assert outcome.frame.intent == "action"
    assert len(outcome.frame.actions) == 1
    assert outcome.mode is CognitiveMode.STUDY
    assert outcome.learning is not None
    assert outcome.learning.update_type is UpdateType.NEW_ENTITY
    assert outcome.inference is None
    assert memory.get_concept("apple slice") is not None
    evidence = memory.evidence.list_by_source("user")
    assert len(evidence) == 3
    assert sum(record.predicate == "action:slice" for record in evidence) == 1
    assert sum(record.predicate in {"part_of", "is_a"} for record in evidence) == 2
    assert all(record.status.value == "accepted" for record in evidence)


def test_followup_statement_uses_learner_dialogue_context():
    memory = MemoryStore(":memory:", seed_ontology=False)
    kernel = CognitiveKernel(memory)
    kernel.process("A falcon is a bird.")

    outcome = kernel.process("It is an animal.")

    assert outcome.mode is CognitiveMode.STUDY
    assert outcome.frame.claims[0].subject == "falcon"
    assert outcome.learning is not None
    assert memory.find_relation_by_names("falcon", "is_a", "animal") is not None
