import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from little.core.contracts import CandidateAction, CandidateClaim, CandidateEntity
from little.language.laya_adapter import (
    LayaAdapterPolicy,
    LayaDecision,
    LayaPerceptionAdapter,
)
from little.memory.store import MemoryStore


@dataclass
class StubBackend:
    decision: LayaDecision | object

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    def decide(self, text: str) -> LayaDecision | object:
        self.calls.append(text)
        return self.decision


def _decision(**overrides: object) -> LayaDecision:
    values: dict[str, object] = {
        "intent": "assertion",
        "uncertainty": 0.12,
        "intent_probabilities": {"assertion": 0.88, "query": 0.12},
        "intent_alternatives": ("query",),
        "claims": (CandidateClaim("falcon", "is_a", "bird", probability=0.88),),
        "entities": (CandidateEntity("falcon", probability=0.95),),
        "actions": (),
        "model_id": "laya-runtime",
        "model_version": "2026.09",
    }
    values.update(overrides)
    return LayaDecision(**values)


def test_constructor_requires_an_injected_backend():
    with pytest.raises(TypeError):
        LayaPerceptionAdapter()  # type: ignore[call-arg]


def test_statement_question_and_action_intents_are_mapped_from_policy():
    memory = MemoryStore(":memory:", seed_ontology=False)

    statement_backend = StubBackend(_decision())
    statement = LayaPerceptionAdapter(statement_backend).perceive(
        "A falcon is a bird.", memory=memory, context=object()  # type: ignore[arg-type]
    )
    assert statement.intent == "statement"
    assert statement.claims[0].subject == "falcon"
    assert statement.entities[0].name == "falcon"
    assert statement.intent_probabilities == {
        "assertion": 0.88,
        "query": 0.12,
    }
    assert statement.intent_alternatives == ("query",)
    assert statement.model_id == "laya-runtime"
    assert statement.model_version == "2026.09"
    assert statement_backend.calls == ["A falcon is a bird."]

    question_backend = StubBackend(
        _decision(
            intent="query",
            uncertainty=0.2,
            intent_probabilities={"query": 0.8, "assertion": 0.2},
            intent_alternatives=("assertion",),
            claims=(),
            entities=(CandidateEntity("falcon", probability=0.9),),
            parsed_query=("falcon", "is_a", "bird"),
            parsed_queries=(("falcon", "is_a", "bird"),),
            question_parts=("Is a falcon a bird?",),
        )
    )
    question = LayaPerceptionAdapter(question_backend).perceive(
        "Is a falcon a bird?", memory=memory
    )
    assert question.intent == "question"
    assert question.claims == []
    assert question.parsed_query == ("falcon", "is_a", "bird")
    assert question.parsed_queries == [("falcon", "is_a", "bird")]
    assert question.question_parts == ["Is a falcon a bird?"]

    action_backend = StubBackend(
        _decision(
            intent="command",
            uncertainty=0.18,
            intent_probabilities={"command": 0.82, "query": 0.18},
            intent_alternatives=("query",),
            claims=(),
            entities=(),
            actions=(
                CandidateAction(
                    "slice",
                    {"object": "apple"},
                    probability=0.82,
                ),
            ),
        )
    )
    action = LayaPerceptionAdapter(action_backend).perceive(
        "Slice the apple.", memory=memory
    )
    assert action.intent == "action"
    assert action.actions[0].name == "slice"
    assert memory.count_relations() == 0


def test_unmapped_intent_fails_closed_without_mutating_memory():
    memory = MemoryStore(":memory:", seed_ontology=False)
    before = (len(memory.list_concepts()), memory.count_relations())
    backend = StubBackend(
        _decision(
            intent="unsupported-intent",
            intent_probabilities={"unsupported-intent": 1.0},
            intent_alternatives=(),
            claims=(CandidateClaim("falcon", "is_a", "bird", probability=1.0),),
            actions=(CandidateAction("slice", {}, probability=1.0),),
        )
    )

    frame = LayaPerceptionAdapter(backend).perceive(
        "An unsupported model output.", memory=memory
    )

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.actions == []
    assert frame.model_id == "laya-runtime"
    assert frame.model_version == "2026.09"
    assert frame.intent_probabilities == {"unsupported-intent": 1.0}
    assert before == (len(memory.list_concepts()), memory.count_relations())


def test_missing_required_payload_fails_closed():
    memory = MemoryStore(":memory:", seed_ontology=False)
    backend = StubBackend(
        _decision(
            intent="assertion",
            claims=(),
            actions=(CandidateAction("slice", {}, probability=0.8),),
        )
    )

    frame = LayaPerceptionAdapter(backend).perceive(
        "A missing candidate payload.", memory=memory
    )

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.actions == []
    assert frame.entities == []


def test_disallowed_mixed_payload_fails_closed():
    memory = MemoryStore(":memory:", seed_ontology=False)
    backend = StubBackend(
        _decision(
            claims=(CandidateClaim("falcon", "is_a", "bird", probability=1.0),),
            actions=(CandidateAction("slice", {"object": "apple"}, probability=1.0),),
            intent_probabilities={"assertion": 1.0},
            intent_alternatives=(),
        )
    )

    frame = LayaPerceptionAdapter(backend).perceive(
        "A falcon is a bird.", memory=memory
    )

    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.actions == []


def test_malformed_query_payload_is_rejected():
    with pytest.raises(ValueError, match="subject"):
        _decision(parsed_query=(None, "is_a", "bird"))
    with pytest.raises(ValueError, match="ordinary query targets"):
        adapter = LayaPerceptionAdapter(
            StubBackend(_decision(parsed_query=("falcon", "is_a", None)))
        )
        adapter.perceive(
            "Is a falcon a bird?",
            memory=MemoryStore(":memory:", seed_ontology=False),
        )


def test_structured_query_targets_are_allowed_by_policy():
    decision = _decision(
        intent="query",
        claims=(),
        parsed_query=("little", "__math__", {"a": 1}),
        parsed_queries=(("little", "__math__", {"a": 1}),),
        question_parts=("Calculate it",),
        intent_probabilities={"query": 1.0},
        intent_alternatives=(),
    )
    frame = LayaPerceptionAdapter(StubBackend(decision)).perceive(
        "Calculate it", memory=MemoryStore(":memory:", seed_ontology=False)
    )
    assert frame.intent == "question"


def test_entity_span_is_validated_and_copied():
    with pytest.raises(ValueError, match="entity span"):
        _decision(entities=(CandidateEntity("falcon", span=(4, 2)),))


def test_decision_requires_consistent_intent_probabilities():
    with pytest.raises(ValueError, match="non-empty"):
        _decision(intent_probabilities={})
    with pytest.raises(ValueError, match="selected intent"):
        _decision(intent_probabilities={"query": 1.0})


def test_decision_freezes_nested_backend_payloads():
    claim = CandidateClaim(
        "falcon",
        "is_a",
        "bird",
        probability=1.0,
        attributes={"evidence": {"source": "test"}},
    )
    decision = _decision(
        claims=(claim,),
        intent_probabilities={"assertion": 1.0},
        intent_alternatives=(),
    )

    with pytest.raises(TypeError):
        decision.claims[0].attributes["evidence"] = "changed"  # type: ignore[index]
    assert decision.claims[0].attributes["evidence"]["source"] == "test"  # type: ignore[index]


def test_policy_rejects_unknown_payload_fields(tmp_path: Path):
    payload = json.loads(
        Path("data/schemas/laya_adapter_policy.json").read_text(encoding="utf-8")
    )
    payload["laya_adapter_policy"]["allowed_payloads"]["statement"] = [
        "source_text"
    ]
    (tmp_path / "laya_adapter_policy.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="supported payload fields"):
        LayaAdapterPolicy.load(tmp_path)


def test_deterministic_parser_identity_is_rejected():
    backend = StubBackend(_decision(model_id="deterministic-parser"))

    with pytest.raises(ValueError, match="deterministic-parser"):
        LayaPerceptionAdapter(backend).perceive(
            "A falcon is a bird.",
            memory=MemoryStore(":memory:", seed_ontology=False),
        )


def test_malformed_backend_output_and_metadata_are_rejected():
    with pytest.raises(ValueError, match="probabilit"):
        LayaDecision(
            intent="assertion",
            uncertainty=0.1,
            intent_probabilities={"assertion": 1.2},
            model_id="laya-runtime",
            model_version="2026.09",
        )

    with pytest.raises(ValueError, match="model_id"):
        LayaDecision(
            intent="assertion",
            uncertainty=0.1,
            intent_probabilities={"assertion": 1.0},
            model_id="",
            model_version="2026.09",
        )

    backend = StubBackend(object())
    with pytest.raises(TypeError, match="LayaDecision"):
        LayaPerceptionAdapter(backend).perceive(
            "A falcon is a bird.",
            memory=MemoryStore(":memory:", seed_ontology=False),
        )


def test_adapter_policy_loads_and_rejects_malformed_data(tmp_path: Path):
    policy = LayaAdapterPolicy.default()
    assert policy.fallback_intent == "unknown"
    assert policy.intent_mapping["assertion"] == "statement"
    assert policy.candidate_requirements["question"] == (
        "parsed_query",
        "parsed_queries",
        "question_parts",
    )

    policy_path = tmp_path / "laya_adapter_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "laya_adapter_policy": {
                    "version": "1",
                    "fallback_intent": "unknown",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rejected_model_ids"):
        LayaAdapterPolicy.load(tmp_path)


def test_adapter_does_not_forward_memory_or_context():
    class ForbiddenMemory:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"memory was accessed: {name}")

    class ForbiddenContext:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"context was accessed: {name}")

    backend = StubBackend(_decision())
    frame = LayaPerceptionAdapter(backend).perceive(
        "A falcon is a bird.",
        memory=ForbiddenMemory(),  # type: ignore[arg-type]
        context=ForbiddenContext(),  # type: ignore[arg-type]
    )

    assert frame.intent == "statement"
    assert backend.calls == ["A falcon is a bird."]
