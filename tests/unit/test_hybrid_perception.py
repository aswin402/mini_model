import json
from dataclasses import replace
from pathlib import Path

import pytest

from little.core.contracts import CandidateAction, CandidateClaim, CandidateEntity, CandidateFrame
from little.core.kernel import CognitiveKernel
from little.core.models import BeliefStatus, UpdateType
from little.inference.thinking_controller import CognitiveMode
from little.language.hybrid_perception import HybridPerceptionAdapter, HybridPerceptionPolicy
from little.language.laya_adapter import LayaAdapterPolicy, LayaDecision
from little.memory.store import MemoryStore


def test_hybrid_policy_loads_data_owned_reconciliation_settings():
    policy = HybridPerceptionPolicy.load(Path("data/schemas"))

    assert policy.version == "1"
    assert policy.unknown_intent == "unknown"
    assert policy.require_route_match is True
    assert policy.uncertainty_strategy == "maximum"
    assert policy.model_id_format == "{system1_model_id}+{grounding_model_id}"
    assert policy.model_version_format == (
        "{system1_model_version}+{grounding_model_version}"
    )


def test_hybrid_policy_default_loads_repository_policy():
    assert HybridPerceptionPolicy.default() == HybridPerceptionPolicy.load(
        Path("data/schemas")
    )


def _write_policy(tmp_path: Path, **updates) -> None:
    data = {
        "version": "1",
        "unknown_intent": "unknown",
        "require_route_match": True,
        "uncertainty_strategy": "maximum",
        "model_id_format": "{system1_model_id}+{grounding_model_id}",
        "model_version_format": "{system1_model_version}+{grounding_model_version}",
    }
    data.update(updates)
    (tmp_path / "hybrid_perception_policy.json").write_text(
        json.dumps({"hybrid_perception_policy": data}), encoding="utf-8"
    )


def test_hybrid_policy_rejects_missing_field(tmp_path):
    _write_policy(tmp_path)
    payload = json.loads((tmp_path / "hybrid_perception_policy.json").read_text())
    del payload["hybrid_perception_policy"]["unknown_intent"]
    (tmp_path / "hybrid_perception_policy.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unknown_intent"):
        HybridPerceptionPolicy.load(tmp_path)


def test_hybrid_policy_requires_boolean_route_match(tmp_path):
    _write_policy(tmp_path, require_route_match=1)

    with pytest.raises(ValueError, match="require_route_match"):
        HybridPerceptionPolicy.load(tmp_path)


def test_hybrid_policy_rejects_disabled_route_match(tmp_path):
    _write_policy(tmp_path, require_route_match=False)

    with pytest.raises(ValueError, match="require_route_match"):
        HybridPerceptionPolicy.load(tmp_path)


def test_hybrid_policy_rejects_unsupported_uncertainty_strategy(tmp_path):
    _write_policy(tmp_path, uncertainty_strategy="guess")

    with pytest.raises(ValueError, match="uncertainty_strategy"):
        HybridPerceptionPolicy.load(tmp_path)


def test_hybrid_policy_rejects_malformed_metadata_template(tmp_path):
    _write_policy(
        tmp_path,
        model_id_format="{system1_model_id}+{grounding_model_id}{",
    )

    with pytest.raises(ValueError, match="model_id_format.*format syntax"):
        HybridPerceptionPolicy.load(tmp_path)


def test_hybrid_policy_requires_actual_substitution_fields(tmp_path):
    _write_policy(
        tmp_path,
        model_id_format="{{system1_model_id}}+{grounding_model_id}",
    )

    with pytest.raises(ValueError, match="model_id_format.*placeholders"):
        HybridPerceptionPolicy.load(tmp_path)


@pytest.mark.parametrize(
    ("template", "invalid_part"),
    [
        (
            "{system1_model_id}+{grounding_model_id}{system1_model_id!x}",
            "conversion",
        ),
        (
            "{system1_model_id}+{grounding_model_id}{system1_model_id:>10}",
            "format specifier",
        ),
        (
            "{system1_model_id}+{grounding_model_id}{other_model_id}",
            "field",
        ),
    ],
)
def test_hybrid_policy_rejects_unsupported_template_components(
    tmp_path, template, invalid_part
):
    _write_policy(tmp_path, model_id_format=template)

    with pytest.raises(ValueError, match=f"model_id_format.*{invalid_part}"):
        HybridPerceptionPolicy.load(tmp_path)


@pytest.mark.parametrize(
    ("field", "missing_placeholder"),
    [
        ("model_id_format", "{system1_model_id}"),
        ("model_id_format", "{grounding_model_id}"),
        ("model_version_format", "{system1_model_version}"),
        ("model_version_format", "{grounding_model_version}"),
    ],
)
def test_hybrid_policy_requires_metadata_placeholders(
    tmp_path, field, missing_placeholder
):
    value = (
        "{system1_model_id}+{grounding_model_id}"
        if field == "model_id_format"
        else "{system1_model_version}+{grounding_model_version}"
    )
    _write_policy(tmp_path, **{field: value.replace(missing_placeholder, "")})

    with pytest.raises(ValueError, match=field):
        HybridPerceptionPolicy.load(tmp_path)


@pytest.mark.parametrize(
    "field", ["version", "unknown_intent", "uncertainty_strategy", "model_id_format", "model_version_format"]
)
def test_hybrid_policy_rejects_empty_string_fields(tmp_path, field):
    _write_policy(tmp_path, **{field: "  "})

    with pytest.raises(ValueError, match=field):
        HybridPerceptionPolicy.load(tmp_path)


class FakeSystem1:
    def __init__(self, decision):
        self.decision = decision
        self.inputs = []

    def decide(self, text):
        self.inputs.append(text)
        return self.decision


class FakeGrounding:
    def __init__(self, frame):
        self.frame = frame
        self.inputs = []

    def perceive(self, text, *, memory, context=None):
        self.inputs.append((text, memory, context))
        return self.frame


def make_hybrid(decision, frame):
    return HybridPerceptionAdapter(FakeSystem1(decision), FakeGrounding(frame))


def decision_for(intent="assertion", **updates):
    values = {
        "intent": intent,
        "uncertainty": 0.2,
        "model_id": "laya",
        "model_version": "checkpoint",
        "intent_probabilities": {intent: 1.0},
    }
    values.update(updates)
    return LayaDecision(**values)


def frame_for(intent="statement", **updates):
    values = {
        "claims": [],
        "intent": intent,
        "uncertainty": 0.1,
        "model_id": "grounder",
        "model_version": "policy-1",
    }
    values.update(updates)
    return CandidateFrame.from_text("grounded source", **values)


def assert_empty_unknown(frame):
    assert frame.intent == "unknown"
    assert frame.claims == []
    assert frame.entities == []
    assert frame.actions == []
    assert frame.parsed_query is None
    assert frame.parsed_queries == []
    assert frame.question_parts == []


def test_matching_statement_routes_preserve_grounded_claims_and_injection():
    claim = CandidateClaim("falcon", "is_a", "bird", probability=0.9)
    entity = CandidateEntity("falcon", probability=0.95)
    decision = decision_for(
        uncertainty=0.2,
        intent_probabilities={"assertion": 0.8, "query": 0.2},
        intent_alternatives=("query",),
        claims=(CandidateClaim("wrong", "is_a", "claim", probability=1.0),),
    )
    grounding = frame_for(claims=[claim], entities=[entity])
    system1_provider = FakeSystem1(decision)
    grounding_provider = FakeGrounding(grounding)
    memory = MemoryStore(":memory:", seed_ontology=False)
    context = object()

    frame = HybridPerceptionAdapter(system1_provider, grounding_provider).perceive(
        "A falcon is a bird.", memory=memory, context=context
    )

    assert frame.intent == "statement"
    assert frame.claims == [claim]
    assert frame.entities == [entity]
    assert frame.claims is not grounding.claims
    assert frame.source_text == "A falcon is a bird."
    assert frame.uncertainty == pytest.approx(0.2)
    assert frame.model_id == "laya+grounder"
    assert frame.model_version == "checkpoint+policy-1"
    assert frame.intent_probabilities == {"assertion": 0.8, "query": 0.2}
    assert frame.intent_alternatives == ("query",)
    assert system1_provider.inputs == ["A falcon is a bird."]
    assert grounding_provider.inputs == [("A falcon is a bird.", memory, context)]
    assert grounding.claims == [claim]


def test_matching_hybrid_statement_uses_kernel_verification_and_ledger():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = make_hybrid(
        decision_for("assertion"),
        frame_for(claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.9)]),
    )
    kernel = CognitiveKernel(memory, perception=adapter)

    outcome = kernel.process("A falcon is a bird.")

    assert kernel.perception is adapter
    assert kernel.learner.perception is adapter
    assert outcome.frame.intent == "statement"
    assert outcome.frame.model_id == "laya+grounder"
    assert outcome.mode is CognitiveMode.STUDY
    assert outcome.learning is not None
    assert outcome.learning.update_type is UpdateType.NEW_RELATION
    assert outcome.inference is None
    assert memory.find_relation_by_names("falcon", "is_a", "bird") is not None
    evidence = memory.evidence.list_by_source("user")
    assert len(evidence) == 1
    assert evidence[0].status.value == "accepted"
    assert evidence[0].source_reference == f"frame:{outcome.frame.frame_id}"
    assert evidence[0].extraction_confidence == pytest.approx(0.9)
    decision = memory._conn.execute(
        "SELECT status, checks_json FROM commit_decisions WHERE evidence_id = ?",
        (evidence[0].evidence_id,),
    ).fetchone()
    assert decision[0] == "accepted"
    checks = json.loads(decision[1])
    assert {gate: checks.get(gate) for gate in ("I_DAG", "I_MUTEX", "I_SORT", "I_GROUND")} == {
        "I_DAG": True,
        "I_MUTEX": True,
        "I_SORT": True,
        "I_GROUND": True,
    }


def test_hybrid_route_mismatch_cannot_commit_graph_state():
    memory = MemoryStore(":memory:", seed_ontology=False)
    adapter = make_hybrid(
        decision_for("query"),
        frame_for(claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.9)]),
    )
    kernel = CognitiveKernel(memory, perception=adapter)
    before = (
        len(memory.list_concepts()),
        memory.count_relations(),
        memory._conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0],
        memory._conn.execute("SELECT COUNT(*) FROM commit_decisions").fetchone()[0],
        memory._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0],
    )

    outcome = kernel.process("A falcon is a bird.")

    assert outcome.frame.intent == "unknown"
    assert outcome.frame.claims == []
    assert outcome.mode is CognitiveMode.UNKNOWN
    assert outcome.learning is None
    assert outcome.inference is not None
    assert outcome.inference.status is BeliefStatus.UNKNOWN
    after = (
        len(memory.list_concepts()),
        memory.count_relations(),
        memory._conn.execute("SELECT COUNT(*) FROM evidence_records").fetchone()[0],
        memory._conn.execute("SELECT COUNT(*) FROM commit_decisions").fetchone()[0],
        memory._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0],
    )
    assert after == before
    assert memory.evidence.list_by_source("user") == []
    assert memory._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0] == 0


@pytest.mark.parametrize(
    "query_payload",
    [
        {"parsed_query": ("falcon", "is_a", "bird")},
        {"parsed_queries": [("falcon", "is_a", "bird")]},
        {"question_parts": ["Is a falcon a bird?"]},
    ],
)
def test_matching_question_accepts_each_policy_required_payload(query_payload):
    grounding = frame_for("question", **query_payload)

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")), FakeGrounding(grounding)
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert frame.intent == "question"
    assert frame.claims == []
    assert frame.parsed_query == grounding.parsed_query
    assert frame.parsed_queries == grounding.parsed_queries
    assert frame.question_parts == grounding.question_parts


def test_matching_action_preserves_only_grounded_actions_and_conservative_uncertainty():
    action = CandidateAction("slice", {"object": "apple"}, probability=0.8)
    grounding = frame_for("action", actions=[action], uncertainty=0.65)

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("command", uncertainty=0.1)), FakeGrounding(grounding)
    ).perceive("Slice the apple.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert frame.intent == "action"
    assert frame.actions == [action]
    assert frame.actions is not grounding.actions
    assert frame.uncertainty == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("decision_intent", "grounding_intent", "payload"),
    [
        (
            "assertion",
            "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird"), None]},
        ),
        (
            "assertion",
            "statement",
            {
                "claims": [CandidateClaim("falcon", "is_a", "bird")],
                "entities": [CandidateEntity("falcon"), None],
            },
        ),
        (
            "command",
            "action",
            {"actions": [CandidateAction("slice", {}), None]},
        ),
    ],
)
def test_mixed_valid_and_invalid_candidate_items_fail_closed(
    decision_intent, grounding_intent, payload
):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(decision_intent)),
        FakeGrounding(frame_for(grounding_intent, **payload)),
    ).perceive("A request.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)
    assert frame.accepted is False


@pytest.mark.parametrize(
    ("decision_intent", "grounding_intent", "payload"),
    [
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird", attributes=None)]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird", attributes={42: "x"})]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird")], "entities": [CandidateEntity(42)]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird")], "entities": [CandidateEntity("falcon", type_hint=42)]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird")], "entities": [CandidateEntity("falcon", span=(0, "six"))]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird")], "entities": [CandidateEntity("falcon", span=(-1, 4))]},
        ),
        (
            "assertion", "statement",
            {"claims": [CandidateClaim("falcon", "is_a", "bird")], "entities": [CandidateEntity("falcon", span=(4, 2))]},
        ),
        ("command", "action", {"actions": [CandidateAction("slice", None)]}),
        ("command", "action", {"actions": [CandidateAction("slice", [])]}),
        ("command", "action", {"actions": [CandidateAction("slice", {42: "apple"})]}),
        ("command", "action", {"actions": [CandidateAction("slice", {"object": 42})]}),
    ],
)
def test_malformed_candidate_contents_fail_closed(
    decision_intent, grounding_intent, payload
):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(decision_intent)),
        FakeGrounding(frame_for(grounding_intent, **payload)),
    ).perceive("A request.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)
    assert frame.accepted is False


def test_route_mismatch_returns_unknown_without_payload():
    grounding = frame_for(
        "question",
        parsed_query=("falcon", "is_a", "bird"),
        entities=[CandidateEntity("falcon")],
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("assertion")), FakeGrounding(grounding)
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)
    assert frame.intent_probabilities == {"assertion": 1.0}


@pytest.mark.parametrize("intent", ["unsupported", "unknown"])
def test_unknown_system1_route_fails_closed(intent):
    grounding = frame_for(
        claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.9)]
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(intent)), FakeGrounding(grounding)
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize(
    ("decision_intent", "grounding_intent"),
    [("assertion", "statement"), ("query", "question"), ("command", "action")],
)
def test_missing_required_grounding_payload_fails_closed(
    decision_intent, grounding_intent
):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(decision_intent)),
        FakeGrounding(frame_for(grounding_intent)),
    ).perceive("A request.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize(
    "query_payload",
    [
        {"parsed_queries": [None]},
        {"parsed_queries": [None, None]},
        {"question_parts": ["   "]},
    ],
)
def test_question_without_usable_payload_fails_closed(query_payload):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", **query_payload)),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_null_only_plural_queries_are_absent_when_question_parts_are_valid():
    grounding = frame_for(
        "question",
        parsed_queries=[None, None],
        question_parts=["Is a falcon a bird?", "Can it fly?"],
    )
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")), FakeGrounding(grounding)
    ).perceive("Is a falcon a bird? Can it fly?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert frame.intent == "question"
    assert frame.parsed_queries == []
    assert frame.question_parts == grounding.question_parts
    assert grounding.parsed_queries == [None, None]


def test_null_plural_query_does_not_mask_invalid_entry_with_question_parts():
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(
            frame_for(
                "question",
                parsed_queries=[None, ("falcon", "is_a")],
                question_parts=["Is a falcon a bird?"],
            )
        ),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize(
    "query",
    [
        ("", "is_a", "bird"),
        ("falcon", "   ", "bird"),
        ("falcon", "is_a"),
        ["falcon", "is_a", "bird"],
    ],
)
@pytest.mark.parametrize("field", ["parsed_query", "parsed_queries"])
def test_question_with_invalid_query_entries_fails_closed(field, query):
    payload = {field: query if field == "parsed_query" else [query]}
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", **payload)),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize("field", ["parsed_query", "parsed_queries"])
@pytest.mark.parametrize("target", [None, "", "   ", 42])
def test_ordinary_query_requires_nonblank_string_target(field, target):
    query = ("falcon", "is_a", target)
    payload = {field: query if field == "parsed_query" else [query]}
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", **payload)),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_configured_structured_query_accepts_targetless_payload():
    laya_policy = replace(
        LayaAdapterPolicy.default(),
        structured_query_predicate_prefixes=("meta:",),
    )
    query = ("falcon", "meta:properties", None)
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", parsed_query=query)),
        laya_policy=laya_policy,
    ).perceive("What properties?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert frame.intent == "question"
    assert frame.parsed_query == query


def test_unconfigured_structured_query_rejects_targetless_payload():
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", parsed_query=("falcon", "meta:properties", None))),
    ).perceive("What properties?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_valid_query_does_not_mask_invalid_ordinary_target():
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(
            frame_for(
                "question",
                parsed_queries=[("falcon", "is_a", "bird"), ("falcon", "is_a", None)],
                question_parts=["Is a falcon a bird?"],
            )
        ),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize("field", ["parsed_query", "parsed_queries"])
def test_question_parts_do_not_mask_invalid_query_fields(field):
    invalid_query = ("falcon", "is_a")
    payload = {
        "question_parts": ["Is a falcon a bird?"],
        field: invalid_query if field == "parsed_query" else [invalid_query],
    }
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", **payload)),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_plural_queries_with_valid_and_invalid_entries_fail_closed():
    grounding = frame_for(
        "question",
        parsed_queries=[("falcon", "is_a"), ("falcon", "is_a", "bird")],
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")), FakeGrounding(grounding)
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize("question_parts", [[None], ["", "  "], [42]])
def test_question_with_invalid_question_parts_fails_closed(question_parts):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(frame_for("question", question_parts=question_parts)),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize("invalid_part", [None, "  ", 42])
def test_question_with_mixed_valid_and_invalid_parts_fails_closed(invalid_part):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("query")),
        FakeGrounding(
            frame_for("question", question_parts=["Is a falcon a bird?", invalid_part])
        ),
    ).perceive("Is a falcon a bird?", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize(
    "disallowed_payload",
    [
        {"parsed_query": ("falcon", "is_a")},
        {"parsed_queries": [("falcon", "is_a")]},
        {"question_parts": [None]},
    ],
)
def test_disallowed_malformed_payload_fails_closed(disallowed_payload):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for()),
        FakeGrounding(
            frame_for(
                claims=[CandidateClaim("falcon", "is_a", "bird")],
                **disallowed_payload,
            )
        ),
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_rejected_system1_model_id_cannot_produce_hybrid_statement():
    rejected_id = LayaAdapterPolicy.default().rejected_model_ids[0]
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(model_id=rejected_id.upper())),
        FakeGrounding(frame_for(claims=[CandidateClaim("falcon", "is_a", "bird")])),
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)
    assert frame.accepted is False
    assert rejected_id.casefold() not in frame.model_id.casefold()


def test_injected_policy_rejected_model_id_cannot_produce_hybrid_statement():
    laya_policy = replace(
        LayaAdapterPolicy.default(), rejected_model_ids=("reserved-system-one",)
    )
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(model_id="RESERVED-SYSTEM-ONE")),
        FakeGrounding(frame_for(claims=[CandidateClaim("falcon", "is_a", "bird")])),
        laya_policy=laya_policy,
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)
    assert "reserved-system-one" not in frame.model_id.casefold()


@pytest.mark.parametrize(
    ("decision_intent", "grounding_intent", "payload"),
    [
        (
            "assertion",
            "statement",
            {
                "claims": [CandidateClaim("falcon", "is_a", "bird")],
                "actions": [CandidateAction("slice", {})],
            },
        ),
        (
            "query",
            "question",
            {
                "parsed_query": ("falcon", "is_a", "bird"),
                "claims": [CandidateClaim("falcon", "is_a", "bird")],
            },
        ),
        (
            "command",
            "action",
            {
                "actions": [CandidateAction("slice", {})],
                "question_parts": ["What next?"],
            },
        ),
    ],
)
def test_disallowed_mixed_grounding_payload_fails_closed(
    decision_intent, grounding_intent, payload
):
    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for(decision_intent)),
        FakeGrounding(frame_for(grounding_intent, **payload)),
    ).perceive("A request.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


@pytest.mark.parametrize("invalid_source", ["system1", "grounding"])
def test_malformed_provider_return_type_becomes_empty_unknown(invalid_source):
    decision = object() if invalid_source == "system1" else decision_for()
    grounding = (
        object()
        if invalid_source == "grounding"
        else frame_for(claims=[CandidateClaim("falcon", "is_a", "bird")])
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision), FakeGrounding(grounding)
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_injected_policies_control_mapping_and_metadata():
    hybrid_policy = replace(
        HybridPerceptionPolicy.default(),
        model_id_format="{grounding_model_id}/{system1_model_id}",
        model_version_format="{grounding_model_version}/{system1_model_version}",
    )
    laya_policy = replace(
        LayaAdapterPolicy.default(),
        intent_mapping={"custom": "statement"},
    )
    claim = CandidateClaim("falcon", "is_a", "bird")

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("custom")),
        FakeGrounding(frame_for("statement", claims=[claim])),
        policy=hybrid_policy,
        laya_policy=laya_policy,
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert frame.intent == "statement"
    assert frame.claims == [claim]
    assert frame.model_id == "grounder/laya"
    assert frame.model_version == "policy-1/checkpoint"


def test_injected_policy_cannot_disable_route_agreement():
    hybrid_policy = replace(HybridPerceptionPolicy.default(), require_route_match=False)
    laya_policy = replace(
        LayaAdapterPolicy.default(), intent_mapping={"custom": "statement"}
    )
    claim = CandidateClaim("falcon", "is_a", "bird")

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for("custom")),
        FakeGrounding(frame_for("question", claims=[claim])),
        policy=hybrid_policy,
        laya_policy=laya_policy,
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_custom_policy_rejects_payload_removed_from_every_allowed_route():
    base_policy = LayaAdapterPolicy.default()
    laya_policy = replace(
        base_policy,
        allowed_payloads={
            route: tuple(field for field in fields if field != "entities")
            for route, fields in base_policy.allowed_payloads.items()
        },
    )
    grounding = frame_for(
        claims=[CandidateClaim("falcon", "is_a", "bird")],
        entities=[CandidateEntity("falcon")],
    )

    frame = HybridPerceptionAdapter(
        FakeSystem1(decision_for()), FakeGrounding(grounding), laya_policy=laya_policy
    ).perceive("A falcon is a bird.", memory=MemoryStore(":memory:", seed_ontology=False))

    assert_empty_unknown(frame)


def test_constructor_requires_both_injected_providers():
    with pytest.raises(TypeError):
        HybridPerceptionAdapter(None, FakeGrounding(frame_for()))
    with pytest.raises(TypeError):
        HybridPerceptionAdapter(FakeSystem1(decision_for()), object())


@pytest.mark.parametrize("text", ["", "  ", None])
def test_perceive_rejects_empty_input_before_calling_providers(text):
    system1_provider = FakeSystem1(decision_for())
    grounding_provider = FakeGrounding(frame_for())

    with pytest.raises(ValueError, match="non-empty"):
        HybridPerceptionAdapter(system1_provider, grounding_provider).perceive(
            text, memory=MemoryStore(":memory:", seed_ontology=False)
        )

    assert system1_provider.inputs == []
    assert grounding_provider.inputs == []
