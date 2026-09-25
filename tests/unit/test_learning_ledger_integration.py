from little.core.contracts import (
    CandidateAction,
    CandidateClaim,
    CandidateFrame,
    SourceType,
)
from little.core.models import BeliefStatus, Construction, LearningResult, UpdateType
from little.inference.invariant_gates import InvariantGateResult
from little.knowledge.registry import ActionSchema, RelationSchema, SchemaRegistry
from little.language.parser import LearningEngine
from little.language.perception import DeterministicPerceptionAdapter
from little.memory.store import MemoryStore


def test_learning_records_user_provenance_and_commits_verified_fact():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)

    result = engine.learn("A falcon is a bird.")

    assert result.relations_created
    records = store.evidence.list_by_source("user")
    assert len(records) == 1
    assert records[0].status.value == "accepted"
    assert records[0].source_text == "A falcon is a bird."
    assert (
        records[0].extraction_confidence
        == engine.perception.policy.deterministic_claim_probability
    )
    assert store.find_relation_by_names("falcon", "is_a", "bird") is not None


def test_learn_perceives_once_and_commits_only_the_supplied_frame():
    class OneFrameAdapter:
        def __init__(self):
            self.calls = 0

        def perceive(self, text, *, memory, context):
            self.calls += 1
            return CandidateFrame.from_text(
                text,
                claims=[CandidateClaim("falcon", "is_a", "bird", probability=0.73)],
            )

    store = MemoryStore(":memory:", seed_ontology=False)
    adapter = OneFrameAdapter()
    result = LearningEngine(store, perception=adapter).learn("unparseable words")

    assert adapter.calls == 1
    assert result.update_type is UpdateType.NEW_RELATION
    assert store.find_relation_by_names("falcon", "is_a", "bird") is not None
    assert store.evidence.list_by_source("user")[0].extraction_confidence == 0.73


def test_commit_frame_uses_supplied_source_and_preserves_negative_claim():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "A bird is not a vehicle.",
        claims=[CandidateClaim("bird", "is_a", "vehicle", 0.8, positive=False)],
    )

    result = LearningEngine(store).commit_frame(frame, source_type=SourceType.DOCUMENT)

    assert result.update_type is UpdateType.NEW_RELATION
    record = store.evidence.list_by_source("document")[0]
    assert record.positive is False
    assert record.source_reference == f"frame:{frame.frame_id}"
    assert store.find_relation_by_names("bird", "is_a", "vehicle").weight_negative == 1


def test_negated_category_statement_still_refutes_inherited_membership():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)
    engine.learn("A dog is a mammal.")
    engine.learn("A mammal is not a reptile.")

    assert engine.ask("Is a dog a reptile?").status is BeliefStatus.REFUTED
    negative = store.evidence.list_for_claim("mammal", "disjoint_with", "reptile")
    assert len(negative) == 1 and negative[0].positive is False


def test_property_commit_records_decision_without_creating_relation():
    store = MemoryStore(":memory:", seed_ontology=False)
    result = LearningEngine(store).learn("The apple is red.")

    assert result.update_type is UpdateType.PROPERTY_UPDATE
    assert store.get_concept("apple").attributes["color"] == "red"
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")
    assert len(evidence) == 1 and evidence[0].status.value == "accepted"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence[0].evidence_id,),
    ).fetchone()[0] == "accepted"


def test_registered_attribute_schema_overrides_false_property_metadata():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "The apple is red.",
        claims=[CandidateClaim("apple", "color", "red", is_property=False)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.PROPERTY_UPDATE
    assert store.get_concept("apple").attributes["color"] == "red"
    assert store.count_relations() == 0


def test_property_verification_uses_active_ledger_registry():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={
            "shade": RelationSchema(
                name="shade",
                domain=("concept", "entity"),
                range=("value",),
                attribute_key="shade",
            )
        },
        actions={},
        state_variables={},
    )
    frame = CandidateFrame.from_text(
        "The apple is crimson.",
        claims=[CandidateClaim("apple", "shade", "crimson", is_property=False)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.PROPERTY_UPDATE
    assert store.get_concept("apple").attributes["shade"] == "crimson"
    assert store.count_relations() == 0


def test_relation_schema_overrides_property_metadata_for_graph_relations():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "The eagle can fly.",
        claims=[CandidateClaim("eagle", "can", "fly", is_property=True)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NEW_RELATION
    assert store.find_relation_by_names("eagle", "can", "fly") is not None
    assert store.count_relations() == 1
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "accepted"


def test_unregistered_property_remains_candidate_without_attribute_write():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "The apple has an unregistered property.",
        claims=[CandidateClaim("apple", "unregistered_property", "red", is_property=True)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.get_concept("apple") is None
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "candidate"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "unknown"


def test_user_name_uses_one_accepted_property_write():
    store = MemoryStore(":memory:", seed_ontology=False)
    result = LearningEngine(store).learn("My name is Aswin.")

    assert "Aswin" in result.message
    assert store.get_concept("user").attributes == {"name": "Aswin"}
    records = store.evidence.list_by_source("user")
    assert len(records) == 1 and records[0].status.value == "accepted"


def test_user_name_replacement_keeps_scalar_identity_and_capitalization():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)

    first = engine.learn("My name is Aswin.")
    second = engine.commit_frame(CandidateFrame.from_text(
        "My name is Priya.",
        claims=[CandidateClaim("user", "name", "priya", is_property=True)],
    ))

    assert first.update_type is UpdateType.PROPERTY_UPDATE
    assert second.update_type is UpdateType.PROPERTY_UPDATE
    assert "Priya" in second.message
    assert store.get_concept("user").attributes["name"] == "Priya"
    assert engine.ask("what is my name").answer == "Your name is Priya."
    assert all(record.status.value == "accepted" for record in store.evidence.list_by_source("user"))


def test_default_property_update_policy_still_appends_distinct_values():
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)
    for color in ("red", "green"):
        engine.commit_frame(CandidateFrame.from_text(
            f"The apple is {color}.",
            claims=[CandidateClaim("apple", "color", color, is_property=True)],
        ))

    assert store.get_concept("apple").attributes["color"] == ["red", "green"]


def test_failed_user_name_verification_does_not_write_attribute(monkeypatch):
    store = MemoryStore(":memory:", seed_ontology=False)
    engine = LearningEngine(store)
    monkeypatch.setattr(
        engine.verifier,
        "verify_relation",
        lambda *_: InvariantGateResult(
            passed=False, violated_gate="I_SORT", error_message="rejected"
        ),
    )

    result = engine.learn("My name is Aswin.")

    assert result.update_type is UpdateType.NO_OP
    assert store.get_concept("user") is None
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "rejected"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "rejected"


def test_unregistered_relation_remains_candidate_only():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "falcon glimmers sky",
        claims=[CandidateClaim("falcon", "glimmers", "sky", 0.6)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")
    assert len(evidence) == 1 and evidence[0].status.value == "candidate"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence[0].evidence_id,),
    ).fetchone()[0] == "unknown"


def test_construction_does_not_register_its_unapproved_relation_schema():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.save_construction(Construction.create(
        name="cxn_originated_in",
        pattern_tokens=["{x}", "originated", "in", "{y}"],
        slot_roles={"x": "subject", "y": "object"},
        predicate_template="originated_in",
    ))
    frame = CandidateFrame.from_text(
        "Apples originated in Central Asia.",
        claims=[CandidateClaim("apple", "originated_in", "central asia", 0.8)],
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "candidate"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "unknown"


def test_registered_action_preserves_transformation_result():
    store = MemoryStore(":memory:", seed_ontology=False)
    assert store.get_concept("apple") is None
    assert store.ledger.registry.action("slice").preconditions == ()
    result = LearningEngine(store).learn("Slice an apple into 4 pieces.")

    assert result.update_type is UpdateType.NEW_ENTITY
    assert result.concepts_created == ["apple slice"]
    assert len(result.relations_created) == 2
    assert store.get_concept("apple slice") is not None
    action_evidence = [
        record
        for record in store.evidence.list_by_source("user")
        if record.predicate == "action:slice"
    ]
    assert len(action_evidence) == 1 and action_evidence[0].status.value == "accepted"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (action_evidence[0].evidence_id,),
    ).fetchone()[0] == "accepted"
    assert store._conn.execute(
        "SELECT COUNT(*) AS count FROM experiences"
    ).fetchone()["count"] == 1
    for subject, predicate, object_ in (
        ("apple slice", "part_of", "apple"),
        ("apple slice", "is_a", "apple"),
    ):
        relation = store.find_relation_by_names(subject, predicate, object_)
        assert relation is not None
        evidence = store.evidence.list_for_claim(
            subject, predicate, object_
        )
        assert len(evidence) == 1 and evidence[0].status.value == "accepted"
        assert relation.source_experience_id == evidence[0].source_reference
        assert store._conn.execute(
            "SELECT status FROM commit_decisions WHERE evidence_id = ?",
            (evidence[0].evidence_id,),
        ).fetchone()[0] == "accepted"


def test_ledger_action_effects_have_one_evidence_and_decision_per_relation():
    store = MemoryStore(":memory:", seed_ontology=False)

    result = LearningEngine(store).learn("Slice an apple into 4 pieces.")

    assert result.relations_created == [
        "(apple slice part_of apple)",
        "(apple slice is_a apple)",
    ]
    assert store.count_relations() == 2
    assert store._conn.execute(
        "SELECT COUNT(*) AS count FROM evidence_records WHERE predicate != ?",
        ("action:slice",),
    ).fetchone()["count"] == 2
    for predicate in ("part_of", "is_a"):
        relation = store.find_relation_by_names("apple slice", predicate, "apple")
        assert relation is not None
        assert relation.weight_positive == 1
        evidence = store.evidence.list_for_claim("apple slice", predicate, "apple")
        assert len(evidence) == 1
        decision = store._conn.execute(
            "SELECT status FROM commit_decisions WHERE evidence_id = ?",
            (evidence[0].evidence_id,),
        ).fetchone()
        assert decision is not None and decision["status"] == "accepted"


def test_registered_split_action_uses_the_registered_transformation():
    store = MemoryStore(":memory:", seed_ontology=False)
    assert store.get_concept("apple") is None
    assert store.ledger.registry.action("split").preconditions == ()
    frame = CandidateFrame.from_text(
        "split apple into 2 pieces",
        claims=[],
        actions=[CandidateAction("split", {"object": "apple", "count": "2"})],
        intent="action",
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NEW_ENTITY
    assert len(result.relations_created) == 2
    assert store.get_concept("apple slice") is not None


def test_action_evidence_preserves_frame_source_and_confidence():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "slice apple into 2 pieces",
        claims=[],
        actions=[CandidateAction("slice", {"object": "apple", "count": "2"}, 0.72)],
        intent="action",
    )

    LearningEngine(store).commit_frame(frame, source_type=SourceType.DOCUMENT)

    evidence = [
        record
        for record in store.evidence.list_by_source("document")
        if record.predicate == "action:slice"
    ]
    assert len(evidence) == 1
    assert evidence[0].source_reference == f"frame:{frame.frame_id}"
    assert evidence[0].extraction_confidence == 0.72
    assert evidence[0].status.value == "accepted"
    effect_evidence = store.evidence.list_for_claim(
        "apple slice", "part_of", "apple"
    )
    assert len(effect_evidence) == 1
    assert effect_evidence[0].source_reference == f"action:{evidence[0].evidence_id}"


def test_registered_action_without_executor_remains_candidate():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={"inspect": ActionSchema("inspect", {"object": "entity"})},
        state_variables={},
    )
    frame = CandidateFrame.from_text(
        "inspect apple", claims=[],
        actions=[CandidateAction("inspect", {"object": "apple"})], intent="action",
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.get_concept("apple") is None
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "candidate"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "unknown"


def test_unregistered_action_has_no_graph_mutations():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "levitate an apple",
        claims=[],
        actions=[CandidateAction("levitate", {"object": "apple"}, probability=0.9)],
        intent="action",
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    assert store.get_concept("apple") is None
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "candidate"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "unknown"


def test_invalid_registered_action_records_rejection_without_mutation():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = CandidateFrame.from_text(
        "slice apple into 0 pieces",
        claims=[],
        actions=[CandidateAction("slice", {"object": "apple", "count": "0"})],
        intent="action",
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.get_concept("apple") is None
    assert store.count_relations() == 0
    evidence = store.evidence.list_by_source("user")[0]
    assert evidence.status.value == "rejected"
    assert store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (evidence.evidence_id,),
    ).fetchone()[0] == "rejected"


def test_action_executor_is_selected_from_schema_metadata():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={"divide": ActionSchema(
            name="divide",
            argument_types={"object": "entity", "count": "quantity"},
            executor="slice_object",
        )},
        state_variables={},
    )
    frame = CandidateFrame.from_text(
        "divide apple into two pieces",
        claims=[],
        actions=[CandidateAction("divide", {"object": "apple", "count": "2"})],
        intent="action",
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NEW_ENTITY
    assert store.get_concept("apple slice") is not None


def test_action_decision_is_persisted_before_executor_runs():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={"probe": ActionSchema("probe", {}, executor="probe")},
        state_variables={},
    )

    def probe(memory, frame, arguments):
        evidence = memory.list_evidence_by_source("user")
        assert len(evidence) == 1 and evidence[0].status.value == "accepted"
        assert memory.get_commit_status(evidence[0].evidence_id) == "accepted"
        assert memory.count_relations() == 0
        return LearningResult(frame.source_text, UpdateType.NEW_ENTITY, "")

    store.ledger.executors.register("probe", probe)
    frame = CandidateFrame.from_text(
        "probe", claims=[], actions=[CandidateAction("probe", {})], intent="action"
    )

    assert LearningEngine(store).commit_frame(frame).update_type is UpdateType.NEW_ENTITY


def test_action_executor_cannot_write_semantic_memory_directly():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={"probe": ActionSchema("probe", {}, executor="probe")},
        state_variables={},
    )

    def probe(memory, frame, arguments):
        memory.add_relation("subject", "predicate", "object")
        return LearningResult(frame.source_text, UpdateType.NEW_ENTITY, "")

    store.ledger.executors.register("probe", probe)
    frame = CandidateFrame.from_text(
        "probe", claims=[], actions=[CandidateAction("probe", {})], intent="action"
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    action_evidence = store.evidence.list_by_source("user")[0]
    assert action_evidence.status.value == "rejected"
    assert store._conn.execute(
        "SELECT COUNT(*) AS count FROM commit_decisions WHERE evidence_id = ?",
        (action_evidence.evidence_id,),
    ).fetchone()["count"] == 1


def test_action_executor_cannot_store_relation_like_physical_metadata():
    store = MemoryStore(":memory:", seed_ontology=False)
    default_registry = SchemaRegistry.default()
    store.ledger.registry = SchemaRegistry(
        relations={
            name: default_registry.relation(name)
            for name in ("part_of", "is_a")
            if default_registry.relation(name) is not None
        },
        actions={"probe": ActionSchema("probe", {}, executor="probe")},
        state_variables={},
    )

    def probe(memory, frame, arguments):
        concept = memory.create_concept(
            "slice", attributes={" PART_OF ": "apple"}
        )
        memory.create_entity(
            "slice_1", concept.id, properties={"part_of": "apple"}
        )
        return LearningResult(frame.source_text, UpdateType.NEW_ENTITY, "")

    store.ledger.executors.register("probe", probe)
    frame = CandidateFrame.from_text(
        "probe", claims=[], actions=[CandidateAction("probe", {})], intent="action"
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    assert store.get_concept("slice") is None


def test_action_executor_relations_are_not_trusted_without_schema_effects():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={"probe": ActionSchema("probe", {}, executor="probe")},
        state_variables={},
    )

    def probe(memory, frame, arguments):
        return LearningResult(
            frame.source_text,
            UpdateType.NEW_ENTITY,
            "",
            relations_created=["(forged relation)"],
        )

    store.ledger.executors.register("probe", probe)
    frame = CandidateFrame.from_text(
        "probe", claims=[], actions=[CandidateAction("probe", {})], intent="action"
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NEW_ENTITY
    assert result.relations_created == []
    assert store.count_relations() == 0


def test_malformed_schema_effect_is_rejected_and_recorded():
    store = MemoryStore(":memory:", seed_ontology=False)
    store.ledger.registry = SchemaRegistry(
        relations={},
        actions={
            "probe": ActionSchema(
                "probe",
                {},
                effects=({"type": "relation", "subject": "{missing}"},),
                executor="probe",
            )
        },
        state_variables={},
    )

    def probe(memory, frame, arguments):
        return LearningResult(frame.source_text, UpdateType.NEW_ENTITY, "")

    store.ledger.executors.register("probe", probe)
    frame = CandidateFrame.from_text(
        "probe", claims=[], actions=[CandidateAction("probe", {})], intent="action"
    )

    result = LearningEngine(store).commit_frame(frame)

    assert result.update_type is UpdateType.NEW_ENTITY
    effect_evidence = [
        record
        for record in store.evidence.list_by_source("user")
        if record.predicate == "action_effect"
    ]
    assert len(effect_evidence) == 1
    assert effect_evidence[0].status.value == "rejected"
    decision = store._conn.execute(
        "SELECT status FROM commit_decisions WHERE evidence_id = ?",
        (effect_evidence[0].evidence_id,),
    ).fetchone()
    assert decision["status"] == "rejected"


def test_unrecognized_input_is_no_op_with_zero_edges():
    store = MemoryStore(":memory:", seed_ontology=False)
    result = LearningEngine(store).learn("qwerty zorp")

    assert result.update_type is UpdateType.NO_OP
    assert store.count_relations() == 0
    assert result.experience_id


def test_candidate_frame_alone_cannot_create_relation():
    store = MemoryStore(":memory:", seed_ontology=False)
    frame = DeterministicPerceptionAdapter().perceive(
        "A falcon is a bird.", memory=store
    )

    assert frame.accepted is False
    assert store.count_relations() == 0
