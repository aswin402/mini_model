"""Regressions for audited evidence and persistent state defects."""

import sqlite3

import pytest

from little.core.contracts import (
    CandidateAction,
    CandidateClaim,
    CandidateFrame,
    SourceType,
)
from little.core.models import Skill
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def propose(store):
    claim = CandidateClaim("subject", "is_a", "object", probability=0.9)
    frame = CandidateFrame.from_text("A subject is an object.", [claim])
    evidence = store.ledger.propose_relation(frame, claim, SourceType.DOCUMENT, "paper:test")
    verification = store.ledger.verifier.verify_relation("subject", "is_a", "object").to_verification_result()
    return evidence, verification


def test_same_evidence_does_not_increase_support_on_retry():
    with MemoryStore() as store:
        evidence, verification = propose(store)
        first = store.ledger.commit_relation(evidence, verification)
        second = store.ledger.commit_relation(evidence, verification)
        assert store.find_relation_by_names("subject", "is_a", "object").weight_positive == 1
        assert first.decision_id == second.decision_id
        assert store._conn.execute("SELECT COUNT(*) FROM commit_decisions").fetchone()[0] == 1


def test_failed_promotion_rolls_back_graph_and_new_concepts():
    with MemoryStore() as store:
        evidence, verification = propose(store)
        store._conn.execute(
            "CREATE TRIGGER fail_status BEFORE UPDATE ON evidence_records "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
            store.ledger.commit_relation(evidence, verification)
        assert store.count_relations() == 0
        assert store.get_concept("subject") is None
        assert store.evidence.get(evidence.evidence_id).status.value == "candidate"


def test_relation_source_resolves_to_an_actual_experience():
    with MemoryStore() as store:
        LearningEngine(store).learn("A falcon is a bird.")
        relation = store.find_relation_by_names("falcon", "is_a", "bird")
        experience = store.get_experience(relation.source_experience_id)
        assert experience is not None
        assert experience.input_text == "A falcon is a bird."


def test_bootstrap_preserves_existing_customized_builtin():
    with MemoryStore() as store:
        store.save_skill(Skill.create("ADD", ["a", "b"], "a + b + 100", "Custom procedure"))
        LearningEngine(store)
        assert store.get_skill("ADD").code_body == "a + b + 100"


def test_weighted_relation_import_is_exact_and_idempotent():
    with MemoryStore() as store:
        subject = store.create_concept("subject")
        object_ = store.create_concept("object")
        for _ in range(2):
            store.import_relation_record(subject.id, "is_a", object_.id, 2, 5)
        relation = store.find_relation_by_names("subject", "is_a", "object")
        assert (relation.weight_positive, relation.weight_negative) == (2, 5)


def test_document_ingestion_keeps_document_identity():
    from types import SimpleNamespace

    from little.active.stream_learner import ContinuousStreamLearner
    with MemoryStore() as store:
        learner = LearningEngine(store)
        kernel = SimpleNamespace(memory=store, learner=learner, parser=learner.parser,
                                 perception=learner.perception, wiki=None)
        ContinuousStreamLearner(kernel).ingest_chunk("A falcon is a bird.", source_id="paper:a")
        records = store.evidence.list_by_source("document")
        assert len(records) == 1
        assert records[0].source_reference == "paper:a"


def test_failed_action_rejects_evidence_and_rolls_back_physical_changes():
    from little.knowledge.registry import ActionSchema, SchemaRegistry
    with MemoryStore() as store:
        store.ledger.registry = SchemaRegistry({}, {"probe": ActionSchema("probe", {}, executor="probe")}, {})
        def fail(context, frame, arguments):
            context.create_concept("partial")
            raise RuntimeError("injected worker failure")
        store.ledger.executors.register("probe", fail)
        frame = CandidateFrame.from_text("probe", [], actions=[CandidateAction("probe", {})], intent="action")
        result = LearningEngine(store).commit_frame(frame)
        assert result.update_type.value == "NO_OP"
        assert "injected worker failure" in result.message
        assert store.get_concept("partial") is None
        assert store.evidence.list_by_source("user")[0].status.value == "rejected"


def test_replaced_property_supersedes_previous_evidence():
    with MemoryStore() as store:
        learner = LearningEngine(store)
        learner.learn("My name is Aswin.")
        learner.learn("My name is Priya.")
        assert [record.status.value for record in store.evidence.list_by_source("user")] == ["superseded", "accepted"]
