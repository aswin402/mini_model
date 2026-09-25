"""SQLite persistence for provenance-bearing evidence and proof decisions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from little.core.contracts import (
    CommitDecision,
    EvidenceRecord,
    EvidenceStatus,
    ProofTrace,
    SourceType,
)

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


class EvidenceStore:
    """Persist candidate evidence separately from accepted graph relations."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    @property
    def _conn(self):
        return self.memory._conn

    def append(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO evidence_records (
                    evidence_id, subject_id, predicate, object_id, value_json,
                    source_type, source_reference, source_text, observed_at,
                    valid_from, valid_to, support_weight, opposition_weight,
                    extraction_confidence, polarity, status, derivation_proof_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.evidence_id,
                    record.subject_id,
                    record.predicate.strip().lower(),
                    record.object_id,
                    None,
                    record.source_type.value,
                    record.source_reference,
                    record.source_text,
                    record.observed_at,
                    None,
                    None,
                    0,
                    0,
                    record.extraction_confidence,
                    1 if record.positive else 0,
                    record.status.value,
                    record.derivation_proof_id,
                    record.created_at,
                ),
            )
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        row = self._conn.execute(
            """
            SELECT evidence_id, subject_id, predicate, object_id,
                   source_type, source_reference, source_text,
                   extraction_confidence, polarity, status, derivation_proof_id,
                   observed_at, created_at
            FROM evidence_records
            WHERE evidence_id = ?;
            """,
            (evidence_id,),
        ).fetchone()
        return self._row_to_record(row) if row else None

    def list_for_claim(
        self, subject_id: str, predicate: str, object_id: str
    ) -> list[EvidenceRecord]:
        rows = self._conn.execute(
            """
            SELECT evidence_id, subject_id, predicate, object_id,
                   source_type, source_reference, source_text,
                   extraction_confidence, polarity, status, derivation_proof_id,
                   observed_at, created_at
            FROM evidence_records
            WHERE subject_id = ? AND predicate = ? AND object_id = ?
            ORDER BY observed_at ASC;
            """,
            (subject_id, predicate.strip().lower(), object_id),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_by_source(self, source_type: str) -> list[EvidenceRecord]:
        rows = self._conn.execute(
            """
            SELECT evidence_id, subject_id, predicate, object_id,
                   source_type, source_reference, source_text,
                   extraction_confidence, polarity, status, derivation_proof_id,
                   observed_at, created_at
            FROM evidence_records
            WHERE source_type = ?
            ORDER BY created_at ASC;
            """,
            (source_type,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_status(self, evidence_id: str, status: EvidenceStatus) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE evidence_records SET status = ? WHERE evidence_id = ?;",
                (status.value, evidence_id),
            )

    def save_proof(self, proof: ProofTrace) -> str:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO proof_traces (
                    proof_id, conclusion_json, premises_json, operations_json,
                    tool_outputs_json, verifier_results_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'));
                """,
                (
                    proof.proof_id,
                    json.dumps(proof.conclusion),
                    json.dumps(proof.premises),
                    json.dumps(proof.operations),
                    json.dumps(proof.tool_outputs),
                    json.dumps(proof.verifier_results),
                    proof.status.value,
                ),
            )
        return proof.proof_id

    def save_decision(self, decision: CommitDecision) -> str:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO commit_decisions (
                    decision_id, evidence_id, status, checks_json, reasons_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                (
                    decision.decision_id,
                    decision.evidence_id,
                    decision.status.value,
                    json.dumps(decision.checks),
                    json.dumps(decision.reasons),
                    decision.created_at,
                ),
            )
        return decision.decision_id

    def replace_decision(self, decision: CommitDecision) -> str:
        """Replace the current decision for evidence while preserving one row."""
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE commit_decisions
                SET status = ?, checks_json = ?, reasons_json = ?, created_at = ?
                WHERE evidence_id = ?;
                """,
                (
                    decision.status.value,
                    json.dumps(decision.checks),
                    json.dumps(decision.reasons),
                    decision.created_at,
                    decision.evidence_id,
                ),
            )
        if cursor.rowcount:
            row = self._conn.execute(
                "SELECT decision_id FROM commit_decisions WHERE evidence_id = ? LIMIT 1;",
                (decision.evidence_id,),
            ).fetchone()
            return row["decision_id"]
        return self.save_decision(decision)

    @staticmethod
    def _row_to_record(row) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=row["evidence_id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            source_type=SourceType(row["source_type"]),
            source_reference=row["source_reference"],
            source_text=row["source_text"],
            extraction_confidence=row["extraction_confidence"],
            positive=bool(row["polarity"]),
            status=EvidenceStatus(row["status"]),
            derivation_proof_id=row["derivation_proof_id"],
            observed_at=row["observed_at"],
            created_at=row["created_at"],
        )
