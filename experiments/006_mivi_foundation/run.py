"""Run the MIVI foundation gate against a temporary SQLite database."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from little.core.contracts import (
    CandidateClaim,
    CandidateFrame,
    SourceType,
    VerificationResult,
    VerificationStatus,
)
from little.knowledge.registry import SchemaRegistry
from little.language.parser import LearningEngine
from little.memory.store import MemoryStore


def run() -> dict[str, object]:
    schema_dir = Path(__file__).resolve().parents[2] / "data" / "schemas"
    registry = SchemaRegistry.load(schema_dir)

    with tempfile.TemporaryDirectory(prefix="mivi-foundation-") as temp_dir:
        db_path = Path(temp_dir) / "foundation.db"
        store = MemoryStore(db_path, seed_ontology=False)
        try:
            action_results = [
                registry.validate_action(
                    "split", {"object": "entity", "count": "quantity"}
                ).valid
                for _material in ("organic", "grain", "metal")
            ]

            frame = CandidateFrame.from_text(
                "A falcon has an invented relation to a bird.",
                claims=[
                    CandidateClaim(
                        "falcon", "invented_predicate", "bird", probability=0.99
                    )
                ],
            )
            evidence = store.ledger.propose_relation(
                frame,
                frame.claims[0],
                source_type=SourceType.USER,
                source_reference="experiment:006:unknown",
            )
            unknown_decision = store.ledger.commit_relation(
                evidence,
                VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    checks={"schema": False},
                    reasons=["unregistered predicate"],
                ),
            )

            LearningEngine(store).learn("A falcon is a bird.")
            accepted = [
                record
                for record in store.evidence.list_by_source("user")
                if record.status.value == "accepted"
            ]

            return {
                "temporary_database": str(db_path),
                "split_schema_valid_for_materials": all(action_results),
                "unknown_commit_status": unknown_decision.status.value,
                "accepted_relations": len(accepted),
                "accepted_provenance_complete": all(
                    record.source_text and record.source_reference
                    for record in accepted
                ),
                "accepted_graph_edges": store.count_relations(),
            }
        finally:
            store.close()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
