"""Seed semantic memory from an external, versioned knowledge pack.

The executable seeder is deliberately generic: domain entities, predicates,
and facts belong in ``data/knowledge`` (or an injected pack), not in Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from little.knowledge.bundle import get_commonsense_triples

if TYPE_CHECKING:
    from little.memory.store import MemoryStore


SEED_MARKER = "__knowledge_pack_seeded__"


def seed_commonsense_ontology(
    memory: MemoryStore,
    pack_path: str | Path | None = None,
) -> None:
    """Load and persist the configured knowledge pack exactly once.

    A knowledge record is represented as a concept-to-concept edge. Property
    predicates are kept as graph edges too, so the same loader supports facts
    whose values are concepts, literals, or other typed registry values.
    """
    if memory.get_concept(SEED_MARKER):
        return

    for subject, predicate, object_, _weight, positive in get_commonsense_triples(
        pack_path
    ):
        subject_concept = memory.get_or_create_concept(subject)
        object_concept = memory.get_or_create_concept(object_)
        memory.add_relation(
            subject_id=subject_concept.id,
            predicate=predicate,
            object_id=object_concept.id,
            positive=positive,
        )

    memory.create_concept(SEED_MARKER, category="metadata")
