"""Physical entity transformations and topological part decomposition for LITTLE.

Implements the user's core vision: When an object (e.g. apple) is cut into pieces,
the cognitive system automatically decomposes the entity, establishes part_of relations,
inherits interior properties, and initializes exposed physical dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from little.core.models import Entity
from little.dynamics.cfc import ContinuousDynamicsEngine, PhysicalState
from little.memory.store import MemoryStore


@dataclass
class SlicingResult:
    """Result of slicing an entity or concept into parts."""

    parent_concept: str
    slice_concept: str
    num_pieces: int
    entities_created: list[Entity]
    relations_created: list[str]
    initial_state: PhysicalState
    message: str


class TransformationEngine:
    """Performs topological modifications and actions on concepts and physical entities."""

    @classmethod
    def slice_object(
        cls,
        memory: MemoryStore,
        object_name: str,
        num_pieces: int = 4,
    ) -> SlicingResult:
        """Perform the slicing action: decompose an object into pieces with part_of relations."""
        clean_name = object_name.strip().lower()
        parent_concept = memory.get_concept(clean_name)
        if not parent_concept:
            # If the concept was not yet known, register it as an entity/concept first
            parent_concept = memory.create_concept(name=clean_name)

        slice_concept_name = f"{clean_name} slice"
        slice_concept = memory.get_concept(slice_concept_name)
        if not slice_concept:
            # Slices inherit interior flesh attributes
            interior_color = parent_concept.attributes.get("interior_color", "white")
            slice_attrs: dict[str, Any] = {
                "interior_color": interior_color,
                "exposed_flesh": True,
                "shape": "wedge",
                "part_of": clean_name,
            }
            slice_concept = memory.create_concept(
                name=slice_concept_name,
                category=parent_concept.category or "food",
                attributes=slice_attrs,
            )

        # Establish explicit part_of relation in semantic knowledge graph
        memory.add_relation(
            subject_id=slice_concept.id,
            predicate="part_of",
            object_id=parent_concept.id,
            positive=True,
        )
        relations_created = [f"({slice_concept.name} part_of {parent_concept.name})"]

        # Also add relation: apple slice is_a fruit part (or food)
        memory.add_relation(
            subject_id=slice_concept.id,
            predicate="is_a",
            object_id=parent_concept.id,
            positive=True,
        )
        relations_created.append(f"({slice_concept.name} is_a {parent_concept.name})")

        # Create physical initial state: exposed to air -> low tau -> rapid oxidation
        initial_state = ContinuousDynamicsEngine.create_initial_state(exposed_to_air=True)

        # Instantiate physical slice entities
        entities_created: list[Entity] = []
        for i in range(1, num_pieces + 1):
            entity_name = f"{clean_name}_slice_{i}"
            entity = memory.create_entity(
                name=entity_name,
                concept_id=slice_concept.id,
                properties={
                    "piece_number": i,
                    "of_total": num_pieces,
                    "physical_state": initial_state.to_dict(),
                },
            )
            entities_created.append(entity)

        # Log episodic experience of the slicing event
        memory.add_experience(
            input_text=f"Sliced {clean_name} into {num_pieces} pieces.",
            extracted_triples=[
                {"subject": slice_concept.name, "predicate": "part_of", "object": parent_concept.name},
                {"subject": slice_concept.name, "predicate": "is_a", "object": parent_concept.name},
            ],
            source="action_execution",
        )

        return SlicingResult(
            parent_concept=parent_concept.name,
            slice_concept=slice_concept.name,
            num_pieces=num_pieces,
            entities_created=entities_created,
            relations_created=relations_created,
            initial_state=initial_state,
            message=(
                f"Successfully sliced {clean_name} into {num_pieces} pieces. "
                f"Established 'part_of' and 'is_a' relations with parent '{parent_concept.name}'. "
                f"Initial state: fresh, flesh exposed to air (tau = {initial_state.tau_seconds:.0f}s)."
            ),
        )
